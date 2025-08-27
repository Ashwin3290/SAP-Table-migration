import os
import logging
import sqlite3
import re
from typing import Optional, Tuple, Dict, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class SourceTableManager:
    """
    Self-contained manager for source table operations
    """
    DATA_PRESERVING_PATTERNS = [
        r'INSERT\s+INTO\s+\w+\s*\([^)]+\)\s*SELECT\s+.*FROM\s+\w+',
        r'UPDATE\s+\w+\s+SET\s+.*=\s*\(\s*SELECT\s+.*FROM\s+\w+',
        r'UPDATE\s+\w+\s+SET\s+.*FROM\s+.*JOIN\s+',
        r'UPDATE\s+\w+\s+SET\s+\w+\s*=\s*\w+\.\w+\s+FROM\s+',
    ]
    DATA_MODIFYING_PATTERNS = [
        r'(substr|length|trim|ltrim|rtrim|upper|lower|replace)\s*\(',
        r'regexp_replace\s*\(',
        r'[\+\-\*\/]\s*\d',
        r'CASE\s+WHEN\s+.*THEN\s+[\'"][^\'"]',
        r'SET\s+\w+\s*=\s*[\'"][^\'"]',
        r'(date|datetime|strftime)\s*\(',
        r'(safe_divide|percentage|format_date|proper_case)\s*\(',
    ]
    SCHEMA_PATTERNS = [
        r'ALTER\s+TABLE\s+\w+\s+ADD\s+COLUMN',
        r'ALTER\s+TABLE\s+\w+\s+DROP\s+COLUMN',
        r'CREATE\s+TABLE',
        r'DROP\s+TABLE',
    ]
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize Source Table Manager
        
        Args:
            db_path (str, optional): Path to SQLite database
        """
        self.db_path = db_path or os.environ.get('DB_PATH')
        self.enabled = os.environ.get('ENABLE_SOURCE_TABLE_BACKUP', 'true').lower() == 'true'
    
    def should_sync_to_source(self, query: str, target_table: str) -> Tuple[bool, str]:
        """
        Determine if a query should be synchronized to the _src table
        
        Args:
            query (str): SQL query to analyze
            target_table (str): Name of the target table
            
        Returns:
            Tuple[bool, str]: (should_sync, reason)
        """
        if not self.enabled:
            return False, "Source table sync disabled"
        
        if not query or not target_table:
            return False, "Missing query or target table"
        
        query_upper = query.upper().strip()
        if not self._is_target_table_operation(query_upper, target_table):
            return False, "Not a target table operation"
        for pattern in self.DATA_MODIFYING_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Data-modifying operation detected: {pattern}"
        for pattern in self.DATA_PRESERVING_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True, f"Data-preserving operation: {pattern}"
        for pattern in self.SCHEMA_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True, f"Schema operation: {pattern}"
        if self._is_simple_insert_or_update(query_upper):
            return True, "Simple INSERT/UPDATE operation"
        
        return False, "Query pattern not recognized as safe for source sync"
    
    def _is_target_table_operation(self, query_upper: str, target_table: str) -> bool:
        """Check if query operates on the target table"""
        target_upper = target_table.upper()
        if f'INSERT INTO {target_upper}' in query_upper:
            return True
        if f'UPDATE {target_upper}' in query_upper:
            return True
        if f'ALTER TABLE {target_upper}' in query_upper:
            return True
        
        return False
    
    def _is_simple_insert_or_update(self, query_upper: str) -> bool:
        """Check if it's a simple INSERT or UPDATE without complex transformations"""
        if 'INSERT INTO' in query_upper and 'SELECT' in query_upper:
            if not re.search(r'\w+\s*\(', query_upper):
                return True
        if 'UPDATE' in query_upper and 'SET' in query_upper:
            if '=' in query_upper and not re.search(r'CASE\s+WHEN', query_upper):
                return True
        
        return False
    
    def ensure_src_table_exists(self, target_table: str) -> bool:
        """
        Ensure the _src table exists for the given target table
        
        Args:
            target_table (str): Name of the target table
            
        Returns:
            bool: True if _src table exists or was created successfully
        """
        if not self.enabled or not self.db_path:
            return False
        
        src_table = f"{target_table}_src"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (src_table,))
            
            if cursor.fetchone():
                conn.close()
                return True
            cursor.execute(f"CREATE TABLE {src_table} AS SELECT * FROM {target_table}")
            conn.commit()
            conn.close()
            
            logger.info(f"Created source table: {src_table}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error creating source table {src_table}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating source table {src_table}: {e}")
            return False
    
    def execute_on_src_table(self, query: str, target_table: str, params: Optional[Dict] = None) -> bool:
        """
        Execute the query on the _src table
        
        Args:
            query (str): SQL query to execute
            target_table (str): Original target table name
            params (dict, optional): Query parameters
            
        Returns:
            bool: True if execution was successful
        """
        if not self.enabled or not self.db_path:
            return False
        
        src_table = f"{target_table}_src"
        src_query = query.replace(target_table, src_table, 1)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if params:
                cursor.execute(src_query, params)
            else:
                cursor.execute(src_query)
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Successfully executed query on source table {src_table}")
            return True
            
        except sqlite3.Error as e:
            logger.warning(f"Failed to execute query on source table {src_table}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error executing on source table {src_table}: {e}")
            return False
    
    def handle_source_table_sync(self, query: str, target_table: str, 
                                 params: Optional[Dict] = None, 
                                 main_execution_successful: bool = True) -> Dict[str, Any]:
        """
        Main function to handle source table synchronization
        This is the single function to call from execute_query
        
        Args:
            query (str): The SQL query that was executed
            target_table (str): Name of the target table  
            params (dict, optional): Query parameters used
            main_execution_successful (bool): Whether the main query execution succeeded
            
        Returns:
            Dict[str, Any]: Result information about the sync operation
        """
        result = {
            "sync_attempted": False,
            "sync_successful": False,
            "should_sync": False,
            "reason": "",
            "src_table_exists": False,
            "enabled": self.enabled
        }
        if not self.enabled:
            result["reason"] = "Source table sync disabled"
            return result
        
        if not main_execution_successful:
            result["reason"] = "Main execution failed"
            return result
        should_sync, reason = self.should_sync_to_source(query, target_table)
        result["should_sync"] = should_sync
        result["reason"] = reason
        
        if not should_sync:
            return result
        src_exists = self.ensure_src_table_exists(target_table)
        result["src_table_exists"] = src_exists
        
        if not src_exists:
            result["reason"] = f"Could not create/access _src table for {target_table}"
            return result
        result["sync_attempted"] = True
        sync_success = self.execute_on_src_table(query, target_table, params)
        result["sync_successful"] = sync_success
        
        if sync_success:
            result["reason"] = f"Successfully synced to {target_table}_src"
        else:
            result["reason"] = f"Failed to sync to {target_table}_src"
        
        return result

# Global instance for easy access
_source_manager = None

def get_source_manager() -> SourceTableManager:
    """Get global source table manager instance"""
    global _source_manager
    if _source_manager is None:
        _source_manager = SourceTableManager()
    return _source_manager

def handle_source_sync(query: str, target_table: str, 
                      params: Optional[Dict] = None,
                      main_execution_successful: bool = True) -> Dict[str, Any]:
    """
    Convenience function for handling source table sync
    This is the function to call from execute_query
    
    Args:
        query (str): The SQL query that was executed
        target_table (str): Name of the target table
        params (dict, optional): Query parameters
        main_execution_successful (bool): Whether main execution succeeded
        
    Returns:
        Dict[str, Any]: Sync operation results
    """
    manager = get_source_manager()
    return manager.handle_source_table_sync(
        query, target_table, params, main_execution_successful
    )