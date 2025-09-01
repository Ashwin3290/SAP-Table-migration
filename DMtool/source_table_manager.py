import os
import logging
import sqlite3
import re
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class SourceTableManager:
    """
    Manager for source table (_src) operations with comprehensive logging
    and proper schema replication
    """
    
    # Patterns that preserve original data
    DATA_PRESERVING_PATTERNS = [
        r'INSERT\s+(?:OR\s+IGNORE\s+)?INTO\s+\w+\s*\([^)]+\)\s*SELECT\s+.*FROM\s+\w+',
        r'UPDATE\s+\w+\s+SET\s+.*=\s*\(\s*SELECT\s+.*FROM\s+\w+',
        r'UPDATE\s+\w+\s+SET\s+.*FROM\s+.*JOIN\s+',
        r'UPDATE\s+\w+\s+SET\s+\w+\s*=\s*\w+\.\w+\s+FROM\s+',
    ]
    
    # Patterns that modify data (should not sync)
    DATA_MODIFYING_PATTERNS = [
        r'(substr|length|trim|ltrim|rtrim|upper|lower|replace)\s*\(',
        r'regexp_replace\s*\(',
        r'[\+\-\*\/]\s*\d',
        r'CASE\s+WHEN\s+.*THEN\s+[\'"][^\'"]',
        r'SET\s+\w+\s*=\s*[\'"][^\'"]',
        r'(date|datetime|strftime)\s*\(',
        r'(safe_divide|percentage|format_date|proper_case)\s*\(',
    ]
    
    # Schema modification patterns
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
        
        logger.info(f"SourceTableManager initialized - Enabled: {self.enabled}, DB Path: {self.db_path}")
        
        # Track statistics
        self.stats = {
            'sync_attempts': 0,
            'sync_successes': 0,
            'sync_failures': 0,
            'tables_created': 0,
            'last_sync': None
        }
    
    def get_table_schema(self, table_name: str, conn: sqlite3.Connection) -> str:
        """
        Get the complete CREATE TABLE statement for a table
        
        Args:
            table_name (str): Name of the table
            conn (sqlite3.Connection): Database connection
            
        Returns:
            str: CREATE TABLE statement
        """
        try:
            cursor = conn.cursor()
            
            # Get the CREATE TABLE statement from sqlite_master
            cursor.execute("""
                SELECT sql FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            
            result = cursor.fetchone()
            if result:
                return result[0]
            
            logger.warning(f"No schema found for table {table_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {e}")
            return None
    
    def replicate_table_schema(self, source_table: str, target_table: str, conn: sqlite3.Connection) -> bool:
        """
        Replicate the exact schema from source table to target table
        
        Args:
            source_table (str): Source table name
            target_table (str): Target table name  
            conn (sqlite3.Connection): Database connection
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Replicating schema from {source_table} to {target_table}")
            
            # Get the original CREATE TABLE statement
            create_sql = self.get_table_schema(source_table, conn)
            if not create_sql:
                logger.error(f"Could not get schema for {source_table}")
                return False
            
            # Replace the table name in the CREATE statement
            create_sql = re.sub(
                r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["\']?' + re.escape(source_table) + r'["\']?',
                f'CREATE TABLE IF NOT EXISTS {target_table}',
                create_sql,
                flags=re.IGNORECASE
            )
            
            cursor = conn.cursor()
            
            # Create the table with the same schema
            cursor.execute(create_sql)
            
            # Copy indexes if any
            cursor.execute("""
                SELECT sql FROM sqlite_master 
                WHERE type='index' AND tbl_name=? AND sql IS NOT NULL
            """, (source_table,))
            
            indexes = cursor.fetchall()
            for index_sql in indexes:
                if index_sql[0]:
                    # Modify index to work with new table
                    new_index_sql = index_sql[0].replace(source_table, target_table)
                    # Make index name unique
                    new_index_sql = re.sub(
                        r'CREATE\s+(UNIQUE\s+)?INDEX\s+(\w+)',
                        lambda m: f"CREATE {m.group(1) or ''}INDEX {m.group(2)}_src",
                        new_index_sql
                    )
                    try:
                        cursor.execute(new_index_sql)
                        logger.debug(f"Created index for {target_table}")
                    except sqlite3.Error as e:
                        logger.warning(f"Could not create index for {target_table}: {e}")
            
            conn.commit()
            logger.info(f"Successfully replicated schema for {target_table}")
            return True
            
        except Exception as e:
            logger.error(f"Error replicating schema from {source_table} to {target_table}: {e}")
            return False
    
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
            logger.debug("Source sync disabled globally")
            return False, "Source table sync disabled"
        
        if not query or not target_table:
            logger.warning(f"Missing query or target table - Query: {bool(query)}, Table: {target_table}")
            return False, "Missing query or target table"
        
        query_upper = query.upper().strip()
        
        # Check if this operates on the target table
        if not self._is_target_table_operation(query_upper, target_table):
            logger.debug(f"Query does not operate on target table {target_table}")
            return False, "Not a target table operation"
        
        # Check for data-modifying patterns (should NOT sync)
        for pattern in self.DATA_MODIFYING_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.info(f"Data-modifying pattern detected: {pattern[:30]}...")
                return False, f"Data-modifying operation detected"
        
        # Check for data-preserving patterns (should sync)
        for pattern in self.DATA_PRESERVING_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.info(f"Data-preserving pattern detected for {target_table}")
                return True, f"Data-preserving operation"
        
        # Check for schema operations (should sync)
        for pattern in self.SCHEMA_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.info(f"Schema operation detected for {target_table}")
                return True, f"Schema operation"
        
        # Check for simple INSERT/UPDATE
        if self._is_simple_insert_or_update(query_upper):
            logger.info(f"Simple INSERT/UPDATE detected for {target_table}")
            return True, "Simple INSERT/UPDATE operation"
        
        logger.debug(f"Query pattern not recognized for sync: {query[:100]}...")
        return False, "Query pattern not recognized as safe for source sync"
    
    def _is_target_table_operation(self, query_upper: str, target_table: str) -> bool:
        """Check if query operates on the target table"""
        target_upper = target_table.upper()
        
        patterns = [
            f'INSERT INTO {target_upper}',
            f'UPDATE {target_upper}',
            f'ALTER TABLE {target_upper}',
            f'DELETE FROM {target_upper}',
            f'TRUNCATE TABLE {target_upper}',
        ]
        
        return any(pattern in query_upper for pattern in patterns)
    
    def _is_simple_insert_or_update(self, query_upper: str) -> bool:
        """Check if it's a simple INSERT or UPDATE without complex transformations"""
        if 'INSERT INTO' in query_upper and 'SELECT' in query_upper:
            # Check if SELECT has no function calls
            if not re.search(r'\w+\s*\(', query_upper):
                return True
        
        if 'UPDATE' in query_upper and 'SET' in query_upper:
            # Simple UPDATE without CASE statements
            if '=' in query_upper and not re.search(r'CASE\s+WHEN', query_upper):
                return True
        
        return False
    
    def ensure_src_table_exists(self, target_table: str) -> bool:
        """
        Ensure the _src table exists with the same schema as the target table
        
        Args:
            target_table (str): Name of the target table
            
        Returns:
            bool: True if _src table exists or was created successfully
        """
        if not self.enabled or not self.db_path:
            logger.debug(f"Source table creation skipped - Enabled: {self.enabled}, DB Path: {bool(self.db_path)}")
            return False
        
        src_table = f"{target_table}_src"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if source table already exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (src_table,))
            
            if cursor.fetchone():
                logger.debug(f"Source table {src_table} already exists")
                conn.close()
                return True
            
            logger.info(f"Creating source table {src_table}")
            
            # Replicate the schema exactly
            if self.replicate_table_schema(target_table, src_table, conn):
                # Copy existing data if any
                cursor.execute(f"INSERT INTO {src_table} SELECT * FROM {target_table}")
                conn.commit()
                
                # Get row count for logging
                cursor.execute(f"SELECT COUNT(*) FROM {src_table}")
                row_count = cursor.fetchone()[0]
                
                logger.info(f"Created source table {src_table} with {row_count} rows")
                self.stats['tables_created'] += 1
                
                conn.close()
                return True
            else:
                logger.error(f"Failed to replicate schema for {src_table}")
                conn.close()
                return False
                
        except sqlite3.Error as e:
            logger.error(f"SQLite error creating source table {src_table}: {e}")
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
        
        # Use regex for more accurate table name replacement
        src_query = re.sub(
            r'\b' + re.escape(target_table) + r'\b',
            src_table,
            query,
            count=1
        )
        
        logger.debug(f"Executing on source table {src_table}: {src_query[:100]}...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Execute the query
            if params:
                cursor.execute(src_query, params)
            else:
                cursor.execute(src_query)
            
            rows_affected = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully executed on source table {src_table}, {rows_affected} rows affected")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Failed to execute on source table {src_table}: {e}")
            logger.debug(f"Failed query: {src_query}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error executing on source table {src_table}: {e}")
            return False
    
    def handle_source_table_sync(self, query: str, target_table: str, 
                                 params: Optional[Dict] = None, 
                                 main_execution_successful: bool = True) -> Dict[str, Any]:
        """
        Main function to handle source table synchronization with comprehensive logging
        
        Args:
            query (str): The SQL query that was executed
            target_table (str): Name of the target table  
            params (dict, optional): Query parameters used
            main_execution_successful (bool): Whether the main query execution succeeded
            
        Returns:
            Dict[str, Any]: Detailed result information about the sync operation
        """
        start_time = datetime.now()
        
        result = {
            "sync_attempted": False,
            "sync_successful": False,
            "should_sync": False,
            "reason": "",
            "src_table_exists": False,
            "enabled": self.enabled,
            "target_table": target_table,
            "execution_time_ms": 0
        }
        
        # Update statistics
        self.stats['sync_attempts'] += 1
        
        if not self.enabled:
            result["reason"] = "Source table sync disabled"
            logger.debug(f"Source sync skipped for {target_table}: disabled")
            return result
        
        if not main_execution_successful:
            result["reason"] = "Main execution failed"
            logger.warning(f"Source sync skipped for {target_table}: main execution failed")
            return result
        
        # Check if we should sync
        should_sync, reason = self.should_sync_to_source(query, target_table)
        result["should_sync"] = should_sync
        result["reason"] = reason
        
        if not should_sync:
            logger.info(f"Source sync not needed for {target_table}: {reason}")
            return result
        
        logger.info(f"Starting source sync for {target_table}")
        
        # Ensure source table exists
        src_exists = self.ensure_src_table_exists(target_table)
        result["src_table_exists"] = src_exists
        
        if not src_exists:
            result["reason"] = f"Could not create/access _src table for {target_table}"
            logger.error(f"Source sync failed for {target_table}: could not create source table")
            self.stats['sync_failures'] += 1
            return result
        
        # Attempt the sync
        result["sync_attempted"] = True
        sync_success = self.execute_on_src_table(query, target_table, params)
        result["sync_successful"] = sync_success
        
        if sync_success:
            result["reason"] = f"Successfully synced to {target_table}_src"
            logger.info(f"Source sync successful for {target_table}")
            self.stats['sync_successes'] += 1
            self.stats['last_sync'] = datetime.now().isoformat()
        else:
            result["reason"] = f"Failed to sync to {target_table}_src"
            logger.error(f"Source sync failed for {target_table}")
            self.stats['sync_failures'] += 1
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        result["execution_time_ms"] = round(execution_time, 2)
        
        logger.debug(f"Source sync completed in {execution_time:.2f}ms for {target_table}")
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get source table manager statistics"""
        return self.stats.copy()
    
    def verify_sync_integrity(self, target_table: str) -> Dict[str, Any]:
        """
        Verify that source and target tables are in sync
        
        Args:
            target_table (str): Target table name
            
        Returns:
            Dict with comparison results
        """
        if not self.enabled or not self.db_path:
            return {"error": "Source sync disabled or no DB path"}
        
        src_table = f"{target_table}_src"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if source table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (src_table,))
            
            if not cursor.fetchone():
                return {"error": f"Source table {src_table} does not exist"}
            
            # Compare row counts
            cursor.execute(f"SELECT COUNT(*) FROM {target_table}")
            target_count = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM {src_table}")
            src_count = cursor.fetchone()[0]
            
            # Compare schemas
            target_schema = self.get_table_schema(target_table, conn)
            src_schema = self.get_table_schema(src_table, conn)
            
            # Normalize schemas for comparison (remove table name differences)
            target_schema_normalized = re.sub(r'\b' + re.escape(target_table) + r'\b', 'TABLE', target_schema or '')
            src_schema_normalized = re.sub(r'\b' + re.escape(src_table) + r'\b', 'TABLE', src_schema or '')
            
            schemas_match = target_schema_normalized == src_schema_normalized
            
            conn.close()
            
            result = {
                "target_table": target_table,
                "source_table": src_table,
                "target_row_count": target_count,
                "source_row_count": src_count,
                "row_counts_match": target_count == src_count,
                "schemas_match": schemas_match,
                "in_sync": (target_count == src_count) and schemas_match
            }
            
            if not result["in_sync"]:
                logger.warning(f"Tables not in sync: {target_table} vs {src_table}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error verifying sync integrity: {e}")
            return {"error": str(e)}

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