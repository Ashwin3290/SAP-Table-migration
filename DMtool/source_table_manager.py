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
    Manager for source table (_src) operations with inverted logic:
    Sync everything EXCEPT transformations/calculations
    """
    
    # Patterns that indicate data transformation/modification - DO NOT SYNC these
    TRANSFORMATION_PATTERNS = [
        # String manipulations that change data
        r'\b(substr|substring|trim|ltrim|rtrim|upper|lower|replace|reverse)\s*\(',
        r'regexp_replace\s*\(',
        r'split_string\s*\(',
        r'proper_case\s*\(',
        r'left_pad\s*\(',
        r'right_pad\s*\(',
        
        # Mathematical operations (except simple column references)
        r'[\+\-\*\/\%]\s*(?:\d+|[\'"][^\'"]+[\'"])',  # Math with literals
        r'\b(round|abs|ceil|floor|sqrt|power|log)\s*\(',
        r'safe_divide\s*\(',
        r'percentage\s*\(',
        
        # Date/time manipulations that transform data
        r'date\s*\(\s*[\'"]now[\'\"]\s*\)',  # Current date
        r'datetime\s*\(\s*[\'"]now[\'\"]\s*\)',
        r'strftime\s*\(',  # Date formatting
        r'date_add_days\s*\(',
        r'date_diff_days\s*\(',
        r'format_date\s*\(',
        r'to_date\s*\(',
        
        # CASE statements with hardcoded values (business logic)
        r'CASE\s+WHEN\s+.*?\s+THEN\s+[\'"][^\'"]+[\'"]',  # CASE with literal strings
        r'CASE\s+WHEN\s+.*?\s+THEN\s+\d+',  # CASE with literal numbers
        
        # Hardcoded value assignments (not from another table)
        r'SET\s+\w+\s*=\s*[\'"][^\'"]+[\'"](?!\s*FROM)',  # String literal not from subquery
        r'SET\s+\w+\s*=\s*\d+(?:\.\d+)?(?!\s*FROM)',  # Number literal not from subquery
        r'SET\s+\w+\s*=\s*NULL',  # Setting to NULL (data deletion)
        
        # Calculated/derived values
        r'length\s*\(',  # String length calculation
        r'count\s*\(',  # Aggregations
        r'sum\s*\(',
        r'avg\s*\(',
        r'min\s*\(',
        r'max\s*\(',
        
        # Conditional transformations
        r'COALESCE\s*\([^,]+,[^)]*[\'"][^\'"]+[\'"][^)]*\)',  # COALESCE with literals
        r'IFNULL\s*\([^,]+,\s*[\'"][^\'"]+[\'\"]\s*\)',  # IFNULL with literals
        r'IIF\s*\(',  # IIF function
        
        # JSON operations
        r'json_extract_value\s*\(',
        
        # Validation functions (these compute new values)
        r'is_numeric\s*\(',
        r'is_email\s*\(',
        r'is_phone\s*\(',
        r'is_valid_json\s*\(',
    ]
    
    # Operations that should NEVER sync (even if they don't transform data)
    NEVER_SYNC_PATTERNS = [
        r'DROP\s+TABLE',
        r'TRUNCATE\s+TABLE',
        r'CREATE\s+(?:TEMP|TEMPORARY)\s+',  # Temporary objects
        r'CREATE\s+VIEW',  # Views
        r'CREATE\s+INDEX',  # Indexes
    ]
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize Source Table Manager"""
        self.db_path = db_path or os.environ.get('DB_PATH')
        self.enabled = os.environ.get('ENABLE_SOURCE_TABLE_BACKUP', 'true').lower() == 'true'
        
        logger.info(f"SourceTableManager initialized - Enabled: {self.enabled}, DB Path: {self.db_path}")
        
        self.stats = {
            'sync_attempts': 0,
            'sync_successes': 0,
            'sync_failures': 0,
            'tables_created': 0,
            'last_sync': None
        }
    
    def should_sync_to_source(self, query: str, target_table: str) -> Tuple[bool, str]:
        """
        Determine if a query should be synchronized to the _src table.
        New logic: Sync everything EXCEPT transformations
        """
        if not self.enabled:
            return False, "Source table sync disabled"
        
        if not query or not target_table:
            return False, "Missing query or target table"
        
        query_upper = query.upper().strip()
        
        # Check if this operates on the target table
        if not self._is_target_table_operation(query_upper, target_table):
            return False, "Not a target table operation"
        
        # NEVER sync these operations
        for pattern in self.NEVER_SYNC_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.info(f"Never-sync pattern detected: {pattern[:30]}...")
                return False, f"Operation type not suitable for sync"
        
        # Check for transformation patterns - if found, DON'T sync
        for pattern in self.TRANSFORMATION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.info(f"Transformation pattern detected: {pattern[:50]}...")
                return False, f"Data transformation detected - preserving original data"
        
        # Special checks for complex cases
        if self._has_complex_transformation(query):
            return False, "Complex transformation detected"
        
        # If we get here, it's likely a simple data movement operation - SYNC IT
        query_type = self._identify_query_type(query_upper)
        logger.info(f"Query type '{query_type}' approved for sync to {target_table}_src")
        return True, f"Data preservation operation - {query_type}"
    
    def _has_complex_transformation(self, query: str) -> bool:
        """
        Additional checks for complex transformations that might not be caught by patterns
        """
        query_upper = query.upper()
        
        # Check for UPDATE with complex SET clause
        if 'UPDATE' in query_upper and 'SET' in query_upper:
            set_match = re.search(r'SET\s+(.*?)(?:WHERE|FROM|$)', query_upper, re.DOTALL)
            if set_match:
                set_clause = set_match.group(1)
                
                # If SET clause has any function calls (except simple column references)
                if re.search(r'\w+\s*\([^)]*\)', set_clause):
                    # Check if it's just a simple subquery
                    if not re.search(r'^\s*\(\s*SELECT\s+\w+\s+FROM\s+\w+', set_clause):
                        return True
                
                # Multiple CASE statements usually indicate complex business logic
                if set_clause.count('CASE') > 1:
                    return True
        
        # Check for INSERT with complex SELECT
        if 'INSERT' in query_upper and 'SELECT' in query_upper:
            select_match = re.search(r'SELECT\s+(.*?)(?:FROM|$)', query_upper, re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                
                # If SELECT has calculations or functions (not just column names)
                if re.search(r'[\+\-\*\/]', select_clause):
                    return True
                if re.search(r'\w+\s*\([^)]*\)', select_clause):
                    # Allow simple column selections from subqueries
                    if not re.search(r'^\s*\w+\s*,?\s*$', select_clause):
                        return True
        
        return False
    
    def _identify_query_type(self, query_upper: str) -> str:
        """Identify the type of query for logging purposes"""
        if 'INSERT' in query_upper and 'SELECT' in query_upper:
            return "INSERT-SELECT"
        elif 'INSERT' in query_upper:
            return "INSERT"
        elif 'UPDATE' in query_upper and 'FROM' in query_upper:
            return "UPDATE-FROM"
        elif 'UPDATE' in query_upper:
            return "UPDATE"
        elif 'ALTER TABLE' in query_upper and 'ADD COLUMN' in query_upper:
            return "ALTER-ADD"
        elif 'ALTER TABLE' in query_upper and 'DROP COLUMN' in query_upper:
            return "ALTER-DROP"
        elif 'DELETE' in query_upper:
            return "DELETE"
        else:
            return "OTHER"
    
    def _is_target_table_operation(self, query_upper: str, target_table: str) -> bool:
        """Check if query operates on the target table"""
        target_upper = target_table.upper()
        
        # Use word boundary for more accurate matching
        patterns = [
            rf'\bINSERT\s+(?:OR\s+\w+\s+)?INTO\s+\[?{re.escape(target_upper)}\]?\b',
            rf'\bUPDATE\s+\[?{re.escape(target_upper)}\]?\b',
            rf'\bALTER\s+TABLE\s+\[?{re.escape(target_upper)}\]?\b',
            rf'\bDELETE\s+FROM\s+\[?{re.escape(target_upper)}\]?\b',
        ]
        
        return any(re.search(pattern, query_upper) for pattern in patterns)
    
    def get_table_schema(self, table_name: str, conn: sqlite3.Connection) -> str:
        """Get the complete CREATE TABLE statement for a table"""
        try:
            cursor = conn.cursor()
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
        """Replicate the exact schema from source table to target table"""
        try:
            logger.info(f"Replicating schema from {source_table} to {target_table}")
            
            create_sql = self.get_table_schema(source_table, conn)
            if not create_sql:
                logger.error(f"Could not get schema for {source_table}")
                return False
            
            # Replace table name in CREATE statement
            create_sql = re.sub(
                r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["\']?\w+["\']?',
                f'CREATE TABLE IF NOT EXISTS {target_table}',
                create_sql,
                count=1,
                flags=re.IGNORECASE
            )
            
            cursor = conn.cursor()
            cursor.execute(create_sql)
            
            # Copy indexes
            cursor.execute("""
                SELECT sql FROM sqlite_master 
                WHERE type='index' AND tbl_name=? AND sql IS NOT NULL
            """, (source_table,))
            
            indexes = cursor.fetchall()
            for index_sql in indexes:
                if index_sql[0]:
                    new_index_sql = index_sql[0].replace(source_table, target_table)
                    new_index_sql = re.sub(
                        r'CREATE\s+(UNIQUE\s+)?INDEX\s+(\w+)',
                        lambda m: f"CREATE {m.group(1) or ''}INDEX {m.group(2)}_src",
                        new_index_sql
                    )
                    try:
                        cursor.execute(new_index_sql)
                    except sqlite3.Error as e:
                        logger.warning(f"Could not create index for {target_table}: {e}")
            
            conn.commit()
            logger.info(f"Successfully replicated schema for {target_table}")
            return True
            
        except Exception as e:
            logger.error(f"Error replicating schema: {e}")
            return False
    
    def ensure_src_table_exists(self, target_table: str) -> bool:
        """Ensure the _src table exists with the same schema as the target table"""
        if not self.enabled or not self.db_path:
            return False
        
        src_table = f"{target_table}_src"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if source table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (src_table,))
            
            if cursor.fetchone():
                logger.debug(f"Source table {src_table} already exists")
                conn.close()
                return True
            
            logger.info(f"Creating source table {src_table}")
            
            if self.replicate_table_schema(target_table, src_table, conn):
                # Copy existing data
                cursor.execute(f"INSERT INTO {src_table} SELECT * FROM {target_table}")
                conn.commit()
                
                cursor.execute(f"SELECT COUNT(*) FROM {src_table}")
                row_count = cursor.fetchone()[0]
                
                logger.info(f"Created source table {src_table} with {row_count} rows")
                self.stats['tables_created'] += 1
                
                conn.close()
                return True
            else:
                conn.close()
                return False
                
        except Exception as e:
            logger.error(f"Error creating source table {src_table}: {e}")
            return False
    
    def execute_on_src_table(self, query: str, target_table: str, params: Optional[Dict] = None) -> bool:
        """Execute the query on the _src table"""
        if not self.enabled or not self.db_path:
            return False
        
        src_table = f"{target_table}_src"
        
        # Replace table name in query
        src_query = re.sub(
            r'\b' + re.escape(target_table) + r'\b',
            src_table,
            query
        )
        
        logger.debug(f"Executing on source table {src_table}: {src_query[:100]}...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if params:
                cursor.execute(src_query, params)
            else:
                cursor.execute(src_query)
            
            rows_affected = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully executed on source table {src_table}, {rows_affected} rows affected")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute on source table {src_table}: {e}")
            return False
    
    def handle_source_table_sync(self, query: str, target_table: str, 
                                 params: Optional[Dict] = None, 
                                 main_execution_successful: bool = True) -> Dict[str, Any]:
        """Main function to handle source table synchronization"""
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
        
        self.stats['sync_attempts'] += 1
        
        if not self.enabled:
            result["reason"] = "Source table sync disabled"
            return result
        
        if not main_execution_successful:
            result["reason"] = "Main execution failed"
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
            self.stats['sync_failures'] += 1
            return result
        
        # Attempt the sync
        result["sync_attempted"] = True
        sync_success = self.execute_on_src_table(query, target_table, params)
        result["sync_successful"] = sync_success
        
        if sync_success:
            result["reason"] = f"Successfully synced to {target_table}_src"
            self.stats['sync_successes'] += 1
            self.stats['last_sync'] = datetime.now().isoformat()
        else:
            result["reason"] = f"Failed to sync to {target_table}_src"
            self.stats['sync_failures'] += 1
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        result["execution_time_ms"] = round(execution_time, 2)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get source table manager statistics"""
        return self.stats.copy()
    
    def verify_sync_integrity(self, target_table: str) -> Dict[str, Any]:
        """Verify that source and target tables are in sync"""
        if not self.enabled or not self.db_path:
            return {"error": "Source sync disabled or no DB path"}
        
        src_table = f"{target_table}_src"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (src_table,))
            
            if not cursor.fetchone():
                return {"error": f"Source table {src_table} does not exist"}
            
            cursor.execute(f"SELECT COUNT(*) FROM {target_table}")
            target_count = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM {src_table}")
            src_count = cursor.fetchone()[0]
            
            target_schema = self.get_table_schema(target_table, conn)
            src_schema = self.get_table_schema(src_table, conn)
            
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
    """Convenience function for handling source table sync"""
    manager = get_source_manager()
    return manager.handle_source_table_sync(
        query, target_table, params, main_execution_successful
    )