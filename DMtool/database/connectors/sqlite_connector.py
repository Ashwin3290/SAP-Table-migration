import sqlite3
import re
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from DMtool.database.base.base_connector import BaseConnector, ConnectionException, QueryException
from DMtool.enums.database_types import DatabaseType

logger = logging.getLogger(__name__)

class SQLiteConnector(BaseConnector):
    """SQLite database connector implementation"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params)
        self.database_type = DatabaseType.SQLITE
        self.db_path = connection_params.get('database_path', ':memory:')
        self._transaction_active = False
        
    def connect(self) -> bool:
        """Establish SQLite connection"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            self.is_connected = True
            self._setup_sqlite_functions()
            logger.info(f"Connected to SQLite database: {self.db_path}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise ConnectionException(f"SQLite connection failed: {e}")
            
    def disconnect(self) -> bool:
        """Close SQLite connection"""
        try:
            if self.connection:
                self.connection.close()
                self.is_connected = False
                logger.info("SQLite connection closed")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error closing SQLite connection: {e}")
            return False
            
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute SQLite query"""
        if not self.is_connected:
            raise ConnectionException("Not connected to database")
            
        try:
            cursor = self.connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            # Auto-commit if not in transaction
            if not self._transaction_active:
                self.connection.commit()
                
            # Return appropriate result based on query type
            if query.strip().upper().startswith(('SELECT', 'WITH')):
                return cursor.fetchall()
            else:
                return cursor
                
        except sqlite3.Error as e:
            logger.error(f"SQLite query execution failed: {e}")
            raise QueryException(f"Query failed: {e}")
            
    def execute_batch(self, queries: List[str], params: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
        """Execute multiple SQLite queries"""
        results = []
        
        for i, query in enumerate(queries):
            query_params = params[i] if params and i < len(params) else None
            result = self.execute_query(query, query_params)
            results.append(result)
            
        return results
        
    def get_tables(self) -> List[str]:
        """Get list of all tables in SQLite database"""
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        result = self.execute_query(query)
        return [row[0] for row in result]
        
    def get_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Get columns for SQLite table"""
        safe_table = self.validate_identifier(table_name)
        query = f"PRAGMA table_info({safe_table})"
        result = self.execute_query(query)
        
        columns = []
        for row in result:
            columns.append({
                'name': row[1],
                'type': row[2],
                'nullable': not row[3],
                'default': row[4],
                'primary_key': bool(row[5])
            })
        return columns
        
    def get_schema(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive schema information"""
        columns = self.get_columns(table_name)
        
        # Get table creation SQL
        query = "SELECT sql FROM sqlite_master WHERE type='table' AND name=?"
        result = self.execute_query(query, {'name': table_name})
        
        return {
            'table_name': table_name,
            'columns': columns,
            'creation_sql': result[0][0] if result else None,
            'row_count': self._get_row_count(table_name)
        }
        
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in SQLite"""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.execute_query(query, {'name': table_name})
        return len(result) > 0
        
    def validate_identifier(self, identifier: str) -> str:
        """Validate SQLite identifier"""
        if not identifier:
            raise ValueError("Empty identifier")
            
        # SQLite identifier validation
        dangerous_patterns = [';', '--', '/*', '*/', 'DROP', 'DELETE', 'TRUNCATE']
        for pattern in dangerous_patterns:
            if pattern.lower() in identifier.lower():
                raise ValueError(f"Dangerous pattern found: {pattern}")
                
        # SQLite allows letters, digits, underscore, and dollar sign
        if not re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', identifier):
            # If contains special chars, wrap in square brackets
            return f"[{identifier}]"
            
        return identifier
        
    def escape_string(self, value: str) -> str:
        """Escape string value for SQLite"""
        if value is None:
            return 'NULL'
        return f"'{value.replace(chr(39), chr(39)+chr(39))}'"
        
    def get_dialect_info(self) -> Dict[str, Any]:
        """Get SQLite dialect information"""
        return {
            'database_type': 'sqlite',
            'supports_right_join': False,
            'supports_full_join': False,
            'supports_window_functions': True,
            'supports_cte': True,
            'string_functions': ['substr', 'replace', 'trim', 'upper', 'lower'],
            'date_functions': ['date', 'datetime', 'strftime', 'julianday'],
            'math_functions': ['abs', 'round', 'random', 'max', 'min'],
            'null_function': 'IFNULL',
            'limit_syntax': 'LIMIT',
            'offset_syntax': 'OFFSET',
            'quote_identifier': '[]',
            'quote_string': "'"
        }
        
    def begin_transaction(self) -> None:
        """Begin SQLite transaction"""
        if not self._transaction_active:
            self.connection.execute("BEGIN")
            self._transaction_active = True
            logger.debug("SQLite transaction started")
            
    def commit_transaction(self) -> None:
        """Commit SQLite transaction"""
        if self._transaction_active:
            self.connection.commit()
            self._transaction_active = False
            logger.debug("SQLite transaction committed")
            
    def rollback_transaction(self) -> None:
        """Rollback SQLite transaction"""
        if self._transaction_active:
            self.connection.rollback()
            self._transaction_active = False
            logger.debug("SQLite transaction rolled back")
            
    def get_table_sample(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Get sample data from table as DataFrame"""
        safe_table = self.validate_identifier(table_name)
        query = f"SELECT * FROM {safe_table} LIMIT ?"
        
        try:
            return pd.read_sql_query(query, self.connection, params=[limit])
        except Exception as e:
            logger.error(f"Error getting table sample: {e}")
            return pd.DataFrame()
            
    def _get_row_count(self, table_name: str) -> int:
        """Get row count for table"""
        try:
            safe_table = self.validate_identifier(table_name)
            query = f"SELECT COUNT(*) FROM {safe_table}"
            result = self.execute_query(query)
            return result[0][0] if result else 0
        except Exception:
            return 0
            
    def _setup_sqlite_functions(self):
        """Setup custom SQLite functions"""
        try:
            # Add REGEXP function
            def regexp_match(pattern, string):
                import re
                if string is None or pattern is None:
                    return False
                return bool(re.search(str(pattern), str(string)))
                
            self.connection.create_function("REGEXP", 2, regexp_match)
            
            # Add IFNULL (though SQLite has it built-in)
            def ifnull(val1, val2):
                return val2 if val1 is None else val1
                
            self.connection.create_function("IFNULL", 2, ifnull)
            
            logger.debug("Custom SQLite functions registered")
            
        except Exception as e:
            logger.warning(f"Error setting up SQLite functions: {e}")
            
    def execute_script(self, script: str) -> None:
        """Execute SQLite script with multiple statements"""
        try:
            self.connection.executescript(script)
            logger.debug("SQLite script executed successfully")
        except sqlite3.Error as e:
            logger.error(f"SQLite script execution failed: {e}")
            raise QueryException(f"Script execution failed: {e}")