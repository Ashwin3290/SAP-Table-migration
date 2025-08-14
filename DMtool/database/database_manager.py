# database/database_manager.py
import os
import logging
from typing import Dict, List, Any, Optional, Type
from contextlib import contextmanager
import pandas as pd

from DMtool.enums.database_types import DatabaseType, ExecutionResult
from DMtool.database.base.base_connector import BaseConnector
from DMtool.database.connectors.sqlite_connector import SQLiteConnector

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Main database manager that handles different database types"""
    
    # Registry of available connectors
    _connector_registry: Dict[DatabaseType, Type[BaseConnector]] = {
        DatabaseType.SQLITE: SQLiteConnector,
        # Future connectors will be added here
        # DatabaseType.POSTGRESQL: PostgreSQLConnector,
        # DatabaseType.MYSQL: MySQLConnector,
    }
    
    def __init__(self, db_type: Optional[DatabaseType] = None, connection_params: Optional[Dict[str, Any]] = None):
        """Initialize database manager"""
        
        # Get database type from environment or parameter
        if db_type is None:
            db_type_str = os.getenv('DB_TYPE', 'sqlite').lower()
            try:
                self.db_type = DatabaseType(db_type_str)
            except ValueError:
                logger.warning(f"Unknown DB_TYPE: {db_type_str}, defaulting to SQLite")
                self.db_type = DatabaseType.SQLITE
        else:
            self.db_type = db_type
            
        # Setup connection parameters
        self.connection_params = connection_params or self._load_connection_params()
        
        # Initialize connector
        self.connector = self._create_connector()
        
        # Connect to database
        if not self.connector.connect():
            raise ConnectionError(f"Failed to connect to {self.db_type.value} database")
            
        logger.info(f"Database manager initialized for {self.db_type.value}")
        
    def _load_connection_params(self) -> Dict[str, Any]:
        """Load connection parameters from environment"""
        if self.db_type == DatabaseType.SQLITE:
            return {
                'database_path': os.getenv('DB_PATH', 'database.db')
            }
        elif self.db_type == DatabaseType.POSTGRESQL:
            return {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'database': os.getenv('DB_NAME', 'postgres'),
                'username': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', ''),
            }
        elif self.db_type == DatabaseType.MYSQL:
            return {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 3306)),
                'database': os.getenv('DB_NAME', 'mysql'),
                'username': os.getenv('DB_USER', 'root'),
                'password': os.getenv('DB_PASSWORD', ''),
            }
        else:
            return {}
            
    def _create_connector(self) -> BaseConnector:
        """Create appropriate database connector"""
        connector_class = self._connector_registry.get(self.db_type)
        
        if connector_class is None:
            raise ValueError(f"No connector available for database type: {self.db_type}")
            
        return connector_class(self.connection_params)
        
    @classmethod
    def register_connector(cls, db_type: DatabaseType, connector_class: Type[BaseConnector]):
        """Register a new database connector"""
        cls._connector_registry[db_type] = connector_class
        logger.info(f"Registered connector for {db_type.value}: {connector_class.__name__}")
        
    # Delegate methods to connector
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a single query"""
        return self.connector.execute_query(query, params)
        
    def execute_batch(self, queries: List[str], params: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
        """Execute multiple queries"""
        return self.connector.execute_batch(queries, params)
        
    def execute_with_rollback(self, queries: List[str], 
                            params: Optional[List[Dict[str, Any]]] = None) -> ExecutionResult:
        """Execute queries with automatic rollback on failure"""
        return self.connector.execute_with_rollback(queries, params)
        
    def get_tables(self) -> List[str]:
        """Get list of all tables"""
        return self.connector.get_tables()
        
    def get_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Get columns for a specific table"""
        return self.connector.get_columns(table_name)
        
    def get_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table"""
        return self.connector.get_schema(table_name)
        
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        return self.connector.table_exists(table_name)
        
    def validate_identifier(self, identifier: str) -> str:
        """Validate and sanitize SQL identifier"""
        return self.connector.validate_identifier(identifier)
        
    def get_dialect_info(self) -> Dict[str, Any]:
        """Get database dialect information"""
        return self.connector.get_dialect_info()
        
    def get_syntax_rules(self) -> Dict[str, Any]:
        """Get database-specific syntax rules"""
        dialect_info = self.get_dialect_info()
        return {
            'supports_right_join': dialect_info.get('supports_right_join', True),
            'supports_full_join': dialect_info.get('supports_full_join', True),
            'null_function': dialect_info.get('null_function', 'COALESCE'),
            'string_functions': dialect_info.get('string_functions', []),
            'date_functions': dialect_info.get('date_functions', []),
            'limit_syntax': dialect_info.get('limit_syntax', 'LIMIT'),
            'quote_identifier': dialect_info.get('quote_identifier', '"'),
        }
        
    def get_table_sample(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Get sample data from table"""
        if hasattr(self.connector, 'get_table_sample'):
            return self.connector.get_table_sample(table_name, limit)
        else:
            # Fallback implementation
            safe_table = self.validate_identifier(table_name)
            query = f"SELECT * FROM {safe_table} LIMIT {limit}"
            result = self.execute_query(query)
            
            # Convert to DataFrame (basic implementation)
            if result:
                columns = [desc[0] for desc in result.description] if hasattr(result, 'description') else []
                data = [dict(zip(columns, row)) for row in result] if columns else []
                return pd.DataFrame(data)
            
            return pd.DataFrame()
            
    def get_multiple_schemas(self, table_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get schemas for multiple tables"""
        schemas = {}
        for table_name in table_names:
            try:
                schemas[table_name] = self.get_schema(table_name)
            except Exception as e:
                logger.warning(f"Failed to get schema for {table_name}: {e}")
                schemas[table_name] = {'error': str(e)}
        return schemas
        
    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        with self.connector.transaction():
            yield self.connector
            
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            is_healthy = self.connector.health_check()
            tables_count = len(self.get_tables()) if is_healthy else 0
            
            return {
                'status': 'healthy' if is_healthy else 'unhealthy',
                'database_type': self.db_type.value,
                'connection_active': self.connector.is_connected,
                'tables_count': tables_count,
                'dialect_info': self.get_dialect_info() if is_healthy else {}
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'database_type': self.db_type.value,
                'connection_active': False
            }
            
    def close(self):
        """Close database connection"""
        if self.connector:
            self.connector.disconnect()
            logger.info("Database manager closed")
            
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close()
        except Exception:
            pass  