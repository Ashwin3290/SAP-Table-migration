from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import logging
from contextlib import contextmanager
from DMtool.enums.database_types import DatabaseType, ExecutionResult, ExecutionStatus

logger = logging.getLogger(__name__)

class DatabaseException(Exception):
    """Base exception for database operations"""
    pass

class ConnectionException(DatabaseException):
    """Exception for connection-related errors"""
    pass

class QueryException(DatabaseException):
    """Exception for query execution errors"""
    pass

class BaseConnector(ABC):
    """Abstract base class for database connectors"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        self.connection = None
        self.is_connected = False
        self.dialect_info = {}
        
    @abstractmethod
    def connect(self) -> bool:
        """Establish database connection"""
        pass
        
    @abstractmethod
    def disconnect(self) -> bool:
        """Close database connection"""
        pass
        
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query and return results"""
        pass
        
    @abstractmethod
    def execute_batch(self, queries: List[str], params: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
        """Execute multiple queries"""
        pass
        
    @abstractmethod
    def get_tables(self) -> List[str]:
        """Get list of all tables"""
        pass
        
    @abstractmethod
    def get_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Get columns for a specific table"""
        pass
        
    @abstractmethod
    def get_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table"""
        pass
        
    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        pass
        
    @abstractmethod
    def validate_identifier(self, identifier: str) -> str:
        """Validate and sanitize SQL identifier"""
        pass
        
    @abstractmethod
    def escape_string(self, value: str) -> str:
        """Escape string value for SQL"""
        pass
        
    @abstractmethod
    def get_dialect_info(self) -> Dict[str, Any]:
        """Get database dialect specific information"""
        pass
        
    @abstractmethod
    def begin_transaction(self) -> None:
        """Begin a database transaction"""
        pass
        
    @abstractmethod
    def commit_transaction(self) -> None:
        """Commit current transaction"""
        pass
        
    @abstractmethod
    def rollback_transaction(self) -> None:
        """Rollback current transaction"""
        pass
        
    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        try:
            self.begin_transaction()
            yield self
            self.commit_transaction()
        except Exception as e:
            self.rollback_transaction()
            logger.error(f"Transaction failed: {e}")
            raise
            
    def execute_with_rollback(self, queries: List[str], 
                            params: Optional[List[Dict[str, Any]]] = None) -> ExecutionResult:
        """Execute queries with automatic rollback on failure"""
        executed_queries = []
        
        try:
            with self.transaction():
                results = []
                for i, query in enumerate(queries):
                    query_params = params[i] if params and i < len(params) else None
                    result = self.execute_query(query, query_params)
                    results.append(result)
                    executed_queries.append(query)
                    
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    results=results,
                    queries_executed=executed_queries,
                    rows_affected=sum(getattr(r, 'rowcount', 0) for r in results if hasattr(r, 'rowcount'))
                )
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                results=[],
                error_message=str(e),
                queries_executed=executed_queries
            )
            
    def get_database_type(self) -> DatabaseType:
        """Get the database type"""
        return self.database_type
        
    def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            self.execute_query("SELECT 1")
            return True
        except Exception:
            return False