import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Centralized database configuration for local SQL Server"""
    
    def __init__(self):
        self.server = os.environ.get('MSSQL_SERVER', 'localhost\\SQLEXPRESS')
        self.database = os.environ.get('MSSQL_DATABASE')
        self.username = os.environ.get('MSSQL_USERNAME')
        self.password = os.environ.get('MSSQL_PASSWORD')
        self.driver = os.environ.get('MSSQL_DRIVER', '{ODBC Driver 17 for SQL Server}')
        
        if not self.database:
            raise ValueError("MSSQL_DATABASE environment variable is required")
    
    @property
    def connection_string(self):
        """Build SQL Server connection string"""
        if not self.username:
            # Windows Authentication
            conn_str = (
                f"Driver={self.driver};"
                f"Server={self.server};"
                f"Database={self.database};"
                f"Trusted_Connection=yes;"
                f"Connection Timeout=30;"
            )
        else:
            # SQL Server Authentication
            conn_str = (
                f"Driver={self.driver};"
                f"Server={self.server};"
                f"Database={self.database};"
                f"Uid={self.username};"
                f"Pwd={self.password};"
                f"Connection Timeout=30;"
            )
        
        logger.info(f"Using connection string: {conn_str.replace(self.password or '', '***')}")
        return conn_str
    
    def test_connection(self):
        """Test database connection"""
        import pyodbc
        try:
            conn = pyodbc.connect(self.connection_string)
            conn.close()
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False