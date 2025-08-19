import os
import logging
from dotenv import load_dotenv
import sqlalchemy as sa
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import pyodbc

load_dotenv(".env")

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Centralized database configuration for local SQL Server"""
    
    def __init__(self):
        self.sql_connection_string = os.getenv("SQL_CONNECTION_STRING", "")
        self.server = os.environ.get('SQL_SERVER', 'localhost\\SQLEXPRESS')
        self.database = os.environ.get('SQL_DATABASE')
        self.username = os.environ.get('SQL_USERNAME')
        self.password = os.environ.get('SQL_PASSWORD')
        self.driver = os.environ.get('SQL_DRIVER', '{ODBC Driver 17 for SQL Server}')
        
        # Cache for engines
        self._engine = None
        self._fast_engine = None
    
        if not self.database:
            raise ValueError("SQL_DATABASE environment variable is required")
    
    @property
    def connection_string(self):
        """Build SQL Server connection string for pyodbc"""
        if self.sql_connection_string:
            return self.sql_connection_string
        
        elif not self.username:
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
    
    @property
    def sqlalchemy_url(self):
        """Build SQLAlchemy connection URL"""
        if not self.username:
            # Windows Authentication
            params = quote_plus(
                f"DRIVER={self.driver};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes;"
                f"Connection Timeout=30"
            )
        else:
            # SQL Server Authentication
            params = quote_plus(
                f"DRIVER={self.driver};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
                f"Connection Timeout=30"
            )
        
        return f"mssql+pyodbc:///?odbc_connect={params}"
    
    def get_engine(self, fast_executemany=True):
        """Get SQLAlchemy engine with optimized settings"""
        if fast_executemany:
            if not self._fast_engine:
                self._fast_engine = create_engine(
                    self.sqlalchemy_url,
                    fast_executemany=True,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False
                )
            return self._fast_engine
        else:
            if not self._engine:
                self._engine = create_engine(
                    self.sqlalchemy_url,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False
                )
            return self._engine
    
    def get_pyodbc_connection(self):
        """Get a pyodbc connection (for backward compatibility)"""
        return pyodbc.connect(self.connection_string)
    
    def test_connection(self):
        """Test database connection"""
        try:
            # Test with pyodbc
            conn = pyodbc.connect(self.connection_string)
            conn.close()
            
            # Test with SQLAlchemy
            engine = self.get_engine()
            with engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
            
            logger.info("Database connection test successful (both pyodbc and SQLAlchemy)")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def dispose_engines(self):
        """Dispose of SQLAlchemy engines to close all connections"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
        if self._fast_engine:
            self._fast_engine.dispose()
            self._fast_engine = None