import sqlite3
import threading
import pandas as pd
import logging
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ThreadSafeDBManager:
    """
    Thread-safe database connection manager for SQLite
    Ensures each thread gets its own connection and handles cleanup properly
    """
    
    def __init__(self, db_path="db.sqlite3"):
        self.db_path = db_path
        self.local = threading.local()
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection for the current thread.
        Creates a new connection if one doesn't exist.
        
        Usage:
            with db_manager.get_connection() as conn:
                # use conn here
        """
        if not hasattr(self.local, 'connection') or self.local.connection is None:
            # Create a new connection for this thread
            self.local.connection = sqlite3.connect(self.db_path)
            self.local.connection._thread_id = threading.get_ident()
            logger.info(f"Created new SQLite connection in thread {threading.get_ident()}")
            
        try:
            yield self.local.connection
        finally:
            pass  # Keep connection open for reuse within the thread
    
    def close(self):
        """
        Close the connection for the current thread if it exists
        """
        if hasattr(self.local, 'connection') and self.local.connection is not None:
            try:
                # Check if connection has thread_id and if it matches current thread
                current_thread_id = threading.get_ident()
                if hasattr(self.local.connection, '_thread_id'):
                    conn_thread_id = self.local.connection._thread_id
                    if conn_thread_id != current_thread_id:
                        logger.warning(f"Attempting to close connection from different thread: created in {conn_thread_id}, closing from {current_thread_id}")
                
                self.local.connection.close()
                logger.info(f"Closed SQLite connection in thread {threading.get_ident()}")
                self.local.connection = None
            except Exception as e:
                logger.error(f"Error closing connection in thread {threading.get_ident()}: {e}")
    
    def cleanup_all(self):
        """
        Attempt to close all connections
        This should only be called when the application is shutting down
        """
        current_thread_id = threading.get_ident()
        if hasattr(self.local, 'connection') and self.local.connection is not None:
            try:
                # First check if _thread_id exists on the connection
                if hasattr(self.local.connection, '_thread_id'):
                    conn_thread_id = self.local.connection._thread_id
                    if conn_thread_id == current_thread_id:
                        self.local.connection.close()
                        logger.info(f"Cleaned up SQLite connection in thread {current_thread_id}")
                    else:
                        logger.warning(f"Cannot close connection from thread {current_thread_id} that was created in thread {conn_thread_id}")
                else:
                    # If _thread_id doesn't exist, close it anyway
                    logger.warning(f"Connection doesn't have _thread_id attribute, closing from thread {current_thread_id}")
                    self.local.connection.close()
            except Exception as e:
                logger.error(f"Error during connection cleanup: {e}")
            
            self.local.connection = None
    
    def execute_query(self, query, params=None):
        """
        Execute a query on the current thread's connection
        
        Parameters:
        query (str): SQL query to execute
        params (tuple/list): Optional parameters for the query
        
        Returns:
        list: Query results as a list of tuples
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            return cursor.fetchall()
    
    def execute_update(self, query, params=None):
        """
        Execute an update query (INSERT, UPDATE, DELETE)
        
        Parameters:
        query (str): SQL query to execute
        params (tuple/list): Optional parameters for the query
        
        Returns:
        int: Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            conn.commit()
            return cursor.rowcount
    
    def get_dataframe(self, query, params=None):
        """
        Execute a query and return results as a pandas DataFrame
        
        Parameters:
        query (str): SQL query to execute
        params (tuple/list): Optional parameters for the query
        
        Returns:
        DataFrame: Query results as a pandas DataFrame
        """
        with self.get_connection() as conn:
            if params:
                return pd.read_sql_query(query, conn, params=params)
            else:
                return pd.read_sql_query(query, conn)
    
    def insert_dataframe(self, df, table_name, if_exists='replace'):
        """
        Insert a DataFrame into a table
        
        Parameters:
        df (DataFrame): DataFrame to insert
        table_name (str): Target table name
        if_exists (str): What to do if table exists ('fail', 'replace', 'append')
        
        Returns:
        bool: Success status
        """
        with self.get_connection() as conn:
            try:
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
                return True
            except Exception as e:
                logger.error(f"Error inserting DataFrame into {table_name}: {e}")
                return False


# Create a singleton instance
db_manager = ThreadSafeDBManager()

# Module-level functions for easier access
@contextmanager
def get_connection(db_path="db.sqlite3"):
    """Get a thread-safe connection from the manager"""
    with db_manager.get_connection() as conn:
        yield conn

def close_connections():
    """Close connections for the current thread"""
    db_manager.close()

def cleanup_all_connections():
    """Clean up all connections - call during app shutdown"""
    db_manager.cleanup_all()

def get_dataframe(query, params=None):
    """Get query results as a DataFrame"""
    return db_manager.get_dataframe(query, params)

def execute_query(query, params=None):
    """Execute a query and return results"""
    return db_manager.execute_query(query, params)

def execute_update(query, params=None):
    """Execute an update query"""
    return db_manager.execute_update(query, params)

def insert_dataframe(df, table_name, if_exists='replace'):
    """Insert a DataFrame into a table"""
    return db_manager.insert_dataframe(df, table_name, if_exists)
