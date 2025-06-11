import logging
import sqlite3
import re
import pandas as pd
import uuid
import os
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, Union, Tuple

from sqlite_utils import add_sqlite_functions

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class SQLExecutor:
    """Executes SQL queries against the database"""
    
    def __init__(self, db_path=os.environ.get('DB_PATH')):
        """Initialize the SQL executor
        
        Parameters:
        db_path (str): Path to the SQLite database
        """
        self.db_path = db_path
    
    def execute_query(self, 
                     query: str, 
                     params: Optional[Dict[str, Any]] = None, 
                     fetch_results: bool = True,
                     commit: bool = True) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute an SQL query with parameter binding
        
        Parameters:
        query (str): SQL query to execute
        params (Optional[Dict[str, Any]]): Parameters to bind to the query
        fetch_results (bool): Whether to fetch and return results
        commit (bool): Whether to commit changes
        
        Returns:
        Union[List[Dict[str, Any]], Dict[str, Any]]: 
            Query results as a list of dictionaries, or error information
        """
        conn = None
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            print(query)
            add_sqlite_functions(conn)
            
            # Configure connection to return rows as dictionaries
            conn.row_factory = sqlite3.Row
            
            cursor = conn.cursor()
            
            # Execute the query with parameters if provided
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            # Commit changes if requested
            if commit:
                conn.commit()
            
            # Fetch results if requested
            if fetch_results:
                # Convert rows to dictionaries
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]
                return result
            else:
                # Return row count for operations that don't return results
                return {"rowcount": cursor.rowcount}
                
        except sqlite3.Error as e:
            # Handle SQLite errors
            if conn and commit:
                conn.rollback()
                
            logger.error(f"SQLite error: {e}")
            return {
                "error_type": "SQLiteError", 
                "error_message": str(e),
                "query": query
            }
        except Exception as e:
            # Handle other exceptions
            if conn and commit:
                conn.rollback()
                
            logger.error(f"Error executing query: {e}")
            return {
                "error_type": "ExecutionError", 
                "error_message": str(e),
                "query": query
            }
        finally:
            # Close the connection
            if conn:
                conn.close()
    

    def execute_and_fetch_df(self, query: str, params: Optional[Dict[str, Any]] = None) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Execute a query and return results as a pandas DataFrame
        
        Parameters:
        query (str): SQL query to execute
        params (Optional[Dict[str, Any]]): Parameters to bind to the query
        
        Returns:
        Union[pd.DataFrame, Dict[str, Any]]: DataFrame with results or error information
        """
        conn = None
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            
            # Execute the query and load results directly into a DataFrame
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
                
            return df
                
        except sqlite3.Error as e:
            # Handle SQLite errors
            logger.error(f"SQLite error in execute_and_fetch_df: {e}")
            return {
                "error_type": "SQLiteError", 
                "error_message": str(e),
                "query": query
            }
        except Exception as e:
            # Handle other exceptions
            logger.error(f"Error in execute_and_fetch_df: {e}")
            return {
                "error_type": "ExecutionError", 
                "error_message": str(e),
                "query": query
            }
        finally:
            # Close the connection
            if conn:
                conn.close()
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database
        
        Parameters:
        table_name (str): Name of the table to check
        
        Returns:
        bool: True if the table exists, False otherwise
        """
        query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=? Limit 1
        """
        
        result = self.execute_query(query, {"table_name": table_name})
        
        if isinstance(result, list):
            return len(result) > 0
        
        return False
    
    def get_table_schema(self, table_name: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Get the schema information for a table
        
        Parameters:
        table_name (str): Name of the table
        
        Returns:
        Union[List[Dict[str, Any]], Dict[str, Any]]: Schema information or error
        """
        query = f"PRAGMA table_info({table_name})"
        
        return self.execute_query(query)
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Get a sample of rows from a table as a DataFrame
        
        Parameters:
        table_name (str): Name of the table
        limit (int): Maximum number of rows to return
        
        Returns:
        Union[pd.DataFrame, Dict[str, Any]]: DataFrame with sample rows or error
        """
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        
        return self.execute_and_fetch_df(query)
    
    def backup_table(self, table_name: str) -> Tuple[bool, str]:
        """
        Create a backup of a table
        
        Parameters:
        table_name (str): Name of the table to backup
        
        Returns:
        Tuple[bool, str]: Success status and backup table name
        """
        # Generate a unique backup table name
        backup_name = f"{table_name}_backup_{uuid.uuid4().hex[:8]}"
        
        query = f"CREATE TABLE {backup_name} AS SELECT * FROM {table_name}"
        
        result = self.execute_query(query, fetch_results=False)
        
        if isinstance(result, dict) and "error_type" in result:
            return False, ""
        
        return True, backup_name