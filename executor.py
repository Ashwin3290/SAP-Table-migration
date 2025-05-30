import logging
import sqlite3
import re
import pandas as pd
import uuid
import os
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, Union, Tuple

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
    
    def create_temp_view(self, view_name: str, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a temporary view in the database
        
        Parameters:
        view_name (str): Name of the view to create
        query (str): SELECT query for the view
        params (Optional[Dict[str, Any]]): Parameters to bind to the query
        
        Returns:
        Dict[str, Any]: Result information or error
        """
        # Sanitize the view name
        if not re.match(r'^[a-zA-Z0-9_]+$', view_name):
            logger.error(f"Invalid view name: {view_name}")
            return {
                "success": False,
                "error_type": "ValidationError",
                "error_message": f"Invalid view name: {view_name}"
            }
        
        # Create the CREATE VIEW statement
        create_view = f"CREATE TEMPORARY VIEW IF NOT EXISTS {view_name} AS {query}"
        
        return self.execute_query(create_view, params, fetch_results=False)
    
    def drop_temp_view(self, view_name: str) -> Dict[str, Any]:
        """
        Drop a temporary view from the database
        
        Parameters:
        view_name (str): Name of the view to drop
        
        Returns:
        Dict[str, Any]: Result information or error
        """
        # Sanitize the view name
        if not re.match(r'^[a-zA-Z0-9_]+$', view_name):
            logger.error(f"Invalid view name: {view_name}")
            return {
                "success": False,
                "error_type": "ValidationError",
                "error_message": f"Invalid view name: {view_name}"
            }
        
        # Create the DROP VIEW statement
        drop_view = f"DROP VIEW IF EXISTS {view_name}"
        
        return self.execute_query(drop_view, fetch_results=False)
    
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
    
    def execute_with_target_update(self, 
                                 source_query: str, 
                                 target_table: str,
                                 key_field: str,
                                 field_mapping: Dict[str, str],
                                 params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a query and update a target table with the results
        
        Parameters:
        source_query (str): Query to get source data
        target_table (str): Target table to update
        key_field (str): Key field for matching records
        field_mapping (Dict[str, str]): Mapping from source to target fields
        params (Optional[Dict[str, Any]]): Parameters for the source query
        
        Returns:
        Dict[str, Any]: Result information
        """
        conn = None
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            
            # Start a transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Execute the source query
            cursor = conn.cursor()
            if params:
                cursor.execute(source_query, params)
            else:
                cursor.execute(source_query)
                
            # Fetch all results
            source_rows = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            updates = 0
            inserts = 0
            
            # Process each source row
            for row in source_rows:
                # Create a dictionary from the row
                row_dict = {column_names[i]: row[i] for i in range(len(column_names))}
                
                # Check if the key exists in the target table
                key_value = row_dict.get(key_field)
                if key_value is None:
                    logger.warning(f"Key field {key_field} has NULL value, skipping row")
                    continue
                    
                check_query = f"SELECT COUNT(*) FROM {target_table} WHERE {key_field} = ?"
                check_cursor = conn.cursor()
                check_cursor.execute(check_query, (key_value,))
                exists = check_cursor.fetchone()[0] > 0
                
                if exists:
                    # Update existing record
                    set_clauses = []
                    update_params = {}
                    
                    for target_field, source_field in field_mapping.items():
                        if source_field in row_dict and target_field != key_field:
                            set_clauses.append(f"{target_field} = :{target_field}")
                            update_params[target_field] = row_dict[source_field]
                    
                    if set_clauses:
                        update_query = f"""
                        UPDATE {target_table}
                        SET {', '.join(set_clauses)}
                        WHERE {key_field} = :{key_field}
                        """
                        update_params[key_field] = key_value
                        
                        update_cursor = conn.cursor()
                        update_cursor.execute(update_query, update_params)
                        updates += update_cursor.rowcount
                else:
                    # Insert new record
                    target_fields = list(field_mapping.keys())
                    source_fields = [field_mapping[target] for target in target_fields]
                    
                    insert_values = {}
                    for i, target_field in enumerate(target_fields):
                        source_field = source_fields[i]
                        if source_field in row_dict:
                            insert_values[target_field] = row_dict[source_field]
                        else:
                            insert_values[target_field] = None
                    
                    placeholders = [f":{field}" for field in target_fields]
                    
                    insert_query = f"""
                    INSERT INTO {target_table} ({', '.join(target_fields)})
                    VALUES ({', '.join(placeholders)})
                    """
                    
                    insert_cursor = conn.cursor()
                    insert_cursor.execute(insert_query, insert_values)
                    inserts += 1
            
            # Commit the transaction
            conn.commit()
            
            return {
                "success": True,
                "updates": updates,
                "inserts": inserts,
                "total_processed": len(source_rows)
            }
                
        except sqlite3.Error as e:
            # Handle SQLite errors
            if conn:
                conn.rollback()
                
            logger.error(f"SQLite error in execute_with_target_update: {e}")
            return {
                "success": False,
                "error_type": "SQLiteError", 
                "error_message": str(e)
            }
        except Exception as e:
            # Handle other exceptions
            if conn:
                conn.rollback()
                
            logger.error(f"Error in execute_with_target_update: {e}")
            return {
                "success": False,
                "error_type": "ExecutionError", 
                "error_message": str(e)
            }
        finally:
            # Close the connection
            if conn:
                conn.close()
                
    def get_table_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table
        
        Parameters:
        table_name (str): Name of the table
        
        Returns:
        int: Number of rows or 0 if error
        """
        query = f"SELECT COUNT(*) AS row_count FROM {table_name}"
        
        result = self.execute_query(query)
        
        if isinstance(result, list) and result:
            return result[0].get("row_count", 0)
        
        return 0
    
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