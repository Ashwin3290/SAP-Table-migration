"""
SQL utilities for TableLLM
"""
import re
import sqlite3
import pandas as pd
from utils.logging_utils import main_logger as logger

class SQLInjectionError(Exception):
    """Exception raised for potential SQL injection attempts."""
    pass

def validate_sql_identifier(identifier):
    """
    Validate that an SQL identifier doesn't contain injection attempts
    Returns sanitized identifier or raises exception
    
    Parameters:
    identifier (str): The SQL identifier to validate
    
    Returns:
    str: The validated identifier
    
    Raises:
    SQLInjectionError: If the identifier contains potentially dangerous patterns
    """
    if not identifier:
        raise SQLInjectionError("Empty SQL identifier provided")

    # Check for common SQL injection patterns
    dangerous_patterns = [
        ";", "--", "/*", "*/", "DROP", "DELETE", "INSERT", "UPDATE", "UNION", "EXEC", "EXECUTE",
    ]
    for pattern in dangerous_patterns:
        if pattern.lower() in identifier.lower():
            raise SQLInjectionError(f"Potentially dangerous SQL pattern found: {pattern}")

    # Only allow alphanumeric characters, underscores, and some specific characters
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", identifier):
        raise SQLInjectionError("SQL identifier contains invalid characters")
    
    return identifier

def safe_execute_query(conn, query, params=None):
    """
    Safely execute an SQL query with parameterized values
    
    Parameters:
    conn (sqlite3.Connection): Database connection
    query (str): SQL query to execute
    params (tuple, optional): Parameters for the query
    
    Returns:
    pandas.DataFrame: Result of the query
    
    Raises:
    sqlite3.Error: If there's an error executing the query
    """
    try:
        if params:
            return pd.read_sql_query(query, conn, params=params)
        else:
            return pd.read_sql_query(query, conn)
    except sqlite3.Error as e:
        logger.error(f"SQLite error executing query: {e}")
        logger.error(f"Query: {query}")
        if params:
            logger.error(f"Params: {params}")
        raise

def get_table_schema(conn, table_name):
    """
    Get the schema of a table
    
    Parameters:
    conn (sqlite3.Connection): Database connection
    table_name (str): Name of the table
    
    Returns:
    pandas.DataFrame: Schema information
    
    Raises:
    sqlite3.Error: If there's an error executing the query
    """
    try:
        # Validate table name
        safe_table = validate_sql_identifier(table_name)
        
        # Get schema information
        query = f"PRAGMA table_info({safe_table})"
        return pd.read_sql_query(query, conn)
    except sqlite3.Error as e:
        logger.error(f"SQLite error getting schema for {table_name}: {e}")
        raise

def table_exists(conn, table_name):
    """
    Check if a table exists in the database
    
    Parameters:
    conn (sqlite3.Connection): Database connection
    table_name (str): Name of the table
    
    Returns:
    bool: True if the table exists, False otherwise
    """
    try:
        # Validate table name
        safe_table = validate_sql_identifier(table_name)
        
        # Check if table exists
        query = f"SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = pd.read_sql_query(query, conn, params=(safe_table,))
        return not result.empty
    except (sqlite3.Error, SQLInjectionError) as e:
        logger.error(f"Error checking if table {table_name} exists: {e}")
        return False

def get_all_tables(conn):
    """
    Get all tables in the database
    
    Parameters:
    conn (sqlite3.Connection): Database connection
    
    Returns:
    list: List of table names
    """
    try:
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        result = pd.read_sql_query(query, conn)
        return result['name'].tolist()
    except sqlite3.Error as e:
        logger.error(f"SQLite error getting all tables: {e}")
        return []

def get_table_statistics(conn, table_name):
    """
    Get basic statistics about a table
    
    Parameters:
    conn (sqlite3.Connection): Database connection
    table_name (str): Name of the table
    
    Returns:
    dict: Dictionary with table statistics
    """
    try:
        # Validate table name
        safe_table = validate_sql_identifier(table_name)
        
        # Get row count
        row_count_query = f"SELECT COUNT(*) AS count FROM {safe_table}"
        row_count = pd.read_sql_query(row_count_query, conn)['count'].iloc[0]
        
        # Get schema
        schema = get_table_schema(conn, safe_table)
        
        # Get column statistics
        column_stats = {}
        for column in schema['name']:
            # Count distinct values
            distinct_query = f"SELECT COUNT(DISTINCT {column}) AS distinct_count FROM {safe_table}"
            try:
                distinct_count = pd.read_sql_query(distinct_query, conn)['distinct_count'].iloc[0]
            except sqlite3.Error:
                distinct_count = None
            
            # Count null values
            null_query = f"SELECT COUNT(*) AS null_count FROM {safe_table} WHERE {column} IS NULL"
            try:
                null_count = pd.read_sql_query(null_query, conn)['null_count'].iloc[0]
            except sqlite3.Error:
                null_count = None
            
            column_stats[column] = {
                'distinct_count': distinct_count,
                'null_count': null_count,
                'null_percentage': (null_count / row_count * 100) if null_count is not None and row_count > 0 else None
            }
        
        return {
            'table_name': safe_table,
            'row_count': row_count,
            'column_count': len(schema),
            'columns': schema['name'].tolist(),
            'column_stats': column_stats
        }
    except (sqlite3.Error, SQLInjectionError) as e:
        logger.error(f"Error getting statistics for table {table_name}: {e}")
        return {
            'table_name': table_name,
            'error': str(e)
        }
