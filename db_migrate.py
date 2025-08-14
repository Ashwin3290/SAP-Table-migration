import pandas as pd
import pyodbc
from sqlalchemy import create_engine, inspect

def migrate_sqlite_table_to_azure(sqlite_file_path, sqlite_table_name, azure_connection_string, azure_table_name=None):
    """
    Migrates a table from a given SQLite file to Azure SQL Database using pyodbc.
   
    Args:
        sqlite_file_path (str): Path to the SQLite database file.
        sqlite_table_name (str): Name of the table to migrate.
        azure_connection_string (str): Azure SQL Database connection string.
        azure_table_name (str): Name for the new table in Azure SQL (defaults to same as SQLite).
    """
    # Read data from SQLite
    sqlite_engine = create_engine(f'sqlite:///{sqlite_file_path}')
    df = pd.read_sql_table(sqlite_table_name, sqlite_engine)
    inspector = inspect(sqlite_engine)
    columns = inspector.get_columns(sqlite_table_name)
   
    # Build SQL CREATE TABLE statement for Azure SQL (simplified, adjust types as needed)
    field_defs = []
    for col in columns:
        name = col['name']
        # Map SQLite types to SQL Server types (basic mapping)
        sqlite_type = str(col['type']).upper()
        if 'INT' in sqlite_type:
            sql_type = 'INT'
        elif 'FLOAT' in sqlite_type or 'REAL' in sqlite_type:
            sql_type = 'FLOAT'
        elif 'TEXT' in sqlite_type or 'VARCHAR' in sqlite_type:
            sql_type = 'NVARCHAR(MAX)'
        else:
            sql_type = 'NVARCHAR(MAX)'  # Default to text
        
        field_defs.append(f'[{name}] {sql_type}')
    
    fields_str = ', '.join(field_defs)
    table_name = azure_table_name or sqlite_table_name
   
    # Connect to Azure SQL Database
    try:
        conn = pyodbc.connect(azure_connection_string)
        cursor = conn.cursor()
        
        # Create table in Azure SQL
        create_sql = f"CREATE TABLE [{table_name}] ({fields_str})"
        try:
            cursor.execute(create_sql)
            conn.commit()
            print(f"Created table '{table_name}' in Azure SQL Database")
        except pyodbc.Error as e:
            print(f"Warning: Table creation may have failed (already exists?): {e}")
       
        # Insert data in batches for better performance
        if not df.empty:
            # Prepare insert statement
            placeholders = ', '.join(['?'] * len(df.columns))
            insert_sql = f"INSERT INTO [{table_name}] VALUES ({placeholders})"
            
            # Convert DataFrame to list of tuples
            records = [tuple(row) for row in df.values]
            
            # Insert in batches
            batch_size = 1000
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                cursor.executemany(insert_sql, batch)
                conn.commit()
                print(f"Inserted batch {i//batch_size + 1} ({len(batch)} records)")
        
        cursor.close()
        conn.close()
        print(f"Migration of '{sqlite_table_name}' to Azure SQL as '{table_name}' complete.")
        
    except pyodbc.Error as e:
        print(f"Error connecting to Azure SQL Database: {e}")
        raise

def get_all_sqlite_tables(sqlite_file_path):
    """
    Get all table names from SQLite database.
    
    Args:
        sqlite_file_path (str): Path to the SQLite database file
        
    Returns:
        list: List of table names
    """
    sqlite_engine = create_engine(f'sqlite:///{sqlite_file_path}')
    inspector = inspect(sqlite_engine)
    
    # Get all table names, excluding system tables
    table_names = inspector.get_table_names()
    
    # Filter out SQLite system tables (optional)
    filtered_tables = [table for table in table_names if not table.startswith('sqlite_')]
    
    return filtered_tables

def sqlite_to_azure_migration(azure_connection_string, sqlite_file_path=None, tablenames=None):
    """
    Migrates tables from SQLite to Azure SQL Database.
    If tablenames is not provided, migrates ALL tables from SQLite.
    
    Args:
        azure_connection_string (str): Azure SQL Database connection string
        sqlite_file_path (str): Path to SQLite file (optional, uses default if not provided)
        tablenames (str): Comma-separated table names to migrate (optional, migrates all if not provided)
    
    Returns:
        str: Success message with details
    """
    # Default SQLite file path (you can modify this)
    if sqlite_file_path is None:
        sqlite_file_path = r"db.sqlite3"
    
    try:
        # If no specific tables provided, get all tables from SQLite
        if tablenames is None:
            tables_to_migrate = get_all_sqlite_tables(sqlite_file_path)
            print(f"Found {len(tables_to_migrate)} tables in SQLite database: {tables_to_migrate}")
        else:
            tables_to_migrate = [table.strip() for table in tablenames.split(",") if table.strip()]
        
        if not tables_to_migrate:
            return "No tables found to migrate"
        
        migrated_tables = []
        failed_tables = []
        
        for table_name in tables_to_migrate:
            try:
                migrate_sqlite_table_to_azure(sqlite_file_path, table_name, azure_connection_string)
                migrated_tables.append(table_name)
            except Exception as e:
                print(f"Failed to migrate table '{table_name}': {e}")
                failed_tables.append(table_name)
        
        # Prepare result message
        result_message = f"Migration completed. Successfully migrated {len(migrated_tables)} tables: {migrated_tables}"
        if failed_tables:
            result_message += f"\nFailed to migrate {len(failed_tables)} tables: {failed_tables}"
        
        return result_message
    
    except Exception as e:
        return f"Migration failed: {str(e)}"

# Example usage:
if __name__ == "__main__":
    # Azure SQL Database connection string example
    # Replace with your actual connection details
    AZURE_SQL_CONNECTION_STRING=r"Driver={ODBC Driver 17 for SQL Server};Server=(LocalDB)\MSSQLLocalDB;Database=LLM;Integrated Security=True;Connect Timeout=30;Trust Server Certificate=True;Application Name=vscode-mssql"
    # Method 1: Migrate ALL tables from SQLite to MSSQL
    result = sqlite_to_azure_migration(AZURE_SQL_CONNECTION_STRING)
    print(result)
    
    # Method 2: Migrate specific tables only
    # result = sqlite_to_azure_migration(azure_conn_str, tablenames="users,products,orders")
    # print(result)
    
    # Method 3: Migrate all tables from custom SQLite file path
    # result = sqlite_to_azure_migration(azure_conn_str, sqlite_file_path="/path/to/database.sqlite3")
    # print(result)
    
    # Method 4: First check what tables are available
    # tables = get_all_sqlite_tables("path/to/your/database.sqlite3")
    # print(f"Available tables: {tables}")
    # result = sqlite_to_azure_migration(azure_conn_str, tablenames=",".join(tables))
    # print(result)