import sqlite3
import pandas as pd
from pathlib import Path

def insert_excel_to_sqlite(excel_file_path, sqlite_db_path, table_name='product_data'):
    """
    Insert Excel data into SQLite3 database table
    
    Args:
        excel_file_path (str): Path to the Excel file
        sqlite_db_path (str): Path to SQLite database file
        table_name (str): Name of the table to create/insert into
    """
    
    try:
        # Read Excel file
        print(f"Reading Excel file: {excel_file_path}")
        df = pd.read_excel(excel_file_path, sheet_name=0)
        
        print(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns[:10])}...")  # Show first 10 columns
        
        # Connect to SQLite database
        print(f"Connecting to SQLite database: {sqlite_db_path}")
        conn = sqlite3.connect(sqlite_db_path)
        cursor = conn.cursor()
        
        # Insert data using pandas to_sql method (recommended approach)
        print(f"Inserting data into table '{table_name}'...")
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Verify insertion
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"Successfully inserted {row_count} rows")
        
        # Show table schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        schema_info = cursor.fetchall()
        print(f"\nTable schema:")
        for col_info in schema_info[:10]:  # Show first 10 columns
            print(f"  {col_info[1]} ({col_info[2]})")
        if len(schema_info) > 10:
            print(f"  ... and {len(schema_info) - 10} more columns")
            
        conn.commit()
        
    except Exception as e:
        print(f"Error: {e}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()
            print("Database connection closed")

insert_excel_to_sqlite('table_data.xlsx', 'db.sqlite3', table_name='t_24_Product_Basic_Data_mandatory_Ext_src')