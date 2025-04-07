import sqlite3

def delete_all_data_from_table(db_path, table_name):
    """Deletes all data from a specified table in an SQLite database.

    Args:
        db_path (str): The path to the SQLite database file.
        table_name (str): The name of the table to delete data from.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(f"DELETE FROM {table_name}")
        conn.commit()
        print(f"All data deleted from table '{table_name}' in database '{db_path}'.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

# Example usage:
db_path = 'db.sqlite3'
table_name = 't_24_Product_Basic_Data_mandatory'
delete_all_data_from_table(db_path, table_name)
