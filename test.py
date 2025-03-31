import sqlite3
import pandas as pd
import json

conn = sqlite3.connect('db.sqlite3') 
def proccess_info(resolved_data,conn):

    source_df=pd.read_sql_query(f"Select * from {resolved_data['source_table_name']} limit 5", conn)[resolved_data['source_field_names']]
    target_df=pd.read_sql_query(f"Select * from {resolved_data['target_table_name']} limit 5", conn)[resolved_data['target_sap_fields']]

    print(source_df.info())
    print(target_df.info())
    print(source_df.describe())
    print(target_df.describe())
    conn.close()
    
    return {
        "source_df":source_df,
        "target_df":target_df,
        "source_info":source_df.info(),
        "target_info":target_df.info(),
        "source_describe":source_df.describe(),
        "target_describe":target_df.describe()
    }

resolved_data = json.load(open("resolved_query.json","r"))
proccess_info(resolved_data,conn)