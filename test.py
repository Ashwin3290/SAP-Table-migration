import sqlite3
import pandas as pd


conn = sqlite3.connect('db.sqlite3') 


df=pd.read_sql_query("PRAGMA table_info([MARA])", conn)
print(df)