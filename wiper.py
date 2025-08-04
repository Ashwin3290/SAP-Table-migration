import sqlite3

conn=sqlite3.connect('db.sqlite3')
c = conn.cursor()
c.execute("delete from t_24_Product_Plant_Data_Ext")
conn.commit()
