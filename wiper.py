import sqlite3

conn=sqlite3.connect('db.sqlite3')
c = conn.cursor()

c.execute("DELETE from t_24_Product_Basic_Data_mandatory_Ext")
c.execute("DELETE from t_24_Product_Basic_Data_mandatory_Ext_src")

conn.commit()
