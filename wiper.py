import sqlite3

conn=sqlite3.connect('db.sqlite3')
c = conn.cursor()
# c.execute("delete from MARA")
c.execute("delete from t_24_Product_Basic_Data_mandatory_Ext")
conn.commit()
