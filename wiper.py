import sqlite3

conn=sqlite3.connect('db.sqlite3')
c = conn.cursor()
# # c.execute("select * from t_24_Product_Plant_Data_Ext")
# # print(c.fetchall())
# c.execute("Delete from t_25_Customer_General_Data_mandatory")
# conn.commit()

# c.execute("""INSERT INTO t_24_Customer_Additional_Addresses (KUNNR, POST_CODE1, CITY1, COUNTRY) SELECT t_24_Customer_General_Data_mandatory.KUNNR, ADRC.POST_CODE1, ADRC.CITY1, ADRC.COUNTRY FROM t_24_Customer_General_Data_mandatory JOIN BUT020 ON t_24_Customer_General_Data_mandatory.KUNNR = BUT020.PARTNER JOIN ADRC ON BUT020.ADDRNUMBER = ADRC.ADDRNUMBER;""")
# conn.commit()

c.execute("DELETE from t_24_Product_Basic_Data_mandatory_Ext")
# c.execute("DELETE from t_24_Product_Plant_Data_Ext")
conn.commit()

# c.execute("ALTER TABLE t_24_Product_Basic_Data_mandatory_Ext DROP COLUMN LIQDT_Day ")