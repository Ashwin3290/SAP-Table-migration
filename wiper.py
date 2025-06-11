import sqlite3

conn=sqlite3.connect('db.sqlite3')
c = conn.cursor()
# # c.execute("select * from t_24_Product_Plant_Data_Ext")
# # print(c.fetchall())
c.execute("Delete from t_25_Product_Additional_Descriptions")
conn.commit()

# c.execute("""INSERT INTO t_24_Product_Plant_Data_Ext (PRODUCT, WERKS)
# SELECT
#   MARC.MATNR,
#   MARC.WERKS
# FROM t_24_Product_Basic_Data_mandatory_Ext
# JOIN MARC ON t_24_Product_Basic_Data_mandatory_Ext.PRODUCT = MARC.MATNR
# WHERE MARC.WERKS IN ('1710', '9999');""")