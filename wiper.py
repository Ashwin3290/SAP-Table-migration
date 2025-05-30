import sqlite3

conn=sqlite3.connect('db.sqlite3')
c = conn.cursor()
# # c.execute("select * from t_24_Product_Plant_Data_Ext")
# # print(c.fetchall())
c.execute("Delete from t_24_Product_Plant_Data_Ext")
conn.commit()


# c.execute("INSERT INTO t_24_Product_Basic_Data_mandatory_Ext (PRODUCT) SELECT MATNR FROM MARA WHERE MTART = 'ROH';")
# conn.commit()

# c.execute("""UPDATE t_24_Product_Basic_Data_mandatory_Ext
# SET MEINS = (
#     SELECT COALESCE(
#         (SELECT T1.MEINS FROM MARA_500 AS T1 WHERE T1.MATNR = t_24_Product_Basic_Data_mandatory_Ext.Product),
#         (SELECT T2.MEINS FROM MARA_700 AS T2 WHERE T2.MATNR = t_24_Product_Basic_Data_mandatory_Ext.Product AND T2.MTART = 'ROH'),     
#         (SELECT T3.MEINS FROM MARA AS T3 WHERE T3.MATNR = t_24_Product_Basic_Data_mandatory_Ext.Product)
#     )
# );""")
# conn.commit()