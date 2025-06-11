from dmtool import DMTool

# Initialize the system
dm_tool = DMTool()
#Extract Description from MAKT table for the Materials got from t_24_Product_Basic_Data_mandatory_Ext where MATNR is equal to product

# Process a natural language query
query = """
Remove special characters from MAKT Descriptions
"""
result, session_id = dm_tool.process_sequential_query(
    query,
    object_id=41,
    segment_id=578,
    project_id=24,
    session_id = '6239dcf9-1dd0-423f-8f0e-a065335d58d1'
) 
print(f"Result: {result}")
print(f"Session ID: {session_id}")