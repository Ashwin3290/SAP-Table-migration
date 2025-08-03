from DMtool.dmtool import DMTool

dm_tool = DMTool()

query = """
Join Product from Basic Sgement with Product from MARC and Bring Product and Plant field from MARC Table for the plants ( 1710, 9999 )"""
result, session_id = dm_tool.process_sequential_query(
    query,
    object_id=41,
    segment_id=592,
    project_id=24,
    session_id = 'a733264c-a858-4f22-a981-a8a5c9910be4'
) 
print(f"Result: {result}")
print(f"Session ID: {session_id}")