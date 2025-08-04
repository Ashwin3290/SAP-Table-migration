from DMtool.dmtool import DMTool

dm_tool = DMTool()

query = """

Bring Material Number with Material Type = ROH from MARA Table
"""
result, session_id = dm_tool.process_sequential_query(
    query,
    object_id=41,
    segment_id=577,
    project_id=24,
) 
print(f"Result: {result}")
print(f"Session ID: {session_id}")