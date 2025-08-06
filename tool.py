from DMtool.dmtool import DMTool

dm_tool = DMTool()

query = """
Extract Description from MAKT table for the Materials got in Transformation 1"""
result, session_id = dm_tool.process_sequential_query(
    query,
    object_id=41,
    segment_id=578,
    project_id=24,
    session_id="1169593f-b2b8-4744-83ad-04b15355fdd8"
) 
print(f"Result: {result}")
print(f"Session ID: {session_id}")