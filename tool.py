from dmtool import DMTool

dm_tool = DMTool()

query = """
From Land1_Mapping Table take the transportation_zone value corresponding to the COUNTRY value and put into LZONE for all customers of target table"""
result, session_id = dm_tool.process_sequential_query(
    query,
    object_id=42,
    segment_id=604,
    project_id=24,
    session_id = '6239dcf9-1dd0-423f-8f0e-a065335d58d1'
    # session_id='f40e92c5-46cf-4670-9bf4-337f3d09d690'
) 
print(f"Result: {result}")
print(f"Session ID: {session_id}")