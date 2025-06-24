from DMtool.dmtool import DMTool

dm_tool = DMTool()

query = """

Add a new column into the target table with column name "LIQDT_Day" and find the Day of the week (Monday , Tuesday , ......) of the date present in the LIQDT field of target table and update in LIQDT_Day field
"""
result, session_id = dm_tool.process_sequential_query(
    query,
    object_id=41,
    segment_id=577,
    project_id=24,
    session_id = '6239dcf9-1dd0-423f-8f0e-a065335d58d1'
    # session_id='f40e92c5-46cf-4670-9bf4-337f3d09d690'
) 
print(f"Result: {result}")
print(f"Session ID: {session_id}")