from dmtool import DMTool

# Initialize the system
dm_tool = DMTool()

# Process a natural language query
query = """

bring customer FROM BUT000 table for BU_GROUPING = BP03 and check if same number is available from Group (KTOKD = CUST ) in KNA1. matching entries should come under customer

"""
result, session_id = dm_tool.process_sequential_query(
    query,
    object_id=37,
    segment_id=463,
    project_id=25,
    session_id = '6239dcf9-1dd0-423f-8f0e-a065335d58d1'

)
print(f"Result: {result}")
print(f"Session ID: {session_id}")