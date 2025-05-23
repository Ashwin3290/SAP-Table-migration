from dmtool_sql import DMToolSQL

# Initialize the system
sql_tool = DMToolSQL()

# Process a natural language query
query = """
For every material in the target table, fetch the LAEDA field from the MARA table (using  MATNR as the key), and update the LIQDT field in your target table with this value.
"""
result, session_id = sql_tool.process_sequential_query(
    query,
    object_id=41,
    segment_id=577,
    project_id=24,
    session_id = '6239dcf9-1dd0-423f-8f0e-a065335d58d1'

)
print(f"Result: {result}")
print(f"Session ID: {session_id}")