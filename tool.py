from dmtool_sql import DMToolSQL

# Initialize the system
sql_tool = DMToolSQL()

# Process a natural language query
query = """
Join Material from Basic Sgement with Material from MARC segment and Bring Material and Plant field from MARC Table for the plants ( 1710, 9999 )
"""
sql_query, result, session_id = sql_tool.process_sequential_query(
    query,
    object_id=41,
    segment_id=592,
    project_id=24,
    session_id = '6239dcf9-1dd0-423f-8f0e-a065335d58d1'

)
print(f"SQL Query: {sql_query}")
print(f"Result: {result}")
print(f"Session ID: {session_id}")