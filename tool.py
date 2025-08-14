# from DMtool.dmtool import DMTool

# dm_tool = DMTool()

# query = """ 
# Delete column LIQDT_Day from the target table"""
# result, session_id = dm_tool.process_sequential_query(
#     query,
#     object_id=41,
#     segment_id=577,
#     project_id=24,
#     session_id="1169593f-b2b8-4744-83ad-04b15355fdd8"
# ) 
# print(f"Result: {result}")
# print(f"Session ID: {session_id}")

# Simple usage
from DMtool.dmtool import create_dmtool

# Create DMTool instance
dmtool = create_dmtool('sqlite', database_path='db.sqlite3')

# Process natural language query
result = dmtool.process_query(
    query="Bring material number from MARA table where material type equals ROH",
    object_id=1,
    segment_id=1, 
    project_id=1
)

print(f"Success: {result['success']}")
print(f"Rows affected: {result['execution']['rows_affected']}")