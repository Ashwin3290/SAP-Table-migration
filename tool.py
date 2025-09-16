from DMtool.dmtool import DMTool

dm_tool = DMTool()

query = """ 
Check Materials which you have got from Transaofmration rule 1 In MARA_500 table and
IF
matching Entries found, then bring Unit of Measure   field from MARA_500 table to the Target Table
ELSE,
If no entries found in MARA_500, then check ROH  Material  ( found in Transformation 2 ) in MARA_700 Table and bring the Unit of Measure
ELSE,
If no entries found in MARA_700, then bring the Unit of measure from MARA table
"""
# session_id = dm_tool.create_session_id()
session_id = '283a90d3-366f-453e-9690-84149706868b'
result, affected_indexes = dm_tool.process_sequential_query(
    query,
    object_id=41,
    segment_id=577,
    project_id=24,
    session_id=session_id
) 
# result, affected_indexes = dm_tool.process_selection_criteria(
#     "Filter and update based on Material type = ROH",
#     object_id=41,
#     segment_id=577,
#     project_id=24,
#     session_id=session_id
# ) 
print("session id:", session_id)
print(f"Result: {result}")
print(f"Session ID: {affected_indexes}")