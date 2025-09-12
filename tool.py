from DMtool.dmtool import DMTool

dm_tool = DMTool()

query = """ 
If Material Group in ( L002, ZMGRP1,202R ) then "GENAI01'
ELSE IF 
Material Group in ( '01','L001','ZSUK1RM1') then "GENAI02'
ELSE IF Material Group in ( 'CH001','02') then "GenAI03'
Else 'GenAINo'(MARA Table)
"""
session_id = dm_tool.create_session_id()
# result, affected_indexes = dm_tool.process_sequential_query(
#     query,
#     object_id=41,
#     segment_id=577,
#     project_id=24,
#     session_id=session_id
# ) 
result, affected_indexes = dm_tool.process_selection_criteria(
    "Filter based on Material type = ROH",
    object_id=41,
    segment_id=577,
    project_id=24,
    session_id=session_id
) 
print(f"Result: {result}")
print(f"Session ID: {affected_indexes}")