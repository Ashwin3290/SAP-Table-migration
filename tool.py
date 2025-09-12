from DMtool.dmtool import DMTool

dm_tool = DMTool()

query = """ 
If Material Group in ( L002, ZMGRP1,202R ) then "GENAI01'
ELSE IF 
Material Group in ( '01','L001','ZSUK1RM1') then "GENAI02'
ELSE IF Material Group in ( 'CH001','02') then "GenAI03'
Else 'GenAINo'(MARA Table)
"""
result, session_id,validation_report = dm_tool.process_sequential_query(
    query,
    object_id=41,
    segment_id=577,
    project_id=24,
    session_id="6bf68a4b-ffb5-4c19-8a45-907b8bd47fcb"
) 
print(f"Result: {result}")
print(f"Session ID: {session_id}")