from DMtool.dmtool import DMTool

dm_tool = DMTool()

query = """ 
Check Materials of target table In MARA_500 table and IF matching Entries found, then bring Material Type field from MARA_500 table to the Target Table ELSE, If no entries found in MARA_500, then check Materials of target table in MARA_700 Table and bring the Material Type ELSE, bring the Material Type from MARA table
"""
result, session_id = dm_tool.process_sequential_query(
    query,
    object_id=41,
    segment_id=577,
    project_id=24,
) 
print(f"Result: {result}")
print(f"Session ID: {session_id}")