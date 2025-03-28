import sqlite3
import pandas as pd

def fetch_data_by_ids(db_path, object_id, segment_id, project_id):
    conn = sqlite3.connect(db_path)   
    joined_query = """
SELECT 
    f.fields,
    f.description,
    f.isMandatory,
    f.isKey,
    f.sap_structure,
    r.source_table,
    r.source_field_name,
    r.target_sap_table,
    r.target_sap_field,
    s.segement_name,
    s.table_name
FROM connection_fields f
LEFT JOIN (
    SELECT r1.*
    FROM connection_rule r1
    INNER JOIN (
        SELECT field_id_id, MAX(version_id) as max_version
        FROM connection_rule
        WHERE object_id_id = ? 
        AND segment_id_id = ? 
        AND project_id_id = ? 
        GROUP BY field_id_id
    ) r2 ON r1.field_id_id = r2.field_id_id AND r1.version_id = r2.max_version
    WHERE r1.object_id_id = ? 
    AND r1.segment_id_id = ? 
    AND r1.project_id_id = ? 
) r ON f.field_id = r.field_id_id
JOIN connection_segments s ON f.segement_id_id = s.segment_id
    AND f.obj_id_id = s.obj_id_id
    AND f.project_id_id = s.project_id_id
WHERE f.obj_id_id = ? 
AND f.segement_id_id = ? 
AND f.project_id_id = ? 
    """
    
    params = [object_id, segment_id, project_id] * 3
    joined_df = pd.read_sql_query(joined_query, conn, params=params)
    conn.close()    
    return joined_df

