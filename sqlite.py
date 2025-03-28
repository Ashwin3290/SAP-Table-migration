import sqlite3
import os
import pandas as pd
import pandas as pd    
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

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
            SELECT field_id, MAX(version_id) as max_version
            FROM connection_rule
            WHERE object_id_id = ? 
            AND segment_id_id = ? 
            AND project_id_id = ? 
            GROUP BY field_id
        ) r2 ON r1.field_id = r2.field_id AND r1.version_id = r2.max_version
        WHERE r1.object_id_id = ? 
        AND r1.segment_id_id = ? 
        AND r1.project_id_id = ? 
    ) r ON f.field_id = r.field_id
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

    prompt = """
    Role: You are an expert data mapping assistant.

    Goal: Extract specific technical data mapping details based on a user's query. Use the provided table description to find the relevant information, paying close attention to matching the user's descriptive terms to the correct technical field names.

    Inputs:
    1.  User Query (`question`):{question}
           This query will contain descriptive terms referring to one or more data fields.
    2.  Table Description (`table_desc`):{table_desc}
           This contains the mapping rules and metadata in a structured format with the following columns: `fields`, `description`, `isMandatory`, `isKey`, `source_field`, `target_sap_table`, `segement`, `table_name`,'source_table','target_sap_field.

    Task:

    1.  Identify Relevant Row(s):
           Analyze the `User Query` to identify the descriptive terms the user is asking about.
           Match these descriptive terms against the values in the `description` column of the `Table Description` to find the corresponding row(s) and field name from `fields column`. The match should be robust enough to handle partial descriptions.

    2.  Extract Information: For each matched row, extract the following details precisely from the corresponding columns in the `Table Description`:
        *   `target_sap_field`: The value from the `target_sap_fields` column. This is the primary technical field name you need to identify based on the user's description.
        *   `table_name`: The value from the `table_name` column.
        *   `sap_structure`: The value from the `sap_struct` column.
        *   `source_field_name`: The value from the `source_field` column.
        *   `target_sap_table`: The value from the `target_sap_table` column.
        *   `segment_name`: The value from the `segement` column.

    Output Format:

    *   Present the extracted information in a clear, structured JSON format.
    *   If the query potentially matches multiple fields/rows, list the details for each distinct match.
    *   If no match is found for a term in the query, explicitly state that.

    Example JSON Output Structure (if one match found):

    ```json
    [
    {{
    "query_term_matched": "[Descriptive term from query]",
    "target_sap_field": "[Value from 'fields' column]",
    "table_name": "[Value from 'table_name' column]",
    "sap_structure": "[Value from 'sap_struct' column]",
    "source_field_name": "[Value from 'source_fie' column]",
    "target_sap_table": "[Value from 'target_sap_tab' column]",
    "segment_name": "[Value from 'segement' column]"
    }}
    ]
    ```
    """ 

    table_desc = pd.read_csv("joined_data.csv")   
    prompt = prompt.format(question = "Extract the description and Product number",table_desc = table_desc.to_csv(index=False))
    api_key = os.environ.get('GEMINI_API_KEY')
    genai.configure(api_key=api_key)
    """Generate response using Gemini API"""
    try:
        model_instance = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.5,
                "max_output_tokens": 2048,
                "top_p": 0.95,
                "top_k": 40,
            },
        )
        response = model_instance.generate_content(prompt)
        print(response.text)
    except Exception as e:
        return None
    finally:
        return joined_df
    
    
if __name__ == "__main__":
    db_path = "db.sqlite3"
    object_id = 29
    segment_id = 336
    project_id = 24
    
    results = fetch_data_by_ids(db_path, object_id, segment_id, project_id)
    
    joined_df = results

    print("\nJoined Data DataFrame:")
    print(joined_df.head())

    joined_df.to_csv('joined_data.csv', index=False)
 


