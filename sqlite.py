import sqlite3
import os
import re
import json
import pandas as pd
import pandas as pd    
from dotenv import load_dotenv
from google import genai
from google.genai import types

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
           Match these descriptive terms against the values in the `description` column of the `Table Description` to find the corresponding rows and field names from `fields column`. The match should be robust enough to handle partial descriptions.

    2.  Extract Information: For all matched rows, extract the following details precisely from the corresponding columns in the `Table Description` and consolidate them into a single JSON object:
        *   `query_terms_matched`: List of all descriptive terms from the query that were matched
        *   `target_sap_fields`: List of values from the `target_sap_fields` column for all matched terms
        *   `table_name`: The value from the `table_name` column (should be the same for all matched fields in this context)
        *   `sap_structure`: The value from the `sap_struct` column (should be the same for all matched fields in this context)
        *   `source_field_names`: List of values from the `source_field` column for all matched terms
        *   `target_sap_table`: The value from the `target_sap_table` column (should be the same for all matched fields in this context)
        *   `segment_name`: The value from the `segement` column (should be the same for all matched fields in this context)
        *   `restructured_question`: The original question with descriptive terms replaced by their technical field names

    Output Format:

    *   Present all extracted information in a single, consolidated JSON object
    *   If no match is found for any term in the query, explicitly state that
    *   Group common values that should be the same across all matches (like table_name, sap_structure, etc.)

    Example JSON Output Structure (if matches found):

    ```json
    {{
    "query_term_matched": "[All Descriptive terms from query]",
    "target_sap_field": "[All Values from 'fields' column]",
    "table_name": "[All Values from 'table_name' column if distinct]",
    "sap_structure": "[All Values from 'sap_struct' column if distinct]",
    "source_field_name": "[Value from 'source_field' column]",
    "target_sap_table": "[Value from 'target_sap_table' column]",
    "segment_name": "[Value from 'segement' column]"
    "Question" : "[Restructured Question]"
    }}
    ```
"""

    # table_desc = pd.read_csv("joined_data.csv") 
    joined_df.to_csv( "joined_data.csv", index=False) 
    table_desc = joined_df
    prompt = prompt.format(question = "Bring Material Number with Material Type = ROH from MARA Table"
,table_desc = table_desc.to_csv(index=False))
    api_key = os.environ.get('GEMINI_API_KEY')
    client = genai.Client(api_key = api_key)
    """Generate response using Gemini API"""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents = prompt,
            config=types.GenerateContentConfig(
                temperature=0.5,  # Corrected: keyword argument, not a string
                max_output_tokens=2048, # Corrected: keyword argument, not a string
                top_p=0.95,    # Corrected: keyword argument, not a string
                top_k=40      #
            )
        )
        try:
    # Try to find JSON block between ```json and ```
            json_str = re.search(r'```json(.*?)```', response.text, re.DOTALL)
            if json_str:
                json_data = json.loads(json_str.group(1).strip())
            else:
                json_data = json.loads(response.text.strip())
            
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print("Raw response:")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")
    except Exception as e:
        return None

    

    results = fetch_data_by_ids(db_path, object_id, segment_id, project_id)
    
    joined_df = results

    print("\nJoined Data DataFrame:")
    print(joined_df)
    print("\nJSON Data:")
    with open("resolved_query.json", 'r') as file:
        json_data = json.load(file)
        print(json.dumps(json_data, indent=2))

