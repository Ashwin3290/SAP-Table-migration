from dotenv import load_dotenv
import json
import os
import pandas as pd
import re
import sqlite3
from google import genai
from google.genai import types

load_dotenv()

joined_df = pd.read_csv("joined_data.csv")


def fetch_data_by_ids(object_id, segment_id, project_id,conn):  
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

    return joined_df

def parse_data(joined_df,query):

    prompt = """
    Role: You are an expert data mapping assistant.

    Goal: Extract specific technical data mapping details based on a user's query, clearly distinguishing between fields used for filtering conditions versus fields targeted for data insertion.

    Inputs:
    1. User Query (`question`): {question}
    This query will contain descriptive terms referring to data fields, some used for filtering and others for selection/insertion.
    2. Table Description (`table_desc`): {table_desc}
    This contains the mapping rules and metadata in a structured format with columns: `fields`, `description`, `isMandatory`, `isKey`, `source_field_name`, `target_sap_table`, `target_sap_field`, `segment_name`, `table_name`, `source_table`.

    Task:

    1. Identify Relevant Row(s):
    - Analyze the `User Query` to identify all descriptive terms the user is asking about.
    - Match these terms against values in the `description` column of the `Table Description`.
    - Determine which fields are being used for filtering conditions versus which fields are being selected/inserted.
    - The match should be robust enough to handle partial descriptions.

    2. Extract Information: Extract the following details precisely from the corresponding columns in the `Table Description` and consolidate them into a single JSON object:
    * `query_terms_matched`: List of all descriptive terms from the query that were matched
    * `target_sap_fields`: List of values from the `target_sap_field` column for all matched terms
    * `target_table_name`: The value from the `table_name` column
    * `source_table_name`: The value from the `source_table` column
    * `sap_structure`: The value from the `sap_structure` column
    * `source_field_names`: List of values from the `source_field_name` column for all matched terms
    * `target_sap_table`: The value from the `target_sap_table` column
    * `segment_name`: The value from the `segment_name` column
    * `filtering_fields`: List of field names (from `source_field_name`) that are used for filtering conditions
    * `insertion_fields`: List of field pairs as {{source_field: "X", target_field: "Y"}} that represent data to be inserted
    * `restructured_question`: The original question with descriptive terms replaced by their technical field names

    Output Format:
    * Present all extracted information in a single, consolidated JSON object
    * If no match is found for any term in the query, explicitly state that
    * Group common values that should be the same across all matches (like table_name, sap_structure, etc.)

    Example JSON Output Structure:
    ```json
    {{
    "query_terms_matched": ["Material Number", "Material Type"],
    "target_sap_fields": ["PRODUCT", "MTART"],
    "target_table_name": "t_24_Product_Basic_Data_mandatory",
    "source_table_name": "MARA",
    "sap_structure": "S_MARA",
    "source_field_names": ["MATNR", "MTART"],
    "target_sap_table": "S_MARA",
    "segment_name": "Basic Data (mandatory)",
    "filtering_fields": ["MTART"],
    "insertion_fields": [{{"source_field": "MATNR", "target_field": "PRODUCT"}}],
    "restructured_question": "Select MATNR from source table where MTART = 'ROH' and insert into PRODUCT field of target table"
    }}
    ```
    """
        # table_desc = pd.read_csv("joined_data.csv") 
    joined_df.to_csv( "joined_data.csv", index=False) 
    table_desc = joined_df
    prompt = prompt.format(question = query
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
            
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print("Raw response:")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")
    except Exception as e:
        return None

def proccess_info(resolved_data,conn):
    source_df=pd.read_sql_query(f"Select * from {resolved_data['source_table_name']} limit 5", conn)[resolved_data['source_field_names']]
    target_df=pd.read_sql_query(f"Select * from {resolved_data['target_table_name']} limit 5", conn)[resolved_data['target_sap_fields']]
    
    return {
        "source_info":source_df.info(),
        "target_info":target_df.info(),
        "source_describe":source_df.describe(),
        "target_describe":target_df.describe(),
        "restructured_question": resolved_data['restructured_question'],
        "filtering_fields": resolved_data['filtering_fields'],
        "insertion_fields": resolved_data['insertion_fields'],
        "target_table_name": resolved_data['target_table_name'],
        "source_table_name": resolved_data['source_table_name'],
    }

def process_query(object_id,segment_id,project_id,query):
    conn = sqlite3.connect('db.sqlite3')
    joined_df = fetch_data_by_ids(object_id, segment_id, project_id,conn)
    resolved_data = parse_data(joined_df,query)
    results = proccess_info(resolved_data,conn)
    conn.close()
    return results
