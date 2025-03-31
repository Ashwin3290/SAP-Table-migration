from dotenv import load_dotenv
import google.generativeai as genai
import json
import os
import pandas as pd
import re

joined_df = pd.read_csv("joined_data.csv")

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
   * `table_name`: The value from the `table_name` column
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
  "table_name": "t_24_Product_Basic_Data_mandatory",
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


table_desc = joined_df
prompt = prompt.format(question = "Bring Material Number with Material Type = ROH from MARA Table"
,table_desc = table_desc.to_csv(index=False))
api_key = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=api_key)
"""Generate response using Gemini API"""
try:
    model_instance = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 2048,
            "top_p": 0.95,
            "top_k": 40,
        },
    )
    response = model_instance.generate_content(prompt)
    try:
# Try to find JSON block between ```json and ```
        json_str = re.search(r'```json(.*?)```', response.text, re.DOTALL)
        if json_str:
            json_data = json.loads(json_str.group(1).strip())
        else:
            # If no code block markers, try parsing the whole response as JSON
            json_data = json.loads(response.text.strip())
        
        # Save to file
        with open("resolved_query.json", 'w') as file:
            json.dump(json_data, file, indent=2)
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print("Raw response:")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
except Exception as e:
    None

# prompt ="""You are expert planner and you given a user's query, and a json object containing the exact extracted mapping details from query
# You task is to make detailed plan for creating a python code to help with the code generation agents, You have to given a logical plan and not the code info.
# We will be giving 2 dataframes to the agents as we have extracted into dataframes with the columns and info u see. consider dataframes as tables and the columns as fields.
# you have to identify the method required to get the data and what exact 1 column is getting affected since the queries will be related to transformation and we will be given a query that requires to change or insert a column
# query: {query}
# details: {json}
# """

# # model_instance = genai.GenerativeModel(
# #         model_name="gemini-2.0-flash",
# #         generation_config={
# #             "temperature": 0.3,
# #             "max_output_tokens": 2048,
# #             "top_p": 0.95,
# #             "top_k": 40,
# #         },
# #     )
# prompt = prompt.format(query = "Bring Material Number with Material Type = ROH from MARA Table"
# ,json = json_data)
# response = model_instance.generate_content(prompt)
# print(response.text)


