from dotenv import load_dotenv
import json
import uuid
import os
import pandas as pd
import re
import sqlite3
from io import StringIO
from datetime import datetime
from google import genai
from google.genai import types
from token_tracker import track_token_usage, get_token_usage_stats
from pathlib import Path

load_dotenv()

class ContextualSessionManager:
    """
    Manages context and state for sequential data transformations
    """
    def __init__(self, storage_path="sessions"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def create_session(self):
        """Create a new session and return its ID"""
        session_id = str(uuid.uuid4())
        session_path = f"{self.storage_path}/{session_id}"
        os.makedirs(session_path, exist_ok=True)
        
        # Initialize empty context
        context = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "transformation_history": [],
            "target_table_state": {
                "populated_fields": [],
                "remaining_mandatory_fields": [],
                "total_rows": 0,
                "rows_with_data": 0
            }
        }
        
        # Save initial context
        with open(f"{session_path}/context.json", "w") as f:
            json.dump(context, f, indent=2)
        
        return session_id
    
    def get_context(self, session_id):
        """Get the current context for a session"""
        context_path = f"{self.storage_path}/{session_id}/context.json"
        if not os.path.exists(context_path):
            return None
        
        with open(context_path, "r") as f:
            return json.load(f)
    
    def update_context(self, session_id, resolved_data):
        """Update the context with new resolved data"""
        context_path = f"{self.storage_path}/{session_id}/context.json"
        
        # Save the entire resolved data object (which includes the updated context)
        with open(context_path, "w") as f:
            json.dump(resolved_data, f, indent=2)
        
        # Also save as a versioned snapshot for history
        snapshot_path = f"{self.storage_path}/{session_id}/context_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        with open(snapshot_path, "w") as f:
            json.dump(resolved_data, f, indent=2)
        
        return resolved_data
    
    def get_transformation_history(self, session_id):
        """Get the transformation history for a session"""
        context = self.get_context(session_id)
        if not context or "context" not in context:
            return []
        
        return context["context"]["transformation_history"]


def fetch_data_by_ids(object_id, segment_id, project_id, conn):  
    """Fetch data mappings from the database"""
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


def missing_values_handling(df):
    """Handle missing values in the dataframe"""
    # Debug: Show initial state
    if 'source_table' in df.columns:
    # Convert empty strings and whitespace-only to NaN first
        df['source_table'] = df['source_table'].replace(r'^\s*$', pd.NA, regex=True)
    
    # # Now handle both original NaNs and newly converted ones
    # if not df['source_table'].dropna().empty:
    #     fill_value = df['source_table'].dropna().iloc[0]
    #     df['source_table'] = df['source_table'].fillna(fill_value)
    #     print(f"\nFilled {df['source_table'].isna().sum()} nulls in source_table with '{fill_value}'")

    # Handle source_field_name
    if 'source_field_name' in df.columns and 'target_sap_field' in df.columns:
        # Convert empty strings to NaN first
        df['source_field_name'] = df['source_field_name'].replace(r'^\s*$', pd.NA, regex=True)
        
        null_count = df['source_field_name'].isna().sum()
        if null_count > 0:
            # Ensure we don't propagate empty strings from target_sap_field
            valid_targets = df['target_sap_field'].replace(r'^\s*$', pd.NA, regex=True).notna()
            df.loc[df['source_field_name'].isna() & valid_targets, 'source_field_name'] = \
                df.loc[df['source_field_name'].isna() & valid_targets, 'target_sap_field']
            print(f"Filled {null_count} nulls in source_field_name from target_sap_field")
    
    return df

@track_token_usage(log_to_file=True, log_path='gemini_planner_usage.log')
def parse_data_with_context(joined_df, query, previous_context=None):
    """
    Parse data using Gemini API with token usage tracking and context awareness
    
    Parameters:
    joined_df (DataFrame): The joined dataframe with field mappings
    query (str): The natural language query
    previous_context (dict): Context from previous transformations
    
    Returns:
    dict: The parsed data with mapping information and updated context
    """
    prompt = """
    Role: You are an expert data mapping assistant with context awareness.

    Goal: Extract specific technical data mapping details based on a user's query, while maintaining awareness of previous transformations.

    Inputs:
    1. User Query (`question`): {question}
       This query will contain descriptive terms referring to data fields, some used for filtering and others for selection/insertion.
    2. Table Description (`table_desc`): {table_desc}
       This contains the mapping rules and metadata in a structured format with columns: `fields`, `description`, `isMandatory`, `isKey`, `source_field_name`, `target_sap_table`, `target_sap_field`, `segment_name`, `table_name`, `source_table`.
    3. Previous Context (`previous_context`): {previous_context}
       This contains information about previous transformations if any have been performed.

    Task:

    1. Identify Relevant Row(s):
       - Analyze the `User Query` to identify all descriptive terms the user is asking about.
       - Match these terms against values in the `description` column of the `Table Description`.
       - Determine which fields are being used for filtering conditions versus which fields are being selected/inserted.
       - The match should be robust enough to handle partial descriptions.

    2. Consider Previous Context:
       - Review the previous transformations to understand what data has already been populated.
       - Consider how the current query relates to previous transformations (e.g., is it adding to previously populated data, or working with different fields).
       - Ensure consistency with previous transformations.

    3. Extract Information: Extract the following details precisely from the corresponding columns in the `Table Description` and consolidate them into a single JSON object:
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
       * `context`: Updated context information including:
       * - transformation_history: List of previous transformations plus this one
       * - target_table_state: Updated state information about populated fields

    Note:
    * Incase you encounter any other tables used as source tables then in that case add the following fields:
    * `source_table_name`: The name of the base source table identified from the table description
    * `Additional_source_table`: List of the names of the additional source table identified from the table description
    * This also is true if the source table is not the one being used in the query, then add the identified table name in the additional_source_table field with the identified target field and mark source table field as empty
    * Do not assume that only single table is used in the source table and do not mix source tables for different target fields 
    * If you dont find any table name for the related to the qery of the user in tbale description , in that case do the following:
       * Identify the table name and the apparent source field from the query and add it to the source table name and source field instead of putting it in the additional source table
       * Assume that the user is directly mentioning the table name and the source field in the query.
       
    Output Format:
    * Present all extracted information in a single, consolidated JSON object
    * If no match is found for any term in the query, explicitly state that
    * Group common values that should be the same across all matches (like table_name, sap_structure, etc.)
    * Include the updated context that builds upon the previous context

    Example JSON Output Structure:
    ```json
{{
  "query_terms_matched": ["<source_field_1>", "<source_field_2>"],
  "target_sap_fields": ["<target_field_1>", "<target_field_2>"],
  "target_table_name": "<target_table>",
  "source_table_name": "<source_table>",
  "sap_structure": "<sap_structure>",
  "source_field_names": ["<source_field_1>", "<source_field_2>"],
  "target_sap_table": "<target_sap_table>",
  "segment_name": "<segment_name>",
  "filtering_fields": ["<filtering_field>"],
  "insertion_fields": [{{
    "source_field": "<source_field_1>", 
    "target_field": "<target_field_1>"
  }}],
  "additional_source_table": ["<additional_source_table>"],
  "restructured_question": "Select <source_field_1> from <source_table> where <filtering_field> = '<filter_value>' and insert into <target_field_1> field of <target_table>",
  "context": {{
    "transformation_history": [
      {{
        "description": "Populated <target_field_1> with <source_field_1> values where <filtering_field> = '<filter_value>'",
        "fields_modified": ["<target_field_1>"],
        "filter_conditions": {{"<filtering_field>": "<filter_value>"}}
      }}
    ],
    "target_table_state": {{
      "populated_fields": ["<target_field_1>"],
      "remaining_mandatory_fields": ["<target_field_2>", "<target_field_3>"]
    }}
  }}
}}
    ```
    """
    
    # Format the previous context for the prompt
    context_str = "None" if previous_context is None else json.dumps(previous_context, indent=2)
    
    # Format the prompt with all inputs
    # joined_df.to_csv("joined_data.csv", index=False) 
    table_desc = joined_df
    formatted_prompt = prompt.format(
        question=query,
        table_desc=table_desc.to_csv(index=False),
        previous_context=context_str
    )
    
    # Call Gemini API with token tracking
    api_key = os.environ.get('GEMINI_API_KEY')
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                temperature=0.5,
                top_p=0.95,
                top_k=40
            )
        )
        
        # Log token usage statistics
        print(f"Current token usage: {get_token_usage_stats()}")
        
        # Parse the JSON response
        try:
            json_str = re.search(r'```json(.*?)```', response.text, re.DOTALL)
            if json_str:
                parsed_data = json.loads(json_str.group(1).strip())
            else:
                parsed_data = json.loads(response.text.strip())
            json.dump(parsed_data,open('response.json',"w"), indent=2)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print("Raw response:")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def process_info(resolved_data, conn):
    """Process the resolved data to extract table information"""
    # Extract source and target dataframes based on the resolved data
    try:

        source_df = pd.read_sql_query(
            f"SELECT * FROM {resolved_data['source_table_name']} LIMIT 5", 
            conn
        )[resolved_data['source_field_names']]
        
        target_df = pd.read_sql_query(
            f"SELECT * FROM {resolved_data['target_table_name']} LIMIT 5", 
            conn
        )[resolved_data['target_sap_fields']]

        if "Additional_source_table" in resolved_data:
            additional_source_table = resolved_data["Additional_source_table"]
            if additional_source_table != "None":
                additional_source_tables = {}
                for table in additional_source_table:
                    additional_source_tables[table] = pd.read_sql_query(
                        f"SELECT * FROM {table} LIMIT 5", 
                        conn
                    )
                    additional_source_tables[table] = additional_source_tables[table][resolved_data['source_field_names']]
                    additional_source_info_buffer = StringIO()
                    additional_source_tables[table].info(buf=additional_source_info_buffer)
                    additional_source_info = additional_source_info_buffer.getvalue()
                    additional_source_tables[table] = {
                        "info": additional_source_info,
                        "describe": additional_source_tables[table].describe()
                    }    
        if isinstance(resolved_data['insertion_fields'] , list):
            if resolved_data["insertion_fields"]:
                insertion_fields = resolved_data['insertion_fields'][0]["target_field"]
            else:
                insertion_fields = "None identified yet try to identify the insertion fields"
        
        elif isinstance(resolved_data['insertion_fields'] , None):
            insertion_fields = "None identified yet try to identify the insertion fields"
        return {
            "source_info": source_df,
            "target_info": target_df,
            "source_describe": source_df.describe(),
            "target_describe": target_df.describe(),
            "restructured_question": resolved_data['restructured_question'],
            "filtering_fields": resolved_data['filtering_fields'],
            "insertion_fields": insertion_fields,
            "target_table_name": resolved_data['target_table_name'],
            "source_table_name": resolved_data['source_table_name'],
            "target_sap_fields": resolved_data['target_sap_fields'],
            "source_field_names": resolved_data['source_field_names'],
            "context": resolved_data.get('context', {}),
            "additional_source_table": resolved_data.get('Additional_source_table', []),
            "additional_source_tables": additional_source_tables if "Additional_source_table" in resolved_data else None,

        }
    except Exception as e:
        print(f"Error processing info: {e}")
        return None

def process_query(object_id, segment_id, project_id, query, session_id=None):
    """
    Process a query with context awareness
    
    Parameters:
    object_id (int): Object ID
    segment_id (int): Segment ID
    project_id (int): Project ID
    query (str): The natural language query
    session_id (str): Optional session ID for context tracking
    
    Returns:
    dict: Processed information including context
    """
    # Initialize context manager
    context_manager = ContextualSessionManager()
    
    # Create a session if none provided
    if not session_id:
        session_id = context_manager.create_session()
    
    # Get existing context
    previous_context = context_manager.get_context(session_id)
    
    # Connect to database
    conn = sqlite3.connect('db.sqlite3')
    
    # Fetch mapping data
    joined_df = fetch_data_by_ids(object_id, segment_id, project_id, conn)
    print(joined_df.head())
    # Handle missing values in the dataframe
    joined_df = missing_values_handling(joined_df)
    # Check if joined_df is emptys

    joined_df.to_csv("joined_data.csv", index=False)
    # Process query with context awareness
    resolved_data = parse_data_with_context(
        joined_df, 
        query, 
        previous_context.get("context") if previous_context else None
    )
    
    if not resolved_data:
        conn.close()
        return None
    # Process the resolved data to get table information
    json.dump(resolved_data,open('resolved_data.json',"w"), indent=2)
    results = process_info(resolved_data, conn)
    if not results:
        conn.close()
        return None
    # Update the context in our session manager
    context_manager.update_context(session_id, resolved_data)
    
    # Add session_id to the results
    results["session_id"] = session_id
    
    conn.close()
    return results


def get_session_context(session_id):
    """
    Get the current context for a session
    
    Parameters:
    session_id (str): The session ID
    
    Returns:
    dict: The session context
    """
    context_manager = ContextualSessionManager()
    return context_manager.get_context(session_id)


def get_or_create_session_target_df(session_id, target_table, conn):
    """
    Get existing target dataframe for a session or create a new one
    
    Parameters:
    session_id (str): Session ID
    target_table (str): Target table name
    target_fields (list): List of target fields
    conn (Connection): SQLite connection
    
    Returns:
    DataFrame: The target dataframe
    """
    session_path = f"sessions/{session_id}"
    target_path = f"{session_path}/target_latest.csv"
    
    if os.path.exists(target_path):
        # Load existing target data
        target_df = pd.read_csv(target_path)
    else:
        # Get fresh target data from the database
        target_df = pd.read_sql_query(f"SELECT * FROM {target_table}", conn)
    
    return target_df


def save_session_target_df(session_id, target_df:pd.DataFrame):
    """
    Save the updated target dataframe for a session
    
    Parameters:
    session_id (str): Session ID
    target_df (DataFrame): The target dataframe to save
    
    Returns:
    bool: True if successful
    """
    session_path = f"sessions/{session_id}"
    os.makedirs(session_path, exist_ok=True)
    target_path = f"{session_path}/target_latest.csv"
    target_df.to_csv(target_path)
    
    return True

# if __name__ == "__main__":
#     # Example usage
#     conn = sqlite3.connect('db.sqlite3')
#     object_id = 29
#     segment_id = 336
#     project_id = 24
#     query = """Check Materials which you have got from Transaofmration rule In MARA_500 table and
# IF
# matching Entries found, then bring Unit of Measure   field from MARA_500 table to the Target Table
# ELSE,
# If no entries found in MARA_500, then check ROH  Material  ( found in Transformation 2 ) in MARA_700 Table and bring the Unit of Measure
# ELSE,
# If no entries found in MARA_700, then bring the Unit of measure from MARA table

# """
    
#     # Process the query and get results
#     results = process_query(object_id, segment_id, project_id, query)
#     print(results)