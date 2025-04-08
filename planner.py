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
def parse_data_with_context(joined_df, query, previous_context=None, target_table_desc=None):
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
    You are a data transformation assistant specializing in SAP data mappings. 
Your task is to analyze a natural language query about data transformations and match it to the appropriate source and target tables and fields.
 
CONTEXT DATA SCHEMA: {table_desc}
 
USER QUERY: {question}

INSTRUCTIONS:
1. Identify key entities in the query:
   - Source table(s)
   - Source field(s)
   - Target table(s)
   - Target field(s)
   - Filtering or transformation conditions
   - Logical flow (IF/THEN/ELSE statements)
 
2. Match these entities to the corresponding entries in the joined_data.csv schema
   - For each entity, find the closest match in the schema
   - Resolve ambiguities using the description field
   - Validate that the identified fields exist in the mentioned tables
 
3. Generate a structured representation of the transformation logic:
   - JSON format showing the transformation flow
   - Include all source tables, fields, conditions, and targets
   - Map conditional logic to proper syntax
   - Handle fallback scenarios (ELSE conditions)
 
4. Provide a summary of the identified transformation in both natural language and structured format.
 
Respond with:
```json
{{
source_table_name: [List of all source_tables],
source_field_names: [List of all source_fields],
filtering_fields: [List of filtering fields],
transformation_logic: [Detailed transformation logic],
}}
```
    """
    
    # Format the previous context for the prompt
    context_str = "None" if previous_context is None else json.dumps(previous_context, indent=2)
    
    # Format the prompt with all inputs
    joined_df.to_csv("joined_data.csv", index=False) 
    table_desc = joined_df
    formatted_prompt = prompt.format(
        question=query,
        table_desc=table_desc.to_csv(index=False),
        previous_context=context_str,
        table_desc = joined_df
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
    """Process the resolved data to extract table information based on the specified JSON structure"""
    try:
        # Initialize result dictionary with only the requested fields
        result = {
            "source_table_name": resolved_data['source_table_name'],
            "source_field_names": resolved_data['source_field_names'],
            "target_table_name": resolved_data['target_table_name'],
            "target_sap_fields": resolved_data['target_sap_fields'],
            "filtering_fields": resolved_data['filtering_fields'],
            "transformation_logic": resolved_data['transformation_logic']
        }
        
        # Add data samples from each source table (first 5 rows)
        source_data = {}
        for table in resolved_data['source_table_name']:
            source_df = pd.read_sql_query(
                f"SELECT {','.join(resolved_data['source_field_names'])} FROM {table} LIMIT 5", 
                conn
            )
            source_data[table] = source_df.to_dict('records')
        result['source_data_samples'] = source_data
        
        # Add target table data sample (first 5 rows)
        target_df = pd.read_sql_query(
            f"SELECT {','.join(resolved_data['target_sap_fields'])} FROM {resolved_data['target_table_name']} LIMIT 5", 
            conn
        )
        result['target_data_sample'] = target_df.to_dict('records')
        
        return result
        
    except Exception as e:
        print(f"Error processing info: {e}")
        return None

def process_query(object_id, segment_id, project_id, query, session_id=None, target_sap_fields=None):
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

if __name__ == "__main__":
    # Example usage
    conn = sqlite3.connect('db.sqlite3')
    object_id = 41
    segment_id = 577
    project_id = 24
    query = """Check Materials which you have got from Transaofmration rule In MARA_500 table and
IF
matching Entries found, then bring Unit of Measure   field from MARA_500 table to the Target Table
ELSE,
If no entries found in MARA_500, then check ROH  Material  ( found in Transformation 2 ) in MARA_700 Table and bring the Unit of Measure
ELSE,
If no entries found in MARA_700, then bring the Unit of measure from MARA table

"""
    
    # Process the query and get results
    results = process_query(object_id, segment_id, project_id, query)
    print(results)