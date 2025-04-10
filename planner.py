from dotenv import load_dotenv
import json
import uuid
import os
import pandas as pd
import re
import sqlite3
from io import StringIO
from datetime import datetime
import logging
from google import genai
from google.genai import types
from token_tracker import track_token_usage, get_token_usage_stats
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SQLInjectionError(Exception):
    """Exception raised for potential SQL injection attempts."""
    pass

class SessionError(Exception):
    """Exception raised for session-related errors."""
    pass

class APIError(Exception):
    """Exception raised for API-related errors."""
    pass

class DataProcessingError(Exception):
    """Exception raised for data processing errors."""
    pass

def validate_sql_identifier(identifier):
    """
    Validate that an SQL identifier doesn't contain injection attempts
    Returns sanitized identifier or raises exception
    """
    if not identifier:
        raise SQLInjectionError("Empty SQL identifier provided")
    
    # Check for common SQL injection patterns
    dangerous_patterns = [';', '--', '/*', '*/', 'DROP', 'DELETE', 'INSERT', 'UPDATE', 'UNION', 'EXEC', 'EXECUTE']
    for pattern in dangerous_patterns:
        if pattern.lower() in identifier.lower():
            raise SQLInjectionError(f"Potentially dangerous SQL pattern found: {pattern}")
    
    # Only allow alphanumeric characters, underscores, and some specific characters
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', identifier):
        raise SQLInjectionError("SQL identifier contains invalid characters")
    
    return identifier

class ContextualSessionManager:
    """
    Manages context and state for sequential data transformations
    """
    def __init__(self, storage_path="sessions"):
        self.storage_path = storage_path
        try:
            os.makedirs(storage_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create session storage directory: {e}")
            raise SessionError(f"Failed to create session storage: {e}")
    
    def create_session(self):
        """Create a new session and return its ID"""
        try:
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
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise SessionError(f"Failed to create session: {e}")
    
    def get_context(self, session_id):
        """Get the current context for a session"""
        try:
            if not session_id:
                logger.warning("No session ID provided for get_context")
                return None
                
            context_path = f"{self.storage_path}/{session_id}/context.json"
            if not os.path.exists(context_path):
                logger.warning(f"Context file not found for session {session_id}")
                return None
            
            with open(context_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in context file for session {session_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get context for session {session_id}: {e}")
            return None
    
    def update_context(self, session_id, resolved_data):
        """Update the context with new resolved data"""
        try:
            if not session_id:
                logger.warning("No session ID provided for update_context")
                raise SessionError("No session ID provided")
                
            session_path = f"{self.storage_path}/{session_id}"
            if not os.path.exists(session_path):
                logger.warning(f"Session directory not found: {session_path}")
                os.makedirs(session_path, exist_ok=True)
                
            context_path = f"{session_path}/context.json"
            
            # Save the entire resolved data object (which includes the updated context)
            with open(context_path, "w") as f:
                json.dump(resolved_data, f, indent=2)
            
            # Also save as a versioned snapshot for history
            snapshot_path = f"{session_path}/context_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            with open(snapshot_path, "w") as f:
                json.dump(resolved_data, f, indent=2)
            
            return resolved_data
        except Exception as e:
            logger.error(f"Failed to update context for session {session_id}: {e}")
            raise SessionError(f"Failed to update context for session {session_id}: {e}")
    
    def get_transformation_history(self, session_id):
        """Get the transformation history for a session"""
        try:
            context = self.get_context(session_id)
            if not context or "context" not in context:
                return []
            
            return context["context"]["transformation_history"]
        except Exception as e:
            logger.error(f"Failed to get transformation history for session {session_id}: {e}")
            return []

    def add_key_mapping(self, session_id, target_col, source_col):
        """Add a key mapping for a session"""
        try:
            if not session_id:
                logger.warning("No session ID provided for add_key_mapping")
                raise SessionError("No session ID provided")
                
            # Validate parameters to prevent injection
            if not target_col or not isinstance(target_col, str):
                logger.warning(f"Invalid target column: {target_col}")
                return []
            
            if not source_col or not isinstance(source_col, str):
                logger.warning(f"Invalid source column: {source_col}")
                return []
                
            file_path = f"{self.storage_path}/{session_id}/key_mapping.json"
            
            # If directory doesn't exist, create it
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if not os.path.exists(file_path):
                key_mappings = []
            else:
                try:
                    with open(file_path, "r") as f:
                        key_mappings = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in key mapping file, creating new mapping")
                    key_mappings = []
                except Exception as e:
                    logger.error(f"Error reading key mapping file: {e}")
                    key_mappings = []
            
            # Add the new mapping
            key_mappings.append({"target_col": target_col, "source_col": source_col})
            
            # Save the updated mappings
            with open(file_path, "w") as f:
                json.dump(key_mappings, f, indent=2)
                
            return key_mappings
        except Exception as e:
            logger.error(f"Failed to add key mapping for session {session_id}: {e}")
            return []
        
    def get_key_mapping(self, session_id):
        """Get key mappings for a session"""
        try:
            if not session_id:
                logger.warning("No session ID provided for get_key_mapping")
                return []
                
            file_path = f"{self.storage_path}/{session_id}/key_mapping.json"
            if not os.path.exists(file_path):
                return []
                
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in key mapping file for session {session_id}")
            return []
        except Exception as e:
            logger.error(f"Failed to get key mapping for session {session_id}: {e}")
            return []

def fetch_data_by_ids(object_id, segment_id, project_id, conn):  
    """Fetch data mappings from the database"""
    try:
        # Validate parameters
        if not all(isinstance(x, int) for x in [object_id, segment_id, project_id]):
            logger.error("Invalid parameter types for fetch_data_by_ids")
            raise ValueError("Object ID, segment ID, and project ID must be integers")
        
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

        if joined_df.empty:
            logger.warning(f"No data found for object_id={object_id}, segment_id={segment_id}, project_id={project_id}")

        return joined_df
    except sqlite3.Error as e:
        logger.error(f"SQLite error in fetch_data_by_ids: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in fetch_data_by_ids: {e}")
        raise


def missing_values_handling(df):
    """Handle missing values in the dataframe"""
    try:
        # Check if dataframe is empty or None
        if df is None or df.empty:
            logger.warning("Empty dataframe passed to missing_values_handling")
            return df
            
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Handle source_table column
        if 'source_table' in df_processed.columns:
            # Convert empty strings and whitespace-only to NaN first
            df_processed['source_table'] = df_processed['source_table'].replace(r'^\s*$', pd.NA, regex=True)
            
            # Fill NaN values if there are any non-NaN values
            if not df_processed['source_table'].dropna().empty:
                non_na_values = df_processed['source_table'].dropna()
                if len(non_na_values) > 0:
                    fill_value = non_na_values.iloc[0]
                    df_processed['source_table'] = df_processed['source_table'].fillna(fill_value)
                    logger.info(f"Filled {df_processed['source_table'].isna().sum()} nulls in source_table with '{fill_value}'")
        
        # Handle source_field_name based on target_sap_field
        if 'source_field_name' in df_processed.columns and 'target_sap_field' in df_processed.columns:
            # Convert empty strings to NaN first
            df_processed['source_field_name'] = df_processed['source_field_name'].replace(r'^\s*$', pd.NA, regex=True)
            
            # Count null values
            null_count = df_processed['source_field_name'].isna().sum()
            if null_count > 0:
                # Ensure we don't propagate empty strings from target_sap_field
                df_processed['target_sap_field'] = df_processed['target_sap_field'].replace(r'^\s*$', pd.NA, regex=True)
                valid_targets = df_processed['target_sap_field'].notna()
                
                # Fill missing source fields with target fields where available
                missing_sources = df_processed['source_field_name'].isna()
                fill_indices = missing_sources & valid_targets
                
                if fill_indices.any():
                    df_processed.loc[fill_indices, 'source_field_name'] = df_processed.loc[fill_indices, 'target_sap_field']
                    logger.info(f"Filled {fill_indices.sum()} nulls in source_field_name from target_sap_field")
        
        return df_processed
    except Exception as e:
        logger.error(f"Error in missing_values_handling: {e}")
        # Return the original dataframe if there's an error
        return df

@track_token_usage(log_to_file=True, log_path='gemini_planner_usage.log')
def parse_data_with_context(joined_df, query, session_id=None, previous_context=None, target_table_desc=None):
    """
    Parse data using Gemini API with token usage tracking and context awareness
    
    Parameters:
    joined_df (DataFrame): The joined dataframe with field mappings
    query (str): The natural language query
    session_id (str): The current session ID for retrieving key mappings
    previous_context (dict): Context from previous transformations
    
    Returns:
    dict: The parsed data with mapping information and updated context
    """
    try:
        # Validate inputs
        if joined_df is None or joined_df.empty:
            logger.error("Empty dataframe passed to parse_data_with_context")
            raise ValueError("Empty dataframe provided")
            
        if not query or not isinstance(query, str):
            logger.error(f"Invalid query passed to parse_data_with_context: {type(query)}")
            raise ValueError("Query must be a non-empty string")
        
        # Get key mapping if session_id is provided
        key_mapping = []
        target_df_sample = None
        if session_id:
            try:
                # Initialize context manager to get key mappings
                context_manager = ContextualSessionManager()
                key_mapping = context_manager.get_key_mapping(session_id)
                
                # Get the current state of target data if available
                try:
                    # Get the target table name from joined_df
                    target_table = joined_df["table_name"].unique().tolist()
                    if target_table and len(target_table) > 0:
                        # Get a connection to fetch current target data
                        conn = sqlite3.connect('db.sqlite3')
                        target_df = get_or_create_session_target_df(session_id, target_table[0], conn)
                        target_df_sample = target_df.head(5).to_dict('records') if not target_df.empty else []
                        conn.close()
                except Exception as e:
                    logger.warning(f"Error getting target data sample: {e}")
                    target_df_sample = []
            except Exception as e:
                logger.warning(f"Error retrieving key mapping for session {session_id}: {e}")
                # Continue with empty key mapping if there's an error
        
        prompt = """
        You are a data transformation assistant specializing in SAP data mappings. 
    Your task is to analyze a natural language query about data transformations and match it to the appropriate source and target tables and fields.
     
    CONTEXT DATA SCHEMA: {table_desc}
    
    CURRENT TARGET TABLE STATE:
    {target_df_sample}
    
    KEY MAPPINGS (target_field:source_field format):
    {key_mapping}
     
    USER QUERY: {question}

    INSTRUCTIONS:
    1. Identify key entities in the query:
       - Source table(s)
       - Source field(s)
       - Filtering or transformation conditions
       - Logical flow (IF/THEN/ELSE statements)
       - insertion fields
     
    2. Match these entities to the corresponding entries in the joined_data.csv schema
       - For each entity, find the closest match in the schema
       - Resolve ambiguities using the description field
       - Validate that the identified fields exist in the mentioned tables
     
    3. Generate a structured representation of the transformation logic:
       - JSON format showing the transformation flow
       - Include all source tables, fields, conditions, and targets
       - Map conditional logic to proper syntax
       - Handle fallback scenarios (ELSE conditions)
       - Use the provided key mappings to connect source and target fields correctly
       - Consider the current state of the target data shown above
     
    4. Provide a summary of the identified transformation in both natural language and structured format.

    5. For the insertion fields, identify the fields that need to be inserted into the target table based on the transformation logic. Take note to not add the filtering fields to the insertion fields if not specifically requested.

    6. Restructure the user query with resolved data and transformation logic.

    Note:
    - When working with key mappings, make sure your transformation logic uses them to properly match records between source and target tables.
    - Use the key mappings to determine the relationship between source and target fields (format is target_field:source_field).
    - Consider the current state of the target table when determining what needs to be inserted or updated.
    - if the user says about something that previous transformations, use target table as a way to know what is already done.
     
    Respond with:
    ```json
    {{
    "source_table_name": [List of all source_tables],
    "source_field_names": [List of all source_fields],
    "filtering_fields": [List of filtering fields],
    "insertion_fields": [List of fields to be inserted],
    "transformation_logic": [Detailed transformation logic that uses key mappings],
    "Resolved_query": [Rephrased query with resolved data]
    }}
    ```
        """
        
        # Format the previous context for the prompt
        context_str = "None" if previous_context is None else json.dumps(previous_context, indent=2)
        
        # Format key mapping for the prompt
        key_mapping_str = "No key mappings available"
        if key_mapping:
            try:
                formatted_mappings = []
                for mapping in key_mapping:
                    if isinstance(mapping, dict) and "target_col" in mapping and "source_col" in mapping:
                        formatted_mappings.append(f"{mapping['target_col']}:{mapping['source_col']}")
                key_mapping_str = ", ".join(formatted_mappings) if formatted_mappings else "No key mappings available"
            except Exception as e:
                logger.warning(f"Error formatting key mappings: {e}")
                key_mapping_str = "Error processing key mappings"
        
        # Format target data sample for the prompt
        target_df_sample_str = "No current target data available"
        if target_df_sample:
            try:
                target_df_sample_str = json.dumps(target_df_sample, indent=2)
            except Exception as e:
                logger.warning(f"Error formatting target data sample: {e}")
                target_df_sample_str = "Error processing target data sample"
        
        # Save joined data for debugging
        try:
            joined_df.to_csv("joined_data.csv", index=False)
        except Exception as e:
            logger.warning(f"Failed to save joined_data.csv: {e}")
            
        # Format the prompt with all inputs
        table_desc = joined_df[joined_df.columns.tolist()[1:7]]
        formatted_prompt = prompt.format(
            question=query,
            table_desc=table_desc.to_csv(index=False),
            key_mapping=key_mapping_str,
            target_df_sample=target_df_sample_str
        )
        
        # Get Gemini API key from environment
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise APIError("Gemini API key not configured")
            
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
        # Call Gemini API with token tracking and error handling
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
            
            # Check if response is valid
            if not response or not hasattr(response, 'text'):
                logger.error("Invalid response from Gemini API")
                raise APIError("Failed to get valid response from Gemini API")
                
            # Log token usage statistics
            logger.info(f"Current token usage: {get_token_usage_stats()}")
            
            # Parse the JSON response with error handling
            try:
                json_str = re.search(r'```json(.*?)```', response.text, re.DOTALL)
                if json_str:
                    parsed_data = json.loads(json_str.group(1).strip())
                else:
                    # Try to parse the whole response as JSON
                    parsed_data = json.loads(response.text.strip())
                
                # Validate the parsed data structure
                required_keys = ["source_table_name", "source_field_names", "filtering_fields", 
                                "insertion_fields", "transformation_logic", "Resolved_query"]
                
                for key in required_keys:
                    if key not in parsed_data:
                        logger.warning(f"Missing key in Gemini response: {key}")
                        parsed_data[key] = [] if key != "transformation_logic" and key != "Resolved_query" else ""
                
                # Add target table information
                parsed_data["target_table"] = joined_df["table_name"].unique().tolist()
                
                # Add key mapping information to the result
                parsed_data["key_mapping"] = key_mapping
                
                # Save response for debugging
                try:
                    with open('response.json', 'w') as f:
                        json.dump(parsed_data, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to save response.json: {e}")
                
                return parsed_data
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {e}")
                logger.error(f"Raw response: {response.text}")
                raise DataProcessingError(f"Failed to parse Gemini response as JSON: {e}")
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise APIError(f"Failed to call Gemini API: {e}")
    except Exception as e:
        logger.error(f"Error in parse_data_with_context: {e}")
        return None


def process_info(resolved_data, conn, target_sap_fields):
    """Process the resolved data to extract table information based on the specified JSON structure"""
    try:
        # Validate inputs
        if resolved_data is None:
            logger.error("None resolved_data passed to process_info")
            return None
            
        if conn is None:
            logger.error("None database connection passed to process_info")
            return None
            
        # Validate required fields in resolved_data
        required_fields = ['source_table_name', 'source_field_names', 'target_table', 
                          'filtering_fields', 'transformation_logic', 'Resolved_query', 'insertion_fields']
        
        for field in required_fields:
            if field not in resolved_data:
                logger.error(f"Missing required field in resolved_data: {field}")
                return None
                
        # Initialize result dictionary with only the requested fields
        result = {
            "source_table_name": resolved_data['source_table_name'],
            "source_field_names": resolved_data['source_field_names'],
            "target_table_name": resolved_data['target_table'],
            "target_sap_fields": target_sap_fields,
            "filtering_fields": resolved_data['filtering_fields'],
            "transformation_logic": resolved_data['transformation_logic'],
            "restructured_query": resolved_data['Resolved_query'],
            "insertion_fields": resolved_data['insertion_fields']
        }
        
        # Add data samples from each source table (first 5 rows)
        source_data = {}
        try:
            for table in resolved_data['source_table_name']:
                # Validate table name to prevent SQL injection
                safe_table = validate_sql_identifier(table)
                
                # Validate field names
                safe_fields = []
                for field in resolved_data['source_field_names']:
                    safe_fields.append(validate_sql_identifier(field))
                
                if not safe_fields:
                    logger.warning(f"No valid fields for table {safe_table}")
                    source_data[table] = pd.DataFrame()
                    continue
                    
                # Build query with parameterized values
                query = f"SELECT {','.join(safe_fields)} FROM {safe_table} LIMIT 5"
                
                try:
                    source_df = pd.read_sql_query(query, conn)
                    source_data[table] = source_df
                except sqlite3.Error as e:
                    logger.error(f"SQLite error querying {safe_table}: {e}")
                    source_data[table] = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting source data samples: {e}")
            # Continue with empty source data
            source_data = {}
        
        result['source_data_samples'] = source_data
        return result
    except Exception as e:
        logger.error(f"Error in process_info: {e}")
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
    target_sap_fields (str/list): Optional target SAP fields
    
    Returns:
    dict: Processed information including context
    """
    conn = None
    try:
        # Validate inputs
        if not query or not isinstance(query, str):
            logger.error(f"Invalid query: {query}")
            return None
            
        # Validate IDs
        if not all(isinstance(x, int) for x in [object_id, segment_id, project_id]):
            logger.error(f"Invalid ID types: object_id={type(object_id)}, segment_id={type(segment_id)}, project_id={type(project_id)}")
            return None
        
        # Initialize context manager
        context_manager = ContextualSessionManager()
        
        # Create a session if none provided
        if not session_id:
            session_id = context_manager.create_session()
            logger.info(f"Created new session: {session_id}")
        
        # Get existing context
        previous_context = context_manager.get_context(session_id)
        
        # Connect to database
        conn = sqlite3.connect('db.sqlite3')
        
        # Fetch mapping data
        joined_df = fetch_data_by_ids(object_id, segment_id, project_id, conn)
        
        # Check if joined_df is empty
        if joined_df.empty:
            logger.error(f"No data found for object_id={object_id}, segment_id={segment_id}, project_id={project_id}")
            if conn:
                conn.close()
            return None
        
        # Handle missing values in the dataframe
        joined_df = missing_values_handling(joined_df)
        
        # Save joined data for debugging
        try:
            joined_df.to_csv("joined_data.csv", index=False)
        except Exception as e:
            logger.warning(f"Failed to save joined_data.csv: {e}")
        
        # Process query with context awareness
        resolved_data = parse_data_with_context(
            joined_df, 
            query,
            session_id,  # Pass session_id to access key mappings and target data
            previous_context.get("context") if previous_context else None
        )
        
        if not resolved_data:
            logger.error("Failed to resolve query")
            if conn:
                conn.close()
            return None
            
        # Process the resolved data to get table information
        results = process_info(resolved_data, conn, target_sap_fields)
        
        if not results:
            logger.error("Failed to process resolved data")
            if conn:
                conn.close()
            return None
        
        # Process key mapping with error handling
        key_mapping = []
        try:
            # Check if we have a target field and it's a key
            target_field_filter = joined_df["target_sap_field"] == target_sap_fields
            if target_field_filter.any() and joined_df[target_field_filter]["isKey"].values[0] == "True":
                # Check if we have insertion fields to map
                if results["insertion_fields"] and len(results["insertion_fields"]) > 0:
                    key_mapping = context_manager.add_key_mapping(session_id, target_sap_fields, results["insertion_fields"][0])
                else:
                    logger.warning("No insertion fields found for key mapping")
                    key_mapping = context_manager.get_key_mapping(session_id)
            else:
                # Not a key field, just get existing mappings
                key_mapping = context_manager.get_key_mapping(session_id)
        except Exception as e:
            logger.error(f"Error processing key mapping: {e}")
            # Continue with empty key mapping
            key_mapping = []
        
        # Safely add key mapping to results
        results["key_mapping"] = key_mapping
        
        # Add session_id to the results
        results["session_id"] = session_id
        
        return results
    except Exception as e:
        logger.error(f"Error in process_query: {e}")
        return None
    finally:
        # Ensure database connection is closed
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")


def get_session_context(session_id):
    """
    Get the current context for a session
    
    Parameters:
    session_id (str): The session ID
    
    Returns:
    dict: The session context
    """
    try:
        if not session_id:
            logger.warning("No session ID provided for get_session_context")
            return None
            
        context_manager = ContextualSessionManager()
        return context_manager.get_context(session_id)
    except Exception as e:
        logger.error(f"Error in get_session_context: {e}")
        return None


def get_or_create_session_target_df(session_id, target_table, conn):
    """
    Get existing target dataframe for a session or create a new one
    
    Parameters:
    session_id (str): Session ID
    target_table (str): Target table name
    conn (Connection): SQLite connection
    
    Returns:
    DataFrame: The target dataframe
    """
    try:
        if not session_id:
            logger.warning("No session ID provided for get_or_create_session_target_df")
            return pd.DataFrame()
            
        if not target_table:
            logger.warning("No target table provided for get_or_create_session_target_df")
            return pd.DataFrame()
            
        if not conn:
            logger.warning("No database connection provided for get_or_create_session_target_df")
            return pd.DataFrame()
        
        session_path = f"sessions/{session_id}"
        target_path = f"{session_path}/target_latest.csv"
        
        if os.path.exists(target_path):
            # Load existing target data
            try:
                target_df = pd.read_csv(target_path)
                return target_df
            except Exception as e:
                logger.error(f"Error reading existing target CSV: {e}")
                # If there's an error reading the file, get fresh data from the database
        
        # Get fresh target data from the database
        try:
            # Validate target table name
            safe_table = validate_sql_identifier(target_table)
            
            # Use a parameterized query for safety
            query = f"SELECT * FROM {safe_table}"
            target_df = pd.read_sql_query(query, conn)
            return target_df
        except sqlite3.Error as e:
            logger.error(f"SQLite error in get_or_create_session_target_df: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in get_or_create_session_target_df: {e}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error in get_or_create_session_target_df: {e}")
        return pd.DataFrame()


def save_session_target_df(session_id, target_df):
    """
    Save the updated target dataframe for a session
    
    Parameters:
    session_id (str): Session ID
    target_df (DataFrame): The target dataframe to save
    
    Returns:
    bool: True if successful
    """
    try:
        if not session_id:
            logger.warning("No session ID provided for save_session_target_df")
            return False
            
        if target_df is None:
            logger.warning("None target_df provided for save_session_target_df")
            return False
            
        if not isinstance(target_df, pd.DataFrame):
            logger.warning(f"Invalid target_df type ({type(target_df)}) for save_session_target_df")
            return False
        
        session_path = f"sessions/{session_id}"
        os.makedirs(session_path, exist_ok=True)
        target_path = f"{session_path}/target_latest.csv"
        
        # Also save a timestamped version for history
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        history_path = f"{session_path}/target_{timestamp}.csv"
        
        # Save the target dataframe
        try:
            target_df.to_csv(target_path, index=False)
            target_df.to_csv(history_path, index=False)
            logger.info(f"Saved target dataframe for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving target dataframe: {e}")
            return False
    except Exception as e:
        logger.error(f"Error in save_session_target_df: {e}")
        return False
