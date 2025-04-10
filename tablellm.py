import os
import logging
import json
import pandas as pd
import numpy as np
import sqlite3
import traceback
from dotenv import load_dotenv
from google import genai
from google.genai import types
from token_tracker import track_token_usage, get_token_usage_stats

# Import planner functions (keeping these since they work well)
from planner import process_query as planner_process_query
from planner import get_session_context, get_or_create_session_target_df, save_session_target_df
from planner import validate_sql_identifier, SQLInjectionError, SessionError, APIError, DataProcessingError
from code_exec import create_code_file, execute_code

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CodeGenerationError(Exception):
    """Exception raised for code generation errors."""
    pass

class ExecutionError(Exception):
    """Exception raised for code execution errors."""
    pass

class TableLLM:
    """Improved TableLLM with optimized code generation and better information flow"""
    
    def __init__(self):
        """Initialize the TableLLM instance"""
        try:
            # Configure Gemini
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                logger.error("GEMINI_API_KEY not found in environment variables")
                raise APIError("Gemini API key not configured")
                
            self.client = genai.Client(api_key=api_key)
            
            # Load code templates
            self.code_templates = self._initialize_templates()
            
            # Current session context
            self.current_context = None
            
            logger.info("TableLLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TableLLM: {e}")
            raise

    def _extract_planner_info(self, resolved_data):
        """
        Extract and organize all relevant information from planner's resolved data
        to make it easily accessible in all prompting phases
        """
        try:
            # Validate input
            if not resolved_data:
                logger.error("No resolved data provided to _extract_planner_info")
                raise ValueError("No resolved data provided")
            
            # Create a comprehensive context object
            planner_info = {
                # Table information
                "source_table": resolved_data.get('source_table_name', []),
                "target_table": resolved_data.get('target_table_name', []),
                
                # Field information
                "source_fields": resolved_data.get('source_field_names', []),
                "target_fields": resolved_data.get('target_sap_fields', []),
                "filtering_fields": resolved_data.get('filtering_fields', []),
                "insertion_fields": resolved_data.get('insertion_fields', []),
                
                # Data samples
                "source_data": {
                    "sample": {},
                    "describe": {}
                },
                "target_data": {
                    "sample": [],
                    "describe": {}
                },
                
                # Query understanding
                "original_query": resolved_data.get('original_query', ''),
                "restructured_query": resolved_data.get('restructured_query', ''),
                "key_columns": resolved_data.get('key_mapping', []),
                "transformation_logic": resolved_data.get('transformation_logic', ''),
                
                # Session information
                "session_id": resolved_data.get('session_id', '')
            }
            
            # Safely process source data samples
            source_data_samples = resolved_data.get('source_data_samples', {})
            if isinstance(source_data_samples, dict):
                for k, v in source_data_samples.items():
                    if isinstance(v, pd.DataFrame) and not v.empty:
                        try:
                            planner_info["source_data"]["sample"][k] = v.head(3).to_dict('records')
                            planner_info["source_data"]["describe"][k] = v.describe().to_dict()
                        except Exception as e:
                            logger.warning(f"Error processing source data sample for {k}: {e}")
                            planner_info["source_data"]["sample"][k] = []
                            planner_info["source_data"]["describe"][k] = {}
            
            # Safely process target data samples
            target_data_samples = resolved_data.get('target_data_samples')
            if isinstance(target_data_samples, pd.DataFrame) and not target_data_samples.empty:
                try:
                    planner_info["target_data"]["sample"] = target_data_samples.head(3).to_dict('records')
                    planner_info["target_data"]["describe"] = target_data_samples.describe().to_dict()
                except Exception as e:
                    logger.warning(f"Error processing target data sample: {e}")
            
            # Extract specific filtering conditions from the restructured query
            query_text = planner_info["restructured_query"]
            conditions = {}
            
            # Only attempt to extract conditions if we have a query
            if query_text:
                # Look for common filter patterns in the restructured query
                if "=" in query_text:
                    for field in planner_info["filtering_fields"]:
                        pattern = f"{field}\\s*=\\s*['\"](.*?)['\"]"
                        import re
                        matches = re.findall(pattern, query_text)
                        if matches:
                            conditions[field] = matches[0]
                
                # Look for IN conditions
                if " in " in query_text.lower():
                    for field in planner_info["filtering_fields"]:
                        pattern = f"{field}\\s+in\\s+\\(([^)]+)\\)"
                        import re
                        matches = re.findall(pattern, query_text, re.IGNORECASE)
                        if matches:
                            # Parse the values in the parentheses
                            values_str = matches[0]
                            values = [v.strip().strip("'\"") for v in values_str.split(',')]
                            conditions[field] = values
            
            # Add specific conditions
            planner_info["extracted_conditions"] = conditions
            
            # Store context for use in all prompting phases
            self.current_context = planner_info
            return planner_info
        except Exception as e:
            logger.error(f"Error in _extract_planner_info: {e}")
            # Return a minimal valid context to prevent further errors
            minimal_context = {
                "source_table": resolved_data.get('source_table_name', []) if resolved_data else [],
                "target_table": resolved_data.get('target_table_name', []) if resolved_data else [],
                "source_fields": [], "target_fields": [], "filtering_fields": [], "insertion_fields": [],
                "source_data": {"sample": {}, "describe": {}},
                "target_data": {"sample": [], "describe": {}},
                "original_query": "", "restructured_query": "", "key_columns": [], "transformation_logic": "",
                "session_id": resolved_data.get('session_id', '') if resolved_data else '',
                "extracted_conditions": {}
            }
            return minimal_context

    @track_token_usage()
    def _classify_query(self, query, planner_info):
        """
        Classify the query type to determine code generation approach
        Uses extracted planner information for better context
        """
        try:
            # Validate inputs
            if not query:
                logger.warning("Empty query provided to _classify_query")
                return "EXTRACTION"  # Default classification
                
            if not planner_info:
                logger.warning("No planner info provided to _classify_query")
                return "EXTRACTION"  # Default classification
            
            # Create a comprehensive prompt with detailed context
            prompt = f"""
Classify this data transformation query into ONE of these categories:
- FILTER_AND_EXTRACT: Filtering records from source and extracting specific fields
- UPDATE_EXISTING: Updating values in existing target records only
- CONDITIONAL_MAPPING: Applying if/else logic to determine values
- EXTRACTION: Extracting data from source to target without complex filtering
- TIERED_LOOKUP: Looking up data in multiple tables in a specific order
- AGGREGATION: Performing calculations or aggregations on source data

QUERY INFORMATION:
Original query: {query}
Restructured query: {planner_info.get('restructured_query', '')}
Source fields: {planner_info.get('source_fields', [])}
Target fields: {planner_info.get('target_fields', [])}
Filter fields: {planner_info.get('filtering_fields', [])}
Insertion fields: {planner_info.get('insertion_fields', [])}

EXTRACTED CONDITIONS:
{json.dumps(planner_info.get('extracted_conditions', {}), indent=2)}

Return ONLY the classification name with no explanation.
"""
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                
                # Validate response
                if not response or not hasattr(response, 'text') or not response.text:
                    logger.warning("Invalid response from Gemini API in _classify_query")
                    return "EXTRACTION"  # Default classification
                
                # Return the classification
                classification = response.text.strip()
                
                # Validate that classification is one of the allowed values
                allowed_classifications = ["FILTER_AND_EXTRACT", "UPDATE_EXISTING", 
                                         "CONDITIONAL_MAPPING", "EXTRACTION", 
                                         "TIERED_LOOKUP", "AGGREGATION"]
                
                if classification not in allowed_classifications:
                    logger.warning(f"Invalid classification returned: {classification}")
                    return "EXTRACTION"  # Default to safest option
                    
                return classification
            except Exception as e:
                logger.error(f"Error calling Gemini API in _classify_query: {e}")
                return "EXTRACTION"  # Default classification
        except Exception as e:
            logger.error(f"Error in _classify_query: {e}")
            return "EXTRACTION"  # Default classification
    
    @track_token_usage()
    def _generate_simple_plan(self, planner_info):
        """
        Generate a simple, step-by-step plan in natural language
        Uses comprehensive planner information
        """
        try:
            # Validate input
            if not planner_info:
                logger.error("No planner info provided to _generate_simple_plan")
                return "1. Make copy of the source dataframe\n2. Return the dataframe unchanged"
            
            # Extract key fields safely
            source_fields = planner_info.get("source_fields", [])
            source_field = source_fields[0] if isinstance(source_fields, list) and len(source_fields) > 0 else "SOURCE_FIELD"
            
            target_fields = planner_info.get("target_fields", [])
            target_field = target_fields[0] if isinstance(target_fields, list) and len(target_fields) > 0 else "TARGET_FIELD"
            
            # Extract filtering conditions safely
            conditions_str = "No specific conditions found"
            if planner_info.get("extracted_conditions"):
                try:
                    conditions_str = json.dumps(planner_info["extracted_conditions"], indent=2)
                except Exception as e:
                    logger.warning(f"Error converting extracted conditions to JSON: {e}")
                    conditions_str = str(planner_info["extracted_conditions"])
            
            keys_in_use = {}
            if planner_info.get("key_columns"):
                for mapping in planner_info["key_columns"]:
                    if isinstance(mapping, dict) and "target_col" in mapping and "source_col" in mapping:
                        target = mapping["target_col"]
                        source = mapping["source_col"]
                        if source in source_fields and target in target_fields:
                            keys_in_use[target] = source
                    elif isinstance(mapping, (list, tuple)) and len(mapping) == 2:
                        target, source = mapping
                        if source in source_fields and target in target_fields:
                            keys_in_use[target] = source
            
            # Generate prompt with different templates based on query type
            base_prompt = f"""
Create a simplified step-by-step plan for code that will perform a data transformation operation.

QUERY DETAILS:
User's intent: {planner_info.get('restructured_query', 'Transform data')}
Transformation logic: {planner_info.get('transformation_logic', 'Simple transformation')}
Source table: {planner_info.get('source_table', ['source_table'])}
Target table: {planner_info.get('target_table', ['target_table'])}
Source field(s): {source_fields}
Target field(s): {target_fields}
Filtering field(s): {planner_info.get('filtering_fields', [])}
Filtering conditions: {conditions_str}
Insertion field(s): {planner_info.get('insertion_fields', [])}

Current state of target table:
Target data sample: {json.dumps(planner_info.get('target_data', {}).get('sample', []), indent=2)}

Primary key mapping: {json.dumps(keys_in_use, indent=2)}
Primary key Mapping is given in target_field:source_field format
Note:
1. Only update the Insertion field in the target table do not add anything else
2. Use the Provided key mapping so to map the filtered values to the actual primary key of the target table
3. Add the condition where the source key matches to target key
3. Use the source table for the initial data 
4. Do not add any additional fields to the target table

Source Data:
Source data sample: {json.dumps(planner_info.get('source_data', {}).get('sample', {}), indent=2)}


Write ONLY simple, clear steps that a code generator must follow exactly, like this example:
1. Make copy of the source dataframe 
2. Filter rows where MTART value is ROH
3. Take only MATNR column from filtered source data
4. Check if target dataframe is empty
5. If empty, create new dataframe with MATNR as target field
6. If not empty, update the target field with source field values
7. Return the updated target dataframe

Your steps (numbered, 5-10 steps maximum):
"""
            
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21",
                    contents=base_prompt
                )
                
                # Validate response
                if not response or not hasattr(response, 'text') or not response.text:
                    logger.warning("Invalid response from Gemini API in _generate_simple_plan")
                    return "1. Make copy of the source dataframe\n2. Return the dataframe unchanged"
                
                logger.info(f"Generated plan: {response.text.strip()}")
                return response.text.strip()
            except Exception as e:
                logger.error(f"Error calling Gemini API in _generate_simple_plan: {e}")
                return "1. Make copy of the source dataframe\n2. Return the dataframe unchanged"
        except Exception as e:
            logger.error(f"Error in _generate_simple_plan: {e}")
            return "1. Make copy of the source dataframe\n2. Return the dataframe unchanged"

    @track_token_usage()
    def _generate_code_from_simple_plan(self, simple_plan, planner_info):
        """
        Generate code based on a simple, step-by-step plan
        With improved context from planner and support for multiple source tables
        """
        try:
            # Validate inputs
            if not simple_plan:
                logger.error("No simple plan provided to _generate_code_from_simple_plan")
                raise CodeGenerationError("No simple plan provided")
                
            if not planner_info:
                logger.error("No planner info provided to _generate_code_from_simple_plan")
                raise CodeGenerationError("No planner info provided")
            
            # Extract key information safely
            source_fields = planner_info.get("source_fields", [])
            source_field = source_fields[0] if isinstance(source_fields, list) and len(source_fields) > 0 else "SOURCE_FIELD"
            
            target_fields = planner_info.get("target_fields", [])
            target_field = target_fields[0] if isinstance(target_fields, list) and len(target_fields) > 0 else "TARGET_FIELD"
            
            # Extract filtering conditions safely
            filter_conditions = []
            source_tables = planner_info.get('source_table', [])
            
            if planner_info.get("extracted_conditions") and source_tables:
                # Convert the extracted conditions to pandas filter syntax
                for field, value in planner_info["extracted_conditions"].items():
                    # Make sure we have at least one source table
                    if not source_tables:
                        continue
                        
                    source_table = source_tables[0]
                    
                    if isinstance(value, list):
                        conditions = [f"source_dfs['{source_table}']['{field}'] == '{v}'" for v in value]
                        filter_conditions.append(f"({' | '.join(conditions)})")
                    else:
                        filter_conditions.append(f"source_dfs['{source_table}']['{field}'] == '{value}'")
            
            # Use a default condition if none found
            if not filter_conditions and source_tables:
                source_table = source_tables[0]
                filter_conditions = [f"source_dfs['{source_table}'][source_dfs['{source_table}'].columns[0]] != ''"]
            elif not filter_conditions:
                filter_conditions = ["source_dfs[source_tables[0]][source_dfs[source_tables[0]].columns[0]] != ''"]
            
            filter_condition_str = " & ".join(filter_conditions)
            
            # Handle multiple source tables safely
            source_tables_str = json.dumps(source_tables) if source_tables else "[]"
            
            # Create template with the simple plan as a guide and extensive context
            prompt = f"""
    Write Python code that follows these EXACT steps:

    {simple_plan}

    DETAILED INFORMATION:
    - Source tables: {source_tables_str}
    - Target table: {planner_info.get('target_table', [])}
    - Source field(s): {source_fields}
    - Target field(s): {target_fields}
    - Filter condition example: {filter_condition_str}
    - Key mapping: {planner_info.get('key_columns', [])}

    SAMPLE DATA:
    Source data samples:
    {json.dumps(planner_info.get('source_data', {}).get('sample', {}), indent=2)}

    Target data sample:
    {json.dumps(planner_info.get('target_data', {}).get('sample', []), indent=2)}

    REQUIREMENTS:
    1. Follow the steps PRECISELY in order
    2. Handle both empty and non-empty target dataframes
    3. Handle multiple source tables correctly - they are provided as a dictionary where keys are table names
    4. Return the modified target dataframe (df2)
    5. Use numpy for conditional logic if needed (already imported as np)
    6. If you don't find target_field in target_table but it is present in the sample data then add the column to the target table
    7. Use the key mapping to match records between source and target tables

    Complete this function:
    
    def analyze_data(source_dfs, target_df):
        # source_dfs is a dictionary where keys are table names and values are dataframes
        # Example: source_dfs = {{'table1': df1, 'table2': df2}}
        # target_df is the target dataframe to update
        
        # Get list of source tables for easier reference
        source_tables = list(source_dfs.keys())
        
        # Your implementation of the steps above
        
        # Make sure to return the modified target_df
        return target_df
    Return ONLY the complete Python function: """
            
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21",
                    contents=prompt
                )
                
                # Validate response
                if not response or not hasattr(response, 'text') or not response.text:
                    logger.error("Invalid response from Gemini API in _generate_code_from_simple_plan")
                    raise CodeGenerationError("Failed to generate code from Gemini API")
            except Exception as e:
                logger.error(f"Error calling Gemini API in _generate_code_from_simple_plan: {e}")
                raise CodeGenerationError(f"Failed to generate code: {e}")

            # Extract the code
            import re
            try:
                # First try to extract code between triple backticks
                code_match = re.search(r'```python\s*(.*?)\s*```', response.text, re.DOTALL)
                if code_match:
                    return code_match.group(1)
                else:
                    # Next try to extract code without backticks
                    code_match = re.search(r'```\s*(.*?)\s*```', response.text, re.DOTALL)
                    if code_match:
                        return code_match.group(1)
                    else:
                        # Try to extract the function definition directly
                        function_match = re.search(r'def analyze_data\s*\(.*?\).*?return\s+target_df', response.text, re.DOTALL)
                        if function_match:
                            return function_match.group(0)
                        else:
                            # Last resort, return the whole text
                            return response.text
            except Exception as e:
                logger.error(f"Error extracting code from response: {e}")
                raise CodeGenerationError(f"Failed to extract code from response: {e}")
        except Exception as e:
            logger.error(f"Error in _generate_code_from_simple_plan: {e}")
            # Generate a safe default function if all else fails
            return """def analyze_data(source_dfs, target_df):
    # Safety function - error occurred during code generation
    # Get list of source tables
    source_tables = list(source_dfs.keys())
    
    # Return unmodified target dataframe
    return target_df"""

    def _initialize_templates(self):
        """Initialize code templates for common operations with support for multiple source tables"""
        try:
            return {
                "filter": """
# Filter source data based on condition
source_table = '{source_table}'
mask = {filter_condition}
filtered_df = source_dfs[source_table][mask].copy()
""",
                "update": """
# Create a copy of target_df to avoid modifying the original
result = target_df.copy()
source_table = '{source_table}'

# Check if target table is empty
if len(result) == 0:
    # For empty target, create a new dataframe with necessary columns
    # and only the data we need from source
    key_data = source_dfs[source_table][['{key_field}']].copy()
    key_data['{target_field}'] = source_dfs[source_table]['{source_field}']
    return key_data
else:
    # For non-empty target, update only the target field
    # Find matching rows using the key field
    for idx, row in source_dfs[source_table].iterrows():
        # Get the key value
        key_value = row['{key_field}']
        
        # Find matching rows in the target
        target_indices = result[result['{key_field}'] == key_value].index
        
        # Update the target field for matching rows
        if len(target_indices) > 0:
            result.loc[target_indices, '{target_field}'] = row['{source_field}']
        
    return result
""",
                "conditional_mapping": """
# Create a copy of target_df to avoid modifying the original
result = target_df.copy()
source_table = '{source_table}'

# Check if target table is empty
if len(result) == 0:
    # For empty target, we need to create initial structure with key fields
    # and apply our conditional logic directly to the source data
    key_data = source_dfs[source_table][['{key_field}']].copy()
    
    # Define conditions and choices
    conditions = [
        {conditions}
    ]
    choices = [
        {choices}
    ]
    default = '{default_value}'
    
    # Apply conditional mapping to create the target field
    key_data['{target_field}'] = np.select(conditions, choices, default=default)
    return key_data
else:
    # Define conditions and choices
    conditions = [
        {conditions}
    ]
    choices = [
        {choices}
    ]
    default = '{default_value}'
    
    # Create temporary mapping from source data
    source_mapping = pd.Series(
        index=source_dfs[source_table]['{key_field}'], 
        data=np.select(conditions, choices, default=default)
    )
    
    # Apply mapping to target based on key field
    for idx, row in result.iterrows():
        key_value = row['{key_field}']
        if key_value in source_mapping.index:
            result.loc[idx, '{target_field}'] = source_mapping[key_value]
    
    return result
"""
            }
        except Exception as e:
            logger.error(f"Error in _initialize_templates: {e}")
            # Return empty templates if there's an error
            return {}
    
    def post_proccess_result(self, result):
        """
        Post-process the result DataFrame to remove any columns added due to reindexing
        
        Parameters:
        result (DataFrame): The result DataFrame to clean
        
        Returns:
        DataFrame: The cleaned DataFrame
        """
        try:
            if not isinstance(result, pd.DataFrame):
                return result
            
            # Create a copy to avoid modifying the original
            cleaned_df = result.copy()
            
            # Find columns that match the pattern "Unnamed: X" where X is a number
            unnamed_cols = [col for col in cleaned_df.columns if 'unnamed' in str(col).lower() and ':' in str(col)]
            
            # Drop these columns
            if unnamed_cols:
                cleaned_df = cleaned_df.drop(columns=unnamed_cols)
                logger.info(f"Removed {len(unnamed_cols)} unnamed columns: {unnamed_cols}")
            
            return cleaned_df
        except Exception as e:
            logger.error(f"Error in post_proccess_result: {e}")
            # Return original result if there's an error
            return result

    def process_sequential_query(self, query, object_id=29, segment_id=336, project_id=24, session_id=None, target_sap_fields=None):
        """
        Process a query as part of a sequential transformation
        With improved information flow from planner and support for multiple source tables
        
        Parameters:
        query (str): The user's query
        object_id (int): Object ID for mapping
        segment_id (int): Segment ID for mapping
        project_id (int): Project ID for mapping
        session_id (str): Optional session ID, creates new session if None
        target_sap_fields (list): Optional list of target SAP fields
        
        Returns:
        tuple: (code, result, session_id)
        """
        conn = None
        try:
            # Validate inputs
            if not query or not isinstance(query, str):
                logger.error(f"Invalid query type: {type(query)}")
                return None, "Query must be a non-empty string", session_id
                
            # Validate object/segment/project IDs
            if not all(isinstance(x, int) for x in [object_id, segment_id, project_id]):
                logger.error(f"Invalid ID types: object_id={type(object_id)}, segment_id={type(segment_id)}, project_id={type(project_id)}")
                return None, "Invalid ID types - must be integers", session_id
                
            # Validate target_sap_fields
            if target_sap_fields is not None and not isinstance(target_sap_fields, (str, list)):
                logger.error(f"Invalid target_sap_fields type: {type(target_sap_fields)}")
                return None, "target_sap_fields must be a string or list of strings", session_id

            # 1. Process query with the planner
            logger.info(f"Processing query: {query}")
            resolved_data = planner_process_query(object_id, segment_id, project_id, query, session_id, target_sap_fields)
            if not resolved_data:
                logger.error("Failed to resolve query with planner")
                return None, "Failed to resolve query", session_id
                
            # Connect to database
            try:
                conn = sqlite3.connect('db.sqlite3')
            except sqlite3.Error as e:
                logger.error(f"Failed to connect to database: {e}")
                return None, f"Database connection error: {e}", session_id
                
            # 2. Extract and organize all relevant information from the planner
            try:
                resolved_data['original_query'] = query  # Add original query for context
                
                # Get target data samples safely
                try:
                    if 'target_table_name' in resolved_data:
                        target_table = resolved_data['target_table_name']
                        if isinstance(target_table, list) and len(target_table) > 0:
                            target_table = target_table[0]
                            
                        # Validate table name to prevent SQL injection
                        target_table = validate_sql_identifier(target_table)
                        
                        resolved_data['target_data_samples'] = get_or_create_session_target_df(
                            session_id, target_table, conn
                        ).head()
                except Exception as e:
                    logger.warning(f"Error getting target data samples: {e}")
                    resolved_data['target_data_samples'] = pd.DataFrame()
                
                # Extract and organize planner information
                planner_info = self._extract_planner_info(resolved_data)
                
                # Get session ID from the results
                session_id = planner_info.get("session_id")
                if not session_id:
                    logger.warning("No session ID returned from planner")
                    # Create a new session ID as fallback
                    import uuid
                    session_id = str(uuid.uuid4())
                
                # 3. Extract table names safely
                source_tables = planner_info.get('source_table', [])
                if not source_tables:
                    logger.error("No source tables found in planner info")
                    if conn:
                        conn.close()
                    return None, "No source tables identified", session_id
                
                target_table = None
                if isinstance(planner_info.get('target_table'), list) and len(planner_info['target_table']) > 0:
                    target_table = planner_info['target_table'][0]
                else:
                    target_table = planner_info.get('target_table')
                    
                if not target_table:
                    logger.error("No target table found in planner info")
                    if conn:
                        conn.close()
                    return None, "No target table identified", session_id
                
                # 4. Get source and target dataframes safely
                source_dfs = {}
                for table in source_tables:
                    try:
                        # Validate table name to prevent SQL injection
                        safe_table = validate_sql_identifier(table)
                        
                        # Use a parameterized query for safety
                        source_dfs[table] = pd.read_sql_query(f"SELECT * FROM {safe_table}", conn)
                    except Exception as e:
                        logger.warning(f"Error reading source table {table}: {e}")
                        # Create an empty dataframe as fallback
                        source_dfs[table] = pd.DataFrame()
                
                # Get target dataframe
                try:
                    target_df = get_or_create_session_target_df(session_id, target_table, conn)
                except Exception as e:
                    logger.warning(f"Error getting target dataframe: {e}")
                    target_df = pd.DataFrame()
                
                # 5. Generate a simple, step-by-step plan in natural language
                try:
                    simple_plan = self._generate_simple_plan(planner_info)
                    logger.info(f"Simple plan generated: {simple_plan}")
                except Exception as e:
                    logger.error(f"Error generating simple plan: {e}")
                    simple_plan = "1. Make copy of the source dataframe\n2. Return the dataframe unchanged"
                
                # 6. Generate code from the simple plan with full context
                try:
                    code_content = self._generate_code_from_simple_plan(simple_plan, planner_info)
                    logger.info(f"Code generated with length: {len(code_content)} chars")
                except CodeGenerationError as e:
                    logger.error(f"Code generation error: {e}")
                    if conn:
                        conn.close()
                    return None, f"Failed to generate code: {e}", session_id
                
                # 7. Execute the generated code with error handling
                try:
                    code_file = create_code_file(code_content, query, is_double=True)
                    result = execute_code(code_file, source_dfs, target_df)
                    
                    # Check if result is an error message
                    if isinstance(result, str) and "Error" in result:
                        logger.error(f"Code execution error: {result}")
                        if conn:
                            conn.close()
                        return code_content, f"Code execution failed: {result}", session_id
                except Exception as e:
                    logger.error(f"Error executing code: {e}")
                    if conn:
                        conn.close()
                    return code_content, f"Error executing code: {e}", session_id
                
                # 8. Save the updated target dataframe if it's a DataFrame
                if isinstance(result, pd.DataFrame):
                    try:
                        result = self.post_proccess_result(result)
                        save_success = save_session_target_df(session_id, result)
                        if not save_success:
                            logger.warning("Failed to save target dataframe")
                    except Exception as e:
                        logger.error(f"Error saving target dataframe: {e}")
                
                # 9. Return the results
                if conn:
                    conn.close()
                return code_content, result, session_id
            
            except Exception as e:
                logger.error(f"Error in process_sequential_query: {e}")
                logger.error(traceback.format_exc())
                if conn:
                    conn.close()
                return None, f"An error occurred during processing: {e}", session_id
        
        except Exception as e:
            logger.error(f"Outer error in process_sequential_query: {e}")
            logger.error(traceback.format_exc())
            if conn:
                try:
                    conn.close()
                except:
                    pass
            return None, f"An error occurred: {e}", session_id