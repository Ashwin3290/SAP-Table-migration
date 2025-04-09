import os
import logging
import json
import pandas as pd
import numpy as np
import sqlite3
from dotenv import load_dotenv
from google import genai
from google.genai import types
from token_tracker import track_token_usage, get_token_usage_stats

# Import planner functions (keeping these since they work well)
from planner import process_query as planner_process_query
from planner import get_session_context, get_or_create_session_target_df, save_session_target_df
from code_exec import create_code_file, execute_code

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TableLLM:
    """Improved TableLLM with optimized code generation and better information flow"""
    
    def __init__(self):
        """Initialize the TableLLM instance"""
        # Configure Gemini
        api_key = os.environ.get('GEMINI_API_KEY')
        self.client = genai.Client(api_key=api_key)
        
        # Load code templates
        self.code_templates = self._initialize_templates()
        
        # Current session context
        self.current_context = None

    def _extract_planner_info(self, resolved_data):
        """
        Extract and organize all relevant information from planner's resolved data
        to make it easily accessible in all prompting phases
        """
        # Create a comprehensive context object
        planner_info = {
            # Table information
            "source_table": resolved_data.get('source_table_name'),
            "target_table": resolved_data.get('target_table_name'),
            
            # Field information
            "source_fields": resolved_data.get('source_field_names', []),
            "target_fields": resolved_data.get('target_sap_fields', []),
            "filtering_fields": resolved_data.get('filtering_fields', []),
            "insertion_fields": resolved_data.get('insertion_fields', []),
            
            # Data samples
            "source_data": {
                "sample": {k:v.head(3).to_dict('records') for k,v in resolved_data.get('source_data_samples', pd.DataFrame()).items()},
                "describe": {k:v.describe().to_dict('records') for k,v in resolved_data.get('source_data_samples', pd.DataFrame()).items()}
            },
            "target_data": {
                "sample": resolved_data.get('target_data_samples', pd.DataFrame()).head(3).to_dict('records'),
                "describe": resolved_data.get('target_data_samples', pd.DataFrame()).describe().to_dict()
            },
            
            # Query understanding
            "original_query": resolved_data.get('original_query', ''),
            "restructured_query": resolved_data.get('restructured_query', ''),
            "key_columns": resolved_data.get('key_mapping', []),
            "transformation_logic" : resolved_data.get('transformation_logic', ''),
            
            # # Transformation history and context
            # "transformation_history": resolved_data.get('context', {}).get('transformation_history', []),
            # "target_table_state": resolved_data.get('context', {}).get('target_table_state', {}),
            
            # Session information
            "session_id": resolved_data.get('session_id')
        }
        
        # Extract specific filtering conditions from the restructured query
        query_text = planner_info["restructured_query"]
        conditions = {}
        
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

    @track_token_usage()
    def _classify_query(self, query, planner_info):
        """
        Classify the query type to determine code generation approach
        Uses extracted planner information for better context
        """
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
Restructured query: {planner_info['restructured_query']}
Source fields: {planner_info['source_fields']}
Target fields: {planner_info['target_fields']}
Filter fields: {planner_info['filtering_fields']}
Insertion fields: {planner_info['insertion_fields']}


# TRANSFORMATION CONTEXT:
# Previously populated fields: {planner_info['target_table_state'].get('populated_fields', [])}
# Previously completed transformations: {len(planner_info['transformation_history'])}

EXTRACTED CONDITIONS:
{json.dumps(planner_info['extracted_conditions'], indent=2)}

Return ONLY the classification name with no explanation.
"""
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Return the classification
        return response.text.strip()
    
    @track_token_usage()
    def _generate_simple_plan(self, planner_info):
        """
        Generate a simple, step-by-step plan in natural language
        Uses comprehensive planner information
        """
        # Extract key fields
        source_fields = planner_info["source_fields"]
        source_field = source_fields[0] if isinstance(source_fields, list) and len(source_fields) > 0 else "SOURCE_FIELD"
        
        target_fields = planner_info["target_fields"]
        target_field = target_fields[0] if isinstance(target_fields, list) and len(target_fields) > 0 else "TARGET_FIELD"
        
        # Extract filtering conditions
        conditions_str = "No specific conditions found"
        if planner_info["extracted_conditions"]:
            conditions_str = json.dumps(planner_info["extracted_conditions"], indent=2)
        
        if planner_info["key_columns"]:
            keys_in_use = {}
            for target,source in planner_info["key_columns"]:
                if source in planner_info["source_fields"] and target in planner_info["target_fields"]:
                    keys_in_use[target] = source
        else:
            keys_in_use = {}
        
        # Add context from transformation history
#         history_context = ""
#         if planner_info["transformation_history"]:
#             last_transform = planner_info["transformation_history"][-1]
#             history_context = f"""
# Last transformation: {last_transform.get('description', 'Unknown')}
# Fields modified: {last_transform.get('fields_modified', [])}
# Filter conditions used: {json.dumps(last_transform.get('filter_conditions', {}))}
# """
        # Generate prompt with different templates based on query type
        base_prompt = f"""
Create a simplified step-by-step plan for code that will perform a  operation.

QUERY DETAILS:
User's intent: {planner_info['restructured_query']}
Transformation logic: {planner_info['transformation_logic']}
Source table: {planner_info['source_table']}
Target table: {planner_info['target_table']}
Source field(s): {source_fields}
Target field(s): {target_fields}
Filtering field(s): {planner_info['filtering_fields']}
Filtering conditions: {conditions_str}
Insertion field(s): {planner_info['insertion_fields']}

Current state of target table:
Target data sample: {json.dumps(planner_info['target_data'], indent=2)}

Primary key mapping: {json.dumps(keys_in_use, indent=2)}
Primary key Mapping is given in target_field:source_field format
Note:
1. Only update the Insertion field in the target table do not add anything else
2. Use the Provided key mapping so to map the filtered values to the actual primary key of the target table
3. Add the condition where the source key matches to target key
3. Use the source table for the initial data 
4. Do not add any additional fields to the target table

Source Data:
Source data sample: {json.dumps(planner_info['source_data'], indent=2)}


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
        
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-thinking-exp-01-21",
            contents=base_prompt
        )
        print(response.text)
        return response.text.strip()

    @track_token_usage()
    def _generate_code_from_simple_plan(self, simple_plan, planner_info):
        """
        Generate code based on a simple, step-by-step plan
        With improved context from planner and support for multiple source tables
        """
        # Extract key information
        source_fields = planner_info["source_fields"]
        source_field = source_fields[0] if isinstance(source_fields, list) and len(source_fields) > 0 else "SOURCE_FIELD"
        
        target_fields = planner_info["target_fields"]
        target_field = target_fields[0] if isinstance(target_fields, list) and len(target_fields) > 0 else "TARGET_FIELD"
        
        # Extract filtering conditions
        filter_conditions = []
        if planner_info["extracted_conditions"]:
            # Convert the extracted conditions to pandas filter syntax
            for field, value in planner_info["extracted_conditions"].items():
                if isinstance(value, list):
                    conditions = [f"source_dfs['{planner_info['source_table'][0]}']['{field}'] == '{v}'" for v in value]
                    filter_conditions.append(f"({' | '.join(conditions)})")
                else:
                    filter_conditions.append(f"source_dfs['{planner_info['source_table'][0]}']['{field}'] == '{value}'")
        
        # Use a default condition if none found
        if not filter_conditions:
            filter_conditions = ["source_dfs[source_tables[0]][source_dfs[source_tables[0]].columns[0]] != ''"]  # Default condition that selects all rows
        
        filter_condition_str = " & ".join(filter_conditions)
        
        # Handle multiple source tables
        source_tables_str = json.dumps(planner_info['source_table'])
        
        # Create template with the simple plan as a guide and extensive context
        prompt = f"""
    Write Python code that follows these EXACT steps:

    {simple_plan}

    DETAILED INFORMATION:
    - Source tables: {source_tables_str}
    - Target table: {planner_info['target_table']}
    - Source field(s): {source_fields}
    - Target field(s): {target_fields}
    - Filter condition example: {filter_condition_str}
    - Key mapping: {planner_info.get('key_columns', [])}

    SAMPLE DATA:
    Source data samples:
    {json.dumps(planner_info['source_data']['sample'], indent=2)}

    Target data sample:
    {json.dumps(planner_info['target_data']['sample'], indent=2)}

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
            
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-thinking-exp-01-21",
            contents=prompt
        )

        # Extract the code
        import re
        code_match = re.search(r'python\s*(.*?)\s*```', response.text, re.DOTALL)
        if code_match:
            return code_match.group(1)
        else:
            # If no code block, try to extract the function
            function_match = re.search(r'def analyze_data.*?return', response.text, re.DOTALL)
            if function_match:
                return function_match.group(0) + " target_df"  # Add return value if missing
            return response.text

    def _initialize_templates(self):
        """Initialize code templates for common operations with support for multiple source tables"""
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
    
    def post_proccess_result(self, result):
        """
        Post-process the result DataFrame to remove any columns added due to reindexing
        
        Parameters:
        result (DataFrame): The result DataFrame to clean
        
        Returns:
        DataFrame: The cleaned DataFrame
        """
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

        # 1. Process query with the planner
        resolved_data = planner_process_query(object_id, segment_id, project_id, query, session_id, target_sap_fields)
        if not resolved_data:
            return None, "Failed to resolve query", session_id
        conn = sqlite3.connect('db.sqlite3')

        # 2. Extract and organize all relevant information from the planner
        resolved_data['original_query'] = query  # Add original query for context
        resolved_data['target_data_samples'] = get_or_create_session_target_df(
            session_id, 
            resolved_data['target_table_name'][0] if isinstance(resolved_data['target_table_name'], list) else resolved_data['target_table_name'], 
            conn
        ).head()
        planner_info = self._extract_planner_info(resolved_data)
        
        # Get session ID from the results
        session_id = planner_info["session_id"]
        
        # 3. Extract table names
        source_tables = planner_info['source_table']
        target_table = planner_info['target_table'][0] if isinstance(planner_info['target_table'], list) else planner_info['target_table']
        
        # 4. Get source and target dataframes
        source_dfs = {}
        for table in source_tables:
            source_dfs[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        
        target_df = get_or_create_session_target_df(session_id, target_table, conn)
        
        # 5. Generate a simple, step-by-step plan in natural language
        simple_plan = self._generate_simple_plan(planner_info)
        logger.info(f"Simple plan generated: {simple_plan}")
        
        # 6. Generate code from the simple plan with full context
        code_content = self._generate_code_from_simple_plan(simple_plan, planner_info)
        logger.info(f"Code generated with context: {len(code_content)} chars")
        
        # 7. Execute the generated code
        code_file = create_code_file(code_content, query, is_double=True)
        result = execute_code(code_file, source_dfs, target_df)
        
        # 8. Save the updated target dataframe if it's a DataFrame
        if isinstance(result, pd.DataFrame):
            result = self.post_proccess_result(result)
            save_session_target_df(session_id, result)
        
        conn.close()
        return code_content, result, session_id
    
    def get_session_info(self, session_id):
        """Get information about a session"""
        context = get_session_context(session_id)
        return {
            "session_id": session_id,
            "context": context,
            "transformation_history": context.get("context", {}).get("transformation_history", []) if context else [],
            "target_table_state": context.get("context", {}).get("target_table_state", {}) if context else {}
        }