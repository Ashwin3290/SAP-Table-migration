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
        With focus on using utility functions with correct parameters
        """
        try:
            # Validate input
            if not planner_info:
                logger.error("No planner info provided to _generate_simple_plan")
                return "1. Import utility functions from transform_utils\n2. Get source dataframe\n3. Return the dataframe unchanged"
            
            # Extract key information for the prompt
            source_fields = planner_info.get("source_fields", [])
            target_fields = planner_info.get("target_fields", [])
            
            # Extract filtering conditions safely
            conditions_str = "No specific conditions found"
            if planner_info.get("extracted_conditions"):
                try:
                    conditions_str = json.dumps(planner_info["extracted_conditions"], indent=2)
                except Exception as e:
                    logger.warning(f"Error converting extracted conditions to JSON: {e}")
                    conditions_str = str(planner_info["extracted_conditions"])
            
            # Generate prompt with focus on utility functions
            base_prompt = f"""
    Create a step-by-step plan for code that will perform this data transformation operation.
    Focus on using the available utility functions with their EXACT parameter formats.

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

    AVAILABLE UTILITY FUNCTIONS AND THEIR PARAMETER FORMATS:
    1. filter_dataframe(df, field, condition_type, value)
    - condition_type MUST be one of: 'equals', 'not_equals', 'contains', 'greater_than', 'less_than', 'in_list', 'not_in_list', 'is_null', 'is_not_null'

    2. map_fields(source_df, target_df, field_mapping, key_field_pair=None)
    - field_mapping format: {{'TARGET_FIELD1': 'SOURCE_FIELD1', 'TARGET_FIELD2': 'SOURCE_FIELD2'}}
    - key_field_pair format: ('TARGET_KEY', 'SOURCE_KEY')

    3. match_by_keys(source_df, target_df, key_mapping)
    - key_mapping format: {{'TARGET_KEY': 'SOURCE_KEY'}}

    4. conditional_mapping(source_df, condition_field, conditions, value_field=None, value_map=None, default=None)
    - conditions format: ["== 'ROH'", "== 'HALB'", "> 100"]
    - value_map format: ['Raw Material', 'Semi-Finished', 'High Value']

    5. initialize_target(target_df, required_columns)
    - required_columns format: ['FIELD1', 'FIELD2', 'FIELD3']

    6. join_tables(main_df, other_df, main_key, other_key, fields_to_add)
    - fields_to_add format: ['FIELD1', 'FIELD2', 'FIELD3']

    7. SAP-specific functions:
    - map_material_type(source_df, source_field='MTART')
    - convert_sap_date(source_df, date_field, output_format='YYYY-MM-DD')
    - map_sap_language_code(source_df, lang_field='SPRAS')
    - map_sap_unit_of_measure(source_df, uom_field='MEINS')
    - handle_sap_leading_zeros(source_df, field, length=10)

    Write a numbered, step-by-step plan that outlines EXACTLY how to perform the transformation.
    Use utility functions whenever possible.
    Specify the EXACT parameter values for each function.
    Be specific about fields, values, and data types.

    Note:
    - Make sure to handle queries with if/else logics by properly adding if-elif-else blocks instead of just adding data directly.
    - Revolve your plans around using keys in the source and target tables to correct add or update data.
    - Use the utility functions as much as possible to avoid writing custom code.
    - If a query talks about if data in source table then he is meaning to look for key values that are common in the target table and source table.

    Example:
    1. Import utility functions from transform_utils
    2. Get source dataframe from source_dfs dictionary using 'MARA' key
    3. Initialize target dataframe with required columns ['PRODUCT', 'MTART', 'MEINS']
    4. Filter source dataframe using filter_dataframe(source_df, 'MTART', 'equals', 'ROH')
    5. Map fields using map_fields(filtered_df, target_df, {{'PRODUCT': 'MATNR', 'MTART': 'MTART', 'MEINS': 'MEINS'}})
    6. Return the updated target dataframe

    Your step-by-step plan (5-10 steps):
    """
            
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21",
                    contents=base_prompt
                )
                
                # Validate response
                if not response or not hasattr(response, 'text') or not response.text:
                    logger.warning("Invalid response from Gemini API in _generate_simple_plan")
                    return "1. Import utility functions from transform_utils\n2. Get source dataframe\n3. Return the dataframe unchanged"
                
                logger.info(f"Generated plan: {response.text.strip()}")
                return response.text.strip()
            except Exception as e:
                logger.error(f"Error calling Gemini API in _generate_simple_plan: {e}")
                return "1. Import utility functions from transform_utils\n2. Get source dataframe\n3. Return the dataframe unchanged"
        except Exception as e:
            logger.error(f"Error in _generate_simple_plan: {e}")
            return "1. Import utility functions from transform_utils\n2. Get source dataframe\n3. Return the dataframe unchanged"
        
    @track_token_usage()
    def _generate_code_from_simple_plan(self, simple_plan, planner_info):
        """
        Generate code based on a simple, step-by-step plan
        With improved context from planner and support for utility functions
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
            target_fields = planner_info.get("target_fields", [])
            
            # Handle multiple source tables safely
            source_tables = planner_info.get('source_table', [])
            source_tables_str = json.dumps(source_tables) if source_tables else "[]"
            
            # Create template with the simple plan as a guide and utility functions information
            prompt = f"""
    Write Python code that follows these EXACT steps:

    {simple_plan}

    DETAILED INFORMATION:
    - Source tables: {source_tables_str}
    - Target table: {planner_info.get('target_table', [])}
    - Source field(s): {source_fields}
    - Target field(s): {target_fields}
    - Filtering field(s): {planner_info.get('filtering_fields', [])}
    - Filtering conditions: {json.dumps(planner_info.get('extracted_conditions', {}), indent=2)}
    - Key mapping: {planner_info.get('key_columns', [])}

    UTILITY FUNCTIONS REFERENCE:
    ===================================================================================

    1. filter_dataframe(df: pd.DataFrame, field: str, condition_type: str, value: Any) -> pd.DataFrame

    PARAMETERS:
    - df: Source pandas DataFrame
    - field: Column name to filter on (must be exact string match)
    - condition_type: MUST be one of these exact strings:
        * 'equals' - Field equals value (==)
        * 'not_equals' - Field not equal to value (!=)
        * 'contains' - Field contains substring (str.contains)
        * 'greater_than' - Field greater than value (>)
        * 'less_than' - Field less than value (<)
        * 'in_list' - Field value is in provided list (isin)
        * 'not_in_list' - Field value is not in provided list (~isin)
        * 'is_null' - Field is null (isna)
        * 'is_not_null' - Field is not null (~isna)
    - value: Value to compare against (not used for is_null/is_not_null)
    
    EXAMPLE: filtered_df = filter_dataframe(source_df, 'MTART', 'equals', 'ROH')

    ===================================================================================

    2. map_fields(source_df: pd.DataFrame, target_df: pd.DataFrame, field_mapping: Dict[str, str], key_field_pair: Optional[Tuple[str, str]] = None) -> pd.DataFrame

    PARAMETERS:
    - source_df: Source pandas DataFrame
    - target_df: Target pandas DataFrame to update
    - field_mapping: Dictionary with format {{'TARGET_FIELD': 'SOURCE_FIELD'}} 
    - key_field_pair: Optional tuple with format (target_key_field, source_key_field)
    
    EXAMPLE: result_df = map_fields(source_df, target_df, {{'PRODUCT': 'MATNR', 'MTART': 'MTART'}}, ('PRODUCT', 'MATNR'))

    ===================================================================================

    3. match_by_keys(source_df: pd.DataFrame, target_df: pd.DataFrame, key_mapping: Dict[str, str]) -> pd.DataFrame

    PARAMETERS:
    - source_df: Source pandas DataFrame
    - target_df: Target pandas DataFrame
    - key_mapping: Dictionary mapping target keys to source keys {{'TARGET_KEY': 'SOURCE_KEY'}}
    
    EXAMPLE: result_df = match_by_keys(source_df, target_df, {{'PRODUCT': 'MATNR'}})

    ===================================================================================

    4. conditional_mapping(source_df: pd.DataFrame, condition_field: str, conditions: List[str], value_field: Optional[str] = None, value_map: Optional[List[Any]] = None, default: Optional[Any] = None) -> pd.Series

    PARAMETERS:
    - source_df: Source pandas DataFrame
    - condition_field: Field to apply conditions to
    - conditions: List of condition strings in these EXACT formats:
        * "== 'ROH'"      (equals string value - note single quotes)
        * "== 50"         (equals numeric value)
        * "!= 'FERT'"     (not equals string value)
        * "> 100"         (greater than numeric value)
        * "< 200"         (less than numeric value)
        * "in ['A', 'B']" (value in list)
    - value_field: Field to get values from if value_map not provided
    - value_map: List of values corresponding to conditions
    - default: Value when no conditions match
    
    EXAMPLE: result_df['CATEGORY'] = conditional_mapping(source_df, 'MTART', ["== 'ROH'", "== 'HALB'"], value_map=['Raw', 'Semi'], default='Other')

    ===================================================================================

    5. initialize_target(target_df: Optional[pd.DataFrame], required_columns: List[str]) -> pd.DataFrame

    PARAMETERS:
    - target_df: Target pandas DataFrame, can be None
    - required_columns: List of column names that must exist in result
    
    EXAMPLE: target_df = initialize_target(target_df, ['PRODUCT', 'MTART', 'MEINS'])

    ===================================================================================

    6. join_tables(main_df: pd.DataFrame, other_df: pd.DataFrame, main_key: str, other_key: str, fields_to_add: List[str]) -> pd.DataFrame

    PARAMETERS:
    - main_df: Main pandas DataFrame to update
    - other_df: Other pandas DataFrame to get data from
    - main_key: Key field in main_df
    - other_key: Key field in other_df
    - fields_to_add: List of field names from other_df to add to main_df
    
    EXAMPLE: result_df = join_tables(source_df, target_df, 'MATNR', 'PRODUCT', ['MAKTX', 'MEINS'])

    ===================================================================================

    7. aggregate_data(df: pd.DataFrame, group_by: Union[str, List[str]], agg_functions: Dict[str, Union[str, List[str]]]) -> pd.DataFrame

    PARAMETERS:
    - df: Pandas DataFrame to aggregate
    - group_by: Column(s) to group by (string or list of strings)
    - agg_functions: Dictionary mapping columns to agg functions. Functions MUST be one of: 'sum', 'mean', 'min', 'max', 'count'
    
    EXAMPLE: result_df = aggregate_data(source_df, 'WERKS', {{'STOCK': 'sum', 'PRICE': 'mean'}})

    ===================================================================================

    8. safe_combine_tables(df1: Optional[pd.DataFrame], df2: Optional[pd.DataFrame], key: Optional[str] = None) -> pd.DataFrame

    PARAMETERS:
    - df1: First pandas DataFrame, can be None
    - df2: Second pandas DataFrame, can be None
    - key: Optional key field for matching records
    
    EXAMPLE: result_df = safe_combine_tables(target_df, new_data, key='PRODUCT')

    ===================================================================================

    SAMPLE DATA:
    Source data samples:
    {json.dumps(planner_info.get('source_data', {}).get('sample', {}), indent=2)}

    Target data sample:
    {json.dumps(planner_info.get('target_data', {}).get('sample', []), indent=2)}

    REQUIREMENTS:
    1. Follow the steps PRECISELY in order
    2. Use the utility functions whenever appropriate
    3. ALWAYS match the exact parameter names and formats shown in the reference above
    4. For conditional operations like filter_dataframe, use ONLY the exact condition_type strings listed
    5. When constructing conditions for conditional_mapping, use the EXACT formats shown
    6. Always import transform_utils at the beginning and do not try to implement it yourself
    7. Handle both empty and non-empty target dataframes
    8. Return the modified target dataframe (target_df)
    9. Use numpy for conditional logic if needed (already imported as np)
    10. Make sure to handle multiple source tables correctly
    11. Do not create sample dataframes and just use the data given in the prompt
    12. Only give the analyze_data function and do not add any other functions or classes
    13. Do not make the transform_utils functions by youself based upon the prompt, just use them as they are already implemented and ready to be imported.

    Make the function like this :

    def analyze_data(source_dfs, target_df):
        # source_dfs is a dictionary where keys are table names and values are dataframes
        # Example: source_dfs = {{'table1': df1, 'table2': df2}}
        # target_df is the target dataframe to update
        
        # Get list of source tables for easier reference
        source_tables = list(source_dfs.keys())
        
        # Your implementation of the steps above

        # Make sure to return the modified target_df
        return target_df
    """
            
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
        # Import the utility functions
        from transform_utils import *
        
        # Get list of source tables
        source_tables = list(source_dfs.keys())
        
        # Return unmodified target dataframe
        return target_df"""

    @track_token_usage()
    def _fix_code(self, code_content, error_info, planner_info, attempt=1, max_attempts=3):
        """
        Attempt to fix code based on error traceback
        
        Parameters:
        code_content (str): The original code that failed
        error_info (dict): Error information with traceback
        planner_info (dict): Context information from planner
        attempt (int): Current attempt number
        max_attempts (int): Maximum number of attempts to fix the code
        
        Returns:
        str: Fixed code or None if max attempts reached
        """
        if attempt > max_attempts:
            logger.error(f"Failed to fix code after {max_attempts} attempts")
            return None
        
        try:
            # Extract error information
            error_type = error_info.get("error_type", "Unknown error") 
            error_message = error_info.get("error_message", "No error message")
            traceback_text = error_info.get("traceback", "No traceback available")
            
            # Create a prompt for the code fixer
            prompt = f"""
    You are an expert code debugging and fixing agent. Your task is to fix code that failed during execution.

    THE CODE THAT FAILED:
    ```python
    {code_content}
    ERROR INFORMATION:
    Error Type: {error_type}
    Error Message: {error_message}
    FULL TRACEBACK:
    {traceback_text}
    CONTEXT INFORMATION:
    Source tables: {planner_info.get('source_table', [])}
    Target table: {planner_info.get('target_table', [])}
    Source fields: {planner_info.get('source_fields', [])}
    Target fields: {planner_info.get('target_fields', [])}
    Filtering fields: {planner_info.get('filtering_fields', [])}
    Insertion fields: {planner_info.get('insertion_fields', [])}
    Extracted conditions: {json.dumps(planner_info.get('extracted_conditions', {}), indent=2)}
    COMMON ERRORS TO CHECK FOR:

    Field name typos or case sensitivity issues
    Incorrect parameter names or order in utility function calls
    Missing or incorrect imports
    Incorrect handling of empty dataframes
    Missing field in target dataframe
    Incorrect data types in filter or map operations
    Wrong condition_type parameter in filter_dataframe
    Syntax errors in condition strings

    TASK:

    Analyze the error carefully
    Identify the root cause of the issue
    Fix the code to address the specific error
    Make sure your solution maintains the original intent of the code
    Return only the complete fixed code with no explanations

    Remember to keep the overall structure and logic of the original code, fixing only what's necessary
    to address the error.
    This is attempt {attempt} of {max_attempts}.
    ONLY PROVIDE THE COMPLETE FIXED CODE, WITH NO EXPLANATIONS:
    """
        
            # Call the AI to fix the code
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21",
                    contents=prompt
                )
                
                # Validate response
                if not response or not hasattr(response, 'text') or not response.text:
                    logger.error("Invalid response from Gemini API in _fix_code")
                    return None
                
                logger.info(f"Generated fixed code on attempt {attempt}")
                
                # Extract the code
                import re
                
                # First try to extract code between triple backticks
                code_match = re.search(r'```python\s*(.*?)\s*```', response.text, re.DOTALL)
                if code_match:
                    return code_match.group(1)
                
                # Next try to extract code without backticks
                code_match = re.search(r'```\s*(.*?)\s*```', response.text, re.DOTALL)
                if code_match:
                    return code_match.group(1)
                
                # Try to extract the function definition directly
                function_match = re.search(r'def analyze_data\s*\(.*?\).*?return\s+target_df', response.text, re.DOTALL)
                if function_match:
                    return function_match.group(0)
                
                # Last resort, return the whole text
                return response.text
                
            except Exception as e:
                logger.error(f"Error calling Gemini API in _fix_code: {e}")
                return None
        except Exception as e:
            logger.error(f"Error in _fix_code: {e}")
            return None
        
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
            print(resolved_data['key_mapping'])

            if isinstance(resolved_data['key_mapping'][0],str):
                return None,resolved_data['key_mapping'][0],session_id
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
                    
                    # Check if result is an error (now it's a dictionary with traceback information)
                    if isinstance(result, dict) and 'error_type' in result:
                        logger.error(f"Code execution error: {result['error_message']}")
                        logger.error(f"Traceback: {result['traceback']}")
                        
                        # Try to fix the code up to 3 times
                        fixed_code = code_content
                        for attempt in range(1, 4):  # 3 attempts maximum
                            logger.info(f"Attempting to fix code (attempt {attempt}/3)")
                            
                            fixed_code = self._fix_code(fixed_code, result, planner_info, attempt=attempt, max_attempts=3)
                            if fixed_code is None:
                                # Failed to fix the code after max attempts
                                if conn:
                                    conn.close()
                                return code_content, f"Failed to generate working code after 3 attempts. Last error: {result['error_message']}", session_id
                            
                            # Try executing the fixed code
                            fixed_code_file = create_code_file(fixed_code, f"{query} (fixed attempt {attempt})", is_double=True)
                            fixed_result = execute_code(fixed_code_file, source_dfs, target_df)
                            
                            # If the fixed code worked (result is not an error dictionary), use it
                            if not isinstance(fixed_result, dict) or 'error_type' not in fixed_result:
                                logger.info(f"Successfully fixed code on attempt {attempt}")
                                code_content = fixed_code  # Update the code content to the fixed version
                                result = fixed_result  # Use the successful result
                                break
                            else:
                                # The fix didn't work, update the error for the next attempt
                                result = fixed_result
                                logger.error(f"Fix attempt {attempt} failed with error: {result['error_message']}")
                        
                        # If we've gone through all attempts and still have an error
                        if isinstance(result, dict) and 'error_type' in result:
                            if conn:
                                conn.close()
                            return code_content, f"Failed to generate working code after 3 attempts. Last error: {result['error_message']}", session_id
                    
                    elif isinstance(result, str) and "Error" in result:
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