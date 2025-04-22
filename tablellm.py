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
from planner import (
    get_session_context,
    get_or_create_session_target_df,
    save_session_target_df,
)
from planner import (
    validate_sql_identifier,
    SQLInjectionError,
    SessionError,
    APIError,
    DataProcessingError,
)
from code_exec import create_code_file, execute_code

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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
            api_key = os.environ.get("GEMINI_API_KEY")
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
                "source_table": resolved_data.get("source_table_name", []),
                "target_table": resolved_data.get("target_table_name", []),
                # Field information
                "source_fields": resolved_data.get("source_field_names", []),
                "target_fields": resolved_data.get("target_sap_fields", []),
                "filtering_fields": resolved_data.get("filtering_fields", []),
                "insertion_fields": resolved_data.get("insertion_fields", []),
                # Data samples
                "source_data": {"sample": {}, "describe": {}},
                "target_data": {"sample": [], "describe": {}},
                # Query understanding
                "original_query": resolved_data.get("original_query", ""),
                "restructured_query": resolved_data.get("restructured_query", ""),
                # Session information
                "session_id": resolved_data.get("session_id", ""),
                "key_columns": resolved_data.get("key_mapping", []),
            }

            # Safely process source data samples
            source_data_samples = resolved_data.get("source_data_samples", {})
            if isinstance(source_data_samples, dict):
                for k, v in source_data_samples.items():
                    if isinstance(v, pd.DataFrame) and not v.empty:
                        try:
                            planner_info["source_data"]["sample"][k] = v.head(
                                3
                            ).to_dict("records")
                            planner_info["source_data"]["describe"][
                                k
                            ] = v.describe().to_dict()
                        except Exception as e:
                            logger.warning(
                                f"Error processing source data sample for {k}: {e}"
                            )
                            planner_info["source_data"]["sample"][k] = []
                            planner_info["source_data"]["describe"][k] = {}

            # Safely process target data samples
            target_data_samples = resolved_data.get("target_data_samples")
            if (
                isinstance(target_data_samples, pd.DataFrame)
                and not target_data_samples.empty
            ):
                try:
                    planner_info["target_data"]["sample"] = target_data_samples.head(
                        3
                    ).to_dict("records")
                    planner_info["target_data"][
                        "describe"
                    ] = target_data_samples.describe().to_dict()
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
                            values = [
                                v.strip().strip("'\"") for v in values_str.split(",")
                            ]
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
                "source_table": (
                    resolved_data.get("source_table_name", []) if resolved_data else []
                ),
                "target_table": (
                    resolved_data.get("target_table_name", []) if resolved_data else []
                ),
                "source_fields": [],
                "target_fields": [],
                "filtering_fields": [],
                "insertion_fields": [],
                "source_data": {"sample": {}, "describe": {}},
                "target_data": {"sample": [], "describe": {}},
                "original_query": "",
                "restructured_query": "",
                "key_columns": [],
                "session_id": (
                    resolved_data.get("session_id", "") if resolved_data else ""
                ),
                "extracted_conditions": {},
            }
            return minimal_context

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
                    conditions_str = json.dumps(
                        planner_info["extracted_conditions"], indent=2
                    )
                except Exception as e:
                    logger.warning(
                        f"Error converting extracted conditions to JSON: {e}"
                    )
                    conditions_str = str(planner_info["extracted_conditions"])

                        # Generate prompt with focus on utility functions
            base_prompt = f"""
You are an expert data transformation architect. Create a detailed, step-by-step plan for the following data transformation task.

TASK CONTEXT:
- Query: {planner_info.get('restructured_query', 'Transform data')}
- Source tables: {planner_info.get('source_table', [])}
- Target table: {planner_info.get('target_table', [])}
- Source fields: {planner_info.get('source_fields', [])}
- Target fields: {planner_info.get('target_fields', [])}
- Filtering fields: {planner_info.get('filtering_fields', [])}
- Key columns: {planner_info.get('key_columns', [])}

PLAN REQUIREMENTS:
1. Include clear steps for filtering if filtering fields are specified
2. Address how both empty AND populated target dataframes will be handled
3. Explicitly include key column mapping between source and target
4. Include steps for handling edge cases like missing data and type mismatches
5. Ensure the plan is complete with no steps missing
6. Ensure all target fields will be populated in the result
7. End with validation of the result before returning

FORMAT:
- Number each step sequentially
- Make each step clear and actionable
- Include 8-12 detailed steps, covering all necessary operations
- Be specific about which fields and tables are used in each step

Example of a good step: "Filter the source_table dataframe where field_name matches the condition value"
Example of a bad step: "Apply filtering as needed"

COMMON QUERY PATTERNS AND HOW TO IMPLEMENT THEM:

1. EXISTENCE CHECK PATTERN:
   "Check Field in Table"

   Implementation Plan:
   - Loop through target records
   - For each record, check if matching key exists in source table
   - If match exists, use data from that source record
   - Do NOT filter source table by any field value unless explicitly stated

2. MULTI-TABLE FALLBACK PATTERN:
   "Check Table1, if not found check Table2, else use Table3"

   Implementation Plan:
   - Create sequential lookup logic
   - Try first table, then only if no match is found, try second table
   - Only use third table if no matches in first two tables
   - Follow key-based matching, not filtering by values

3. CONDITIONAL VALUE MAPPING:
   "If Field1 is Value1, set Field2 to Value2"

   Implementation Plan:
   - Create explicit conditional logic for value mapping
   - Apply conditions based on specified field values
   - Update target fields according to the mapping rules

4. IMPORTANT FIELD NAMING CONSIDERATIONS:
   - Terms like "Material", "Type", "ROH", "Customer" are typically field names
   - Don't interpret these as filter values unless explicitly part of a condition
   - Capitalized terms almost always refer to field names or concepts, not values

Return ONLY the numbered plan with no explanations or additional text.
"""

            try:
                print(base_prompt)
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21", contents=base_prompt
                )

                # Validate response
                if not response or not hasattr(response, "text") or not response.text:
                    logger.warning(
                        "Invalid response from Gemini API in _generate_simple_plan"
                    )
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
                logger.error(
                    "No simple plan provided to _generate_code_from_simple_plan"
                )
                raise CodeGenerationError("No simple plan provided")

            if not planner_info:
                logger.error(
                    "No planner info provided to _generate_code_from_simple_plan"
                )
                raise CodeGenerationError("No planner info provided")

            # Extract key information safely
            source_fields = planner_info.get("source_fields", [])
            target_fields = planner_info.get("target_fields", [])

            # Handle multiple source tables safely
            source_tables = planner_info.get("source_table", [])
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

KEY MAPPING (CRITICALLY IMPORTANT):
- Key columns for mapping between source and target: {planner_info.get('key_columns', [])}
- You MUST implement these key mappings in your code
- Every operation that accesses the target table MUST use these key columns

SAMPLE DATA:
Source data samples:
{json.dumps(planner_info.get('source_data', {}).get('sample', {}), indent=2)}

Target data sample:
{json.dumps(planner_info.get('target_data', {}).get('sample', []), indent=2)}

FILTERING AND CONDITION IMPLEMENTATION:
1. For simple equality filters:
   - Use: df[df['column'] == value]
   - Example: source_df[source_df['STATUS'] == 'ACTIVE']

2. For IN conditions:
   - Use: df[df['column'].isin([val1, val2, ...])]
   - Example: source_df[source_df['TYPE'].isin(['A', 'B', 'C'])]

3. For numeric comparisons:
   - Use: df[df['column'] > value]
   - Example: source_df[source_df['COUNT'] > 100]

4. For text patterns:
   - Use: df[df['column'].str.contains(pattern)]
   - Example: source_df[source_df['NAME'].str.contains('Test')]

5. Combining conditions:
   - AND: df[(condition1) & (condition2)]
   - OR: df[(condition1) | (condition2)]
   - Example: source_df[(source_df['STATUS'] == 'ACTIVE') & (source_df['COUNT'] > 100)]

KEY MAPPING IMPLEMENTATION:
1. For inserting new records:
   ```python
   # Extract key columns from source
   new_records = source_df[key_columns + target_fields].copy()
   # Append to target
   updated_target = pd.concat([target_df, new_records], ignore_index=True)

For updating existing records:
python# For each source record
for _, source_row in source_df.iterrows():
    # Find matching target rows using ALL key columns
    mask = pd.Series(True, index=target_df.index)
    for key in key_columns:
        mask &= target_df[key] == source_row[key]
    # Update fields in matching rows
    if mask.any():
        target_df.loc[mask, target_field] = source_row[source_field]

For checking existence before insert:
python# For each source record
for _, source_row in source_df.iterrows():
    # Create match condition using ALL key columns
    exists = False
    mask = pd.Series(True, index=target_df.index)
    for key in key_columns:
        mask &= target_df[key] == source_row[key]
    exists = mask.any()
    # Insert only if not exists
    if not exists:
        new_row = pd.DataFrame([source_row[key_columns + target_fields]])
        target_df = pd.concat([target_df, new_row], ignore_index=True)


SAP-specific utils functions in transform_utils.py:

map_material_type(source_df, source_field='MTART')
convert_sap_date(source_df, date_field, output_format='YYYY-MM-DD')
map_sap_language_code(source_df, lang_field='SPRAS')
map_sap_unit_of_measure(source_df, uom_field='MEINS')
handle_sap_leading_zeros(source_df, field, length=10)

COMMON MISTAKES TO AVOID:

DO NOT skip key column validation
DO NOT create new key columns - use only provided ones
DO NOT modify key column values during transformation
DO NOT assume single-column keys - always use ALL provided key columns
DO NOT ignore filtering conditions - apply them EXACTLY as specified
DO NOT add columns not specified in target fields
DO NOT transform data that isn't needed in the target
DO NOT use incompatible data types in comparisons
DO NOT drop or ignore key columns during any operation

REQUIREMENTS:

Follow the provided step-by-step plan in the exact order given.
Use the provided source_dfs and target_df dataframes.
Properly map source and target fields using key mapping provided.
For conditional operations (e.g., filter_dataframe), use ONLY the specified condition_type strings.
For conditional_mapping, ensure condition strings and value maps strictly follow the provided formats.
Always import transform_utils at the top of the function.
Handle both empty and non-empty target dataframes correctly.
The function must return the modified target dataframe (target_df).
Use numpy (np) for conditional logic only if necessary (already imported).
Support multiple source tables as described in the prompt.
Do NOT create or use sample dataframes; only use the data provided in the prompt.
Only define the analyze_data function; do NOT add other functions or classes.
Do NOT add or remove columns from the target except as specified.
Do NOT output explanations, comments, or extra text—only the code.
Use transform_utils module for sap utils functions.

EXAMPLE IMPLEMENTATIONS FOR COMMON PATTERNS:

1. EXISTENCE CHECK (correct implementation):
```python
# CORRECT - Check if records exist in source_table by key matching
for idx, target_row in target_df.iterrows():
    key_value = target_row['KEY_COLUMN']
    matching_rows = source_dfs['SOURCE_TABLE'][source_dfs['SOURCE_TABLE']['KEY_COLUMN'] == key_value]
    
    if not matching_rows.empty:
        # Found matching record - use its values
        target_df.loc[idx, 'TARGET_FIELD'] = matching_rows.iloc[0]['SOURCE_FIELD']

EXISTENCE CHECK (incorrect implementation - avoid this):

python# INCORRECT - This filters by a value rather than checking existence
filtered_source = source_dfs['SOURCE_TABLE'][source_dfs['SOURCE_TABLE']['SOME_FIELD'] == 'FIELD_NAME']

# This ignores the key-based matching requirement
for idx, row in filtered_source.iterrows():
    target_df.loc[target_df['KEY_COLUMN'] == row['KEY_COLUMN'], 'TARGET_FIELD'] = row['SOURCE_FIELD']

MULTI-TABLE FALLBACK (correct implementation):

python# CORRECT - Try tables in sequence, only moving to next if no match found
for idx, target_row in target_df.iterrows():
    key_value = target_row['KEY_COLUMN']
    
    # Try first table
    primary_matches = source_dfs['PRIMARY_TABLE'][source_dfs['PRIMARY_TABLE']['KEY_COLUMN'] == key_value]
    
    if not primary_matches.empty:
        # Found in primary table
        target_df.loc[idx, 'TARGET_FIELD'] = primary_matches.iloc[0]['SOURCE_FIELD']
    else:
        # Not found in primary, try secondary table
        secondary_matches = source_dfs['SECONDARY_TABLE'][source_dfs['SECONDARY_TABLE']['KEY_COLUMN'] == key_value]
        
        if not secondary_matches.empty:
            # Found in secondary table
            target_df.loc[idx, 'TARGET_FIELD'] = secondary_matches.iloc[0]['SOURCE_FIELD']
        else:
            # Not found in either table, use fallback
            fallback_value = source_dfs['FALLBACK_TABLE'].iloc[0]['SOURCE_FIELD']
            target_df.loc[idx, 'TARGET_FIELD'] = fallback_value

MULTI-TABLE FALLBACK (incorrect implementation - avoid this):

python# INCORRECT - This incorrectly filters each table by a field value
# Rather than checking existence by key

# Filter first table by some field value
filtered_primary = source_dfs['PRIMARY_TABLE'][source_dfs['PRIMARY_TABLE']['SOME_FIELD'] == 'SOME_VALUE']

if not filtered_primary.empty:
    # Use filtered primary records
    for idx, row in filtered_primary.iterrows():
        target_df.loc[target_df['KEY_COLUMN'] == row['KEY_COLUMN'], 'TARGET_FIELD'] = row['SOURCE_FIELD']
else:
    # Filter second table by the same value
    filtered_secondary = source_dfs['SECONDARY_TABLE'][source_dfs['SECONDARY_TABLE']['SOME_FIELD'] == 'SOME_VALUE']
    # This approach incorrectly uses filtering instead of existence checking
CRITICAL REQUIREMENTS:

ALWAYS use key-based matching to check for record existence, not filtering by field values
For multi-table fallbacks, ALWAYS check tables in sequence, only moving to next if no match found
NEVER interpret field names as filter values - terms like "Material", "Type", "ROH" are typically fields
ALWAYS prioritize matching records by key columns, not by other field values
When looping through records, check existence in source tables for EACH target record

Make the function like this:
def analyze_data(source_dfs, target_df):
# Import required utilities
# source_dfs is a dictionary where keys are table names and values are dataframes
# Example: source_dfs = {{'table1': df1, 'table2': df2}}
# target_df is the target dataframe to update

# Get list of source tables for easier reference
source_tables = list(source_dfs.keys())

# Your implementation of the steps above

# VALIDATION: Verify key columns exist in the result
key_columns = {json.dumps(planner_info.get('key_columns', []))}
for key in key_columns:
    if key not in target_df.columns:
        # If missing, try to add from source if possible
        for source_table in source_tables:
            if key in source_dfs[source_table].columns:
                # This is a fallback, proper implementation should handle keys correctly
                target_df[key] = source_dfs[source_table][key]
                break

# Make sure to return the modified target_df
return target_df
"""
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21", contents=prompt
                )

                # Validate response
                if not response or not hasattr(response, "text") or not response.text:
                    logger.error(
                        "Invalid response from Gemini API in _generate_code_from_simple_plan"
                    )
                    raise CodeGenerationError("Failed to generate code from Gemini API")
            except Exception as e:
                logger.error(
                    f"Error calling Gemini API in _generate_code_from_simple_plan: {e}"
                )
                raise CodeGenerationError(f"Failed to generate code: {e}")

            # Extract the code
            import re

            try:
                # First try to extract code between triple backticks
                code_match = re.search(
                    r"```python\s*(.*?)\s*```", response.text, re.DOTALL
                )
                if code_match:
                    return code_match.group(1)
                else:
                    # Next try to extract code without backticks
                    code_match = re.search(
                        r"```\s*(.*?)\s*```", response.text, re.DOTALL
                    )
                    if code_match:
                        return code_match.group(1)
                    else:
                        # Try to extract the function definition directly
                        function_match = re.search(
                            r"def analyze_data\s*\(.*?\).*?return\s+target_df",
                            response.text,
                            re.DOTALL,
                        )
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
        
        # Get list of source tables
        source_tables = list(source_dfs.keys())
        
        # Return unmodified target dataframe
        return target_df"""

    @track_token_usage()
    def _fix_code(
        self, code_content, error_info, planner_info, attempt=1, max_attempts=3
    ):
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
                    model="gemini-2.0-flash-thinking-exp-01-21", contents=prompt
                )

                # Validate response
                if not response or not hasattr(response, "text") or not response.text:
                    logger.error("Invalid response from Gemini API in _fix_code")
                    return None

                logger.info(f"Generated fixed code on attempt {attempt}")

                # Extract the code
                import re

                # First try to extract code between triple backticks
                code_match = re.search(
                    r"```python\s*(.*?)\s*```", response.text, re.DOTALL
                )
                if code_match:
                    return code_match.group(1)

                # Next try to extract code without backticks
                code_match = re.search(r"```\s*(.*?)\s*```", response.text, re.DOTALL)
                if code_match:
                    return code_match.group(1)

                # Try to extract the function definition directly
                function_match = re.search(
                    r"def analyze_data\s*\(.*?\).*?return\s+target_df",
                    response.text,
                    re.DOTALL,
                )
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
""",
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
            unnamed_cols = [
                col
                for col in cleaned_df.columns
                if "unnamed" in str(col).lower() and ":" in str(col)
            ]

            # Drop these columns
            if unnamed_cols:
                cleaned_df = cleaned_df.drop(columns=unnamed_cols)
                logger.info(
                    f"Removed {len(unnamed_cols)} unnamed columns: {unnamed_cols}"
                )

            return cleaned_df
        except Exception as e:
            logger.error(f"Error in post_proccess_result: {e}")
            # Return original result if there's an error
            return result

    def process_sequential_query(
        self,
        query,
        object_id=29,
        segment_id=336,
        project_id=24,
        session_id=None,
        target_sap_fields=None,
    ):
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
                logger.error(
                    f"Invalid ID types: object_id={type(object_id)}, segment_id={type(segment_id)}, project_id={type(project_id)}"
                )
                return None, "Invalid ID types - must be integers", session_id

            # Validate target_sap_fields
            if target_sap_fields is not None and not isinstance(
                target_sap_fields, (str, list)
            ):
                logger.error(
                    f"Invalid target_sap_fields type: {type(target_sap_fields)}"
                )
                return (
                    None,
                    "target_sap_fields must be a string or list of strings",
                    session_id,
                )

            # 1. Process query with the planner
            logger.info(f"Processing query: {query}")
            resolved_data = planner_process_query(
                object_id, segment_id, project_id, query, session_id, target_sap_fields
            )
            if not resolved_data:
                logger.error("Failed to resolve query with planner")
                return None, "Failed to resolve query", session_id

            # Connect to database
            print(resolved_data["key_mapping"])
            if not len(resolved_data["key_mapping"]) == 0:
                if isinstance(resolved_data["key_mapping"][0], str):
                    return None, resolved_data["key_mapping"][0], session_id
            try:
                conn = sqlite3.connect("db.sqlite3")
            except sqlite3.Error as e:
                logger.error(f"Failed to connect to database: {e}")
                return None, f"Database connection error: {e}", session_id

            # 2. Extract and organize all relevant information from the planner
            try:
                resolved_data["original_query"] = (
                    query  # Add original query for context
                )

                # Get target data samples safely
                try:
                    if "target_table_name" in resolved_data:
                        target_table = resolved_data["target_table_name"]
                        if isinstance(target_table, list) and len(target_table) > 0:
                            target_table = target_table[0]

                        # Validate table name to prevent SQL injection
                        target_table = validate_sql_identifier(target_table)

                        resolved_data["target_data_samples"] = (
                            get_or_create_session_target_df(
                                session_id, target_table, conn
                            ).head()
                        )
                except Exception as e:
                    logger.warning(f"Error getting target data samples: {e}")
                    resolved_data["target_data_samples"] = pd.DataFrame()

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
                source_tables = planner_info.get("source_table", [])
                if not source_tables:
                    logger.error("No source tables found in planner info")
                    if conn:
                        conn.close()
                    return None, "No source tables identified", session_id

                target_table = None
                if (
                    isinstance(planner_info.get("target_table"), list)
                    and len(planner_info["target_table"]) > 0
                ):
                    target_table = planner_info["target_table"][0]
                else:
                    target_table = planner_info.get("target_table")

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
                        source_dfs[table] = pd.read_sql_query(
                            f"SELECT * FROM {safe_table}", conn
                        )
                    except Exception as e:
                        logger.warning(f"Error reading source table {table}: {e}")
                        # Create an empty dataframe as fallback
                        source_dfs[table] = pd.DataFrame()

                # Get target dataframe
                try:
                    target_df = get_or_create_session_target_df(
                        session_id, target_table, conn
                    )
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
                    code_content = self._generate_code_from_simple_plan(
                        simple_plan, planner_info
                    )
                    logger.info(
                        f"Code generated with length: {len(code_content)} chars"
                    )
                except CodeGenerationError as e:
                    logger.error(f"Code generation error: {e}")
                    if conn:
                        conn.close()
                    return None, f"Failed to generate code: {e}", session_id

                # 7. Execute the generated code with error handling
                try:
                    code_file = create_code_file(code_content, query, is_double=True)
                    result = execute_code(code_file, source_dfs, target_df, target_sap_fields)

                    # Check if result is an error (now it's a dictionary with traceback information)
                    if isinstance(result, dict) and "error_type" in result:
                        logger.error(f"Code execution error: {result['error_message']}")
                        logger.error(f"Traceback: {result['traceback']}")

                        # Try to fix the code up to 3 times
                        fixed_code = code_content
                        for attempt in range(1, 4):  # 3 attempts maximum
                            logger.info(f"Attempting to fix code (attempt {attempt}/3)")

                            fixed_code = self._fix_code(
                                fixed_code,
                                result,
                                planner_info,
                                attempt=attempt,
                                max_attempts=3,
                            )
                            if fixed_code is None:
                                # Failed to fix the code after max attempts
                                if conn:
                                    conn.close()
                                return (
                                    code_content,
                                    f"Failed to generate working code after 3 attempts. Last error: {result['error_message']}",
                                    session_id,
                                )

                            # Try executing the fixed code
                            fixed_code_file = create_code_file(
                                fixed_code,
                                f"{query} (fixed attempt {attempt})",
                                is_double=True,
                            )
                            fixed_result = execute_code(
                                fixed_code_file, source_dfs, target_df, target_sap_fields
                            )

                            # If the fixed code worked (result is not an error dictionary), use it
                            if (
                                not isinstance(fixed_result, dict)
                                or "error_type" not in fixed_result
                            ):
                                logger.info(
                                    f"Successfully fixed code on attempt {attempt}"
                                )
                                code_content = fixed_code  # Update the code content to the fixed version
                                result = fixed_result  # Use the successful result
                                break
                            else:
                                # The fix didn't work, update the error for the next attempt
                                result = fixed_result
                                logger.error(
                                    f"Fix attempt {attempt} failed with error: {result['error_message']}"
                                )

                        # If we've gone through all attempts and still have an error
                        if isinstance(result, dict) and "error_type" in result:
                            if conn:
                                conn.close()
                            return (
                                code_content,
                                f"Failed to generate working code after 3 attempts. Last error: {result['error_message']}",
                                session_id,
                            )

                    elif isinstance(result, str) and "Error" in result:
                        logger.error(f"Code execution error: {result}")
                        if conn:
                            conn.close()
                        return (
                            code_content,
                            f"Code execution failed: {result}",
                            session_id,
                        )
                except Exception as e:
                    logger.error(f"Error executing code: {e}")
                    if conn:
                        conn.close()
                    return code_content, f"Error executing code: {e}", session_id

                # 8. Save the updated target dataframe if it's a DataFrame
                if isinstance(result, pd.DataFrame):
                    try:
                        result = self.post_proccess_result(result)
                        print(result.columns)
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
