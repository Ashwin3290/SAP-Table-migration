import os
import logging
import json
import pandas as pd
import numpy as np
import re
import sqlite3
import traceback
from dotenv import load_dotenv
from google import genai
from google.genai import types
from workspace_db import WorkspaceDB
from token_tracker import track_token_usage, get_token_usage_stats

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

class DMTool:
    """Improved DMTool with optimized code generation and better information flow"""

    def __init__(self):
        """Initialize the DMTool instance"""
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

            # Initialize workspace database
            self.workspace_db = WorkspaceDB()
            logger.info("Workspace database initialized")

            logger.info("DMTool initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing DMTool: {e}")
            raise
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'workspace_db'):
                self.workspace_db.close()
                logger.info("Workspace database connection closed")
        except Exception as e:
            logger.error(f"Error closing workspace database: {e}")

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
                    model="gemini-2.0-flash", contents=prompt
                )

                # Validate response
                if not response or not hasattr(response, "text") or not response.text:
                    logger.warning(
                        "Invalid response from Gemini API in _classify_query"
                    )
                    return "EXTRACTION"  # Default classification

                # Return the classification
                classification = response.text.strip()

                # Validate that classification is one of the allowed values
                allowed_classifications = [
                    "FILTER_AND_EXTRACT",
                    "UPDATE_EXISTING",
                    "CONDITIONAL_MAPPING",
                    "EXTRACTION",
                    "TIERED_LOOKUP",
                    "AGGREGATION",
                ]

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
1. Begin with data validation steps for all source tables and fields
2. Include clear steps for filtering if filtering fields are specified
3. Address how both empty AND populated target dataframes will be handled
4. Explicitly include key column mapping between source and target
5. Include steps for handling edge cases like missing data and type mismatches
6. Ensure the plan is complete with no steps missing
7. Ensure all target fields will be populated in the result

FORMAT:
- Number each step sequentially
- Make each step clear and actionable
- Include 8-12 detailed steps, covering all necessary operations
- Be specific about which fields and tables are used in each step

Example of a good step: "Filter the source_table dataframe where field_name matches the condition value"
Example of a bad step: "Apply filtering as needed"

Return ONLY the numbered plan with no explanations or additional text.
"""

            try:
                print(base_prompt)
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21", contents=base_prompt,
                    config=types.GenerateContentConfig(temperature=0.2)
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
DO NOT make dummy dataframes - use the provided source_dfs and target_df
DO NOT use hardcoded values - use the provided key mappings

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
The code should have proper indentation and syntax.

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
                    model="gemini-2.0-flash-thinking-exp-01-21", contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.25, top_p=0.9)
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
    def _fix_code(self, code_content, error_info, planner_info, attempt=1, max_attempts=3):
        """
        Improved code fixer that handles syntax errors better, especially indentation issues
        
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
            
            # Check if this is a syntax error, especially indentation
            is_indentation_error = "IndentationError" in error_type
            is_syntax_error = "SyntaxError" in error_type
            
            # For indentation errors, use a special approach
            if is_indentation_error:
                fixed_code = self._fix_indentation_error(code_content, error_message)
                if fixed_code:
                    logger.info("Fixed indentation error directly")
                    return fixed_code
            
            # Create a prompt for the code fixer with specialized instructions based on error type
            if is_indentation_error or is_syntax_error:
                prompt = f"""
    You are an expert Python code fixer. The following code has a {error_type} that needs to be fixed:

    ```python
    {code_content}
    ```

    ERROR INFORMATION:
    Error Type: {error_type}
    Error Message: {error_message}
    FULL TRACEBACK:
    {traceback_text}

    SPECIFIC INSTRUCTIONS:
    1. This is a SYNTAX ERROR, not a logical error. Focus ONLY on fixing the syntax.
    2. Pay special attention to indentation levels in all blocks.
    3. Check for missing colons after if/for/while/def statements.
    4. Check for missing indentation after if/else/for/try/except blocks.
    5. Make sure all parentheses, brackets, and braces are properly closed.
    6. Look for missing commas in lists, dictionaries, or function calls.
    7. Check proper line continuation in multi-line statements.

    DO NOT try to rewrite the code's logic, ONLY fix the syntax errors.
    DO NOT add any example data, test code, or additional functionality.
    KEEP all imports, comments, and existing functionality intact.

    Return ONLY the fixed code with no explanations. Make sure every line is properly indented.
    """
            else:
                # Standard prompt for other types of errors
                prompt = f"""
    You are an expert code debugging and fixing agent. This code has a {error_type} error:

    ```python
    {code_content}
    ```

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

    INSTRUCTIONS:
    1. Fix ONLY the exact error shown in the traceback
    2. DO NOT change any working code
    3. DO NOT add additional test code or debug prints
    4. Return the complete fixed code with no explanations

    This is attempt {attempt} of {max_attempts}.
    """

            # Call the AI to fix the code
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21", contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2)  # Lower temperature for more precise fixes
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
                    fixed_code = code_match.group(1)
                else:
                    # Next try to extract code without backticks
                    code_match = re.search(r"```\s*(.*?)\s*```", response.text, re.DOTALL)
                    if code_match:
                        fixed_code = code_match.group(1)
                    else:
                        # Try to extract the function definition directly
                        function_match = re.search(
                            r"def analyze_data\s*\(.*?\).*?return\s+\w+",
                            response.text,
                            re.DOTALL,
                        )
                        if function_match:
                            fixed_code = function_match.group(0)
                        else:
                            # Last resort, return the whole text if it looks like code
                            if "def analyze_data" in response.text:
                                fixed_code = response.text
                            else:
                                logger.error("Could not extract code from response")
                                return None

                # Verify the code is actually different - don't return the same code
                if fixed_code.strip() == code_content.strip():
                    logger.warning("Fix attempt produced identical code, making manual adjustment")
                    # For indentation errors, try a simple fix
                    if is_indentation_error:
                        return self._fix_indentation_error(code_content, error_message)
                    # For other errors, try a simple fix if LLM couldn't help
                    return self._apply_simple_fix(code_content, error_info)
                    
                # Basic validation - make sure it has the function definition
                if "def analyze_data" not in fixed_code:
                    logger.error("Fixed code does not contain analyze_data function")
                    return None
                    
                return fixed_code

            except Exception as e:
                logger.error(f"Error calling Gemini API in _fix_code: {e}")
                # Try a simple fix if API call fails
                if is_indentation_error:
                    return self._fix_indentation_error(code_content, error_message)
                return self._apply_simple_fix(code_content, error_info)
        except Exception as e:
            logger.error(f"Error in _fix_code: {e}")
            return None

    def _fix_indentation_error(self, code_content, error_message):
        """
        Directly fix indentation errors without using LLM
        
        Parameters:
        code_content (str): The code with indentation errors
        error_message (str): The error message with line information
        
        Returns:
        str: Fixed code or None if couldn't fix
        """
        try:
            # Parse the line number from the error message
            import re
            line_match = re.search(r"line (\d+)", error_message)
            if not line_match:
                return None
                
            line_number = int(line_match.group(1))
            
            # Split the code into lines for processing
            lines = code_content.split('\n')
            
            # Check if we have enough lines
            if line_number >= len(lines) or line_number < 1:
                return None
                
            # Check for specific indentation issues
            if "expected an indented block" in error_message:
                # Find the last line that should have an indented block following it
                trigger_keywords = ["if", "else:", "elif", "for", "while", "try:", "except", "def", "class"]
                problematic_line = lines[line_number - 1]
                
                # Add indentation to the problematic line
                if any(keyword in lines[line_number - 2] for keyword in trigger_keywords):
                    lines[line_number - 1] = "    " + problematic_line
                    
            # Check for unexpected indent
            elif "unexpected indent" in error_message:
                problematic_line = lines[line_number - 1]
                # Remove one level of indentation
                if problematic_line.startswith("    "):
                    lines[line_number - 1] = problematic_line[4:]
                elif problematic_line.startswith("\t"):
                    lines[line_number - 1] = problematic_line[1:]
                    
            # Check for missing except block
            if "expected an indented block after 'except'" in error_message:
                # Find the except line and add a simple pass statement
                for i in range(line_number - 2, min(line_number + 2, len(lines))):
                    if i >= 0 and "except" in lines[i] and ":" in lines[i]:
                        # Add a pass statement after the except line
                        if i+1 < len(lines):
                            lines.insert(i+1, "        pass  # Added to fix indentation error")
                        else:
                            lines.append("        pass  # Added to fix indentation error")
                        break
            
            # Reassemble the fixed code
            fixed_code = '\n'.join(lines)
            return fixed_code
        except Exception as e:
            logger.error(f"Error in _fix_indentation_error: {e}")
            return None

    def _apply_simple_fix(self, code_content, error_info):
        """
        Apply simple fixes for common errors
        
        Parameters:
        code_content (str): The original code with errors
        error_info (dict): Error information dictionary
        
        Returns:
        str: Fixed code or original code if couldn't fix
        """
        try:
            error_type = error_info.get("error_type", "")
            error_message = error_info.get("error_message", "")
            
            # Split the code into lines
            lines = code_content.split('\n')
            
            # Fix missing closing parentheses/brackets
            if "SyntaxError" in error_type and ("unexpected EOF" in error_message or "unexpected end of file" in error_message):
                # Check for unbalanced parentheses, brackets, and braces
                parens = code_content.count('(') - code_content.count(')')
                brackets = code_content.count('[') - code_content.count(']')
                braces = code_content.count('{') - code_content.count('}')
                
                # Add the missing closing characters
                fixed_code = code_content
                fixed_code += ')' * parens if parens > 0 else ''
                fixed_code += ']' * brackets if brackets > 0 else ''
                fixed_code += '}' * braces if braces > 0 else ''
                
                return fixed_code
                
            # Fix missing colons
            if "SyntaxError" in error_type and "expected ':'" in error_message:
                # Try to find the line with the error
                line_match = re.search(r"line (\d+)", error_message)
                if line_match:
                    line_number = int(line_match.group(1))
                    if 0 < line_number <= len(lines):
                        # Add a colon at the end of the line if it's missing
                        if not lines[line_number - 1].strip().endswith(':'):
                            lines[line_number - 1] = lines[line_number - 1].rstrip() + ':'
                        return '\n'.join(lines)
                        
            # Fix invalid continuation
            if "SyntaxError" in error_type and "invalid continuation" in error_message:
                # Try to find the line with the error
                line_match = re.search(r"line (\d+)", error_message)
                if line_match:
                    line_number = int(line_match.group(1))
                    if 0 < line_number <= len(lines):
                        # Replace tabs with spaces at the beginning of the line
                        lines[line_number - 1] = lines[line_number - 1].replace('\t', '    ')
                        return '\n'.join(lines)
                        
            # Return the original code if no fix was applied
            return code_content
        except Exception as e:
            logger.error(f"Error in _apply_simple_fix: {e}")
            return code_content

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
        session_id=None
    ):
        """
        Process a query as part of a sequential transformation
        With query classification and specialized handling for different operations
        
        Parameters:
        query (str): The user's query
        object_id (int): Object ID for mapping
        segment_id (int): Segment ID for mapping
        project_id (int): Project ID for mapping
        session_id (str): Optional session ID, creates new session if None
        
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

            # 1. Process query with the planner - now with classification
            logger.info(f"Processing query: {query}")
            resolved_data = planner_process_query(
                object_id, segment_id, project_id, query, session_id
            )
            if not resolved_data:
                logger.error("Failed to resolve query with planner")
                return None, "Failed to resolve query", session_id
                
            # Get query type from resolved data
            query_type = resolved_data.get("query_type", "SIMPLE_TRANSFORMATION")
            logger.info(f"Query type determined as: {query_type}")
                
            # Get session ID and segments info from the results
            session_id = resolved_data.get("session_id")
            visited_segments = resolved_data.get("visited_segments", {})
            current_segment = resolved_data.get("current_segment", {})

            # Connect to database
            try:
                conn = sqlite3.connect("db.sqlite3")
            except sqlite3.Error as e:
                logger.error(f"Failed to connect to database: {e}")
                return None, f"Database connection error: {e}", session_id

            # 2. Extract and organize all relevant information from the planner
            try:
                resolved_data["original_query"] = query
                
                # Get target data samples safely - same for all query types
                try:
                    if "target_table_name" in resolved_data:
                        target_table = resolved_data["target_table_name"]
                        if isinstance(target_table, list) and len(target_table) > 0:
                            target_table = target_table[0]
                        target_table = validate_sql_identifier(target_table)
                        resolved_data["target_data_samples"] = get_or_create_session_target_df(
                            session_id, target_table, conn
                        ).head()
                except Exception as e:
                    logger.warning(f"Error getting target data samples: {e}")
                    resolved_data["target_data_samples"] = pd.DataFrame()

                # Extract and organize planner information
                planner_info = self._extract_planner_info(resolved_data)
                
                # Add query type and classification details
                planner_info["query_type"] = query_type
                planner_info["classification_details"] = resolved_data.get("classification_details", {})
                
                # Add segment information to planner info
                planner_info["visited_segments"] = visited_segments
                planner_info["current_segment"] = current_segment
                
                # Add segment references based on query type
                if query_type in ["CROSS_SEGMENT", "JOIN_OPERATION"]:
                    # For these operations, segment references are important
                    segment_references = resolved_data.get("segment_references", [])
                    planner_info["segment_references"] = segment_references
                    
                    # For JOIN_OPERATION, also add join conditions
                    if query_type == "JOIN_OPERATION":
                        join_conditions = resolved_data.get("join_conditions", [])
                        planner_info["join_conditions"] = join_conditions
                
                # 3. Extract table names safely
                source_tables = planner_info.get("source_table", [])
                if not source_tables:
                    logger.error("No source tables found in planner info")
                    if conn:
                        conn.close()
                    return None, "No source tables identified", session_id

                target_table = None
                if isinstance(planner_info.get("target_table"), list) and len(planner_info["target_table"]) > 0:
                    target_table = planner_info["target_table"][0]
                else:
                    target_table = planner_info.get("target_table")

                if not target_table:
                    logger.error("No target table found in planner info")
                    if conn:
                        conn.close()
                    return None, "No target table identified", session_id

                # 4. Get source and target dataframes - handling depends on query type
                source_dfs = self._load_source_tables(source_tables, visited_segments, conn, query_type, session_id=session_id)                
                target_df = get_or_create_session_target_df(session_id, target_table, conn)

                # 5. Generate a simple, step-by-step plan in natural language
                try:
                    simple_plan = self._generate_simple_plan(planner_info)
                    logger.info(f"Simple plan generated: {simple_plan}")
                except Exception as e:
                    logger.error(f"Error generating simple plan: {e}")
                    simple_plan = "1. Make copy of the source dataframe\n2. Return the dataframe unchanged"

                # 6. Generate code based on query type
                try:
                    # Select code generation method based on query type
                    if query_type == "JOIN_OPERATION":
                        code_content = self._generate_join_code(simple_plan, planner_info)
                    elif query_type == "CROSS_SEGMENT":
                        code_content = self._generate_cross_segment_code(simple_plan, planner_info)
                    elif query_type == "VALIDATION_OPERATION":
                        code_content = self._generate_validation_code(simple_plan, planner_info)
                    elif query_type == "AGGREGATION_OPERATION":
                        code_content = self._generate_aggregation_code(simple_plan, planner_info)
                    else:
                        # Default to simple transformation
                        code_content = self._generate_code_from_simple_plan(simple_plan, planner_info)
                        
                    logger.info(f"Code generated with length: {len(code_content)} chars")
                except Exception as e:
                    logger.error(f"Error generating code: {e}")
                    if conn:
                        conn.close()
                    return None, f"Failed to generate code: {e}", session_id

                # 7. Execute the generated code with error handling
                try:
                    code_file = create_code_file(code_content, query, is_double=True)
                    target_sap_fields = resolved_data.get("target_sap_fields")
                    key_mapping = resolved_data.get("key_mapping", [])
                    result = execute_code(code_file, source_dfs, target_df, target_sap_fields, query_type=query_type, key_mapping=key_mapping,session_id=session_id)
                    print(f"Result: {isinstance(result,pd.DataFrame)}\n{result}")
                    # Handle execution errors and fix attempts
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
                            fixed_result = execute_code(fixed_code_file, source_dfs, target_df, target_sap_fields, query_type=query_type, key_mapping=key_mapping, session_id=session_id)

                            # If the fixed code worked, use it
                            if not isinstance(fixed_result, dict) or "error_type" not in fixed_result:
                                logger.info(f"Successfully fixed code on attempt {attempt}")
                                code_content = fixed_code  # Update the code content to the fixed version
                                result = fixed_result  # Use the successful result
                                break
                            else:
                                # The fix didn't work, update the error for the next attempt
                                result = fixed_result
                                logger.error(f"Fix attempt {attempt} failed: {result['error_message']}")

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

                if isinstance(result, pd.DataFrame):
                    try:
                        # Post-process the result dataframe
                        result = self.post_proccess_result(result)
                        
                        # Save to session target
                        save_success = save_session_target_df(session_id, result)
                        if not save_success:
                            logger.warning("Failed to save target dataframe to session")
                        try:

                        # Special handling for different query types
                            if query_type in ["CROSS_SEGMENT", "JOIN_OPERATION"]:
                                # Save to workspace with current segment info for cross-segment use
                                    # Get current segment name
                                    segment_name = current_segment.get("name", f"segment_{segment_id}")
                                    
                                    # Save to workspace DB with current segment name
                                    result = result.copy()
                                    
                                    # Ensure key fields are present for joins
                                    key_fields = []
                                    if key_mapping:
                                        # Extract source fields from key mappings
                                        key_fields = [mapping.get('source_col') for mapping in key_mapping 
                                                    if isinstance(mapping, dict) and 'source_col' in mapping]

                                    # Also check for common key fields
                                    common_key_fields = ['MATNR', 'MATERIAL', 'PRODUCT', 'WERKS', 'PLANT']
                                    for key_field in common_key_fields:
                                        if key_field not in key_fields:
                                            key_fields.append(key_field)

                                    # Add necessary key fields for cross-segment compatibility
                                    for key_field in key_fields:
                                        if key_field not in result.columns:
                                            # Try to add key field from source
                                            for table_name, df in source_dfs.items():
                                                if key_field in df.columns:
                                                    logger.info(f"Adding {key_field} from {table_name} for cross-segment compatibility")
                                                    if len(df) >= len(result):
                                                        result[key_field] = df[key_field].values[:len(result)]
                                                    else:
                                                        field_values = [None] * len(result)
                                                        for i in range(min(len(df), len(result))):
                                                            field_values[i] = df[key_field].iloc[i]
                                                        result[key_field] = field_values
                                                    break
                                    
                                    # Save to workspace
                            workspace_table = self.workspace_db.save_segment_table(
                                    session_id, segment_id, segment_name, result
                                )
                            logger.info(f"Saved result to workspace as {workspace_table} with {len(result)} rows")
                            return code_content, f"Saved result to workspace as {workspace_table}", session_id
                        except Exception as e:
                            logger.error(f"Error saving to workspace: {e}")
                        
                    except Exception as e:
                        logger.error(f"Error saving target dataframe: {e}")

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


    def _load_source_tables(self, source_tables, visited_segments, conn, query_type, session_id=None):
        """
        Load source tables with special handling for different query types
        
        Parameters:
        source_tables (list): List of source table names
        visited_segments (dict): Dictionary of visited segments
        conn (Connection): Database connection
        query_type (str): Type of query
        session_id (str, optional): Current session ID
        
        Returns:
        dict: Dictionary of source DataFrames
        """
        source_dfs = {}
        
        for table in source_tables:
            # Clean the table name first to remove suffixes like "Table"
            from planner import clean_table_name
            cleaned_table = clean_table_name(table)
            
            # Check if this is a workspace table (from a previous segment)
            is_workspace_table = False
            segment_id = None
            segment_name = None
            
            # Check both by exact name and partial match for robust detection
            for seg_id, segment_info in visited_segments.items():
                table_name = segment_info.get("table_name", "")
                seg_name = segment_info.get("name", "")
                
                # Check if table matches either the table_name or segment_name
                if (cleaned_table == table_name or 
                    (table_name and table_name.lower() in cleaned_table.lower()) or
                    (seg_name and seg_name.lower() in cleaned_table.lower())):
                    is_workspace_table = True
                    segment_id = seg_id
                    segment_name = seg_name
                    break
                    
            # Special handling for segment tables in CROSS_SEGMENT queries
            if is_workspace_table and query_type in ["CROSS_SEGMENT", "JOIN_OPERATION"] and session_id:
                try:
                    logger.info(f"Attempting to load workspace table: {cleaned_table} (segment ID: {segment_id}, segment name: {segment_name})")
                    
                    # Try to get the table name from the workspace DB
                    workspace_table = self.workspace_db.get_segment_table_name(
                        session_id, segment_id=segment_id, segment_name=segment_name
                    )
                    
                    if workspace_table:
                        logger.info(f"Loading source table {cleaned_table} from workspace ({workspace_table})")
                        source_dfs[table] = pd.read_sql_query(
                            f"SELECT * FROM '{workspace_table}'", 
                            self.workspace_db.conn
                        )
                    else:
                        # Try with just the segment name 
                        workspace_table = self.workspace_db.get_segment_table_name(
                            session_id, segment_name=segment_name
                        )
                        
                        if workspace_table:
                            logger.info(f"Loading {cleaned_table} using segment name {segment_name} from workspace ({workspace_table})")
                            source_dfs[table] = pd.read_sql_query(
                                f"SELECT * FROM '{workspace_table}'", 
                                self.workspace_db.conn
                            )
                        else:
                            # Try with a partial match on segment name or table name
                            segment_name_parts = segment_name.split() if segment_name else []
                            for part in segment_name_parts:
                                if len(part) > 3:  # Only try with meaningful parts
                                    workspace_table = self.workspace_db.get_segment_table_name(
                                        session_id, segment_name=part
                                    )
                                    if workspace_table:
                                        logger.info(f"Loading {cleaned_table} using partial match '{part}' from workspace ({workspace_table})")
                                        source_dfs[table] = pd.read_sql_query(
                                            f"SELECT * FROM '{workspace_table}'", 
                                            self.workspace_db.conn
                                        )
                                        break
                            
                            # If still not found, check table name in all session tables
                            if table not in source_dfs:
                                tables = self.workspace_db.list_session_tables(session_id)
                                logger.info(f"Available workspace tables: {tables}")
                                
                                # See if any table name contains our cleaned table name
                                for table_info in tables:
                                    if cleaned_table.lower() in table_info['table_name'].lower():
                                        workspace_table = table_info['table_name']
                                        logger.info(f"Found matching workspace table: {workspace_table}")
                                        source_dfs[table] = pd.read_sql_query(
                                            f"SELECT * FROM '{workspace_table}'", 
                                            self.workspace_db.conn
                                        )
                                        break
                    
                    # If still not found, create an empty DataFrame
                    if table not in source_dfs:
                        logger.warning(f"Could not find workspace table for {cleaned_table}")
                        source_dfs[table] = pd.DataFrame()
                        
                except Exception as e:
                    logger.error(f"Error loading segment table {cleaned_table}: {e}")
                    source_dfs[table] = pd.DataFrame()
            else:
                # Regular table from main database
                try:
                    # Validate table name to prevent SQL injection
                    safe_table = validate_sql_identifier(cleaned_table)
                    # Use a parameterized query for safety
                    logger.info(f"Loading source table {cleaned_table} from main database")
                    source_dfs[table] = pd.read_sql_query(
                        f"SELECT * FROM {safe_table}", conn
                    )
                except Exception as e:
                    logger.warning(f"Error reading source table {cleaned_table}: {e}")
                    # Create an empty dataframe as fallback
                    source_dfs[table] = pd.DataFrame()
                        
        return source_dfs
    
    def _generate_join_code(self, simple_plan, planner_info):
        """
        Generate code specifically for JOIN operations
        
        Parameters:
        simple_plan (str): Step-by-step plan in natural language
        planner_info (dict): Context information from planner
        
        Returns:
        str: Generated code
        """
        # Extract join information
        join_conditions = planner_info.get("join_conditions", [])
        
        prompt = f"""
    Write Python code for a JOIN operation following these EXACT steps:

    {simple_plan}

    JOIN DETAILS:
    {json.dumps(join_conditions, indent=2)}

    CONTEXT INFORMATION:
    - Source tables: {json.dumps(planner_info.get('source_table', []), indent=2)}
    - Target table: {planner_info.get('target_table', [])}
    - Source fields: {planner_info.get('source_fields', [])}
    - Target fields: {planner_info.get('target_fields', [])}
    - Filtering fields: {planner_info.get('filtering_fields', [])}
    - Filtering conditions: {json.dumps(planner_info.get('extracted_conditions', {}), indent=2)}

    SAMPLE DATA:
    Source data samples:
    {json.dumps(planner_info.get('source_data', {}).get('sample', {}), indent=2)}

    Target data sample:
    {json.dumps(planner_info.get('target_data', {}).get('sample', []), indent=2)}

    JOIN IMPLEMENTATION REQUIREMENTS:
    1. Use pandas.merge() for joining tables
    2. Properly handle join types (inner, left, right, outer)
    3. Use the exact join fields specified in the join conditions
    4. Apply filtering conditions AFTER joining if applicable
    5. Handle empty dataframes properly
    6. Return a dataframe with the required target fields

    The function must follow this structure:
    def analyze_data(source_dfs, target_df):
        # Import required libraries
        import pandas as pd
        import numpy as np
        
        # Source tables are available in the source_dfs dictionary
        # Example: source_dfs = {{'MARA': df1, 'MARC': df2}}
        
        # Implement the join operations according to the plan
        # ...
        
        # Return the updated target dataframe
        return result_df
    """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21", 
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.25, top_p=0.9)
            )
            
            # Extract code from response
            import re
            code_match = re.search(r"```python\s*(.*?)\s*```", response.text, re.DOTALL)
            if code_match:
                return code_match.group(1)
            
            code_match = re.search(r"```\s*(.*?)\s*```", response.text, re.DOTALL)
            if code_match:
                return code_match.group(1)
                
            function_match = re.search(r"def analyze_data\s*\(.*?\).*?return\s+\w+", response.text, re.DOTALL)
            if function_match:
                return function_match.group(0)
                
            return response.text
        except Exception as e:
            logger.error(f"Error generating join code: {e}")
            # Return a basic join implementation
            return """def analyze_data(source_dfs, target_df):
        import pandas as pd
        import numpy as np
        
        # Get source tables
        source_tables = list(source_dfs.keys())
        if len(source_tables) < 2:
            return target_df  # Not enough tables for join
            
        # Basic implementation - join first two tables
        table1 = source_tables[0]
        table2 = source_tables[1]
        
        if table1 in source_dfs and table2 in source_dfs:
            result = pd.merge(
                source_dfs[table1],
                source_dfs[table2],
                left_on='MATNR',  # Default join field
                right_on='MATNR',
                how='inner'
            )
            return result
        else:
            return target_df
    """

    def _generate_cross_segment_code(self, simple_plan, planner_info):
        """
        Generate code specifically for CROSS_SEGMENT operations
        
        Parameters:
        simple_plan (str): Step-by-step plan in natural language
        planner_info (dict): Context information from planner
        
        Returns:
        str: Generated code
        """
        # Extract cross segment information
        segment_references = planner_info.get("segment_references", [])
        cross_segment_joins = planner_info.get("cross_segment_joins", [])
        
        prompt = f"""
    Write Python code for a CROSS-SEGMENT operation following these EXACT steps:

    {simple_plan}

    SEGMENT REFERENCES:
    {json.dumps(segment_references, indent=2)}

    CROSS-SEGMENT JOINS:
    {json.dumps(cross_segment_joins, indent=2)}

    CONTEXT INFORMATION:
    - Source tables: {json.dumps(planner_info.get('source_table', []), indent=2)}
    - Target table: {planner_info.get('target_table', [])}
    - Source fields: {planner_info.get('source_fields', [])}
    - Target fields: {planner_info.get('target_fields', [])}
    - Filtering fields: {planner_info.get('filtering_fields', [])}
    - Filtering conditions: {json.dumps(planner_info.get('extracted_conditions', {}), indent=2)}

    SAMPLE DATA:
    Source data samples:
    {json.dumps(planner_info.get('source_data', {}).get('sample', {}), indent=2)}

    Target data sample:
    {json.dumps(planner_info.get('target_data', {}).get('sample', []), indent=2)}

    CROSS-SEGMENT IMPLEMENTATION REQUIREMENTS:
    1. Properly access segment tables from the source_dfs dictionary
    2. Join segment data with current data using specified join fields
    3. Apply transformations across segments as needed
    4. Handle empty dataframes properly
    5. Return a dataframe with the required target fields
    6. Make sure to handle cases where segment tables might be empty or missing
    7. Ensure that the join conditions are met and handle any mismatches
    8. Include error handling for potential issues during cross-segment operations
    9. Ensure that the function is robust and can handle various edge cases
    10. DO NOT include any print statements or debugging code in the final output
    11. DO NOT make any dummy dataframes or tables in the final output

    The function must follow this structure:
    def analyze_data(source_dfs, target_df):
        # Import required libraries
        import pandas as pd
        import numpy as np
        
        # Source tables are available in the source_dfs dictionary
        # Example: source_dfs = {{'MARA': df1, 'basic_segment': df2}}
        
        # Implement the cross-segment operations according to the plan
        # ...
        
        # Return the updated target dataframe
        return result_df
    """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21", 
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.25, top_p=0.9)
            )
            
            # Extract code from response
            import re
            code_match = re.search(r"```python\s*(.*?)\s*```", response.text, re.DOTALL)
            if code_match:
                return code_match.group(1)
            
            code_match = re.search(r"```\s*(.*?)\s*```", response.text, re.DOTALL)
            if code_match:
                return code_match.group(1)
                
            function_match = re.search(r"def analyze_data\s*\(.*?\).*?return\s+\w+", response.text, re.DOTALL)
            if function_match:
                return function_match.group(0)
                
            return response.text
        except Exception as e:
            logger.error(f"Error generating cross-segment code: {e}")
            # Return a basic implementation
            return """def analyze_data(source_dfs, target_df):
        import pandas as pd
        import numpy as np
        
        # Get segment tables
        segment_tables = [table for table in source_dfs.keys() if 'segment' in table.lower()]
        current_tables = [table for table in source_dfs.keys() if 'segment' not in table.lower()]
        
        if not segment_tables or not current_tables:
            return target_df  # Not enough tables for cross-segment operation
            
        # Basic implementation - use first segment table and first current table
        segment_table = segment_tables[0]
        current_table = current_tables[0]
        
        if segment_table in source_dfs and current_table in source_dfs:
            # Join the segment data with current data
            result = pd.merge(
                source_dfs[segment_table],
                source_dfs[current_table],
                left_on='MATNR',  # Default join field
                right_on='MATNR',
                how='left'
            )
            return result
        else:
            return target_df
    """

    def _generate_validation_code(self, simple_plan, planner_info):
        """
        Generate code specifically for VALIDATION operations
        
        Parameters:
        simple_plan (str): Step-by-step plan in natural language
        planner_info (dict): Context information from planner
        
        Returns:
        str: Generated code
        """
        validation_rules = planner_info.get("validation_rules", [])
        
        prompt = f"""
    Write Python code for a DATA VALIDATION operation following these EXACT steps:

    {simple_plan}

    VALIDATION RULES:
    {json.dumps(validation_rules, indent=2)}

    CONTEXT INFORMATION:
    - Source tables: {json.dumps(planner_info.get('source_table', []), indent=2)}
    - Target table: {planner_info.get('target_table', [])}
    - Source fields: {planner_info.get('source_fields', [])}
    - Target fields: {planner_info.get('target_fields', [])}
    - Filtering fields: {planner_info.get('filtering_fields', [])} if 'filtering_fields' in planner_info else []

    SAMPLE DATA:
    Source data samples:
    {json.dumps(planner_info.get('source_data', {}).get('sample', {}), indent=2)}

    Target data sample:
    {json.dumps(planner_info.get('target_data', {}).get('sample', []), indent=2)}

    VALIDATION IMPLEMENTATION REQUIREMENTS:
    1. Check data against specified validation rules
    2. For each record, determine if it passes or fails validation
    3. Update target fields with validation results
    4. Include detailed validation status where appropriate
    5. Handle edge cases like empty values or missing fields

    The function must follow this structure:
    def analyze_data(source_dfs, target_df):
        # Import required libraries
        import pandas as pd
        import numpy as np
        
        # Implement validation according to the plan
        # ...
        
        # Return the updated target dataframe with validation results
        return result_df
    """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21", 
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.25, top_p=0.9)
            )
            
            # Extract code from response
            import re
            code_match = re.search(r"```python\s*(.*?)\s*```", response.text, re.DOTALL)
            if code_match:
                return code_match.group(1)
            
            code_match = re.search(r"```\s*(.*?)\s*```", response.text, re.DOTALL)
            if code_match:
                return code_match.group(1)
                
            function_match = re.search(r"def analyze_data\s*\(.*?\).*?return\s+\w+", response.text, re.DOTALL)
            if function_match:
                return function_match.group(0)
                
            return response.text
        except Exception as e:
            logger.error(f"Error generating validation code: {e}")
            # Return a basic validation implementation
            return """def analyze_data(source_dfs, target_df):
        import pandas as pd
        import numpy as np
        
        # Get source table
        if not source_dfs:
            return target_df
            
        source_table = list(source_dfs.keys())[0]
        source_df = source_dfs[source_table]
        
        # Create a copy of target df
        result = target_df.copy()
        
        # Add validation status column if it doesn't exist
        if 'VALIDATION_STATUS' not in result.columns:
            result['VALIDATION_STATUS'] = 'VALID'
        
        # Basic validation - check for null values in key fields
        for col in source_df.columns:
            if 'MATNR' in col or 'KEY' in col.upper():
                mask = source_df[col].isna()
                if mask.any():
                    # Flag invalid records
                    invalid_indices = source_df[mask].index
                    if len(result) >= len(invalid_indices):
                        result.loc[invalid_indices, 'VALIDATION_STATUS'] = 'INVALID - NULL KEY'
        
        return result
    """

    def _generate_aggregation_code(self, simple_plan, planner_info):
        """
        Generate code specifically for AGGREGATION operations
        
        Parameters:
        simple_plan (str): Step-by-step plan in natural language
        planner_info (dict): Context information from planner
        
        Returns:
        str: Generated code
        """
        aggregation_functions = planner_info.get("aggregation_functions", [])
        group_by_fields = planner_info.get("group_by_fields", [])
        
        prompt = f"""
    Write Python code for a DATA AGGREGATION operation following these EXACT steps:

    {simple_plan}

    AGGREGATION FUNCTIONS:
    {json.dumps(aggregation_functions, indent=2)}

    GROUP BY FIELDS:
    {json.dumps(group_by_fields, indent=2)}

    CONTEXT INFORMATION:
    - Source tables: {json.dumps(planner_info.get('source_table', []), indent=2)}
    - Target table: {planner_info.get('target_table', [])}
    - Source fields: {planner_info.get('source_fields', [])}
    - Target fields: {planner_info.get('target_fields', [])}
    - Filtering fields: {planner_info.get('filtering_fields', [])} if 'filtering_fields' in planner_info else []

    SAMPLE DATA:
    Source data samples:
    {json.dumps(planner_info.get('source_data', {}).get('sample', {}), indent=2)}

    Target data sample:
    {json.dumps(planner_info.get('target_data', {}).get('sample', []), indent=2)}

    AGGREGATION IMPLEMENTATION REQUIREMENTS:
    1. Group data by the specified fields
    2. Apply the specified aggregation functions
    3. Handle empty dataframes properly
    4. Format results appropriately
    5. Update the target with aggregated results

    The function must follow this structure:
    def analyze_data(source_dfs, target_df):
        # Import required libraries
        import pandas as pd
        import numpy as np
        
        # Implement aggregation according to the plan
        # ...
        
        # Return the updated target dataframe with aggregation results
        return result_df
    """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21", 
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.25, top_p=0.9)
            )
            
            # Extract code from response
            import re
            code_match = re.search(r"```python\s*(.*?)\s*```", response.text, re.DOTALL)
            if code_match:
                return code_match.group(1)
            
            code_match = re.search(r"```\s*(.*?)\s*```", response.text, re.DOTALL)
            if code_match:
                return code_match.group(1)
                
            function_match = re.search(r"def analyze_data\s*\(.*?\).*?return\s+\w+", response.text, re.DOTALL)
            if function_match:
                return function_match.group(0)
                
            return response.text
        except Exception as e:
            logger.error(f"Error generating aggregation code: {e}")
            # Return a basic aggregation implementation
            return """def analyze_data(source_dfs, target_df):
        import pandas as pd
        import numpy as np
        
        # Get source table
        if not source_dfs:
            return target_df
            
        source_table = list(source_dfs.keys())[0]
        source_df = source_dfs[source_table]
        
        # Create result dataframe
        result = target_df.copy() if not target_df.empty else pd.DataFrame()
        
        # Find numeric columns for aggregation
        numeric_cols = source_df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            return result
            
        # Find potential group by columns (non-numeric)
        group_cols = [col for col in source_df.columns if col not in numeric_cols][:1]
        if not group_cols:
            return result
            
        # Simple aggregation
        agg_df = source_df.groupby(group_cols)[numeric_cols].agg(['sum', 'mean']).reset_index()
        
        # Flatten multi-level columns
        agg_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in agg_df.columns.values]
        
        return agg_df
    """
