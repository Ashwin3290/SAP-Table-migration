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
                    model="gemini-2.0-flash-thinking-exp-01-21", contents=prompt
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
                logger.warning(f"Error configuring Gemini: {e}. Falling back to Ollama.")
    
    def _format_single_table_prompt(self, question, table):
        """Format a single table prompt"""
        # Implementation remains the same
        pass
    
    def _format_context_aware_prompt(self, resolved_data):
        """Format a code generation prompt that includes context"""
        
        # Extract context information
        context = resolved_data.get("context", {})
        history = context.get("transformation_history", [])
        table_state = context.get("target_table_state", {})
        
        # Build context section for the prompt
        context_section = """
Previous Transformations:
"""
        if not history:
            context_section += "None (this is the first transformation)"
        else:
            for i, tx in enumerate(history):
                context_section += f"""
{i+1}. {tx.get('description', 'Unknown transformation')}
   - Fields modified: {', '.join(tx.get('fields_modified', []))}
   - Filter conditions: {json.dumps(tx.get('filter_conditions', {}))}
"""
        
        context_section += f"""
Current Target Table State:
- Populated fields: {', '.join(table_state.get('populated_fields', []))}
- Remaining mandatory fields: {', '.join(table_state.get('remaining_mandatory_fields', []))}
"""
        
        # Handle insertion fields - ensure it's properly formatted
        insertion_fields_str = ""
        if "insertion_fields" in resolved_data and resolved_data["insertion_fields"]:
            if isinstance(resolved_data["insertion_fields"], list) and len(resolved_data["insertion_fields"]) > 0:
                insertion_fields_str = resolved_data["insertion_fields"][0]["source_field"]
                target_field_str = resolved_data["insertion_fields"][0]["target_field"]
            else:
                logger.warning(f"Unexpected insertion_fields format: {resolved_data['insertion_fields']}")
                insertion_fields_str = str(resolved_data["insertion_fields"])
                target_field_str = "Unknown"
        else:
            insertion_fields_str = "None"
            target_field_str = "None"
            
        # Combine with the regular prompt
        prompt = f"""
I need ONLY Python code - DO NOT include any explanations, markdown, or comments outside the code.

Source table info and description:
{resolved_data['source_info']}
{resolved_data['source_describe']}


{"There can also be more than one source table. In that case these are the Additional source table info and description:" if resolved_data["additional_source_table"] else ""}
{resolved_data["additional_source_tables"] if resolved_data["additional_source_table"] else ""}

Target table info and description:
{resolved_data['target_info']}
{resolved_data['target_describe']}

Columns that will be used for filtering:
{resolved_data['filtering_fields']}

Source column from where data has to be picked:
{insertion_fields_str}

Target column where data will be inserted:
{target_field_str}
{context_section}

Question: {resolved_data['restructured_question']}

I want your code in this exact function template:
df1 will be the source table and df2 will be the target table.
The function must update df2 (target table) WITHOUT replacing any previously populated data.

{"If we have additional tables then you will find them in additional_tables dictionary with the table names as the keys. df1 has the " + resolved_data['source_table_name'] if resolved_data["additional_source_table"] else ""}


def analyze_data(df1, df2, additional_tables=None):
    # Your Code comes here
    return result

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
    13. Do not make any mock transform_utils functions by youself based upon the prompt, just use them as they are already implemented and ready to be imported.
    14. Do not make any mock utils function, import them from the transform_utils module, Assume it already exists and is ready to be imported.

    Note:
    - save transformation as intermediates in the code and at the end merge and return the final target dataframe.

    Make the function like this :

    from transform_utils import *

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
        from transform_utils import *
        
        # Get list of source tables
        source_tables = list(source_dfs.keys())
        
        # Return unmodified target dataframe
        return target_df"""

    @track_token_usage()
    def _generate_with_gemini(self, prompt):
        """Generate response using Gemini API"""
        try:
            response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents = prompt,
        )
            # Log token usage statistics after call
            logger.info(f"Current token usage: {get_token_usage_stats()}")
            return response.text
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
            logger.error(f"Error connecting to Ollama: {e}")
            return f"Error connecting to LLM service: {str(e)}"
    
    def generate(self, prompt):
        """Generate a response using the available LLM service"""
        if self.use_gemini:
            response = self._generate_with_gemini(prompt)
            if response:
                return response
            
        # Fall back to Ollama if Gemini fails or isn't configured
        return self._generate_with_ollama(prompt)
    
    def process_sequential_query(self, query, object_id=29, segment_id=336, project_id=24, session_id=None):
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
        # Process query with context awareness
        resolved_data = planner_process_query(object_id, segment_id, project_id, query, session_id)
        if not resolved_data:
            return None, "Failed to resolve query", session_id
        
        # Get session ID from the results
        session_id = resolved_data.get("session_id")
        
        # Connect to database
        conn = sqlite3.connect('db.sqlite3')
        print(resolved_data)
        # Extract table names and field names
        source_table = resolved_data['source_table_name']
        target_table = resolved_data['target_table_name']
        source_fields = resolved_data['source_field_names']
        target_fields = resolved_data['target_sap_fields']
        additional_tables = resolved_data['additional_source_table']
        
        
        # Get source dataframe
        source_df = pd.read_sql_query(f"SELECT * FROM {source_table}", conn)
        
        # Get target dataframe (either existing or new)
        target_df = get_or_create_session_target_df(session_id, target_table , conn)
        
        if additional_tables:
            additional_source_tables = {}
            for table in additional_tables:
                additional_source_tables[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        
        else:
            additional_source_tables = None
        

        # Generate code with context awareness
        code_prompt = self._format_context_aware_prompt(resolved_data)
        code = self.generate(code_prompt)
        
        # Clean the code
        if "# CODE STARTS HERE" in code and "# CODE ENDS HERE" in code:
            start_marker = "# CODE STARTS HERE"
            end_marker = "# CODE ENDS HERE"
            start_idx = code.find(start_marker) + len(start_marker)
            end_idx = code.find(end_marker)
            if start_idx != -1 and end_idx != -1:
                code_content = code[start_idx:end_idx].strip()
            else:
                code_content = code
        else:
            code_content = code

        if "```python" in code_content:
            code_content = code_content.replace("```python", "").replace("```", "")
        if "```" in code_content:
            code_content = code_content.replace("```", "")
        
        # Create code file
        code_file = create_code_file(code_content, query, is_double=True)
        
        # Execute code
        result = execute_code(code_file, (source_df, target_df),additional_tables=additional_source_tables, is_double=True)
        
        # Save the updated target dataframe
        if isinstance(result, pd.DataFrame):
            save_session_target_df(session_id, result)
        
        conn.close()
        return code_content, result, session_id
    
    def process_query(self, question, tables, dataframes, mode='Code', session_id=None):
        """Process a query and execute code or generate QA response"""
        
        is_double = isinstance(tables, tuple) and len(tables) == 2
        
        # Generate the appropriate prompt
        if mode == 'QA':
            description = ''
            if isinstance(tables, dict) and 'description' in tables:
                description = tables['description']
            prompt = self._format_qa_prompt(question, tables, description)
            response = self.generate(prompt)
            return response, None  # No code to execute for QA
        
        elif mode == 'Code':
            if is_double:
                # Use the context-aware sequential processing
                return self.process_sequential_query(question, session_id=session_id)
            else:
                # Single table case remains the same
                prompt = self._format_single_table_prompt(question, tables)
                
                logger.info(f"Generating code for query: {question}")
                
                # Generate code content and extract only the body
                raw_code = self.generate(prompt)
                
                # Clean the code - extract only what would go inside the function
                if "# CODE STARTS HERE" in raw_code and "# CODE ENDS HERE" in raw_code:
                    start_marker = "# CODE STARTS HERE"
                    end_marker = "# CODE ENDS HERE"
                    start_idx = raw_code.find(start_marker) + len(start_marker)
                    end_idx = raw_code.find(end_marker)
                    if start_idx != -1 and end_idx != -1:
                        code_content = raw_code[start_idx:end_idx].strip()
                    else:
                        code_content = raw_code
                else:
                    code_content = raw_code

                
                if "```python" in code_content:
                    code_content = code_content.replace("```python", "").replace("```", "")
                if "```" in code_content:
                    code_content = code_content.replace("```", "")
                
                # Create a code file with the generated content
                code_file = create_code_file(code_content, question, is_double=is_double)
                
                # Execute the code
                result = execute_code(code_file, dataframes, is_double=is_double)
                
                return code_content, result
        
        else:
            return f"Unknown mode: {mode}", None
    
    def get_session_info(self, session_id):
        """
        Get information about a session
        
        Parameters:
        session_id (str): The session ID
        
        Returns:
        dict: Session information
        """
        context = get_session_context(session_id)
        return {
            "session_id": session_id,
            "context": context,
            "transformation_history": context.get("context", {}).get("transformation_history", []) if context else [],
            "target_table_state": context.get("context", {}).get("target_table_state", {}) if context else {}
        }
            
    def save_interaction(self, question, code, result, file_details, db_client=None):
        """Save the interaction to database if a client is provided"""
        if not db_client:
            return None
            
        session_id = str(uuid.uuid4())
        try:
            db_client.chat.insert_one({
                'session_id': session_id,
                'question': question,
                'code': code,
                'result': str(result)[:1000],  # Limit result size
                'file_details': file_details,
                'vote': 0
            })
            return session_id
        except Exception as e:
            logger.error(f"Outer error in process_sequential_query: {e}")
            logger.error(traceback.format_exc())
            if conn:
                try:
                    conn.close()
                except:
                    pass
            return None, f"An error occurred: {e}", session_id
