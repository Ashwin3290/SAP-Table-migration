"""
DMTool with Relational Session Model

This module extends the original DMTool to support the relational model for multi-segment processing.
"""

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

# Import from the original planner
from planner import process_query as planner_process_query
from planner import SQLInjectionError, SessionError, APIError, DataProcessingError
from planner import validate_sql_identifier

# Import from relational model
from relational_functions import (
    get_or_create_segment_target_df, 
    save_segment_target_df,
    handle_segment_switch,
    detect_relationship_from_results,
    get_segment_relationships,
    generate_session_diagram
)
from relational_session import RelationalSessionManager

# Import from the original code_exec
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

class RelationalDMTool:
    """Extended DMTool with relational model support for multi-segment processing"""

    def __init__(self):
        """Initialize the RelationalDMTool instance"""
        try:
            # Configure Gemini
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not found in environment variables")
                raise APIError("Gemini API key not configured")

            self.client = genai.Client(api_key=api_key)

            # Current session context
            self.current_context = None

            logger.info("RelationalDMTool initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RelationalDMTool: {e}")
            raise

    def _extract_planner_info(self, resolved_data):
        """Extract and organize information from planner's resolved data"""
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
                "segment_id": resolved_data.get("segment_id", None),
                "key_columns": resolved_data.get("key_mapping", []),
            }

            # Process source data samples
            source_data_samples = resolved_data.get("source_data_samples", {})
            if isinstance(source_data_samples, dict):
                for k, v in source_data_samples.items():
                    if isinstance(v, pd.DataFrame) and not v.empty:
                        try:
                            planner_info["source_data"]["sample"][k] = v.head(3).to_dict("records")
                            planner_info["source_data"]["describe"][k] = v.describe().to_dict()
                        except Exception as e:
                            logger.warning(f"Error processing source data sample for {k}: {e}")
                            planner_info["source_data"]["sample"][k] = []
                            planner_info["source_data"]["describe"][k] = {}

            # Process target data samples
            target_data_samples = resolved_data.get("target_data_samples")
            if isinstance(target_data_samples, pd.DataFrame) and not target_data_samples.empty:
                try:
                    planner_info["target_data"]["sample"] = target_data_samples.head(3).to_dict("records")
                    planner_info["target_data"]["describe"] = target_data_samples.describe().to_dict()
                except Exception as e:
                    logger.warning(f"Error processing target data sample: {e}")

            # Store context for use in all prompting phases
            self.current_context = planner_info
            return planner_info
        except Exception as e:
            logger.error(f"Error in _extract_planner_info: {e}")
            # Return a minimal valid context to prevent further errors
            minimal_context = {
                "source_table": resolved_data.get("source_table_name", []) if resolved_data else [],
                "target_table": resolved_data.get("target_table_name", []) if resolved_data else [],
                "source_fields": [],
                "target_fields": [],
                "filtering_fields": [],
                "insertion_fields": [],
                "source_data": {"sample": {}, "describe": {}},
                "target_data": {"sample": [], "describe": {}},
                "original_query": "",
                "restructured_query": "",
                "key_columns": [],
                "session_id": resolved_data.get("session_id", "") if resolved_data else "",
                "segment_id": resolved_data.get("segment_id", None) if resolved_data else None,
            }
            return minimal_context

    @track_token_usage()
    def _classify_query(self, query, planner_info):
        """Classify the query type to determine code generation approach"""
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
- JOIN_TABLES: Joining or relating data from multiple tables
- CROSS_SEGMENT: Operation spanning multiple segments or tables

QUERY INFORMATION:
Original query: {query}
Restructured query: {planner_info.get('restructured_query', '')}
Source fields: {planner_info.get('source_fields', [])}
Target fields: {planner_info.get('target_fields', [])}
Filter fields: {planner_info.get('filtering_fields', [])}
Insertion fields: {planner_info.get('insertion_fields', [])}

Return ONLY the classification name with no explanation.
"""
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21", contents=prompt
                )

                # Validate response
                if not response or not hasattr(response, "text") or not response.text:
                    logger.warning("Invalid response from Gemini API in _classify_query")
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
                    "JOIN_TABLES",
                    "CROSS_SEGMENT",
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
        """Generate a simple, step-by-step plan in natural language"""
        try:
            # Validate input
            if not planner_info:
                logger.error("No planner info provided to _generate_simple_plan")
                return "1. Import utility functions from transform_utils\n2. Get source dataframe\n3. Return the dataframe unchanged"

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
8. End with validation of the result before returning

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
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21", contents=base_prompt,
                    config=types.GenerateContentConfig(temperature=0.2)
                )

                # Validate response
                if not response or not hasattr(response, "text") or not response.text:
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
        """Generate code based on a simple, step-by-step plan"""
        try:
            # Validate inputs
            if not simple_plan:
                logger.error("No simple plan provided to _generate_code_from_simple_plan")
                raise CodeGenerationError("No simple plan provided")

            if not planner_info:
                logger.error("No planner info provided to _generate_code_from_simple_plan")
                raise CodeGenerationError("No planner info provided")

            # Create template with the simple plan as a guide
            prompt = f"""
Write Python code that follows these EXACT steps:

{simple_plan}

DETAILED INFORMATION:
- Source tables: {json.dumps(planner_info.get('source_table', []))}
- Target table: {planner_info.get('target_table', [])}
- Source field(s): {planner_info.get('source_fields', [])}
- Target field(s): {planner_info.get('target_fields', [])}
- Filtering field(s): {planner_info.get('filtering_fields', [])}

KEY MAPPING (CRITICALLY IMPORTANT):
- Key columns for mapping between source and target: {planner_info.get('key_columns', [])}
- You MUST implement these key mappings in your code
- Every operation that accesses the target table MUST use these key columns

SAMPLE DATA:
{json.dumps(planner_info.get('source_data', {}).get('sample', {}), indent=2)[:1000]}

Target data sample:
{json.dumps(planner_info.get('target_data', {}).get('sample', []), indent=2)[:1000]}

Make the function like this:
def analyze_data(source_dfs, target_df):
    # Import required utilities
    from transform_utils import filter_dataframe, map_fields, conditional_mapping, join_tables
    
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
                    model="gemini-2.0-flash-thinking-exp-01-21", contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.25, top_p=0.9)
                )

                # Validate response
                if not response or not hasattr(response, "text") or not response.text:
                    logger.error("Invalid response from Gemini API in _generate_code_from_simple_plan")
                    raise CodeGenerationError("Failed to get valid response from Gemini API")
            except Exception as e:
                logger.error(f"Error calling Gemini API in _generate_code_from_simple_plan: {e}")
                raise CodeGenerationError(f"Failed to call Gemini API: {e}")

            # Extract the code
            import re

            try:
                # First try to extract code between triple backticks
                code_match = re.search(r"```python\s*(.*?)\s*```", response.text, re.DOTALL)
                if code_match:
                    return code_match.group(1)
                else:
                    # Next try to extract code without language
                    code_match = re.search(r"```\s*(.*?)\s*```", response.text, re.DOTALL)
                    if code_match:
                        return code_match.group(1)
                    else:
                        # Try to extract the function definition directly
                        function_match = re.search(r"def analyze_data\s*\(.*?\).*?return\s+target_df", response.text, re.DOTALL)
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
    # Import required utilities
    from transform_utils import filter_dataframe, map_fields
    
    # Get list of source tables for easier reference
    source_tables = list(source_dfs.keys())
    
    # Return unmodified target dataframe
    return target_df"""

    @track_token_usage()
    def _fix_code(self, code_content, error_info, planner_info, attempt=1, max_attempts=3):
        """Attempt to fix code based on error traceback"""
        if attempt > max_attempts:
            logger.error(f"Failed to fix code after {max_attempts} attempts")
            return None

        try:
            # Extract error information
            error_type = error_info.get("error_type", "Unknown error")
            error_message = error_info.get("error_message", "No error message")
            traceback_text = error_info.get("traceback", "No traceback available")

            # Create prompt for fixing code
            prompt = f"""
You are an expert code debugging and fixing agent. Fix this code that failed during execution.

THE CODE THAT FAILED:
```python
{code_content}
```

ERROR INFORMATION:
Error Type: {error_type}
Error Message: {error_message}
TRACEBACK:
{traceback_text}

CONTEXT INFORMATION:
Source tables: {planner_info.get('source_table', [])}
Target table: {planner_info.get('target_table', [])}
Source fields: {planner_info.get('source_fields', [])}
Target fields: {planner_info.get('target_fields', [])}
Filtering fields: {planner_info.get('filtering_fields', [])}

RETURN ONLY THE FIXED CODE WITH NO EXPLANATIONS:
```python
"""
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-thinking-exp-01-21", contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2)
                )

                # Validate response
                if not response or not hasattr(response, "text") or not response.text:
                    logger.error("Invalid response from Gemini API in _fix_code")
                    return None

                logger.info(f"Generated fixed code on attempt {attempt}")

                # Extract the code
                fixed_code = response.text.replace("```python", "").replace("```", "").strip()
                return fixed_code
            except Exception as e:
                logger.error(f"Error calling Gemini API in _fix_code: {e}")
                return None
        except Exception as e:
            logger.error(f"Error in _fix_code: {e}")
            return None

    def post_process_result(self, result):
        """Post-process the result DataFrame to remove any columns added due to reindexing"""
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
                logger.info(f"Removed {len(unnamed_cols)} unnamed columns: {unnamed_cols}")

            return cleaned_df
        except Exception as e:
            logger.error(f"Error in post_process_result: {e}")
            # Return original result if there's an error
            return result

    def process_sequential_query(
        self,
        query,
        object_id=29,
        segment_id=336,
        project_id=24,
        session_id=None,
    ):
        """
        Process a query as part of a relational multi-segment sequence
        
        Parameters:
            query (str): The user's query
            object_id (int): Object ID for mapping
            segment_id (int): Segment ID for mapping (can change between queries)
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

            # Validate IDs
            if not all(isinstance(x, int) for x in [object_id, segment_id, project_id]):
                logger.error(f"Invalid ID types: object_id={type(object_id)}, segment_id={type(segment_id)}, project_id={type(project_id)}")
                return None, "Invalid ID types - must be integers", session_id

            # Connect to database
            try:
                conn = sqlite3.connect("db.sqlite3")
            except sqlite3.Error as e:
                logger.error(f"Failed to connect to database: {e}")
                return None, f"Database connection error: {e}", session_id

            # 1. Handle segment switching if this is an existing session
            previous_segment_id = None
            cursor = conn.cursor()
            cursor.execute("Select table_name from connection_segments where segment_id = ?", (segment_id,))
            target_table = cursor.fetchone()
            if target_table:
                target_table = target_table[0]
            if session_id:
                previous_segment_id, target_df = handle_segment_switch(
                    session_id, 
                    segment_id, 
                    object_id, 
                    project_id, 
                    target_table,
                    conn
                )
                
                logger.info(f"Segment switch: {previous_segment_id} -> {segment_id}")
            
            # 2. Process query with the planner
            logger.info(f"Processing query: {query}")
            # Add segment_id to planner parameters
            resolved_data = planner_process_query(
                object_id, segment_id, project_id, query, session_id
            )
            
            if not resolved_data:
                logger.error("Failed to resolve query with planner")
                if conn:
                    conn.close()
                return None, "Failed to resolve query", session_id
            
            # Add segment_id to resolved data for future reference
            resolved_data["segment_id"] = segment_id
            
            # 3. Check for key mapping errors
            if isinstance(resolved_data.get("key_mapping"), list) and len(resolved_data["key_mapping"]) > 0:
                if isinstance(resolved_data["key_mapping"][0], str):
                    if conn:
                        conn.close()
                    return None, resolved_data["key_mapping"][0], session_id
            # 4. Extract and organize information from planner's resolved data
            try:
                resolved_data["original_query"] = query  # Add original query for context
                
                # Get target data samples for the current segment
                try:
                    target_table = None
                    if "target_table_name" in resolved_data:
                        target_table_list = resolved_data["target_table_name"]
                        target_table = target_table_list[0] if isinstance(target_table_list, list) and len(target_table_list) > 0 else target_table_list
                    
                    if target_table:
                        # Get target dataframe for the current segment
                        resolved_data["target_data_samples"] = get_or_create_segment_target_df(
                            session_id, segment_id, target_table, conn
                        )
                except Exception as e:
                    logger.warning(f"Error getting target data samples: {e}")
                    resolved_data["target_data_samples"] = pd.DataFrame()
                
                # Extract planner information
                planner_info = self._extract_planner_info(resolved_data)
                
                # Update the segment ID in planner info
                planner_info["segment_id"] = segment_id
                
                # Get session ID from the results
                session_id = resolved_data.get("session_id")
                if not session_id:
                    logger.warning("No session ID returned from planner")
                    import uuid
                    session_id = str(uuid.uuid4())  # Fallback
                
                # 5. Extract table names safely
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
                
                # 6. Get source dataframes
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
                
                # 7. Get target dataframe for the current segment
                target_df = get_or_create_segment_target_df(
                    session_id, segment_id, target_table, conn
                )
                
                # 8. Generate a simple, step-by-step plan in natural language
                try:
                    simple_plan = self._generate_simple_plan(planner_info)
                    logger.info(f"Simple plan generated: {simple_plan}")
                except Exception as e:
                    logger.error(f"Error generating simple plan: {e}")
                    simple_plan = "1. Make copy of the source dataframe\n2. Return the dataframe unchanged"
                
                # 9. Generate code from the simple plan
                try:
                    code_content = self._generate_code_from_simple_plan(simple_plan, planner_info)
                    logger.info(f"Code generated with length: {len(code_content)} chars")
                except Exception as e:
                    logger.error(f"Code generation error: {e}")
                    if conn:
                        conn.close()
                    return None, f"Failed to generate code: {e}", session_id
                
                # 10. Execute the generated code
                try:
                    code_file = create_code_file(code_content, query, is_double=True)
                    result = execute_code(code_file, source_dfs, target_df, resolved_data['target_sap_fields'])
                    
                    # Check if result is an error dictionary
                    if isinstance(result, dict) and "error_type" in result:
                        logger.error(f"Code execution error: {result['error_message']}")
                        
                        # Try to fix the code
                        fixed_code = code_content
                        for attempt in range(1, 4):
                            logger.info(f"Attempting to fix code (attempt {attempt}/3)")
                            
                            fixed_code = self._fix_code(
                                fixed_code,
                                result,
                                planner_info,
                                attempt=attempt,
                                max_attempts=3
                            )
                            if fixed_code is None:
                                if conn:
                                    conn.close()
                                return code_content, f"Failed to fix code after 3 attempts. Last error: {result['error_message']}", session_id
                            
                            # Try executing the fixed code
                            fixed_code_file = create_code_file(
                                fixed_code,
                                f"{query} (fixed attempt {attempt})",
                                is_double=True
                            )
                            fixed_result = execute_code(
                                fixed_code_file, source_dfs, target_df, resolved_data['target_sap_fields']
                            )
                            
# If the fix worked, use it
                            if not isinstance(fixed_result, dict) or "error_type" not in fixed_result:
                                logger.info(f"Successfully fixed code on attempt {attempt}")
                                code_content = fixed_code
                                result = fixed_result
                                break
                            else:
                                # Update error for next attempt
                                result = fixed_result
                        
                        # If we've gone through all attempts and still have an error
                        if isinstance(result, dict) and "error_type" in result:
                            if conn:
                                conn.close()
                            return code_content, f"Failed to fix code after 3 attempts. Last error: {result['error_message']}", session_id
                    
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
                
                # 11. Save the result to the current segment's table
                if isinstance(result, pd.DataFrame):
                    try:
                        result = self.post_process_result(result)
                        save_success = save_segment_target_df(session_id, segment_id, target_table, result)
                        if not save_success:
                            logger.warning("Failed to save target dataframe")
                    except Exception as e:
                        logger.error(f"Error saving target dataframe: {e}")
                
                # 12. If we switched segments, try to detect and create relationships
                if previous_segment_id is not None and previous_segment_id != segment_id:
                    try:
                        relationship = detect_relationship_from_results(
                            session_id,
                            segment_id,
                            previous_segment_id,
                            resolved_data,
                            result if isinstance(result, pd.DataFrame) else None
                        )
                        
                        if relationship:
                            logger.info(f"Detected relationship between segments {previous_segment_id} and {segment_id}")
                    except Exception as e:
                        logger.warning(f"Error detecting relationship: {e}")
                
                # 13. Return the results
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
    
    def get_session_diagram(self, session_id):
        """
        Generate an ER diagram for the session
        
        Parameters:
            session_id (str): Session ID
            
        Returns:
            str: Mermaid diagram code
        """
        try:
            if not session_id:
                return "No session ID provided"
            
            # Generate diagram
            diagram = generate_session_diagram(session_id)
            return diagram
        except Exception as e:
            logger.error(f"Error generating session diagram: {e}")
            return f"Error generating diagram: {e}"
    
    def get_session_relationships(self, session_id):
        """
        Get all relationships for a session
        
        Parameters:
            session_id (str): Session ID
            
        Returns:
            list: List of relationship dictionaries
        """
        try:
            if not session_id:
                return []
            
            # Initialize relational session manager
            rsm = RelationalSessionManager()
            
            # Get relationships
            return rsm.get_relationships(session_id)
        except Exception as e:
            logger.error(f"Error getting session relationships: {e}")
            return []
    
    def validate_session_integrity(self, session_id):
        """
        Validate the integrity of all tables in a session
        
        Parameters:
            session_id (str): Session ID
            
        Returns:
            dict: Validation results for each table
        """
        try:
            if not session_id:
                return {"valid": False, "reason": "No session ID provided"}
            
            # Initialize relational session manager
            rsm = RelationalSessionManager()
            
            # Get schema
            schema = rsm.get_schema(session_id)
            if not schema:
                return {"valid": False, "reason": "No schema found for session"}
            
            # Validate each table
            results = {}
            for table_name in schema["tables"]:
                results[table_name] = rsm.validate_table_integrity(session_id, table_name)
            
            # Overall validation result
            valid = all(result.get("valid", False) for result in results.values())
            
            return {
                "valid": valid,
                "tables": results
            }
        except Exception as e:
            logger.error(f"Error validating session integrity: {e}")
            return {"valid": False, "reason": str(e)}