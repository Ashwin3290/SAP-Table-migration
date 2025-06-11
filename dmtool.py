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
from typing import Dict, Any

from planner import process_query
from planner import (
    validate_sql_identifier,
    APIError)
from planner import ContextualSessionManager
# Import the new SQLite modules
from generator import SQLGenerator
from executor import SQLExecutor
from logging_config import setup_logging

setup_logging(log_to_file=True, log_to_console=True)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CodeGenerationError(Exception):
    """Exception raised for code generation errors."""
    pass

class ExecutionError(Exception):
    """Exception raised for code execution errors."""
    pass

import os
import json
import logging
import spacy
import re
from typing import Dict, List, Any, Optional


class QueryTemplateRepository:
    """Repository of query templates for common transformation patterns"""
    
    def __init__(self, template_file="query_templates.json"):
        """
        Initialize the template repository
        
        Parameters:
        template_file (str): Path to the JSON file containing templates
        """
        self.template_file = template_file
        self.templates = self._load_templates()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise APIError("Gemini API key not configured")
        self.client = genai.Client(api_key=api_key)
        
    def _load_templates(self):
        """
        Load templates from the JSON file
        
        Returns:
        list: List of template dictionaries
        """
        try:
            if os.path.exists(self.template_file):
                with open(self.template_file, 'r') as f:
                    templates = json.load(f)
                return templates
            else:
                logger.warning(f"Template file {self.template_file} not found.")
                return []
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            return []
    
    def find_matching_template(self, query):
        """
        Find the best matching template for a given query using LLM
        
        Parameters:
        query (str): The natural language query
        
        Returns:
        dict: The best matching template or None if no good match
        """
        try:
            # Check if we have templates and LLM client
            if not self.templates:
                logger.warning("No templates available")
                return None
            
            # Extract template prompts for LLM matching
            template_options = []
            for i, template in enumerate(self.templates):
                template_options.append(f"{i+1}. ID: {template['id']}\n   Pattern: {template['prompt']}")
            
            # Create LLM prompt for template matching
            llm_prompt = f"""You are an expert at matching user queries to data transformation templates.

    USER QUERY: "{query}"

    AVAILABLE TEMPLATES:
    {chr(10).join(template_options)}

    INSTRUCTIONS:
    Analyze the user query and determine which template pattern best matches the intent and structure.

    Consider:
    - Query operations (bring, add, delete, update, check, join, etc.)
    - Data sources (tables, fields, segments)
    - Conditional logic (IF/ELSE, CASE statements)
    - Filtering conditions (WHERE clauses)
    - Transformations (date formatting, string operations, etc.)

    Respond with ONLY the template ID (nothing else).

    Examples:
    - "Bring Material Number from MARA where Material Type = ROH" → simple_filter_transformation
    - "If Plant is 1000 then 'Domestic' else 'International'" → conditional_value_assignment  
    - "Add new column for current date" → get_current_date
    - "Join data from Basic segment with Sales segment" → join_segment_data

    Template ID:"""

            try:
                # Call LLM (assuming Gemini client is available as self.client)
                from google.genai import types
                
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash-preview-04-17",
                    contents=llm_prompt,
                    config=types.GenerateContentConfig(temperature=0.1)
                )
                
                if response and hasattr(response, "text"):
                    # Extract template ID from response
                    template_id = response.text.strip().strip('"').strip("'").lower()
                    
                    # Find the matching template
                    best_match = None
                    for template in self.templates:
                        if template['id'].lower() == template_id:
                            best_match = template
                            break
                    
                    if best_match:
                        return best_match
                    else:
                        logger.warning(f"Template ID '{template_id}' not found in available templates")
                        # Try partial matching as fallback
                        for template in self.templates:
                            if template_id in template['id'].lower() or template['id'].lower() in template_id:
                                return template
                        
                        return {}
                else:
                    logger.warning("Invalid response from LLM")
                    return {}
                    
            except Exception as llm_error:
                logger.error(f"Error calling LLM for template matching: {llm_error}")
                return {}
                
        except Exception as e:
            logger.error(f"Error finding matching template: {e}")
            return None    

class DMTool:
    """SQLite-based DMTool for optimized data transformations using direct SQLite queries"""

    def __init__(self, DB_PATH=os.environ.get('DB_PATH')):
        """Initialize the DMToolSQL instance"""
        try:
            # Configure Gemini
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not found in environment variables")
                raise APIError("Gemini API key not configured")

            self.client = genai.Client(api_key=api_key)
 
            # Initialize SQLite components
            self.sql_generator = SQLGenerator()
            self.sql_executor = SQLExecutor()
            self.query_template_repo = QueryTemplateRepository()

            # Current session context
            self.current_context = None

            logger.info("DMToolSQL initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing DMToolSQL: {e}")
            raise
    
    def __del__(self):
        """Cleanup resources"""
        try:
            # Nothing specific to close with SQLite executor as it creates/closes
            # connections per request, not maintaining persistent connections
            logger.info(f"DMToolSQL cleanup complete")
        except Exception as e:
            logger.error(f"Error during DMToolSQL cleanup: {e}")

    def _extract_planner_info(self, resolved_data):
        """
        Extract and organize all relevant information from planner's resolved data
        to make it easily accessible in SQLite generation
        """
        try:
            # Validate input
            if not resolved_data:
                logger.error("No resolved data provided to _extract_planner_info")
                raise ValueError("No resolved data provided")

            # Create a comprehensive context object
            planner_info = {
                # Table information
                "source_table_name": resolved_data.get("source_table_name", []),
                "target_table_name": resolved_data.get("target_table_name", []),
                # Field information
                "source_field_names": resolved_data.get("source_field_names", []),
                "target_sap_fields": resolved_data.get("target_sap_fields", []),
                "filtering_fields": resolved_data.get("filtering_fields", []),
                "insertion_fields": resolved_data.get("insertion_fields", []),
                # Query understanding
                "original_query": resolved_data.get("original_query", ""),
                "restructured_query": resolved_data.get("Resolved_query", ""),
                # Session information
                "session_id": resolved_data.get("session_id", ""),
                "key_mapping": resolved_data.get("key_mapping", []),
                # Query type
                "query_type": resolved_data.get("query_type", "SIMPLE_TRANSFORMATION"),
            }
            
            # Additional information for different query types
            if planner_info["query_type"] == "JOIN_OPERATION":
                planner_info["join_conditions"] = resolved_data.get("join_conditions", [])
            elif planner_info["query_type"] == "CROSS_SEGMENT":
                planner_info["segment_references"] = resolved_data.get("segment_references", [])
                planner_info["cross_segment_joins"] = resolved_data.get("cross_segment_joins", [])
            elif planner_info["query_type"] == "VALIDATION_OPERATION":
                planner_info["validation_rules"] = resolved_data.get("validation_rules", [])
            elif planner_info["query_type"] == "AGGREGATION_OPERATION":
                planner_info["aggregation_functions"] = resolved_data.get("aggregation_functions", [])
                planner_info["group_by_fields"] = resolved_data.get("group_by_fields", [])

            # Extract specific filtering conditions from the restructured query
            query_text = planner_info["restructured_query"]
            conditions = {}

            # Only attempt to extract conditions if we have a query
            if query_text:
                # Look for common filter patterns
                for field in planner_info["filtering_fields"]:
                    # Equality check
                    pattern = f"{field}\\s*=\\s*['\"](.*?)['\"]"
                    matches = re.findall(pattern, query_text)
                    if matches:
                        conditions[field] = matches[0]
                    
                    # Check for IN conditions
                    pattern = f"{field}\\s+in\\s+\\(([^)]+)\\)"
                    matches = re.findall(pattern, query_text, re.IGNORECASE)
                    if matches:
                        # Parse the values in the parentheses
                        values_str = matches[0]
                        values = [v.strip().strip("'\"") for v in values_str.split(",")]
                        conditions[field] = values

            # Add specific conditions
            planner_info["extracted_conditions"] = conditions
            
            # Add data samples for target table validation
            planner_info["target_data_samples"] = resolved_data.get("target_data_samples", {})

            # Store context for use in SQLite generation
            self.current_context = planner_info
            return planner_info
        except Exception as e:
            logger.error(f"Error in _extract_planner_info: {e}")
            # Return a minimal valid context to prevent further errors
            minimal_context = {
                "source_table_name": (
                    resolved_data.get("source_table_name", []) if resolved_data else []
                ),
                "target_table_name": (
                    resolved_data.get("target_table_name", []) if resolved_data else []
                ),
                "source_field_names": [],
                "target_sap_fields": [],
                "filtering_fields": [],
                "insertion_fields": [],
                "original_query": "",
                "restructured_query": "",
                "session_id": (
                    resolved_data.get("session_id", "") if resolved_data else ""
                ),
                "key_mapping": [],
                "extracted_conditions": {},
                "query_type": "SIMPLE_TRANSFORMATION",
            }
            return minimal_context

    def _create_operation_plan(self, query, planner_info: Dict[str, Any],template: Dict[str, Any]) -> str:
        """
        Use LLM to create a detailed plan for SQLite query generation
        
        Parameters:
        query (str): Original natural language query
        planner_info (Dict): Information extracted by the planner
        
        Returns:
        str: Step-by-step SQLite generation plan
        """
        try:
            # Extract key information for the prompt
            query_type = planner_info.get("query_type", "SIMPLE_TRANSFORMATION")
            source_tables = planner_info.get("source_table_name", [])
            source_fields = planner_info.get("source_field_names", [])
            target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
            target_fields = planner_info.get("target_sap_fields", [])
            filtering_fields = planner_info.get("filtering_fields", [])
            conditions = planner_info.get("extracted_conditions", {})
            key_mapping = planner_info.get("key_mapping", [])
            
            # Check if target table has data (for UPDATE vs INSERT decision)
            target_has_data = False
            target_data_samples = planner_info.get("target_data_samples", {})
            if isinstance(target_data_samples, pd.DataFrame) and not target_data_samples.isnull().all().all():
                target_has_data = True
            
            # Get information about segment references
            
            # Create a comprehensive prompt for the LLM
            prompt = f"""
    You are an expert SQLite database engineer focused on data transformation. I need you to create a step-by-step plan to generate 
    SQLite for the following natural language query:

    ORIGINAL QUERY: "{query}"

    CONTEXT INFORMATION:
    - Query Type: {query_type}
    - Source Tables: {source_tables}
    - Source Fields: {source_fields}
    - Target Table: {target_table}
    - Target Fields: {target_fields}
    - Filtering Fields: {filtering_fields}
    - Filtering Conditions: {json.dumps(conditions, indent=2)}
    - Key Mapping: {json.dumps(key_mapping, indent=2)}
    - Target Data Samples: {target_data_samples}

    Use this Template for the SQLite generation plan:
    {template["plan"]}

    Note:
    Tables with t_[number] like t_24 are target tables , these can act as both source and target tables.
    These can be have different names for same column data as the source table
    key mapping is used to match columns between source and target tables.
    for example, if source table has MATNR and target table has Product, then if key mapping mentions then use them for matching in where clauses.

    Requirements:
    - Generate a step-by-step SQLite generation plan
    - Give 10-20 steps to generate the SQLite query
    - Use the following format for each step:
        1. Step description
        2. SQLite operation (e.g., SELECT, INSERT, UPDATE)
        3. SQLite query template with placeholders for parameters
        4. Any additional notes or considerations
    - Make sure to include any filtering conditions in the SQLite query
    - If the query references segments or previous transformations, make sure to use the appropriate source tables
    - Pay special attention to the segment references - these indicate using data from previous segments

    Note:
    1. ALWAYS assume we need to INSERT or UPDATE the target table with data from source tables
    2. {"use an UPDATE operation (possibly with a subquery)" if target_has_data else " use an INSERT operation"}
    3. For a validation query, use a SELECT operation instead
    4. Always apply any filtering conditions to source data before inserting/updating
    5. Match source fields to target fields exactly in the correct order
    6. When filtering, use exact literal values (e.g., WHERE MTART = 'ROH')
    7. For updates, use key fields to match records between source and target
    8. If no explicit key mapping is provided, identify a likely key field (MATNR, ID, etc.)
    10. DO NOT give any code 
    11. DO NOT use Table alias.
    12. When the query refers to a previous segment or transformation, 
        we need to use the target table from that segment as a source table for this query

    Format your response as a numbered list only, with no explanations or additional text.
    Each step should be clear, concise, and directly actionable for SQLite generation.
    """
            
            # Call the LLM for planning
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17", 
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            
            # Extract and return the plan
            if response and hasattr(response, "text"):
                logger.info(f"response text {response.text.strip()}")
                return response.text.strip()
            else:
                logger.warning("Invalid response from LLM in create_sql_plan")
                return "1. Generate basic SQLite query based on query type\n2. Return query"
                
        except Exception as e:
            logger.error(f"Error in create_sql_plan: {e}")
            return "1. Generate basic SQLite query based on query type\n2. Return query"
        
    def _get_segment_name(self, segment_id,conn):
        cursor = conn.cursor()
        cursor.execute("SELECT segement_name FROM connection_segments WHERE segment_id = ?", (segment_id,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            logger.error(f"Segment ID {segment_id} not found in database")
            return None

    def process_sequential_query(self, query, object_id, segment_id, project_id, session_id=None):
        """
        Process a query as part of a sequential transformation using SQLite generation
        instead of Python code generation
        
        Parameters:
        query (str): The user's query
        object_id (int): Object ID for mapping
        segment_id (int): Segment ID for mapping
        project_id (int): Project ID for mapping
        session_id (str): Optional session ID, creates new session if None
        
        Returns:
        tuple: (generated_sql, result, session_id)
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
            
            context_manager = ContextualSessionManager()

            # Create or get session ID
            if not session_id:
                session_id = context_manager.create_session()
                logger.info(f"Created new session: {session_id}")
                
            # Add source tables from referenced segments
            additional_source_tables = []

            # 1. Process query with the planner - this stays the same
            logger.info(f"Processing query: {query}")
            resolved_data = process_query(
                object_id, segment_id, project_id, query, session_id=session_id
            )
            
            # FIX: Check if resolved_data is None before proceeding
            if not resolved_data:
                logger.error("Failed to resolve query with planner")
                return None, "Failed to resolve query", session_id
            
            # Find matching template
            template = self.query_template_repo.find_matching_template(query)
            if not template:
                logger.error("No matching template found for query")
                # Create a basic fallback template
                template = {
                    "id": "fallback",
                    "prompt": "Basic transformation",
                    "query": "SELECT {field} FROM {table} WHERE {filter_field} = '{filter_value}'",
                    "plan": ["1. Identify source and target", "2. Generate basic SQL query"]
                }
            # Get query type from resolved data
            query_type = resolved_data.get("query_type", "SIMPLE_TRANSFORMATION")
            # Get session ID from the results
            session_id = resolved_data.get("session_id")

            # Connect to database for sample data gathering
            try:
                conn = sqlite3.connect(os.environ.get('DB_PATH'))
            except sqlite3.Error as e:
                logger.error(f"Failed to connect to database: {e}")
                return None, f"Database connection error: {e}", session_id

            try:
                resolved_data["original_query"] = query
                
                # Get target data samples for validation
                try:
                    if "target_table_name" in resolved_data:
                        target_table = resolved_data["target_table_name"]
                        if isinstance(target_table, list) and len(target_table) > 0:
                            target_table = target_table[0]
                        target_table = validate_sql_identifier(target_table)
                        resolved_data["target_data_samples"] = self.sql_executor.get_table_sample(target_table)
                except Exception as e:
                    logger.warning(f"Error getting target data samples: {e}")
                    resolved_data["target_data_samples"] = pd.DataFrame()

                # Add additional source tables from segment references
                if additional_source_tables:
                    source_tables = resolved_data.get("source_table_name", [])
                    if isinstance(source_tables, list):
                        # Add only unique tables not already in the list
                        for table in additional_source_tables:
                            if table not in source_tables:
                                source_tables.append(table)
                        resolved_data["source_table_name"] = source_tables

                # Extract and organize planner information
                planner_info = self._extract_planner_info(resolved_data)
                
                # Create detailed operational plan using LLM
                sql_plan = self._create_operation_plan(query, planner_info, template)
                
                # 3. Generate SQLite query instead of Python code
                sql_query, sql_params = self.sql_generator.generate_sql(sql_plan, planner_info, template)
                
                result = self._execute_sql_query(sql_query, sql_params, planner_info)

                # 5. Process the results
                if isinstance(result, dict) and "error_type" in result:
                    logger.error(f"SQLite execution error: {result}")
                    return None, f"SQLite execution failed: {result.get('error_message', 'Unknown error')}", session_id

                if isinstance(result, dict) and result.get("multi_query_result"):
                    logger.info(f"Processing multi-query result: {result.get('completed_statements', 0)} statements completed")
                    multi_result = self._handle_multi_query_result(result, planner_info, session_id)
                    
                    if result.get("success") and len(multi_result) == 2:
                        try:
                            context_manager = ContextualSessionManager()
                            transformation_data = {
                                "original_query": query,
                                "generated_sql": sql_query,
                                "query_type": query_type,
                                "source_tables": planner_info.get("source_table_name", []),
                                "target_table": planner_info.get("target_table_name", []),
                                "fields_affected": planner_info.get("target_sap_fields", []),
                                "execution_result": {
                                    "success": True,
                                    "rows_affected": len(multi_result[0]) if isinstance(multi_result[0], pd.DataFrame) else 0,
                                    "is_multi_step": True,
                                    "steps_completed": result.get("completed_statements", 0)
                                },
                                "is_multi_step": True,
                                "steps_completed": result.get("completed_statements", 0)
                            }
                            context_manager.add_transformation_record(session_id, transformation_data)
                        except Exception as e:
                            logger.warning(f"Could not save transformation record for multi-query: {e}")
                    
                    return multi_result

                # 6. Register the target table for this segment
                if "target_table_name" in resolved_data:
                    target_table = resolved_data["target_table_name"]
                    if isinstance(target_table, list) and len(target_table) > 0:
                        target_table = target_table[0]
                segment_name = self._get_segment_name(segment_id, conn)
                if segment_name:
                    context_manager.add_segment(
                        session_id,
                        segment_name,
                        planner_info["target_table_name"],
                    )

                try:
                    context_manager = ContextualSessionManager()
                    
                    # Prepare transformation data
                    transformation_data = {
                        "original_query": query,
                        "generated_sql": sql_query,
                        "query_type": query_type,
                        "source_tables": planner_info.get("source_table_name", []),
                        "target_table": target_table,
                        "fields_affected": planner_info.get("target_sap_fields", []),
                        "execution_result": {
                            "success": True,
                            "rows_affected": len(target_data) if isinstance(target_data, pd.DataFrame) else 0,
                            "is_multi_step": isinstance(result, dict) and result.get("multi_query_result", False),
                            "steps_completed": result.get("completed_statements", 1) if isinstance(result, dict) else 1
                        },
                        "is_multi_step": isinstance(result, dict) and result.get("multi_query_result", False),
                        "steps_completed": result.get("completed_statements", 1) if isinstance(result, dict) else 1
                    }
                    
                    context_manager.add_transformation_record(session_id, transformation_data)
                    
                except Exception as e:
                    logger.warning(f"Could not save transformation record: {e}")

                if target_table and query_type in ["SIMPLE_TRANSFORMATION", "JOIN_OPERATION", "CROSS_SEGMENT", "AGGREGATION_OPERATION"]:
                    try:
                        # Get the full target table data to show actual results
                        select_query = f"SELECT * FROM {validate_sql_identifier(target_table)}"
                        target_data = self.sql_executor.execute_and_fetch_df(select_query)
                        
                        if isinstance(target_data, pd.DataFrame) and not target_data.empty:
                            # Return the actual target table data
                            rows_affected = len(target_data)
                            non_null_columns = target_data.dropna(axis=1, how='all').columns.tolist()
                            
                            # Add metadata to the DataFrame
                            target_data.attrs['transformation_summary'] = {
                                'rows': rows_affected,
                                'populated_fields': non_null_columns,
                                'target_table': target_table,
                                'query_type': query_type
                            }
                            
                            return  target_data, session_id
                        else:
                            # Target table exists but is empty
                            empty_df = pd.DataFrame()
                            empty_df.attrs['message'] = f"Target table '{target_table}' is empty after transformation"
                            return  empty_df, session_id
                            
                    except Exception as e:
                        # Fallback to execution result if can't fetch target data
                        return  result, session_id
                        
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

    def _is_multi_statement_query(self, sql_query):
        """Detect if SQL contains multiple statements"""
        if not sql_query or not isinstance(sql_query, str):
            return False        
        statements = self.sql_executor.split_sql_statements(sql_query)
        return len(statements) > 1

    def _execute_sql_query(self, sql_query, sql_params, planner_info):
        """
        Execute SQLite query using the SQLExecutor with multi-statement support
        
        Parameters:
        sql_query (str): The SQLite query to execute
        sql_params (dict): The parameters for the query
        planner_info (dict): Planner information for context
        
        Returns:
        Union[pd.DataFrame, dict]: Results or error information
        """
        
        # NEW: Check if this is a multi-statement query
        if self._is_multi_statement_query(sql_query):
            logger.info("Detected multi-statement query, using multi-query executor")
            return self.sql_executor.execute_multi_statement_query(
                sql_query, sql_params, planner_info.get("session_id")
            )
        
        # Existing single statement logic
        query_type = planner_info.get("query_type", "SIMPLE_TRANSFORMATION")
        
        if query_type == "SIMPLE_TRANSFORMATION":
            # Determine if this is an INSERT or UPDATE operation
            operation_type = sql_query.strip().upper().split()[0]
            
            if operation_type == "INSERT":
                # For inserts, we want to execute and not fetch results
                return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False)
            elif operation_type == "UPDATE":
                # For updates, execute and don't fetch results
                return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False)
            elif operation_type == "WITH":
                # This is likely a WITH clause followed by INSERT or UPDATE
                if "INSERT INTO" in sql_query.upper():
                    return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False)
                elif "UPDATE" in sql_query.upper():
                    return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False)
                else:
                    # Default to fetching results as DataFrame
                    return self.sql_executor.execute_and_fetch_df(sql_query, sql_params)
            else:
                # Default to fetching results as DataFrame for SELECT statements
                return self.sql_executor.execute_and_fetch_df(sql_query, sql_params)
        elif query_type in ["JOIN_OPERATION", "CROSS_SEGMENT"]:
            # For join operations, we want the result as a DataFrame
            if "INSERT INTO" in sql_query.upper() or "UPDATE" in sql_query.upper():
                # If this is a direct update/insert, execute without fetching
                return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False)
            else:
                # Otherwise, fetch as DataFrame
                return self.sql_executor.execute_and_fetch_df(sql_query, sql_params)
        elif query_type == "VALIDATION_OPERATION":
            # For validation operations, we want the results as a DataFrame
            return self.sql_executor.execute_and_fetch_df(sql_query, sql_params)
        elif query_type == "AGGREGATION_OPERATION":
            # For aggregation operations, we want the results as a DataFrame
            return self.sql_executor.execute_and_fetch_df(sql_query, sql_params)
        else:
            # Default to executing query and fetching results
            return self.sql_executor.execute_and_fetch_df(sql_query, sql_params)
            
    def _insert_dataframe_to_table(self, df, table_name):
        """
        Insert a DataFrame into a table
        
        Parameters:
        df (pd.DataFrame): The DataFrame to insert
        table_name (str): The target table name
        
        Returns:
        bool: Success status
        """
        try:
            # Sanitize table_name
            table_name = validate_sql_identifier(table_name)
            
            # Connect directly to the database
            conn = sqlite3.connect(os.environ.get('DB_PATH'))
            
            # Write the DataFrame to the table
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            
            # Close the connection
            conn.close()
            
            return True
        except Exception as e:
            logger.error(f"Error in _insert_dataframe_to_table: {e}")
            return False

def _handle_multi_query_result(self, result, planner_info, session_id):
    """Handle results from multi-statement query execution"""
    
    # Check if this was a resumed execution
    is_resumed = result.get("is_resumed_execution", False)
    resume_attempt = result.get("resume_attempt", 1)
    
    if result.get("success"):
        # All statements completed successfully
        target_table = planner_info["target_table_name"][0] if planner_info.get("target_table_name") else None
        
        if target_table:
            # Get final state of target table
            try:
                select_query = f"SELECT * FROM {validate_sql_identifier(target_table)}"
                target_data = self.sql_executor.execute_and_fetch_df(select_query)
                
                if isinstance(target_data, pd.DataFrame) and not target_data.empty:
                    # Add metadata about the multi-step operation
                    target_data.attrs['transformation_summary'] = {
                        'rows': len(target_data),
                        'target_table': target_table,
                        'query_type': 'MULTI_STEP_OPERATION',
                        'steps_completed': result.get("completed_statements", 0),
                        'is_multi_step': True,
                        'is_resumed_execution': is_resumed,
                        'resume_attempt': resume_attempt
                    }
                    
                    success_message = "Multi-step operation completed successfully"
                    if is_resumed:
                        success_message += f" (resumed from previous failure, attempt #{resume_attempt})"
                    
                    target_data.attrs['message'] = success_message
                    return target_data, session_id
                else:
                    # Target table exists but is empty
                    empty_df = pd.DataFrame()
                    success_message = f"Multi-step operation completed. Target table '{target_table}' is empty after transformation"
                    if is_resumed:
                        success_message += f" (resumed execution, attempt #{resume_attempt})"
                    empty_df.attrs['message'] = success_message
                    return empty_df, session_id
                    
            except Exception as e:
                logger.warning(f"Could not fetch final target data after multi-query: {e}")
                # Return success indicator even if we can't fetch final data
                success_message = "Multi-step operation completed successfully"
                if is_resumed:
                    success_message += f" (resumed execution, attempt #{resume_attempt})"
                success_df = pd.DataFrame({'status': [success_message]})
                return success_df, session_id
        else:
            # No target table specified, return success indicator
            success_message = "Multi-step operation completed successfully"
            if is_resumed:
                success_message += f" (resumed execution, attempt #{resume_attempt})"
            success_df = pd.DataFrame({'status': [success_message]})
            return success_df, session_id
    
    else:
        # Partial failure - return informative error message
        completed = result.get("completed_statements", 0)
        total_statements = completed + 1  # At least one more failed
        failed_statement = result.get("failed_statement", "")
        error_info = result.get("error", {})
        
        resume_info = ""
        if is_resumed:
            resume_info = f"\n🔄 This was resume attempt #{resume_attempt}"
        
        error_message = f"""Multi-step operation partially completed:
✅ Completed steps: {completed}/{total_statements}
❌ Failed at step {completed + 1}: {failed_statement[:100]}{'...' if len(failed_statement) > 100 else ''}
💡 Error: {error_info.get('error_message', 'Unknown error')}
🔄 Can resume from failed step: {result.get('can_resume', False)}
📝 Session ID: {session_id}{resume_info}"""
        
        return None, error_message, session_id