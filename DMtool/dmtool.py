import os
import logging
import json
import pandas as pd
import numpy as np
import re
import pyodbc
import traceback
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Dict, Any

from DMtool.planner import process_query
from DMtool.planner import (
    validate_sql_identifier,
    APIError)
from DMtool.planner import ContextualSessionManager
from DMtool.generator import SQLGenerator
from DMtool.executor import SQLExecutor
from DMtool.logging_config import setup_logging

setup_logging(log_to_file=True, log_to_console=True)

logger = logging.getLogger(__name__)

load_dotenv()

class CodeGenerationError(Exception):
    """Exception raised for code generation errors."""
    pass

class ExecutionError(Exception):
    """Exception raised for code execution errors."""
    pass

class QueryTemplateRepository:
    """Repository of query templates for common transformation patterns"""
    
    def __init__(self, template_file="DMtool/query_templates.json"):
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

            if not self.templates:
                logger.warning("No templates available")
                return None
            

            template_options = []
            for i, template in enumerate(self.templates):
                template_options.append(f"{i+1}. ID: {template['id']}\n   Pattern: {template['prompt']}")
            

            llm_prompt = f"""You are an expert at matching user queries to data transformation templates.

    USER QUERY: "{query}"

    AVAILABLE TEMPLATES:
    {chr(10).join(template_options)}

    INSTRUCTIONS:
    Analyze the user query and determine which template pattern best matches the intent and structure.
    Properly understand what the template is performing for and how it relates to the query.

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

                from google.genai import types
                
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=llm_prompt,
                    config=types.GenerateContentConfig(temperature=0.1)
                )
                
                if response and hasattr(response, "text"):

                    template_id = response.text.strip().strip('"').strip("'").lower()
                    

                    best_match = None
                    for template in self.templates:
                        if template['id'].lower() == template_id:
                            best_match = template
                            break
                    
                    if best_match:
                        return best_match
                    else:
                        logger.warning(f"Template ID '{template_id}' not found in available templates")

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

class ErrorRecoveryManager:
    """Manages error analysis and targeted retry logic"""
    
    def __init__(self):
        self.max_retries = 2
        self.retry_history = {}
    
    def analyze_error_stage(self, error_result: Dict[str, Any], sql_query: str = None) -> str:
        """
        Analyze where the error occurred and determine retry stage
        
        Returns:
        - "PLANNER" - Retry from query classification/planning
        - "GENERATOR" - Retry from SQL generation only  
        - "EXECUTOR" - Retry execution with fixes
        - "FATAL" - Cannot recover
        """
        
        if not isinstance(error_result, dict) or "error_type" not in error_result:
            return "FATAL"
        
        error_type = error_result.get("error_type", "")
        error_message = error_result.get("error_message", "").lower()
        
        # Database/Table/Column issues - retry from PLANNER (re-analyze query)
        if any(indicator in error_message for indicator in [
            "invalid object name", "table does not exist", "invalid column name", 
            "column not found", "invalid table", "table not found"
        ]):
            return "PLANNER"
        
        # SQL Syntax issues - retry from GENERATOR (regenerate SQL)
        if any(indicator in error_message for indicator in [
            "syntax error", "near", "unexpected token", "sql error",
            "invalid sql", "parse error", "malformed", "incorrect syntax"
        ]):
            return "GENERATOR"
        
        # Execution/Runtime issues - retry EXECUTOR with fixes
        if any(indicator in error_message for indicator in [
            "connection", "timeout", "lock", "busy", "constraint"
        ]):
            return "EXECUTOR"
        
        # Variable/Code issues - retry from GENERATOR
        if any(indicator in error_message for indicator in [
            "nonetype", "not iterable", "attribute error", "key error"
        ]):
            return "GENERATOR"
        
        return "FATAL"
    
    def create_retry_prompt_addition(self, stage: str, error_result: Dict[str, Any], 
                                   attempt_number: int) -> str:
        """Create additional prompt context for retry attempts"""
        
        error_msg = error_result.get("error_message", "")
        
        if stage == "PLANNER":
            return f"""
RETRY ATTEMPT #{attempt_number} - QUERY ANALYSIS CORRECTION:
Previous attempt failed with error: {error_msg}

SPECIFIC INSTRUCTIONS FOR THIS RETRY:
- Double-check all table names exist in the database schema
- Verify all mentioned columns are available in their respective tables  
- Use only validated table and column names from the enhanced matching results
- If tables/columns don't exist, suggest alternatives or indicate data unavailability
- Pay extra attention to table name formatting (no spaces, correct suffixes)
"""
        
        elif stage == "GENERATOR":
            return f"""
RETRY ATTEMPT #{attempt_number} - SQL GENERATION CORRECTION:
Previous SQL generation failed with error: {error_msg}

SPECIFIC INSTRUCTIONS FOR THIS RETRY:
- Generate simpler, more conservative SQL syntax
- Avoid complex joins if column validation shows issues
- Use proper T-SQL syntax (no SQLite-specific functions)
- Quote table names with spaces using square brackets [table name]
- For DDL operations (ALTER/DROP), don't try to fetch results
- Validate column existence before including in SELECT/UPDATE clauses
"""
        
        elif stage == "EXECUTOR":
            return f"""
RETRY ATTEMPT #{attempt_number} - EXECUTION OPTIMIZATION:
Previous execution failed with error: {error_msg}

SPECIFIC INSTRUCTIONS FOR THIS RETRY:
- Generate more robust SQL with better error handling
- Add NULL checks and COALESCE where appropriate
- Use simpler table aliases and join conditions
- Avoid operations on non-existent or empty tables
"""
        
        return ""

class DMTool:
    """Azure SQL Server-based DMTool for optimized data transformations using direct SQL queries"""

    def __init__(self, DB_PATH=None):
        """Initialize the DMToolSQL instance"""
        try:

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not found in environment variables")
                raise APIError("Gemini API key not configured")

            self.client = genai.Client(api_key=api_key)
 

            self.sql_generator = SQLGenerator()
            self.sql_executor = SQLExecutor()
            self.query_template_repo = QueryTemplateRepository()


            self.current_context = None

            logger.info("DMToolSQL initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing DMToolSQL: {e}")
            raise
    
    def __del__(self):
        """Cleanup resources"""
        try:


            logger.info(f"DMToolSQL cleanup complete")
        except Exception as e:
            logger.error(f"Error during DMToolSQL cleanup: {e}")

    def _extract_planner_info(self, resolved_data):
        """
        Extract and organize all relevant information from planner's resolved data
        to make it easily accessible in SQL generation
        """
        try:

            if not resolved_data:
                logger.error("No resolved data provided to _extract_planner_info")
                raise ValueError("No resolved data provided")


            planner_info = {

                "source_table_name": resolved_data.get("source_table_name", []),
                "target_table_name": resolved_data.get("target_table_name", []),

                "source_field_names": resolved_data.get("source_field_names", []),
                "target_sap_fields": resolved_data.get("target_sap_fields", []),
                "filtering_fields": resolved_data.get("filtering_fields", []),
                "insertion_fields": resolved_data.get("insertion_fields", []),

                "original_query": resolved_data.get("original_query", ""),
                "restructured_query": resolved_data.get("Resolved_query", ""),

                "session_id": resolved_data.get("session_id", ""),
                "key_mapping": resolved_data.get("key_mapping", []),

                "query_type": resolved_data.get("query_type", "SIMPLE_TRANSFORMATION"),
            }
            

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


            query_text = planner_info["restructured_query"]
            conditions = {}

            planner_info["qualified_source_fields"] = resolved_data.get("qualified_source_fields", [])
            planner_info["qualified_filtering_fields"] = resolved_data.get("qualified_filtering_fields", [])
            planner_info["qualified_insertion_fields"] = resolved_data.get("qualified_insertion_fields", [])
            planner_info["qualified_target_fields"] = resolved_data.get("qualified_target_fields", [])
            planner_info["table_column_mapping"] = resolved_data.get("table_column_mapping", {})
            if query_text:

                for field in planner_info["filtering_fields"]:

                    pattern = f"{field}\\s*=\\s*['\"](.*?)['\"]"
                    matches = re.findall(pattern, query_text)
                    if matches:
                        conditions[field] = matches[0]
                    

                    pattern = f"{field}\\s+in\\s+\\(([^)]+)\\)"
                    matches = re.findall(pattern, query_text, re.IGNORECASE)
                    if matches:

                        values_str = matches[0]
                        values = [v.strip().strip("'\"") for v in values_str.split(",")]
                        conditions[field] = values


            planner_info["extracted_conditions"] = conditions
            

            planner_info["target_data_samples"] = resolved_data.get("target_data_samples", {})


            self.current_context = planner_info
            return planner_info
        except Exception as e:
            logger.error(f"Error in _extract_planner_info: {e}")

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

    def _format_table_column_context_from_planner(self, table_column_mapping):
        """Format table.column context from planner's table_column_mapping"""
        try:
            if not table_column_mapping:
                return "No table column mapping available"
            
            context_parts = ["AVAILABLE TABLE.COLUMN REFERENCES:"]
            source_tables = table_column_mapping.get("source_tables", {})
            for table_name, columns in source_tables.items():
                context_parts.append(f"\nSOURCE TABLE '{table_name}':")
                for col in columns:
                    context_parts.append(f"  {table_name}.{col}")
            
            target_tables = table_column_mapping.get("target_tables", {})
            for table_name, columns in target_tables.items():
                context_parts.append(f"\nTARGET TABLE '{table_name}':")
                for col in columns:
                    context_parts.append(f"  {table_name}.{col}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error formatting table column context: {e}")
            return "Error formatting table column context"

    def _create_operation_plan(self, query, planner_info: Dict[str, Any], template: Dict[str, Any]) -> str:
        """
        Use LLM to create a detailed plan for Azure SQL query generation using enhanced planner info
        """
        try:
            # Extract qualified field information
            qualified_source_fields = planner_info.get("qualified_source_fields", [])
            qualified_filtering_fields = planner_info.get("qualified_filtering_fields", [])
            qualified_insertion_fields = planner_info.get("qualified_insertion_fields", [])
            qualified_target_fields = planner_info.get("qualified_target_fields", [])
            table_column_mapping = planner_info.get("table_column_mapping", {})
            join_conditions = planner_info.get("join_conditions", [])
            
            # Get actual column names from the resolved data
            source_tables = planner_info.get("source_table_name", [])
            target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
            
            # Format table-column context
            table_column_context = self._format_table_column_context_from_planner(table_column_mapping)
            
            # Enhanced join context
            join_context = ""
            if join_conditions:
                join_context = "\nVERIFIED JOIN CONDITIONS:\n"
                for condition in join_conditions:
                    # Use the actual field names from join conditions
                    left_table = condition.get("left_table", "")
                    right_table = condition.get("right_table", "")
                    left_field = condition.get("left_field", "")
                    right_field = condition.get("right_field", "")
                    join_type = condition.get("join_type", "INNER")
                    
                    # Build qualified condition
                    if left_table and right_table and left_field and right_field:
                        qualified_condition = f"{left_table}.{left_field} = {right_table}.{right_field}"
                    else:
                        qualified_condition = condition.get("qualified_condition", "")
                        
                    join_context += f"- {join_type} JOIN: {qualified_condition}\n"
            
            prompt = f"""
    You are an expert Azure SQL Server database engineer focusing on data transformation. I need you to create 
    precise Azure SQL Server generation plan for a data transformation task.

    ORIGINAL QUERY: "{query}"

    SOURCE TABLES INFORMATION:
    {source_tables}

    TARGET TABLE INFORMATION:
    {target_table}

    JOIN CONDITIONS FROM PLANNER:
    {join_conditions}

    {join_context}

    CONTEXT INFORMATION:
    - Query Type: {planner_info.get("query_type", "SIMPLE_TRANSFORMATION")}
    - Source Tables: {planner_info.get("source_table_name", [])}
    - Target Table: {planner_info.get("target_table_name", [])}
    - Insertion Fields: {planner_info.get("insertion_fields", [])}
    - Target Fields: {planner_info.get("target_sap_fields", [])}
    - Key Mapping: {planner_info.get("key_mapping", [])}

    Use this Template for the Azure SQL Server generation plan:
    {template.get("plan", [])}

    CRITICAL RULES FOR AZURE SQL:
    1. For conditional multi-table lookups, use the actual column names provided in the join conditions
    2. The target table is: {target_table}
    3. Use the column names EXACTLY as they appear in the join_conditions
    4. For the query type multi_table_conditional, the key field for joining should be taken from join_conditions
    5. Common join fields in SAP tables: MATNR, PRODUCT, MaterialNumber, etc.
    6. DO NOT invent column names like 'MaterialKey' - use the actual column names from join_conditions
    7. For UPDATE operations with CASE WHEN EXISTS, use the proper join columns from the join_conditions

    REQUIREMENTS:
    1. Generate 10-20 detailed steps for Azure SQL Server query creation
    2. Each step must reference the EXACT column names from the join conditions
    3. Include specific T-SQL syntax examples in each step
    4. For complex operations, reference the verified join conditions
    5. Use TOP 1 instead of LIMIT for Azure SQL

    Format:
    1. Step description using exact column references
    2. Azure SQL operation type (SELECT, INSERT, UPDATE, etc.)
    3. T-SQL query template with exact column names from join_conditions
    4. Verification note confirming the column usage

    EXAMPLE STEP FORMAT FOR YOUR QUERY:
    "1. Check if materials exist in MARA_500 table
    T-SQL operation: EXISTS check
    T-SQL query template: EXISTS (SELECT 1 FROM MARA_500 WHERE MARA_500.{actual_join_field} = {target_table}.{actual_join_field})
    Verification: Using the join field from join_conditions"

    Remember: Use ONLY the column names that appear in the join_conditions - no invented columns!
    """
            
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            
            if response and hasattr(response, "text"):
                logger.info(f"Plan generated with proper column references")
                return response.text.strip()
            else:
                logger.warning("Invalid response from LLM in plan generation")
                return self._generate_fallback_plan_with_qualified_fields(template, planner_info)
                
        except Exception as e:
            logger.error(f"Error in create_operation_plan: {e}")
            return self._generate_fallback_plan_with_qualified_fields(template, planner_info)

    def _format_table_column_context_from_planner(self, table_column_mapping):
        """Format table.column context from planner's table_column_mapping"""
        try:
            if not table_column_mapping:
                return "No table column mapping available"
            
            context_parts = ["AVAILABLE TABLE.COLUMN REFERENCES:"]
            source_tables = table_column_mapping.get("source_tables", {})
            for table_name, columns in source_tables.items():
                context_parts.append(f"\nSOURCE TABLE '{table_name}':")
                for col in columns:
                    context_parts.append(f"  {table_name}.{col}")
            
            target_tables = table_column_mapping.get("target_tables", {})
            for table_name, columns in target_tables.items():
                context_parts.append(f"\nTARGET TABLE '{table_name}':")
                for col in columns:
                    context_parts.append(f"  {table_name}.{col}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error formatting table column context: {e}")
            return "Error formatting table column context"

    def _generate_fallback_plan_with_qualified_fields(self, template, planner_info):
        """Generate fallback plan using qualified field information"""
        try:
            qualified_source = planner_info.get("qualified_source_fields", [])
            qualified_target = planner_info.get("qualified_target_fields", [])
            
            fallback_steps = []
            for i, step in enumerate(template.get("plan", []), 1):
                source_ref = qualified_source[0] if qualified_source else "source_table.source_field"
                target_ref = qualified_target[0] if qualified_target else "target_table.target_field"
                
                filled_step = step.replace("{field}", source_ref).replace("{table}", source_ref.split('.')[0] if '.' in source_ref else "unknown_table")
                fallback_steps.append(f"{i}. {filled_step}")
            
            return "\n".join(fallback_steps)
            
        except Exception as e:
            logger.error(f"Error in fallback plan generation: {e}")
            return "1. Generate basic SQL Server query\n2. Execute transformation"
        
    def _get_segment_name(self, segment_id, conn):
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
        Process a query as part of a sequential transformation using SQL generation
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

            if not query or not isinstance(query, str):
                logger.error(f"Invalid query type: {type(query)}")
                return "Query must be a non-empty string", session_id


            if not all(isinstance(x, int) for x in [object_id, segment_id, project_id]):
                logger.error(
                    f"Invalid ID types: object_id={type(object_id)}, segment_id={type(segment_id)}, project_id={type(project_id)}"
                )
                return "Invalid ID types - must be integers", session_id
            
            context_manager = ContextualSessionManager()


            if not session_id:
                session_id = context_manager.create_session()
                logger.info(f"Created new session: {session_id}")
                

            additional_source_tables = []


            logger.info(f"Processing query: {query}")
            resolved_data = process_query(
                object_id, segment_id, project_id, query, session_id=session_id
            )
            

            if not resolved_data:
                logger.error("Failed to resolve query with planner")
                return "Failed to resolve query", session_id
            

            template = self.query_template_repo.find_matching_template(query)
            logger.info(f"Found template: {template.get('id', 'None')} for query '{query}'")
            if not template:
                logger.error("No matching template found for query")

                template = {
                    "id": "fallback",
                    "prompt": "Basic transformation",
                    "query": "SELECT {field} FROM {table} WHERE {filter_field} = '{filter_value}'",
                    "plan": ["1. Identify source and target", "2. Generate basic SQL query"]
                }

            query_type = resolved_data.get("query_type", "SIMPLE_TRANSFORMATION")

            session_id = resolved_data.get("session_id")


            try:
                conn = pyodbc.connect(self.sql_executor.connection_string)
            except pyodbc.Error as e:
                logger.error(f"Failed to connect to database: {e}")
                return f"Database connection error: {e}", session_id

            try:
                resolved_data["original_query"] = query
                

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


                if additional_source_tables:
                    source_tables = resolved_data.get("source_table_name", [])
                    if isinstance(source_tables, list):

                        for table in additional_source_tables:
                            if table not in source_tables:
                                source_tables.append(table)
                        resolved_data["source_table_name"] = source_tables


                planner_info = self._extract_planner_info(resolved_data)
                

                sql_plan = self._create_operation_plan(query, planner_info, template)
                

                sql_query, sql_params = self.sql_generator.generate_sql(sql_plan, planner_info, template)
                logger.info(f"Generated SQL query: {sql_query}")
                result = self._execute_sql_query(sql_query, sql_params, planner_info)


                if isinstance(result, dict) and "error_type" in result:
                    logger.error(f"SQL execution error: {result}")
                    return f"SQL execution failed: {result.get('error_message', 'Unknown error')}", session_id

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
                    
                    return multi_result[0], session_id

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

                if target_table and query_type in ["SIMPLE_TRANSFORMATION", "JOIN_OPERATION", "CROSS_SEGMENT", "AGGREGATION_OPERATION"]:
                    try:

                        select_query = f"SELECT * FROM [{validate_sql_identifier(target_table)}]"
                        target_data = self.sql_executor.execute_and_fetch_df(select_query)
                        
                        if isinstance(target_data, pd.DataFrame) and not target_data.empty:

                            rows_affected = len(target_data)
                            non_null_columns = target_data.dropna(axis=1, how='all').columns.tolist()
                            
                            target_data.attrs['transformation_summary'] = {
                                'rows': rows_affected,
                                'populated_fields': non_null_columns,
                                'target_table': target_table,
                                'query_type': query_type
                            }
                            try:
                                context_manager = ContextualSessionManager()
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

                            return target_data, session_id
                        else:

                            empty_df = pd.DataFrame()
                            empty_df.attrs['message'] = f"Target table '{target_table}' is empty after transformation"
                            return empty_df, session_id
                            
                    except Exception as e:

                        return  result, session_id
                        
            except Exception as e:
                logger.error(f"Error in process_sequential_query: {e}")
                logger.error(traceback.format_exc())
                if conn:
                    conn.close()
                return f"An error occurred during processing: {e}", session_id
                        

        except Exception as e:
            logger.error(f"Outer error in process_sequential_query: {e}")
            logger.error(traceback.format_exc())
            if conn:
                try:
                    conn.close()
                except:
                    pass
            return f"An error occurred: {e}", session_id

    def _is_multi_statement_query(self, sql_query):
        """Detect if SQL contains multiple statements"""
        if not sql_query or not isinstance(sql_query, str):
            return False        
        statements = self.sql_executor.split_sql_statements(sql_query)
        return len(statements) > 1

    def _execute_sql_query(self, sql_query, sql_params, planner_info):
        """
        Execute SQL query using the SQLExecutor with multi-statement support
        
        Parameters:
        sql_query (str): The SQL query to execute
        sql_params (dict): The parameters for the query
        planner_info (dict): Planner information for context
        
        Returns:
        Union[pd.DataFrame, dict]: Results or error information
        """
        

        if self._is_multi_statement_query(sql_query):
            logger.info("Detected multi-statement query, using multi-query executor")
            return self.sql_executor.execute_multi_statement_query(
                sql_query, sql_params, context_manager=ContextualSessionManager(),session_id=planner_info.get("session_id")
            )
        

        query_type = planner_info.get("query_type", "SIMPLE_TRANSFORMATION")
        operation_type = None
        if query_type == "SIMPLE_TRANSFORMATION":

            operation_type = None
            try:
                operation_type = sql_query.strip().upper().split()[0]
            except:
                if "ALTER TABLE" in sql_query.upper():
                    operation_type = "ALTER"
                elif sql_query.upper() in ["Drop","Delete"]:
                    operation_type = "DELETE"
                
            
            if operation_type == "INSERT":
                return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False)
            elif operation_type == "UPDATE":
                return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False)
            elif operation_type == "DELETE":
                return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False)
            elif operation_type == "ALTER":
                return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False)
            elif operation_type == "WITH":
                if "INSERT INTO" in sql_query.upper():
                    return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False)
                elif "UPDATE" in sql_query.upper():
                    return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False)
                else:

                    return self.sql_executor.execute_and_fetch_df(sql_query, sql_params)
            else:

                return self.sql_executor.execute_and_fetch_df(sql_query, sql_params)
        elif not operation_type :
            return self.sql_executor.execute_query(sql_query, sql_params,fetch_results=False)
        elif query_type in ["JOIN_OPERATION", "CROSS_SEGMENT"]:

            if "INSERT INTO" in sql_query.upper() or "UPDATE" in sql_query.upper():

                return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False)
            else:

                return self.sql_executor.execute_and_fetch_df(sql_query, sql_params)
        elif query_type == "VALIDATION_OPERATION":

            return self.sql_executor.execute_and_fetch_df(sql_query, sql_params)
        elif query_type == "AGGREGATION_OPERATION":

            return self.sql_executor.execute_and_fetch_df(sql_query, sql_params)
        else:

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

            table_name = validate_sql_identifier(table_name)
            

            conn = pyodbc.connect(self.sql_executor.connection_string)
            

            df.to_sql(table_name, conn, if_exists="replace", index=False)
            

            conn.close()
            
            return True
        except Exception as e:
            logger.error(f"Error in _insert_dataframe_to_table: {e}")
            return False

    def _handle_multi_query_result(self, result, planner_info, session_id):
        """Handle results from multi-statement query execution"""
        
        if result.get("success"):

            target_table = planner_info["target_table_name"][0] if planner_info.get("target_table_name") else None
            
            if target_table:

                try:
                    select_query = f"SELECT * FROM [{validate_sql_identifier(target_table)}]"
                    target_data = self.sql_executor.execute_and_fetch_df(select_query)
                    
                    if isinstance(target_data, pd.DataFrame) and not target_data.empty:

                        target_data.attrs['transformation_summary'] = {
                            'rows': len(target_data),
                            'target_table': target_table,
                            'query_type': 'MULTI_STEP_OPERATION',
                            'steps_completed': result.get("completed_statements", 0),
                            'is_multi_step': True
                        }
                        return target_data, session_id
                    else:

                        empty_df = pd.DataFrame()
                        empty_df.attrs['message'] = f"Multi-step operation completed. Target table '{target_table}' is empty after transformation"
                        return empty_df, session_id
                        
                except Exception as e:
                    logger.warning(f"Could not fetch final target data after multi-query: {e}")

                    success_df = pd.DataFrame({'status': ['Multi-step operation completed successfully']})
                    return success_df, session_id
            else:

                success_df = pd.DataFrame({'status': ['Multi-step operation completed successfully']})
                return success_df, session_id
        
        else:

            completed = result.get("completed_statements", 0)
            total_statements = completed + 1
            failed_statement = result.get("failed_statement", "")
            error_info = result.get("error", {})
            
            error_message = f"""Multi-step operation partially completed:
    ✅ Completed steps: {completed}/{total_statements}
    ❌ Failed at step {completed + 1}: {failed_statement[:100]}{'...' if len(failed_statement) > 100 else ''}
    💡 Error: {error_info.get('error_message', 'Unknown error')}
    🔄 Can resume from failed step: {result.get('can_resume', False)}
    📝 Session ID: {session_id}"""
            
            return None, error_message, session_id