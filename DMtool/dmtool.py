import os
import logging
import json
import pandas as pd
import numpy as np
import re
import sqlite3
import traceback
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Tuple

from DMtool.planner import process_query
from DMtool.planner import (
    validate_sql_identifier,
    APIError)
from DMtool.planner import ContextualSessionManager
from DMtool.generator import SQLGenerator
from DMtool.executor import SQLExecutor
from DMtool.logging_config import setup_logging
from DMtool.llm_config import LLMManager
from DMtool.source_table_manager import generate_lineage_report

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
        api_key = os.environ.get("API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise APIError("Gemini API key not configured")
        self.llm = LLMManager(
            provider="google",
            model="gemini/gemini-2.5-flash",
            api_key=api_key
        )
        
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
                
                response = self.llm.generate(llm_prompt, temperature=0.05, max_tokens=50)
                
                if response :

                    template_id = response.strip().strip('"').strip("'").lower()
                    

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

class DMTool:
    """SQLite-based DMTool for optimized data transformations using direct SQLite queries"""

    def __init__(self, DB_PATH=os.environ.get('DB_PATH')):
        """Initialize the DMToolSQL instance"""
        try:

            api_key = os.environ.get("API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not found in environment variables")
                raise APIError("Gemini API key not configured")

            self.llm = LLMManager(
                provider="google",
                model="gemini/gemini-2.5-flash",
                api_key=api_key
            )

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
        to make it easily accessible in SQLite generation
        """
        try:

            if not resolved_data:
                logger.error("No resolved data provided to _extract_planner_info")
                raise ValueError("No resolved data provided")


            planner_info = {

                "source_table_name": resolved_data.get("source_table_name", []),
                "target_table_name": resolved_data.get("target_table_name", []),

                "source_field_names": resolved_data.get("qualified_source_fields", []),
                "target_sap_fields": resolved_data.get("qualified_target_fields", []),
                "filtering_fields": resolved_data.get("qualified_filtering_fields", []),
                "insertion_fields": resolved_data.get("qualified_insertion_fields", []),

                "original_query": resolved_data.get("original_query", ""),
                "restructured_query": resolved_data.get("Resolved_query", ""),

                "session_id": resolved_data.get("session_id", ""),
                "key_mapping": resolved_data.get("key_mapping", []),
                "table_column_mapping": resolved_data.get("table_column_mapping", {}),
                "transformation_context": resolved_data.get("transformation_context", ""),
                "transformation_plan": resolved_data.get("transformation_plan", ""),
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

    def _create_operation_plan(self, query, planner_info: Dict[str, Any], template: Dict[str, Any]) -> str:
        """
        Use LLM to create a detailed plan for SQLite query generation using enhanced planner info
        """
        try:
            qualified_source_fields = planner_info.get("qualified_source_fields", [])
            qualified_filtering_fields = planner_info.get("qualified_filtering_fields", [])
            qualified_insertion_fields = planner_info.get("qualified_insertion_fields", [])
            qualified_target_fields = planner_info.get("qualified_target_fields", [])
            table_column_mapping = planner_info.get("table_column_mapping", {})
            join_conditions = planner_info.get("join_conditions", [])
            table_column_context = self._format_table_column_context_from_planner(table_column_mapping)
            join_context = ""
            if join_conditions:
                join_context = "\nVERIFIED JOIN CONDITIONS:\n"
                for condition in join_conditions:
                    qualified_condition = condition.get("qualified_condition", "")
                    join_type = condition.get("join_type", "INNER")
                    join_context += f"- {join_type} JOIN: {qualified_condition}\n"
            prompt = f"""
    You are an expert SQLite database engineer focusing on data transformation. I need you to create 
    precise SQLite generation plan for a data transformation task.

    ORIGINAL QUERY: "{planner_info.get("restructured_query", "")}"

    VERIFIED TABLE.COLUMN MAPPINGS:
    {table_column_context}

    {join_context}

    QUALIFIED FIELD INFORMATION (Use these exact references):
    - Source Fields: {qualified_source_fields}
    - Filtering Fields: {qualified_filtering_fields}
    - Insertion Fields: {qualified_insertion_fields}
    - Target Fields: {qualified_target_fields}

    CONTEXT INFORMATION:
    - Query Type: {planner_info.get("query_type", "SIMPLE_TRANSFORMATION")}
    - Source Tables: {planner_info.get("source_table_name", [])}
    - Target Table: {planner_info.get("target_table_name", [])}
    - Key Mapping: {planner_info.get("key_mapping", [])}

    Use this Template for the SQLite generation plan:
    {template.get("plan", [])}

    CRITICAL RULES:
    1. Use ONLY the qualified table.column references provided above
    2. Never invent or modify the table.column combinations
    3. Follow the exact qualified field mappings for all operations
    4. For JOIN operations, use the verified join conditions provided
    5. All column references must be exactly as specified in the qualified fields
    6. If you need to create a new column, use the ALTER TABLE statement with the exact qualified table.column reference
    7. If you need to delete a column, use the ALTER TABLE statement with the exact qualified table.column reference
    8. Do not create or delete columns unless explicitly mentioned in the prompt
    9. Do not drop any tables or columns.
    10. If a column is not said to be created, assume it already exists in the tables.
    11. Do not create or delete tables.
    12. Do not create transactions

    REQUIREMENTS:
    1. Generate 10-20 detailed steps for SQLite query creation
    2. Each step must use the EXACT qualified table.column references from above
    3. Include specific SQLite syntax examples in each step
    4. Verify every table.column reference against the provided qualified fields
    5. For complex operations, reference the verified join conditions

    Format:
    1. Step description using exact qualified references
    2. SQLite operation type (SELECT, INSERT, UPDATE, etc.)
    3. SQLite query template with exact qualified table.column names
    4. Verification note confirming the table.column usage

    EXAMPLE STEP FORMAT:
    "1. Select material data from source table
    SQLite operation: SELECT
    SQLite query template: SELECT MARA.MATNR, MARA.MTART FROM MARA
    Verification: Using qualified fields MARA.MATNR and MARA.MTART from source fields list"

    Remember: Use ONLY the qualified table.column references provided - no modifications or additions!

    Notes:
    1. Use Alter table only when the prompt specifically mentions creation or deletion of a column. DO NOT use Alter for anything else
    2. If User does not give to create a column then assume that the column already exists in the tables and there is not need to create a column.
    3. If the prompt does not specify a column, do not include it in the query.
    4. If we have Column given that exist in a table, then use that column in the query.
    """
            response = self.llm.generate(prompt, temperature=0.3, max_tokens=1500)
            
            if response:
                logger.info(f"Plan generated using qualified field references")
                logger.info(f"Generated Plan:\n{response}")
                return response.strip()
            else:
                logger.warning("Invalid response from LLM in enhanced plan generation")
                return self._generate_fallback_plan_with_qualified_fields(template, planner_info)
                
        except Exception as e:
            logger.error(f"Error in enhanced create_sql_plan: {e}")
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

    def _find_affected_indexes(self, old_table,updated_table) -> List[str]:
        """Identify indexes that may be affected by transformations"""
        if len(old_table) == 0 and len(updated_table) != 0:
            return updated_table.index.tolist()
        min_len = min(len(old_table), len(updated_table))
        old_hash = pd.util.hash_pandas_object(old_table.iloc[:min_len], index=False)
        new_hash = pd.util.hash_pandas_object(updated_table.iloc[:min_len], index=False)

        change_mask = old_hash != new_hash
        changed_indexes = updated_table.index[:min_len][change_mask].tolist()
        inserted_indexes = updated_table.index[min_len:].tolist()

        return list(set(changed_indexes + inserted_indexes))


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
            return "1. Generate basic SQLite query\n2. Execute transformation"
        
    def _get_segment_name(self, segment_id,conn):
        cursor = conn.cursor()
        cursor.execute("SELECT segement_name FROM connection_segments WHERE segment_id = ?", (segment_id,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            logger.error(f"Segment ID {segment_id} not found in database")
            return None
        

    def process_sequential_query(self, query, object_id, segment_id, project_id, session_id = None,is_selection_criteria = False):
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

            logger.info(f"Processing query: {query}")
            resolved_data = process_query(
                object_id, segment_id, project_id, query,template=template, session_id=session_id
            )

            if not resolved_data:
                logger.error("Failed to resolve query with planner")
                return "Failed to resolve query", session_id
            
            query_type = resolved_data.get("query_type", "SIMPLE_TRANSFORMATION")

            session_id = resolved_data.get("session_id")


            try:
                conn = sqlite3.connect(os.environ.get('DB_PATH'))
            except sqlite3.Error as e:
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

                
                # planner_info = self._extract_planner_info(resolved_data)
                planner_info = resolved_data

                # sql_plan = self._create_operation_plan(planner_info["restructured_query"], planner_info, template)
                select_query = f"SELECT * FROM {validate_sql_identifier(target_table)}"
                target_data_before = self.sql_executor.execute_and_fetch_df(select_query)
                sql_query, sql_params = self.sql_generator.generate_sql(planner_info, template,sql_plan=planner_info.get("transformation_plan", ""))
                logger.info(f"Generated SQL query: {sql_query}")
                result = self._execute_sql_query(sql_query, sql_params, planner_info,
                object_id=object_id,
                segment_id=segment_id,
                project_id=project_id,
                is_selection_criteria=is_selection_criteria)


                if isinstance(result, dict) and "error_type" in result:
                    logger.error(f"SQL execution error: {result}")
                    return f"SQL execution failed: {result.get('error_message', 'Unknown error')}", session_id

                if isinstance(result, dict) and result.get("multi_query_result"):
                    logger.info(f"Processing multi-query result: {result.get('completed_statements', 0)} statements completed")
                    multi_result = self._handle_multi_query_result(result, planner_info, session_id)
                    
                    if result.get("success") and len(multi_result) > 2:
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
                            target_data_after = self.sql_executor.execute_and_fetch_df(select_query)
                            affected_indexes = self._find_affected_indexes(target_data_before,target_data_after)
                        except Exception as e:
                            logger.warning(f"Could not save transformation record for multi-query: {e}")
                    
                    return multi_result[0], affected_indexes

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
                                target_data_after = self.sql_executor.execute_and_fetch_df(select_query)
                                affected_indexes = self._find_affected_indexes(target_data_before,target_data_after)
                                
                            except Exception as e:
                                logger.warning(f"Could not save transformation record: {e}")

                            return target_data, affected_indexes
                        else:

                            empty_df = pd.DataFrame()
                            empty_df.attrs['message'] = f"Target table '{target_table}' is empty after transformation"

                            return empty_df, []
                            
                    except Exception as e:

                        return  result, []
                        
            except Exception as e:
                logger.error(f"Error in process_sequential_query: {e}")
                logger.error(traceback.format_exc())
                if conn:
                    conn.close()
                return f"An error occurred during processing: {e}", []
                        

        except Exception as e:
            logger.error(f"Outer error in process_sequential_query: {e}")
            logger.error(traceback.format_exc())
            if conn:
                try:
                    conn.close()
                except:
                    pass
            return f"An error occurred: {e}", session_id
        
    def generate_preload_postload_report(self, target_table: str, session_id: Optional[str] = None) -> str:
        """
        Generate pre-load/post-load CSV report for a target table
        
        Parameters:
        target_table (str): Target table name
        session_id (str): Session ID for lineage tracking
        
        Returns:
        str: Path to generated CSV file
        """
        try:
            logger.info(f"Generating pre-load/post-load report for {target_table}")
            
            csv_path = generate_lineage_report(target_table, session_id)
            
            if csv_path:
                logger.info(f"Report generated successfully: {csv_path}")
                return csv_path
            else:
                logger.warning("Failed to generate report - no lineage data found")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""

    def process_selection_criteria(self, selection_criteria, object_id, segment_id, project_id, session_id = None):
        """
        Process selection criteria query using SQL generation
        
        Parameters:
        selection_criteria (str): The selection criteria query
        object_id (int): Object ID for mapping
        segment_id (int): Segment ID for mapping
        project_id (int): Project ID for mapping
        session_id (str): Optional session ID, creates new session if None
        
        Returns:
        tuple: (generated_sql, result, session_id)
        """
        is_selection_criteria = True
        conn = None
        try:
            query = selection_criteria
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

            template={
                "id": "filter_and_keep_rows",
                "prompt": "Filter and update target table where {filter_field} = {filter_value}",
                "query": "DELETE FROM {target_table} WHERE {filter_field} != '{filter_value}'",
                "plan": [
                    "1. Identify the target table to filter",
                    "2. Identify the filter field and value condition", 
                    "3. Use DELETE statement to remove all rows that don't match the condition",
                    "4. This will keep only rows where the condition is true",
                    "5. Consider backing up data before executing the DELETE operation"
                ]
                }
            logger.info(f"Processing query: {query}")
            resolved_data = process_query(
                object_id, segment_id, project_id, query, session_id=session_id,template=template, is_selection_criteria=True
            )
            

            if not resolved_data:
                logger.error("Failed to resolve query with planner")
                return "Failed to resolve query", session_id
            

            query_type = resolved_data.get("query_type", "SIMPLE_TRANSFORMATION")

            session_id = resolved_data.get("session_id")


            try:
                conn = sqlite3.connect(os.environ.get('DB_PATH'))
            except sqlite3.Error as e:
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

                
                planner_info = resolved_data
                select_query = f"SELECT * FROM {validate_sql_identifier(target_table)}"
                target_data_before = self.sql_executor.execute_and_fetch_df(select_query)
                sql_query, sql_params = self.sql_generator.generate_sql(planner_info, template,sql_plan=planner_info.get("transformation_plan", ""))
                logger.info(f"Generated SQL query: {sql_query}")
                result = self._execute_sql_query(sql_query, sql_params, planner_info,
                object_id=object_id,
                segment_id=segment_id,
                project_id=project_id,
                is_selection_criteria=is_selection_criteria)


                if isinstance(result, dict) and "error_type" in result:
                    logger.error(f"SQL execution error: {result}")
                    return f"SQL execution failed: {result.get('error_message', 'Unknown error')}", session_id

                if "selection_criteria_target_table" in resolved_data:
                    target_table = resolved_data["selection_criteria_target_table"]
                    if isinstance(target_table, list) and len(target_table) > 0:
                        target_table = target_table[0]
                segment_name = self._get_segment_name(segment_id, conn)
                if segment_name:
                    context_manager.add_segment(
                        session_id,
                        segment_name,
                        planner_info["target_table_name"],
                    )

                if target_table:
                    try:

                        select_query = f"SELECT * FROM [{validate_sql_identifier(target_table)}]"
                        self.sql_executor.sync_src_to_target(target_table)
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
                                    "query_type": "SELECTION_CRITERIA",
                                    "source_tables": planner_info.get("selection_criteria_source_table", []),
                                    "target_table": planner_info.get("selection_criteria_target_table", []),
                                    "fields_affected": planner_info.get("target_sap_field", []),
                                    "execution_result": {
                                        "success": True,
                                        "rows_affected": len(target_data) if isinstance(target_data, pd.DataFrame) else 0,
                                    }
                                }
                                
                                context_manager.add_transformation_record(session_id, transformation_data)
                                affected_indexes = self._find_affected_indexes(target_data_before,target_data) 
                                return target_data, affected_indexes
                                
                            except Exception as e:
                                logger.warning(f"Could not save transformation record: {e}")

                            return target_data, affected_indexes
                        else:

                            empty_df = pd.DataFrame()
                            empty_df.attrs['message'] = f"Target table '{target_table}' is empty after transformation"
                            return empty_df, []
                            
                    except Exception as e:
                        return  result, []
                        
            except Exception as e:
                logger.error(f"Error in process_sequential_query: {e}")
                logger.error(traceback.format_exc())
                if conn:
                    conn.close()
                return f"An error occurred during processing: {e}", []
                        

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
    
    def create_session_id(self):
        """ Create a new session ID for tracking transformations"""
        context_manager = ContextualSessionManager()
        session_id = context_manager.create_session()
        logger.info(f"Created new session: {session_id}")
        return session_id

    def _execute_sql_query(self, sql_query, sql_params, planner_info, object_id=None, segment_id=None, project_id=None, is_selection_criteria=True):
        """
        Execute SQLite query using the SQLExecutor with multi-statement support
        
        Parameters:
        sql_query (str): The SQLite query to execute
        sql_params (dict): The parameters for the query
        planner_info (dict): Planner information for context
        
        Returns:
        Union[pd.DataFrame, dict]: Results or error information
        """
        

        if self._is_multi_statement_query(sql_query):
            logger.info("Detected multi-statement query, using multi-query executor")
            return self.sql_executor.execute_multi_statement_query(
                sql_query, sql_params, context_manager=ContextualSessionManager(),session_id=planner_info.get("session_id"),
                object_id=object_id,
                segment_id=segment_id,
                project_id=project_id,
                planner_info=planner_info
            )
        

        query_type = planner_info.get("query_type", "SIMPLE_TRANSFORMATION")
        operation_type = None
        if query_type == "SIMPLE_TRANSFORMATION":

            operation_type = None
            try:
                operation_type = sql_query.strip().upper().split()[0]
            except:
                if "AlTER TABLE" in sql_query.upper():
                    operation_type = "ALTER"
                elif sql_query.upper() in ["Drop","Delete"]:
                    operation_type = "DELETE"
                
            
            if operation_type == "INSERT":
                return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False,session_id=planner_info.get("session_id"),planner_info=planner_info,is_selection_criteria=is_selection_criteria)
            elif operation_type == "UPDATE":
                return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False,session_id=planner_info.get("session_id"),planner_info=planner_info,is_selection_criteria=is_selection_criteria)
            elif operation_type == "DELETE":
                return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False,object_id=object_id,segment_id=segment_id,project_id=project_id,session_id=planner_info.get("session_id"),planner_info=planner_info,is_selection_criteria=is_selection_criteria)
            elif operation_type == "ALTER":
                return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False,object_id=object_id,segment_id=segment_id,project_id=project_id,session_id=planner_info.get("session_id"),planner_info=planner_info,is_selection_criteria=is_selection_criteria)
            elif operation_type == "WITH":
                if "INSERT INTO" in sql_query.upper():
                    return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False,session_id=planner_info.get("session_id"),planner_info=planner_info,is_selection_criteria=is_selection_criteria)
                elif "UPDATE" in sql_query.upper():
                    return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False,session_id=planner_info.get("session_id"),planner_info=planner_info,is_selection_criteria=is_selection_criteria)
                else:
                    return self.sql_executor.execute_and_fetch_df(sql_query, sql_params,session_id=planner_info.get("session_id"))
            else:
                return self.sql_executor.execute_and_fetch_df(sql_query, sql_params)
        elif not operation_type :
            return self.sql_executor.execute_query(sql_query, sql_params,fetch_results=False,planner_info=planner_info)
        elif query_type in ["JOIN_OPERATION", "CROSS_SEGMENT"]:

            if "INSERT INTO" in sql_query.upper() or "UPDATE" in sql_query.upper():

                return self.sql_executor.execute_query(sql_query, sql_params, fetch_results=False,planner_info=planner_info)
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
            

            conn = sqlite3.connect(os.environ.get('DB_PATH'))
            

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
                    select_query = f"SELECT * FROM {validate_sql_identifier(target_table)}"
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
            