import logging
import json
import re
import sqlite3
import pandas as pd
from google import genai
from google.genai import types
import traceback
import os
from typing import Dict, List, Any, Optional, Union, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SQLGenerator:
    """Generates SQL queries based on planner output"""
    
    def __init__(self, db_dialect="sqlite"):
        """Initialize the SQL generator
        
        Parameters:
        db_dialect (str): Database dialect to use ('sqlite' by default)
        """
        self.db_dialect = db_dialect
        self.sql_templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize SQL query templates for different operations"""
        templates = {
            # Simple SELECT query
            "select": """
                SELECT {select_fields}
                FROM {table}
                {where_clause}
            """,
            
            # JOIN operation
            "join": """
                SELECT {select_fields}
                FROM {main_table} {main_alias}
                {join_clauses}
                {where_clause}
            """,
            
            # INSERT operation
            "insert": """
                INSERT INTO {target_table} ({target_fields})
                SELECT {source_fields}
                FROM {source_table}
                {where_clause}
            """,
            
            # UPDATE operation
            "update": """
                UPDATE {target_table}
                SET {set_clause}
                {where_clause}
            """,
            
            # CREATE VIEW operation
            "create_view": """
                CREATE TEMPORARY VIEW IF NOT EXISTS {view_name} AS
                SELECT {select_fields}
                FROM {source_table}
                {where_clause}
            """,
            
            # AGGREGATION operation
            "aggregation": """
                SELECT {group_fields}{agg_separator}{agg_functions}
                FROM {table}
                {where_clause}
                GROUP BY {group_fields}
            """
        }
        return templates
    
    def generate_sql(self, sql_plan, planner_info: Dict[str, Any],template: Dict[str, str]=None) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL query based on planner information using LLM for planning and generation
        
        Parameters:
        planner_info (Dict): Information extracted by the planner
        
        Returns:
        Tuple[str, Dict]: The generated SQL query and parameterized values
        """
        try:
            # Log the information we're working with
            logger.info(f"Generating SQLite query for query: {planner_info.get('original_query', '')}")
            logger.info(f"Source tables: {planner_info.get('source_table_name', [])}")
            logger.info(f"Source fields: {planner_info.get('source_field_names', [])}")
            logger.info(f"Target table: {planner_info.get('target_table_name', [])}")
            logger.info(f"Target fields: {planner_info.get('target_sap_fields', [])}")
            logger.info(f"Filtering fields: {planner_info.get('filtering_fields', [])}")
            logger.info(f"Conditions: {planner_info.get('extracted_conditions', {})}")
            
            # 1. Create a step-by-step SQLite generation plan using LLM
            
            # 2. Generate initial SQLite query using LLM based on the plan
            initial_sql_query, initial_sql_params = self.generate_sql_with_llm(sql_plan, planner_info,template["query"])
            
            # 3. Analyze and fix the generated query (new step)
            from query_analyzer import SQLiteQueryAnalyzer
            query_analyzer = SQLiteQueryAnalyzer()
            fixed_sql_query, fixed_sql_params, is_valid = query_analyzer.analyze_and_fix_query(
                initial_sql_query, initial_sql_params, planner_info
            )
            
            if is_valid:
                return fixed_sql_query, fixed_sql_params
            
            # 4. If still not valid after fixes, fall back to rule-based method
            logger.warning(f"Could not generate valid SQLite query even after fixing attempts: {fixed_sql_query}")
            logger.info("Falling back to rule-based query generation")
                    
            # Fallback to rule-based method as a safety measure
            query_type = planner_info.get("query_type", "SIMPLE_TRANSFORMATION")
            
            if query_type == "SIMPLE_TRANSFORMATION":
                return self._generate_simple_transformation(planner_info)
            elif query_type == "JOIN_OPERATION":
                return self._generate_join_operation(planner_info)
            elif query_type == "CROSS_SEGMENT":
                return self._generate_cross_segment(planner_info)
            elif query_type == "VALIDATION_OPERATION":
                return self._generate_validation_operation(planner_info)
            elif query_type == "AGGREGATION_OPERATION":
                return self._generate_aggregation_operation(planner_info)
            else:
                return self._generate_simple_transformation(planner_info)
            
        except Exception as e:
            logger.error(f"Error in generate_sql: {e}")
            logger.error(traceback.format_exc())
            
            # Fallback to rule-based method
            query_type = planner_info.get("query_type", "SIMPLE_TRANSFORMATION")
            
            if query_type == "SIMPLE_TRANSFORMATION":
                return self._generate_simple_transformation(planner_info)
            elif query_type == "JOIN_OPERATION":
                return self._generate_join_operation(planner_info)
            elif query_type == "CROSS_SEGMENT":
                return self._generate_cross_segment(planner_info)
            elif query_type == "VALIDATION_OPERATION":
                return self._generate_validation_operation(planner_info)
            elif query_type == "AGGREGATION_OPERATION":
                return self._generate_aggregation_operation(planner_info)
            else:
                return self._generate_simple_transformation(planner_info)

    def create_sql_plan(self, planner_info: Dict[str, Any]) -> str:
        """
        Use LLM to create a detailed plan for SQLite query generation
        
        Parameters:
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
            original_query = planner_info.get("original_query", "")
            
            # Create a comprehensive prompt for the LLM
            prompt = f"""
    You are an expert SQLite database engineer. I need you to create a step-by-step plan to generate 
    SQLite for the following natural language query:

    ORIGINAL QUERY: "{original_query}"

    CONTEXT INFORMATION:
    - Query Type: {query_type}
    - Source Tables: {source_tables}
    - Source Fields: {source_fields}
    - Target Table: {target_table}
    - Target Fields: {target_fields}
    - Filtering Fields: {filtering_fields}
    - Filtering Conditions: {json.dumps(conditions, indent=2)}

    Based on the query intent, create a detailed plan for generating an efficient SQLite query, 
    structured as numbered steps. Carefully consider:

    1. The query type and appropriate SQLite operation (SELECT, INSERT, UPDATE, etc.)
    2. Proper table references and aliases
    3. Precise column selections
    4. Correct filter conditions
    5. Join conditions (if applicable)
    6. Proper ordering of operations

    IMPORTANT SQLite-SPECIFIC CONSIDERATIONS:
    - SQLite does not support RIGHT JOIN or FULL JOIN (use LEFT JOIN with table order swapped instead)
    - SQLite uses IFNULL instead of ISNULL
    - SQLite UPDATE with JOIN requires a specific syntax (use FROM clause after SET)
    - SQLite has limited support for common table expressions
    - SQLite has no BOOLEAN type (use INTEGER 0/1)

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
                return response.text.strip()
            else:
                logger.warning("Invalid response from LLM in create_sql_plan")
                return "1. Generate basic SQLite query based on query type\n2. Return query"
                
        except Exception as e:
            logger.error(f"Error in create_sql_plan: {e}")
            return "1. Generate basic SQLite query based on query type\n2. Return query"

    def generate_sql_with_llm(self, plan: str, planner_info: Dict[str, Any],template: str=None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate SQLite query using LLM based on the plan
        
        Parameters:
        plan (str): Step-by-step plan for SQLite generation
        planner_info (Dict): Information extracted by the planner
        
        Returns:
        Tuple[str, Dict[str, Any]]: The generated SQLite query and parameters
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
            original_query = planner_info.get("original_query", "")
            key_mapping = planner_info.get("key_mapping", [])
            
            # Check if target table has data
            target_has_data = False
            target_data_samples = planner_info.get("target_data_samples", {})
            if isinstance(target_data_samples, pd.DataFrame) and not target_data_samples.empty:
                target_has_data = True
            
            # Create detailed SQLite generation prompt
            prompt = f"""
    You are an expert SQLite database engineer focusing on data transformation operations. I need you to generate 
    precise SQLite query for a data transformation task based on the following plan and information:

    ORIGINAL QUERY: "{original_query}"

    SQLite GENERATION PLAN:
    {plan}

    Use the given template to generate the SQLite Query:
    {template}


    CONTEXT INFORMATION:
    - Query Type: {query_type}
    - Source Tables: {source_tables}
    - Source Fields: {source_fields}
    - Target Table: {target_table}
    - Target Fields: {target_fields}
    - Filtering Fields: {filtering_fields}
    - Filtering Conditions: {json.dumps(conditions, indent=2)}
    - Key Mapping: {json.dumps(key_mapping, indent=2)}
    - Target Table Has Data: {target_has_data}

    IMPORTANT REQUIREMENTS:
    1. Generate ONLY standard SQLite SQL syntax (not MS SQL, MySQL, PostgreSQL, etc.)
    2. For all queries except validations, use DML operations (INSERT, UPDATE, etc.)
    3. If Target Table Has Data = True, use UPDATE operations with proper key matching
    4. If Target Table Has Data = False, use INSERT operations
    5. For validation queries only, use SELECT operations
    6. Always include WHERE clauses for all filter conditions using exact literal values
    7. If source fields are requested (like MATNR), make sure they are included in the query
    8. Properly handle key fields for matching records in UPDATE operations
    9. Return ONLY the final SQL query with no explanations or markdown formatting

    CRITICAL SQLite-SPECIFIC SYNTAX:
    - SQLite does not support RIGHT JOIN or FULL JOIN (use LEFT JOIN with table order swapped instead)
    - SQLite uses IFNULL instead of ISNULL for handling nulls
    - SQLite UPDATE with JOIN requires FROM clause (different from standard SQL)
    - SQLite has no BOOLEAN type (use INTEGER 0/1)
    - For UPDATE with data from another table, use: UPDATE target SET col = subquery.col FROM (SELECT...) AS subquery WHERE target.key = subquery.key

    EXAMPLES:
    - For "Bring Material Number with Material Type = ROH from MARA Table" with empty target table:
    INSERT INTO target_table (MATNR)
    SELECT MATNR FROM MARA
    WHERE MTART = 'ROH'

    - For "Bring Material Number with Material Type = ROH from MARA Table" with existing target data:
    UPDATE target_table
    SET MATNR = source.MATNR
    FROM (SELECT MATNR FROM MARA WHERE MTART = 'ROH') AS source
    WHERE target_table.MATNR = source.MATNR

    - For "Validate material numbers in MARA":
    SELECT MATNR,
    CASE WHEN MATNR IS NULL THEN 'Invalid' ELSE 'Valid' END AS validation_result
    FROM MARA
    """
        
            # Call the LLM for SQLite generation
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model="gemini-2.5-pro-preview-05-06", 
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            
            # Extract the SQLite query
            if response and hasattr(response, "text"):
                sql_query = response.text.strip()
                
                # Remove markdown code blocks if present
                import re
                sql_match = re.search(r"```(?:sqlite|sql)\s*(.*?)\s*```", sql_query, re.DOTALL)
                if sql_match:
                    sql_query = sql_match.group(1)
                else:
                    sql_match = re.search(r"```\s*(.*?)\s*```", sql_query, re.DOTALL)
                    if sql_match:
                        sql_query = sql_match.group(1)
                
                # Build parameter dict (empty for now as we're using literal values)
                params = {}
                
                return sql_query.strip(), params
            else:
                logger.warning("Invalid response from LLM in generate_sql_with_llm")
                
                # Generate fallback query
                if target_has_data and query_type != "VALIDATION_OPERATION":
                    # UPDATE operation - SQLite specific syntax
                    fallback = f"UPDATE {target_table} SET {target_fields[0]} = source.{source_fields[0]} FROM (SELECT {', '.join(source_fields)} FROM {source_tables[0]}) AS source WHERE {target_table}.{target_fields[0]} = source.{source_fields[0]}"
                elif query_type == "VALIDATION_OPERATION":
                    # SELECT for validation
                    fallback = f"SELECT * FROM {source_tables[0]}"
                else:
                    # INSERT operation
                    fallback = f"INSERT INTO {target_table} ({', '.join(target_fields)}) SELECT {', '.join(source_fields)} FROM {source_tables[0]}"
                    
                return fallback, {}
            
        except Exception as e:
            logger.error(f"Error in generate_sql_with_llm: {e}")
            return "SELECT * FROM " + str(source_tables[0] if source_tables else "unknown_table"), {}
    def _generate_simple_transformation(self, planner_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL for simple transformations
        
        Parameters:
        planner_info (Dict): Information extracted by the planner
        
        Returns:
        Tuple[str, Dict]: The generated SQL query and parameterized values
        """
        # Extract necessary information
        source_tables = planner_info.get("source_table_name", [])
        source_fields = planner_info.get("source_field_names", [])
        target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
        target_fields = planner_info.get("target_sap_fields", [])
        filtering_fields = planner_info.get("filtering_fields", [])
        conditions = planner_info.get("extracted_conditions", {})
        
        # Handle missing or empty values
        if not source_tables or not source_fields or not target_table or not target_fields:
            logger.error("Missing essential information for SQL generation")
            return "", {}
        
        # Use the first source table if multiple are provided
        source_table = source_tables[0]
        
        # Build parameterized values dict
        params = {}
        
        # Determine if this is an INSERT or UPDATE operation
        operation_type = self._determine_operation_type(planner_info)
        
        if operation_type == "INSERT":
            # Generate INSERT query
            return self._build_insert_query(source_table, target_table, source_fields, target_fields, 
                                          filtering_fields, conditions)
        else:
            # Generate UPDATE query
            return self._build_update_query(source_table, target_table, source_fields, target_fields,
                                          filtering_fields, conditions)
    
    def _generate_join_operation(self, planner_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL for join operations
        
        Parameters:
        planner_info (Dict): Information extracted by the planner
        
        Returns:
        Tuple[str, Dict]: The generated SQL query and parameterized values
        """
        # Extract necessary information
        source_tables = planner_info.get("source_table_name", [])
        source_fields = planner_info.get("source_field_names", [])
        target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
        target_fields = planner_info.get("target_sap_fields", [])
        join_conditions = planner_info.get("join_conditions", [])
        filtering_fields = planner_info.get("filtering_fields", [])
        conditions = planner_info.get("extracted_conditions", {})
        
        # Handle missing or empty values
        if not source_tables or len(source_tables) < 2 or not target_table:
            logger.error("Missing essential information for JOIN operation")
            return "", {}
        
        # Build tables and aliases
        main_table = source_tables[0]
        main_alias = f"t1"
        
        # Build join clauses
        join_clauses = []
        
        for i, table in enumerate(source_tables[1:], 2):
            alias = f"t{i}"
            
            # Find join condition for this table pair
            join_info = next((jc for jc in join_conditions if 
                             (jc.get("left_table") == main_table and jc.get("right_table") == table) or
                             (jc.get("right_table") == main_table and jc.get("left_table") == table)), 
                             None)
            
            if join_info:
                join_type = join_info.get("join_type", "INNER").upper()
                
                # Ensure join type is valid
                if join_type not in ["INNER", "LEFT", "RIGHT", "FULL"]:
                    join_type = "INNER"
                
                # Get join fields
                if join_info.get("left_table") == main_table:
                    left_field = join_info.get("left_field")
                    right_field = join_info.get("right_field")
                else:
                    left_field = join_info.get("right_field")
                    right_field = join_info.get("left_field")
                
                join_clauses.append(f"{join_type} JOIN {table} {alias} ON {main_alias}.{left_field} = {alias}.{right_field}")
            else:
                # Default join on common field names if no explicit join condition
                common_fields = self._find_common_fields(main_table, table)
                if common_fields:
                    join_field = common_fields[0]
                    join_clauses.append(f"INNER JOIN {table} {alias} ON {main_alias}.{join_field} = {alias}.{join_field}")
                else:
                    # Fallback to CROSS JOIN if no common fields
                    join_clauses.append(f"CROSS JOIN {table} {alias}")
        
        # Build field selection with appropriate table aliases
        select_fields = []
        field_mapping = {}
        
        for field in source_fields:
            # Determine which table this field belongs to
            for i, table in enumerate(source_tables):
                alias = f"t{i+1}"
                field_mapping[field] = f"{alias}.{field}"
                select_fields.append(f"{alias}.{field} AS {field}")
                break
        
        # Build WHERE clause
        where_clause, params = self._build_where_clause(filtering_fields, conditions, field_mapping)
        
        # Determine operation type
        operation_type = self._determine_operation_type(planner_info)
        
        if operation_type == "INSERT":
            # Generate query to insert join results into target
            query = self.sql_templates["insert"].format(
                target_table=target_table,
                target_fields=", ".join(target_fields),
                source_fields=", ".join(select_fields),
                source_table=f"{main_table} {main_alias} {' '.join(join_clauses)}",
                where_clause=where_clause
            )
        else:
            # Generate a common table expression (CTE) for the join, then update target
            join_query = f"""
            WITH joined_data AS (
                SELECT {', '.join(select_fields)}
                FROM {main_table} {main_alias}
                {' '.join(join_clauses)}
                {where_clause}
            )
            """
            
            # Determine key field for matching
            key_field = self._get_key_field(planner_info, target_fields, source_fields)
            
            if key_field:
                # Build UPDATE query with CTE
                set_clauses = []
                for field in target_fields:
                    if field != key_field and field in source_fields:
                        set_clauses.append(f"{field} = joined_data.{field}")
                
                if set_clauses:
                    query = f"""
                    {join_query}
                    UPDATE {target_table}
                    SET {', '.join(set_clauses)}
                    FROM joined_data
                    WHERE {target_table}.{key_field} = joined_data.{key_field}
                    """
                else:
                    # If no fields to update, fallback to a simple SELECT
                    query = f"""
                    SELECT {', '.join(select_fields)}
                    FROM {main_table} {main_alias}
                    {' '.join(join_clauses)}
                    {where_clause}
                    """
            else:
                # If no key field identified, fallback to a simple SELECT
                query = f"""
                SELECT {', '.join(select_fields)}
                FROM {main_table} {main_alias}
                {' '.join(join_clauses)}
                {where_clause}
                """
        
        return query, params
    
    def _generate_cross_segment(self, planner_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL for cross-segment operations
        
        Parameters:
        planner_info (Dict): Information extracted by the planner
        
        Returns:
        Tuple[str, Dict]: The generated SQL query and parameterized values
        """
        # Extract necessary information
        source_tables = planner_info.get("source_table_name", [])
        source_fields = planner_info.get("source_field_names", [])
        target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
        target_fields = planner_info.get("target_sap_fields", [])
        segment_references = planner_info.get("segment_references", [])
        filtering_fields = planner_info.get("filtering_fields", [])
        conditions = planner_info.get("extracted_conditions", {})
        
        # Handle missing or empty values
        if not source_tables or not target_table or not segment_references:
            logger.error("Missing essential information for CROSS_SEGMENT operation")
            return "", {}
        
        # For cross-segment, we need to create views for the segment data
        view_queries = []
        
        for segment_ref in segment_references:
            segment_id = segment_ref.get("segment_id")
            segment_name = segment_ref.get("segment_name")
            table_name = segment_ref.get("table_name")
            
            if table_name:
                # Create a view name from segment name
                view_name = f"view_{segment_name.lower().replace(' ', '_')}"
                
                # Create view query
                view_query = self.sql_templates["create_view"].format(
                    view_name=view_name,
                    select_fields="*",
                    source_table=table_name,
                    where_clause=""
                )
                
                view_queries.append(view_query)
        
        # Now create the main query as a JOIN operation
        # Treat segment tables as source tables
        join_tables = [t for t in source_tables]
        for segment_ref in segment_references:
            segment_name = segment_ref.get("segment_name")
            view_name = f"view_{segment_name.lower().replace(' ', '_')}"
            if view_name not in join_tables:
                join_tables.append(view_name)
        
        # Create a join query similar to _generate_join_operation
        main_table = join_tables[0]
        main_alias = f"t1"
        
        # Build join clauses
        join_clauses = []
        
        for i, table in enumerate(join_tables[1:], 2):
            alias = f"t{i}"
            
            # Try to find common fields for join
            common_fields = self._find_common_fields(main_table, table)
            if common_fields:
                join_field = common_fields[0]
                join_clauses.append(f"LEFT JOIN {table} {alias} ON {main_alias}.{join_field} = {alias}.{join_field}")
            else:
                # Fallback to CROSS JOIN if no common fields
                join_clauses.append(f"CROSS JOIN {table} {alias}")
        
        # Build field selection
        select_fields = []
        field_mapping = {}
        
        for field in source_fields:
            # Add all needed fields with table aliases
            select_fields.append(f"{main_alias}.{field} AS {field}")
            field_mapping[field] = f"{main_alias}.{field}"
        
        # Build WHERE clause
        where_clause, params = self._build_where_clause(filtering_fields, conditions, field_mapping)
        
        # Determine operation type
        operation_type = self._determine_operation_type(planner_info)
        
        # Combine view creation with main query
        if view_queries:
            views_sql = "\n".join(view_queries)
            
            if operation_type == "INSERT":
                # Generate query to insert join results into target
                query = f"""
                {views_sql}
                
                INSERT INTO {target_table} ({', '.join(target_fields)})
                SELECT {', '.join(select_fields)}
                FROM {main_table} {main_alias}
                {' '.join(join_clauses)}
                {where_clause}
                """
            else:
                # Generate UPDATE query with CTE
                query = f"""
                {views_sql}
                
                WITH joined_data AS (
                    SELECT {', '.join(select_fields)}
                    FROM {main_table} {main_alias}
                    {' '.join(join_clauses)}
                    {where_clause}
                )
                """
                
                # Determine key field for matching
                key_field = self._get_key_field(planner_info, target_fields, source_fields)
                
                if key_field:
                    # Build UPDATE query with CTE
                    set_clauses = []
                    for field in target_fields:
                        if field != key_field and field in source_fields:
                            set_clauses.append(f"{field} = joined_data.{field}")
                    
                    if set_clauses:
                        query += f"""
                        UPDATE {target_table}
                        SET {', '.join(set_clauses)}
                        FROM joined_data
                        WHERE {target_table}.{key_field} = joined_data.{key_field}
                        """
                    else:
                        # If no fields to update, just return the joined data
                        query = f"""
                        {views_sql}
                        
                        SELECT {', '.join(select_fields)}
                        FROM {main_table} {main_alias}
                        {' '.join(join_clauses)}
                        {where_clause}
                        """
                else:
                    # If no key field identified, just return the joined data
                    query = f"""
                    {views_sql}
                    
                    SELECT {', '.join(select_fields)}
                    FROM {main_table} {main_alias}
                    {' '.join(join_clauses)}
                    {where_clause}
                    """
        else:
            # No views needed, just build a regular query
            if operation_type == "INSERT":
                query = self.sql_templates["insert"].format(
                    target_table=target_table,
                    target_fields=", ".join(target_fields),
                    source_fields=", ".join(select_fields),
                    source_table=f"{main_table} {main_alias} {' '.join(join_clauses)}",
                    where_clause=where_clause
                )
            else:
                # Fallback to SELECT query
                query = f"""
                SELECT {', '.join(select_fields)}
                FROM {main_table} {main_alias}
                {' '.join(join_clauses)}
                {where_clause}
                """
        
        return query, params
    
    def _generate_validation_operation(self, planner_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL for validation operations
        
        Parameters:
        planner_info (Dict): Information extracted by the planner
        
        Returns:
        Tuple[str, Dict]: The generated SQL query and parameterized values
        """
        # Extract necessary information
        source_tables = planner_info.get("source_table_name", [])
        source_fields = planner_info.get("source_field_names", [])
        target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
        target_fields = planner_info.get("target_sap_fields", [])
        validation_rules = planner_info.get("validation_rules", [])
        
        # Handle missing or empty values
        if not source_tables or not target_table or not validation_rules:
            logger.error("Missing essential information for VALIDATION operation")
            return "", {}
        
        # Use the first source table if multiple are provided
        source_table = source_tables[0]
        
        # Build CASE expressions for each validation rule
        case_expressions = []
        params = {}
        
        for i, rule in enumerate(validation_rules):
            field = rule.get("field")
            rule_type = rule.get("rule_type")
            parameters = rule.get("parameters", {})
            
            if not field or not rule_type:
                continue
                
            case_expression = self._build_validation_case(field, rule_type, parameters, i, params)
            if case_expression:
                case_expressions.append(case_expression)
        
        # Create validation query
        select_fields = []
        
        # Add validation result fields
        for i, case_expr in enumerate(case_expressions):
            select_fields.append(f"{case_expr} AS validation_result_{i+1}")
        
        # Add original fields
        select_fields.extend([f"{field}" for field in source_fields])
        
        # Build the SQL query
        query = f"""
        SELECT {', '.join(select_fields)}
        FROM {source_table}
        """
        
        return query, params
    
    def _generate_aggregation_operation(self, planner_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL for aggregation operations
        
        Parameters:
        planner_info (Dict): Information extracted by the planner
        
        Returns:
        Tuple[str, Dict]: The generated SQL query and parameterized values
        """
        # Extract necessary information
        source_tables = planner_info.get("source_table_name", [])
        source_fields = planner_info.get("source_field_names", [])
        target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
        target_fields = planner_info.get("target_sap_fields", [])
        group_by_fields = planner_info.get("group_by_fields", [])
        aggregation_functions = planner_info.get("aggregation_functions", [])
        filtering_fields = planner_info.get("filtering_fields", [])
        conditions = planner_info.get("extracted_conditions", {})
        
        # Handle missing or empty values
        if not source_tables or not aggregation_functions:
            logger.error("Missing essential information for AGGREGATION operation")
            return "", {}
        
        # Use the first source table if multiple are provided
        source_table = source_tables[0]
        
        # Build the aggregation functions
        agg_expressions = []
        
        for agg in aggregation_functions:
            field = agg.get("field")
            function = agg.get("function", "").lower()
            alias = agg.get("alias")
            
            if not field or not function:
                continue
                
            # Map function name to SQL aggregation function
            if function == "sum":
                agg_expr = f"SUM({field})"
            elif function == "count":
                agg_expr = f"COUNT({field})"
            elif function == "avg":
                agg_expr = f"AVG({field})"
            elif function == "min":
                agg_expr = f"MIN({field})"
            elif function == "max":
                agg_expr = f"MAX({field})"
            else:
                # Unknown function, use as-is
                agg_expr = f"{function}({field})"
            
            # Add alias if provided
            if alias:
                agg_expr += f" AS {alias}"
            else:
                agg_expr += f" AS {function}_{field}"
            
            agg_expressions.append(agg_expr)
        
        # Build WHERE clause
        where_clause, params = self._build_where_clause(filtering_fields, conditions)
        
        # Generate the SQL query
        if group_by_fields:
            # With GROUP BY
            query = self.sql_templates["aggregation"].format(
                group_fields=", ".join(group_by_fields),
                agg_separator=", " if group_by_fields and agg_expressions else "",
                agg_functions=", ".join(agg_expressions),
                table=source_table,
                where_clause=where_clause
            )
        else:
            # Without GROUP BY (aggregate entire table)
            query = f"""
            SELECT {', '.join(agg_expressions)}
            FROM {source_table}
            {where_clause}
            """
        
        return query, params
    
    def _determine_operation_type(self, planner_info: Dict[str, Any]) -> str:
        """
        Determine if this should be an INSERT or UPDATE operation
        
        Parameters:
        planner_info (Dict): Information extracted by the planner
        
        Returns:
        str: "INSERT" or "UPDATE"
        """
        # Check if target is empty or if this is a first-time operation
        target_data_samples = planner_info.get("target_data_samples", {})
        
        if isinstance(target_data_samples, pd.DataFrame) and target_data_samples.empty:
            return "INSERT"
        
        # Check if the restructured query suggests an INSERT
        query_text = planner_info.get("restructured_query", "").lower()
        
        if any(term in query_text for term in ["insert", "add", "create", "new"]):
            return "INSERT"
        
        # Default to UPDATE for existing data
        return "UPDATE"
    
    def _build_where_clause(self, 
                           filtering_fields: List[str], 
                           conditions: Dict[str, Any],
                           field_mapping: Optional[Dict[str, str]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Build a WHERE clause from filtering fields and conditions
        
        Parameters:
        filtering_fields (List[str]): Fields used for filtering
        conditions (Dict[str, Any]): Conditions to apply
        field_mapping (Dict[str, str]): Optional mapping of fields to table-qualified names
        
        Returns:
        Tuple[str, Dict[str, Any]]: WHERE clause and parameters
        """
        if not filtering_fields or not conditions:
            return "", {}
        
        where_parts = []
        params = {}
        
        for i, field in enumerate(filtering_fields):
            if field in conditions:
                value = conditions[field]
                param_name = f"param_{i}"
                
                field_ref = field_mapping.get(field, field) if field_mapping else field
                
                # Handle different condition types
                if isinstance(value, list):
                    placeholders = [f":param_{i}_{j}" for j in range(len(value))]
                    where_parts.append(f"{field_ref} IN ({', '.join(placeholders)})")
                    
                    for j, val in enumerate(value):
                        params[f"param_{i}_{j}"] = val
                else:
                    where_parts.append(f"{field_ref} = :{param_name}")
                    params[param_name] = value
        
        if where_parts:
            return f"WHERE {' AND '.join(where_parts)}", params
        else:
            return "", {}
    
    def _build_insert_query(self, 
                           source_table: str, 
                           target_table: str,
                           source_fields: List[str], 
                           target_fields: List[str],
                           filtering_fields: List[str],
                           conditions: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Build an INSERT query
        
        Parameters:
        source_table (str): Source table name
        target_table (str): Target table name
        source_fields (List[str]): Fields to select from source
        target_fields (List[str]): Fields to insert into target
        filtering_fields (List[str]): Fields used for filtering
        conditions (Dict[str, Any]): Conditions to apply
        
        Returns:
        Tuple[str, Dict[str, Any]]: INSERT query and parameters
        """
        # Build field mappings
        field_mapping = {}
        select_fields = []
        
        for target_field in target_fields:
            # Find corresponding source field
            source_field = target_field  # Default to same name
            
            # Try to find in source fields
            if target_field in source_fields:
                field_mapping[target_field] = target_field
                select_fields.append(target_field)
            else:
                # Check if any source field might match
                potential_matches = [f for f in source_fields if 
                                     f.lower() == target_field.lower() or
                                     target_field.lower() in f.lower() or
                                     f.lower() in target_field.lower()]
                
                if potential_matches:
                    source_field = potential_matches[0]
                    field_mapping[target_field] = source_field
                    select_fields.append(f"{source_field} AS {target_field}")
                else:
                    # Use NULL as placeholder
                    select_fields.append(f"NULL AS {target_field}")
        
        # Build WHERE clause
        where_clause, params = self._build_where_clause(filtering_fields, conditions)
        
        # Build INSERT query
        query = self.sql_templates["insert"].format(
            target_table=target_table,
            target_fields=", ".join(target_fields),
            source_fields=", ".join(select_fields),
            source_table=source_table,
            where_clause=where_clause
        )
        
        return query, params
    
    def _build_update_query(self, 
                           source_table: str, 
                           target_table: str,
                           source_fields: List[str], 
                           target_fields: List[str],
                           filtering_fields: List[str],
                           conditions: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Build an UPDATE query
        
        Parameters:
        source_table (str): Source table name
        target_table (str): Target table name
        source_fields (List[str]): Fields to select from source
        target_fields (List[str]): Fields to update in target
        filtering_fields (List[str]): Fields used for filtering
        conditions (Dict[str, Any]): Conditions to apply
        
        Returns:
        Tuple[str, Dict[str, Any]]: UPDATE query and parameters
        """
        # Need to identify key field for joining source and target
        key_field = None
        
        # Try to find key field from source_fields that might be in target_fields
        for field in source_fields:
            if field in target_fields:
                key_field = field
                break
        
        if not key_field:
            logger.warning("No key field found for UPDATE operation, using first target field")
            key_field = target_fields[0] if target_fields else None
        
        if not key_field:
            logger.error("Cannot build UPDATE query without key field")
            return "", {}
        
        # Build the UPDATE with subquery approach
        set_clauses = []
        for field in target_fields:
            if field != key_field and field in source_fields:
                set_clauses.append(f"{field} = subquery.{field}")
        
        if not set_clauses:
            logger.warning("No fields to update in UPDATE query")
            return "", {}
        
        # Build WHERE clause for source table
        where_clause, params = self._build_where_clause(filtering_fields, conditions)
        
        # Build the UPDATE query
        query = f"""
        UPDATE {target_table}
        SET {', '.join(set_clauses)}
        FROM (
            SELECT {', '.join(source_fields)}
            FROM {source_table}
            {where_clause}
        ) AS subquery
        WHERE {target_table}.{key_field} = subquery.{key_field}
        """
        
        return query, params
    
    def _build_validation_case(self, 
                              field: str, 
                              rule_type: str, 
                              parameters: Dict[str, Any], 
                              rule_index: int,
                              params: Dict[str, Any]) -> str:
        """
        Build a CASE expression for a validation rule
        
        Parameters:
        field (str): Field to validate
        rule_type (str): Type of validation rule
        parameters (Dict): Parameters for the rule
        rule_index (int): Index of the rule for parameter naming
        params (Dict): Parameter dictionary to update
        
        Returns:
        str: CASE expression for validation
        """
        # Build a CASE expression based on rule type
        if rule_type == "not_null":
            return f"CASE WHEN {field} IS NULL THEN 'Invalid: Null value' ELSE 'Valid' END"
            
        elif rule_type == "unique":
            # Uniqueness is hard to check in a single query
            # Would need a correlated subquery or window function
            return f"CASE WHEN {field} IS NULL THEN 'Invalid: Null value' ELSE 'Valid (uniqueness to be verified)' END"
            
        elif rule_type == "range":
            min_val = parameters.get("min")
            max_val = parameters.get("max")
            
            min_param = f"min_{rule_index}"
            max_param = f"max_{rule_index}"
            
            conditions = []
            
            if min_val is not None:
                conditions.append(f"{field} < :{min_param}")
                params[min_param] = min_val
                
            if max_val is not None:
                conditions.append(f"{field} > :{max_param}")
                params[max_param] = max_val
                
            if conditions:
                return f"CASE WHEN {' OR '.join(conditions)} THEN 'Invalid: Out of range' ELSE 'Valid' END"
            else:
                return f"'Valid'"
                
        elif rule_type == "regex":
            pattern = parameters.get("pattern")
            pattern_param = f"pattern_{rule_index}"
            
            if pattern:
                params[pattern_param] = pattern
                # Note: Regex syntax varies by database engine
                if self.db_dialect == "sqlite":
                    return f"CASE WHEN {field} NOT REGEXP :{pattern_param} THEN 'Invalid: Pattern mismatch' ELSE 'Valid' END"
                else:
                    # Generic fallback
                    return f"CASE WHEN {field} NOT LIKE :{pattern_param} THEN 'Invalid: Pattern mismatch' ELSE 'Valid' END"
            else:
                return f"'Valid'"
                
        elif rule_type == "exists_in":
            ref_table = parameters.get("reference_table")
            ref_field = parameters.get("reference_field")
            
            if ref_table and ref_field:
                # This uses a correlated subquery
                return f"""CASE WHEN NOT EXISTS (
                    SELECT 1 FROM {ref_table} 
                    WHERE {ref_field} = {field}
                ) THEN 'Invalid: Reference not found' ELSE 'Valid' END"""
            else:
                return f"'Valid'"
        
        # Default case for unknown rule type
        return f"'Unknown validation rule: {rule_type}'"
    
    def _find_common_fields(self, table1: str, table2: str) -> List[str]:
        """
        Find common fields between two tables
        
        Parameters:
        table1 (str): First table name
        table2 (str): Second table name
        
        Returns:
        List[str]: List of common fields
        """
        # This is a placeholder implementation
        # In a real implementation, this would query the database schema
        
        # Common SAP key fields that might be used for joins
        common_key_fields = [
            "MATNR",  # Material Number
            "MANDT",  # Client
            "KUNNR",  # Customer Number
            "LIFNR",  # Vendor Number
            "WERKS",  # Plant
            "LGORT",  # Storage Location
            "BUKRS",  # Company Code
        ]
        
        # Return common fields based on SAP table naming conventions
        if table1 == "MARA" and table2 == "MAKT":
            return ["MATNR", "MANDT"]
        elif table1 == "MARA" and table2 == "MARC":
            return ["MATNR", "MANDT"]
        elif table1 == "MAKT" and table2 == "MARA":
            return ["MATNR", "MANDT"]
        elif table1 == "MARC" and table2 == "MARA":
            return ["MATNR", "MANDT"]
        
        # Default to common SAP key fields
        return common_key_fields
    
    def _get_key_field(self, planner_info: Dict[str, Any], target_fields: List[str], source_fields: List[str]) -> Optional[str]:
        """
        Get the key field for the operation
        
        Parameters:
        planner_info (Dict): Information extracted by the planner
        target_fields (List[str]): Target fields
        source_fields (List[str]): Source fields
        
        Returns:
        Optional[str]: Key field or None if not found
        """
        # Try to get key field from mapping
        key_mapping = planner_info.get("key_mapping", [])
        if key_mapping and isinstance(key_mapping, list):
            for mapping in key_mapping:
                if isinstance(mapping, dict) and "target_col" in mapping:
                    return mapping["target_col"]
                elif isinstance(mapping, str):
                    return mapping
        
        # Try to find a field that's in both target and source
        common_fields = [f for f in target_fields if f in source_fields]
        if common_fields:
            # Priority to fields that are likely keys based on naming
            key_indicators = ["MATNR", "ID", "KEY", "NR", "CODE", "NUM"]
            for indicator in key_indicators:
                for field in common_fields:
                    if indicator in field.upper():
                        return field
            
            # If no priority field found, return the first common field
            return common_fields[0]
        
        return None

    # def create_sql_plan(self, planner_info: Dict[str, Any]) -> str:
    #     """
    #     Use LLM to create a detailed plan for SQL query generation
        
    #     Parameters:
    #     planner_info (Dict): Information extracted by the planner
        
    #     Returns:
    #     str: Step-by-step SQL generation plan
    #     """
    #     try:
    #         # Extract key information for the prompt
    #         query_type = planner_info.get("query_type", "SIMPLE_TRANSFORMATION")
    #         source_tables = planner_info.get("source_table_name", [])
    #         source_fields = planner_info.get("source_field_names", [])
    #         target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
    #         target_fields = planner_info.get("target_sap_fields", [])
    #         filtering_fields = planner_info.get("filtering_fields", [])
    #         conditions = planner_info.get("extracted_conditions", {})
    #         original_query = planner_info.get("original_query", "")
            
    #         # Create a comprehensive prompt for the LLM
    #         prompt = f"""
    # You are an expert SQL database engineer. I need you to create a step-by-step plan to generate 
    # SQL for the following natural language query:

    # ORIGINAL QUERY: "{original_query}"

    # CONTEXT INFORMATION:
    # - Query Type: {query_type}
    # - Source Tables: {source_tables}
    # - Source Fields: {source_fields}
    # - Target Table: {target_table}
    # - Target Fields: {target_fields}
    # - Filtering Fields: {filtering_fields}
    # - Filtering Conditions: {json.dumps(conditions, indent=2)}

    # Based on the query intent, create a detailed plan for generating an efficient SQL query, 
    # structured as numbered steps. Carefully consider:

    # 1. The query type and appropriate SQL operation (SELECT, INSERT, UPDATE, etc.)
    # 2. Proper table references and aliases
    # 3. Precise column selections
    # 4. Correct filter conditions
    # 5. Join conditions (if applicable)
    # 6. Proper ordering of operations

    # Format your response as a numbered list only, with no explanations or additional text.
    # Each step should be clear, concise, and directly actionable for SQL generation.
    # """
            
    #         # Call the LLM for planning
    #         client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    #         response = client.models.generate_content(
    #             model="gemini-2.5-flash-preview-04-17", 
    #             contents=prompt,
    #             config=types.GenerateContentConfig(temperature=0.2)
    #         )
            
    #         # Extract and return the plan
    #         if response and hasattr(response, "text"):
    #             return response.text.strip()
    #         else:
    #             logger.warning("Invalid response from LLM in create_sql_plan")
    #             return "1. Generate basic SQL query based on query type\n2. Return query"
                
    #     except Exception as e:
    #         logger.error(f"Error in create_sql_plan: {e}")
    #         return "1. Generate basic SQL query based on query type\n2. Return query"
        
    # def generate_sql_with_llm(self, plan: str, planner_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    #     """
    #     Generate SQLite query using LLM based on the plan
        
    #     Parameters:
    #     plan (str): Step-by-step plan for SQLite generation
    #     planner_info (Dict): Information extracted by the planner
        
    #     Returns:
    #     Tuple[str, Dict[str, Any]]: The generated SQLite query and parameters
    #     """
    #     try:
    #         # Extract key information for the prompt
    #         query_type = planner_info.get("query_type", "SIMPLE_TRANSFORMATION")
    #         source_tables = planner_info.get("source_table_name", [])
    #         source_fields = planner_info.get("source_field_names", [])
    #         target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
    #         target_fields = planner_info.get("target_sap_fields", [])
    #         filtering_fields = planner_info.get("filtering_fields", [])
    #         conditions = planner_info.get("extracted_conditions", {})
    #         original_query = planner_info.get("original_query", "")
    #         key_mapping = planner_info.get("key_mapping", [])
            
    #         # Check if target table has data
    #         target_has_data = False
    #         target_data_samples = planner_info.get("target_data_samples", {})
    #         if isinstance(target_data_samples, pd.DataFrame) and not target_data_samples.empty:
    #             target_has_data = True
            
    #         # Create detailed SQLite generation prompt
    #         prompt = f"""
    # You are an expert SQLite database engineer focusing on data transformation operations. I need you to generate 
    # precise SQLite query for a data transformation task based on the following plan and information:

    # ORIGINAL QUERY: "{original_query}"

    # SQLite GENERATION PLAN:
    # {plan}

    # CONTEXT INFORMATION:
    # - Query Type: {query_type}
    # - Source Tables: {source_tables}
    # - Source Fields: {source_fields}
    # - Target Table: {target_table}
    # - Target Fields: {target_fields}
    # - Filtering Fields: {filtering_fields}
    # - Filtering Conditions: {json.dumps(conditions, indent=2)}
    # - Key Mapping: {json.dumps(key_mapping, indent=2)}
    # - Target Table Has Data: {target_has_data}

    # IMPORTANT REQUIREMENTS:
    # 1. Generate ONLY standard SQLite SQL syntax (not MS SQL, MySQL, PostgreSQL, etc.)
    # 2. For all queries except validations, use DML operations (INSERT, UPDATE, etc.)
    # 3. If Target Table Has Data = True, use UPDATE operations with proper key matching
    # 4. If Target Table Has Data = False, use INSERT operations
    # 5. For validation queries only, use SELECT operations
    # 6. Always include WHERE clauses for all filter conditions using exact literal values
    # 7. If source fields are requested (like MATNR), make sure they are included in the query
    # 8. Properly handle key fields for matching records in UPDATE operations
    # 9. Return ONLY the final SQL query with no explanations or markdown formatting

    # CRITICAL SQLite-SPECIFIC SYNTAX:
    # - SQLite does not support RIGHT JOIN or FULL JOIN (use LEFT JOIN with table order swapped instead)
    # - SQLite uses IFNULL instead of ISNULL for handling nulls
    # - SQLite UPDATE with JOIN requires FROM clause (different from standard SQL)
    # - SQLite has no BOOLEAN type (use INTEGER 0/1)
    # - For UPDATE with data from another table, use: UPDATE target SET col = subquery.col FROM (SELECT...) AS subquery WHERE target.key = subquery.key

    # EXAMPLES:
    # - For "Bring Material Number with Material Type = ROH from MARA Table" with empty target table:
    # INSERT INTO target_table (MATNR)
    # SELECT MATNR FROM MARA
    # WHERE MTART = 'ROH'

    # - For "Bring Material Number with Material Type = ROH from MARA Table" with existing target data:
    # UPDATE target_table
    # SET MATNR = source.MATNR
    # FROM (SELECT MATNR FROM MARA WHERE MTART = 'ROH') AS source
    # WHERE target_table.MATNR = source.MATNR

    # - For "Validate material numbers in MARA":
    # SELECT MATNR,
    # CASE WHEN MATNR IS NULL THEN 'Invalid' ELSE 'Valid' END AS validation_result
    # FROM MARA
    # """
        
    #         # Call the LLM for SQLite generation
    #         client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    #         response = client.models.generate_content(
    #             model="gemini-2.5-flash-preview-04-17", 
    #             contents=prompt,
    #             config=types.GenerateContentConfig(temperature=0.1)
    #         )
            
    #         # Extract the SQLite query
    #         if response and hasattr(response, "text"):
    #             sql_query = response.text.strip()
                
    #             # Remove markdown code blocks if present
    #             import re
    #             sql_match = re.search(r"```(?:sqlite|sql)\s*(.*?)\s*```", sql_query, re.DOTALL)
    #             if sql_match:
    #                 sql_query = sql_match.group(1)
    #             else:
    #                 sql_match = re.search(r"```\s*(.*?)\s*```", sql_query, re.DOTALL)
    #                 if sql_match:
    #                     sql_query = sql_match.group(1)
                
    #             # Build parameter dict (empty for now as we're using literal values)
    #             params = {}
                
    #             return sql_query.strip(), params
    #         else:
    #             logger.warning("Invalid response from LLM in generate_sql_with_llm")
                
    #             # Generate fallback query
    #             if target_has_data and query_type != "VALIDATION_OPERATION":
    #                 # UPDATE operation - SQLite specific syntax
    #                 fallback = f"UPDATE {target_table} SET {target_fields[0]} = source.{source_fields[0]} FROM (SELECT {', '.join(source_fields)} FROM {source_tables[0]}) AS source WHERE {target_table}.{target_fields[0]} = source.{source_fields[0]}"
    #             elif query_type == "VALIDATION_OPERATION":
    #                 # SELECT for validation
    #                 fallback = f"SELECT * FROM {source_tables[0]}"
    #             else:
    #                 # INSERT operation
    #                 fallback = f"INSERT INTO {target_table} ({', '.join(target_fields)}) SELECT {', '.join(source_fields)} FROM {source_tables[0]}"
                    
    #             return fallback, {}
            
    #     except Exception as e:
    #         logger.error(f"Error in generate_sql_with_llm: {e}")
    #         return "SELECT * FROM " + str(source_tables[0] if source_tables else "unknown_table"), {}

    def analyze_and_fix_query(self, sql_query, sql_params, planner_info, max_attempts=3):
        """
        Analyze a SQL query for SQLite compatibility issues and make multiple attempts to fix it
        
        Parameters:
        sql_query (str): The initially generated SQL query
        sql_params (dict): Parameters for the query
        planner_info (dict): Planner information for context
        max_attempts (int): Maximum number of fixing attempts
        
        Returns:
        tuple: (fixed_query, params, success_status)
        """
        try:
            # If the query is already valid, return it
            if self._is_valid_sqlite_query(sql_query):
                return sql_query, sql_params, True
                
            
            # First, analyze the query for issues
            analysis = self._analyze_sqlite_query(sql_query, planner_info)
            
            best_query = sql_query
            best_params = sql_params
            
            # Make up to max_attempts to fix the query
            for attempt in range(max_attempts):
                
                # Generate fixed query based on analysis and previous attempts
                fixed_query, fixed_params = self._fix_sqlite_query(
                    best_query, 
                    sql_params, 
                    planner_info, 
                    analysis, 
                    attempt
                )
                
                # If the fixed query is valid, return it
                if self._is_valid_sqlite_query(fixed_query):
                    return fixed_query, fixed_params, True
                    
                # Check if this fixed query is better than the previous best
                if self._compare_query_quality(fixed_query, best_query, planner_info):
                    best_query = fixed_query
                    best_params = fixed_params
                    
                # Update analysis for next attempt
                if attempt < max_attempts - 1:
                    analysis = self._analyze_sqlite_query(fixed_query, planner_info)
                    
            # Return the best query we found, even if it's not perfect
            logger.warning(f"Could not generate a perfectly valid query after {max_attempts} attempts")
            return best_query, best_params, False
                
        except Exception as e:
            logger.error(f"Error in analyze_and_fix_query: {e}")
            return sql_query, sql_params, False
            
    def _analyze_sqlite_query(self, sql_query, planner_info):
        """
        Analyze a SQL query for SQLite compatibility issues
        
        Parameters:
        sql_query (str): The SQL query to analyze
        planner_info (dict): Planner information for context
        
        Returns:
        str: Analysis of the query issues
        """
        try:
            # Create comprehensive prompt for analysis
            prompt = f"""
    You are an expert SQLite database engineer. Analyze the following SQL query for SQLite compatibility issues and other problems.

    SQL QUERY:
    {sql_query}

    CONTEXT INFORMATION:
    - Query Type: {planner_info.get("query_type", "SIMPLE_TRANSFORMATION")}
    - Source Tables: {planner_info.get("source_table_name", [])}
    - Source Fields: {planner_info.get("source_field_names", [])}
    - Target Table: {planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None}
    - Target Fields: {planner_info.get("target_sap_fields", [])}

    INSTRUCTIONS:
    1. Analyze for SQLite compatibility issues
    2. Check for syntax errors
    3. Check for logical errors
    4. Check for potential performance issues
    5. Verify table and column references
    6. Verify join conditions if present
    7. Verify subqueries if present
    8. Check for proper handling of NULL values

    Your analysis should be in a structured format with clear categories of issues.
    """

            # Call the LLM for analysis
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17", 
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            
            # Extract and return the analysis
            if response and hasattr(response, "text"):
                return response.text.strip()
            else:
                return "Failed to analyze query"
                
        except Exception as e:
            logger.error(f"Error in _analyze_sqlite_query: {e}")
            return f"Error analyzing query: {e}"
            
    def _fix_sqlite_query(self, sql_query, sql_params, planner_info, analysis, attempt_number):
        """
        Fix a SQL query based on analysis
        
        Parameters:
        sql_query (str): The SQL query to fix
        sql_params (dict): Parameters for the query
        planner_info (dict): Planner information for context
        analysis (str): Analysis of query issues
        attempt_number (int): Current attempt number (0-based)
        
        Returns:
        tuple: (fixed_query, fixed_params)
        """
        try:
            # Create comprehensive prompt for fixing
            prompt = f"""
    You are an expert SQLite database engineer. Fix the following SQL query based on the analysis.

    ORIGINAL SQL QUERY:
    {sql_query}

    ANALYSIS OF ISSUES:
    {analysis}

    FIX ATTEMPT: {attempt_number + 1}

    CONTEXT INFORMATION:
    - Query Type: {planner_info.get("query_type", "SIMPLE_TRANSFORMATION")}
    - Source Tables: {planner_info.get("source_table_name", [])}
    - Source Fields: {planner_info.get("source_field_names", [])}
    - Target Table: {planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None}
    - Target Fields: {planner_info.get("target_sap_fields", [])}
    - Filtering Fields: {planner_info.get("filtering_fields", [])}
    - Filtering Conditions: {json.dumps(planner_info.get("extracted_conditions", {}), indent=2)}

    INSTRUCTIONS:
    1. IMPORTANT: Only generate standard SQLite SQL syntax
    2. Fix all identified issues in the analysis
    3. Maintain the original query intent
    4. Ensure proper table and column references
    5. Ensure proper join syntax if needed
    6. Ensure proper handling of parameters
    7. Pay special attention to SQLite-specific syntax (different from other SQL dialects)
    8. Target Table Has Data: {isinstance(planner_info.get("target_data_samples", {}), pd.DataFrame) and not planner_info.get("target_data_samples", {}).empty}

    REQUIREMENTS:
    1. Return ONLY the fixed SQL query, with no explanations or markdown
    2. Ensure the query is valid SQLite syntax
    3. If a parameter should be used, keep the same parameter format
    4. Ensure the generated SQL meets the requirements of the query type
    5. Be especially careful with SQLite-specific features:
    - SQLite uses IFNULL not ISNULL
    - SQLite does not support RIGHT JOIN or FULL JOIN
    - SQLite has limited support for common table expressions
    - SQLite UPDATE with JOIN requires a specific syntax
    - SQLite has no BOOLEAN type (use INTEGER 0/1)
    """

            # Call the LLM for fixing
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17", 
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            
            # Extract the fixed query
            if response and hasattr(response, "text"):
                fixed_query = response.text.strip()
                
                # Remove markdown code blocks if present
                import re
                sql_match = re.search(r"```(?:sqlite|sql)\s*(.*?)\s*```", fixed_query, re.DOTALL)
                if sql_match:
                    fixed_query = sql_match.group(1)
                else:
                    sql_match = re.search(r"```\s*(.*?)\s*```", fixed_query, re.DOTALL)
                    if sql_match:
                        fixed_query = sql_match.group(1)
                
                return fixed_query.strip(), sql_params
            else:
                return sql_query, sql_params
                
        except Exception as e:
            logger.error(f"Error in _fix_sqlite_query: {e}")
            return sql_query, sql_params
            
    def _is_valid_sqlite_query(self, sql_query):
        """
        Check if a SQL query is valid SQLite syntax
        
        Parameters:
        sql_query (str): The SQL query to check
        
        Returns:
        bool: True if valid, False otherwise
        """
        try:
            # Perform basic validation
            if not sql_query or len(sql_query) < 10:
                return False
                
            # Check for basic SQL keywords
            sql_upper = sql_query.upper()
            has_select = "SELECT" in sql_upper
            has_insert = "INSERT" in sql_upper
            has_update = "UPDATE" in sql_upper
            has_create = "CREATE" in sql_upper
            
            if not (has_select or has_insert or has_update or has_create):
                return False
                
            # Check for obvious SQLite incompatible syntax
            if "RIGHT JOIN" in sql_upper or "FULL JOIN" in sql_upper:
                return False
                
            if "ISNULL" in sql_upper:  # Should be IFNULL in SQLite
                return False
                
            # More advanced checks could be added
            
            return True
        except Exception as e:
            logger.error(f"Error in _is_valid_sqlite_query: {e}")
            return False
            
    def _compare_query_quality(self, new_query, old_query, planner_info):
        """
        Compare quality of two queries to determine which is better
        
        Parameters:
        new_query (str): New query to evaluate
        old_query (str): Current best query
        planner_info (dict): Planner information for context
        
        Returns:
        bool: True if new query is better, False otherwise
        """
        try:
            # Basic validation
            if not new_query:
                return False
            
            if not old_query:
                return True
                
            # If old query isn't valid but new one is
            if not self._is_valid_sqlite_query(old_query) and self._is_valid_sqlite_query(new_query):
                return True
                
            # Check for improvement in including required fields
            source_fields = planner_info.get("source_field_names", [])
            target_fields = planner_info.get("target_sap_fields", [])
            
            new_field_count = sum(1 for field in source_fields if field in new_query)
            old_field_count = sum(1 for field in source_fields if field in old_query)
            
            new_target_count = sum(1 for field in target_fields if field in new_query)
            old_target_count = sum(1 for field in target_fields if field in old_query)
            
            # If new query includes more source fields
            if new_field_count > old_field_count:
                return True
                
            # If new query includes more target fields
            if new_target_count > old_target_count:
                return True
                
            # Check for specific SQLite patterns
            sqlite_patterns = ["IFNULL", "CASE WHEN", "COALESCE", "GROUP BY", "LEFT JOIN"]
            sqlite_score_new = sum(1 for pattern in sqlite_patterns if pattern.upper() in new_query.upper())
            sqlite_score_old = sum(1 for pattern in sqlite_patterns if pattern.upper() in old_query.upper())
            
            if sqlite_score_new > sqlite_score_old:
                return True
                
            # Default to keeping the old query
            return False
        except Exception as e:
            logger.error(f"Error in _compare_query_quality: {e}")
            return False