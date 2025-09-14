import logging
import json
import re
import sqlite3
import pandas as pd
import traceback
import os
from typing import Dict, List, Any, Optional, Union, Tuple

from DMtool.query_analyzer import SQLiteQueryAnalyzer
from DMtool.llm_config import LLMManager

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

            "select": """
                SELECT {select_fields}
                FROM {table}
                {where_clause}
            """,
            

            "join": """
                SELECT {select_fields}
                FROM {main_table} {main_alias}
                {join_clauses}
                {where_clause}
            """,
            

            "insert": """
                INSERT INTO {target_table} ({target_fields})
                SELECT {source_fields}
                FROM {source_table}
                {where_clause}
            """,
            

            "update": """
                UPDATE {target_table}
                SET {set_clause}
                {where_clause}
            """,
            

            "create_view": """
                CREATE TEMPORARY VIEW IF NOT EXISTS {view_name} AS
                SELECT {select_fields}
                FROM {source_table}
                {where_clause}
            """,
            

            "aggregation": """
                SELECT {group_fields}{agg_separator}{agg_functions}
                FROM {table}
                {where_clause}
                GROUP BY {group_fields}
            """
        }
        return templates
    
    def generate_sql(self, planner_info: Dict[str, Any],template: Dict[str, str]=None, sql_plan: str=None) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL query based on planner information using LLM for planning and generation
        
        Parameters:
        planner_info (Dict): Information extracted by the planner
        
        Returns:
        Tuple[str, Dict]: The generated SQL query and parameterized values
        """
        try:

            logger.info(f"Generating SQLite query for query: {planner_info.get('original_query', '')}")
            logger.info(f"Source tables: {planner_info.get('source_table_name', [])}")
            logger.info(f"Insertion fields: {planner_info.get('insertion_fields', [])}")
            logger.info(f"Target table: {planner_info.get('target_table_name', [])}")
            logger.info(f"Target fields: {planner_info.get('target_sap_fields', [])}")
            logger.info(f"Filtering fields: {planner_info.get('filtering_fields', [])}")
            logger.info(f"Conditions: {planner_info.get('extracted_conditions', {})}")
            

            

            initial_sql_query, initial_sql_params = self.generate_sql_with_llm(sql_plan, planner_info,template["query"])
            

            query_analyzer = SQLiteQueryAnalyzer()
            fixed_sql_query, fixed_sql_params, is_valid = query_analyzer.analyze_and_fix_query(
                initial_sql_query, initial_sql_params, planner_info
            )
            
            if is_valid:
                return fixed_sql_query, fixed_sql_params
            

            logger.warning(f"Could not generate valid SQLite query even after fixing attempts: {fixed_sql_query}")
            logger.info("Falling back to rule-based query generation")
                    

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

            query_type = planner_info.get("query_type", "SIMPLE_TRANSFORMATION")
            source_tables = planner_info.get("source_table_name", [])
            insertion_fields = planner_info.get("insertion_fields", [])
            target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
            target_fields = planner_info.get("target_sap_fields", [])
            filtering_fields = planner_info.get("filtering_fields", [])
            conditions = planner_info.get("extracted_conditions", {})
            original_query = planner_info.get("original_query", "")
            key_mapping = planner_info.get("key_mapping", [])
            

            target_has_data = False
            target_data_samples = planner_info.get("target_data_samples", {})
            if isinstance(target_data_samples, pd.DataFrame) and not target_data_samples.empty:
                target_has_data = True
            

            prompt = f"""
    You are an expert SQLite database engineer focusing on data transformation operations. I need you to generate 
    precise SQLite query for a data transformation task based on the following information:

    ORIGINAL QUERY: "{planner_info.get("restructured_query", original_query)}"

    {f"""SQLite GENERATION PLAN:
    {plan}""" if plan else "Follow the original query in detail and do not skip any steps and any filtering conditions."}

    Use the given template to generate the SQLite Query:
    {template}

    CONTEXT INFORMATION:
    - Query Type: {query_type}
    - Source Tables: {source_tables}
    - Insertion Fields: {insertion_fields}  
    - Target Table: {target_table}
    - Target Fields: {target_fields}
    - Filtering Fields: {filtering_fields}
    - Filtering Conditions: {json.dumps(conditions, indent=2)}
    - Key Mapping: {json.dumps(key_mapping, indent=2)}
    - Target Table Has Data: {target_has_data}

    CRITICAL FIELD USAGE RULES:
    1. Insertion Fields ({insertion_fields}) are the ONLY fields that should appear in:
       - INSERT INTO field lists
       - UPDATE SET clauses
       - SELECT clauses for data retrieval operations
    2. Filtering Fields ({filtering_fields}) should ONLY appear in:
       - WHERE clauses
       - JOIN conditions for filtering
       - HAVING clauses
    3. NEVER mix insertion fields with filtering fields in INSERT/UPDATE operations
    4. If a field appears in both lists, use it according to the operation context

    IMPORTANT REQUIREMENTS:
    1. Generate ONLY standard SQLite SQL syntax (not MS SQL, MySQL, PostgreSQL, etc.)
    2. For all queries except validations, use DML operations (INSERT, UPDATE, etc.)
    3. If Target Table Has Data = True, use UPDATE operations with proper key matching
    4. If Target Table Has Data = False, use INSERT operations
    5. For validation queries only, use SELECT operations
    6. Always include WHERE clauses for all filter conditions using exact literal values
    7. If insertion fields are requested, make sure ONLY they are included in the INSERT/UPDATE
    8. Properly handle key fields for matching records in UPDATE operations
    9. Return ONLY the final SQL query with no explanations or markdown formatting
    10. Follow the plan step-by-step and do not skip any steps
    11. Do not skip any filtering conditions
    12. 

    CRITICAL SQLite-SPECIFIC SYNTAX:
    - SQLite does not support RIGHT JOIN or FULL JOIN (use LEFT JOIN with table order swapped instead)
    - SQLite uses IFNULL instead of ISNULL for handling nulls
    - SQLite UPDATE with JOIN requires FROM clause (different from standard SQL)
    - SQLite has no BOOLEAN type (use INTEGER 0/1)
    - For UPDATE with data from another table, use: UPDATE target SET col = subquery.col FROM (SELECT...) AS subquery WHERE target.key = subquery.key
    """
        

            llm= LLMManager(
                provider="google",
                model="gemini/gemini-2.5-flash",
                api_key=os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
            )
            
            response = llm.generate(prompt, temperature=0.05, max_tokens=500)

            if response :
                sql_query = response.strip()
                

                import re
                sql_match = re.search(r"```(?:sqlite|sql)\s*(.*?)\s*```", sql_query, re.DOTALL)
                if sql_match:
                    sql_query = sql_match.group(1)
                else:
                    sql_match = re.search(r"```\s*(.*?)\s*```", sql_query, re.DOTALL)
                    if sql_match:
                        sql_query = sql_match.group(1)
                

                params = {}
                
                return sql_query.strip(), params
            else:
                logger.warning("Invalid response from LLM in generate_sql_with_llm")
                

                if target_has_data and query_type != "VALIDATION_OPERATION":

                    if insertion_fields and target_fields:
                        fallback = f"UPDATE {target_table} SET {target_fields[0]} = source.{insertion_fields[0]} FROM (SELECT {', '.join(insertion_fields)} FROM {source_tables[0]}) AS source WHERE {target_table}.{target_fields[0]} = source.{insertion_fields[0]}"
                    else:
                        fallback = f"UPDATE {target_table} SET field = 'value'"
                elif query_type == "VALIDATION_OPERATION":

                    fallback = f"SELECT * FROM {source_tables[0]}"
                else:

                    if insertion_fields and target_fields:
                        fallback = f"INSERT INTO {target_table} ({', '.join(target_fields)}) SELECT {', '.join(insertion_fields)} FROM {source_tables[0]}"
                    else:
                        fallback = f"SELECT * FROM {source_tables[0]}"
                    
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

        source_tables = planner_info.get("source_table_name", [])
        insertion_fields = planner_info.get("insertion_fields", [])
        target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
        target_fields = planner_info.get("target_sap_fields", [])
        filtering_fields = planner_info.get("filtering_fields", [])
        conditions = planner_info.get("extracted_conditions", {})
        

        if not source_tables or not insertion_fields or not target_table or not target_fields:
            logger.error("Missing essential information for SQL generation")
            logger.error(f"source_tables: {source_tables}, insertion_fields: {insertion_fields}, target_table: {target_table}, target_fields: {target_fields}")
            return "", {}
        

        source_table = source_tables[0]
        

        params = {}
        

        operation_type = self._determine_operation_type(planner_info)
        
        if operation_type == "INSERT":

            return self._build_insert_query(source_table, target_table, insertion_fields, target_fields, 
                                          filtering_fields, conditions)
        else:

            return self._build_update_query(source_table, target_table, insertion_fields, target_fields,
                                          filtering_fields, conditions)
    
    def _generate_join_operation(self, planner_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL for join operations
        
        Parameters:
        planner_info (Dict): Information extracted by the planner
        
        Returns:
        Tuple[str, Dict]: The generated SQL query and parameterized values
        """

        source_tables = planner_info.get("source_table_name", [])
        insertion_fields = planner_info.get("insertion_fields", [])
        target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
        target_fields = planner_info.get("target_sap_fields", [])
        join_conditions = planner_info.get("join_conditions", [])
        filtering_fields = planner_info.get("filtering_fields", [])
        conditions = planner_info.get("extracted_conditions", {})
        

        if not source_tables or len(source_tables) < 2 or not target_table:
            logger.error("Missing essential information for JOIN operation")
            return "", {}
        

        main_table = source_tables[0]
        main_alias = f"t1"
        

        join_clauses = []
        
        for i, table in enumerate(source_tables[1:], 2):
            alias = f"t{i}"
            

            join_info = next((jc for jc in join_conditions if 
                             (jc.get("left_table") == main_table and jc.get("right_table") == table) or
                             (jc.get("right_table") == main_table and jc.get("left_table") == table)), 
                             None)
            
            if join_info:
                join_type = join_info.get("join_type", "INNER").upper()
                

                if join_type not in ["INNER", "LEFT", "RIGHT", "FULL"]:
                    join_type = "INNER"
                

                if join_info.get("left_table") == main_table:
                    left_field = join_info.get("left_field")
                    right_field = join_info.get("right_field")
                else:
                    left_field = join_info.get("right_field")
                    right_field = join_info.get("left_field")
                
                join_clauses.append(f"{join_type} JOIN {table} {alias} ON {main_alias}.{left_field} = {alias}.{right_field}")
            else:

                common_fields = self._find_common_fields(main_table, table)
                if common_fields:
                    join_field = common_fields[0]
                    join_clauses.append(f"INNER JOIN {table} {alias} ON {main_alias}.{join_field} = {alias}.{join_field}")
                else:

                    join_clauses.append(f"CROSS JOIN {table} {alias}")
        

        select_fields = []
        field_mapping = {}
        
        for field in insertion_fields:

            for i, table in enumerate(source_tables):
                alias = f"t{i+1}"
                field_mapping[field] = f"{alias}.{field}"
                select_fields.append(f"{alias}.{field} AS {field}")
                break
        

        where_clause, params = self._build_where_clause(filtering_fields, conditions, field_mapping)
        

        operation_type = self._determine_operation_type(planner_info)
        
        if operation_type == "INSERT":

            query = self.sql_templates["insert"].format(
                target_table=target_table,
                target_fields=", ".join(target_fields),
                source_fields=", ".join(select_fields),
                source_table=f"{main_table} {main_alias} {' '.join(join_clauses)}",
                where_clause=where_clause
            )
        else:

            join_query = f"""
            WITH joined_data AS (
                SELECT {', '.join(select_fields)}
                FROM {main_table} {main_alias}
                {' '.join(join_clauses)}
                {where_clause}
            )
            """
            

            key_field = self._get_key_field(planner_info, target_fields, insertion_fields)
            
            if key_field:

                set_clauses = []
                for field in target_fields:
                    if field != key_field and field in insertion_fields:
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

                    query = f"""
                    SELECT {', '.join(select_fields)}
                    FROM {main_table} {main_alias}
                    {' '.join(join_clauses)}
                    {where_clause}
                    """
            else:

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

        source_tables = planner_info.get("source_table_name", [])
        insertion_fields = planner_info.get("insertion_fields", [])
        target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
        target_fields = planner_info.get("target_sap_fields", [])
        segment_references = planner_info.get("segment_references", [])
        filtering_fields = planner_info.get("filtering_fields", [])
        conditions = planner_info.get("extracted_conditions", {})
        

        if not source_tables or not target_table or not segment_references:
            logger.error("Missing essential information for CROSS_SEGMENT operation")
            return "", {}
        

        view_queries = []
        
        for segment_ref in segment_references:
            segment_id = segment_ref.get("segment_id")
            segment_name = segment_ref.get("segment_name")
            table_name = segment_ref.get("table_name")
            
            if table_name:

                view_name = f"view_{segment_name.lower().replace(' ', '_')}"
                

                view_query = self.sql_templates["create_view"].format(
                    view_name=view_name,
                    select_fields="*",
                    source_table=table_name,
                    where_clause=""
                )
                
                view_queries.append(view_query)
        


        join_tables = [t for t in source_tables]
        for segment_ref in segment_references:
            segment_name = segment_ref.get("segment_name")
            view_name = f"view_{segment_name.lower().replace(' ', '_')}"
            if view_name not in join_tables:
                join_tables.append(view_name)
        

        main_table = join_tables[0]
        main_alias = f"t1"
        

        join_clauses = []
        
        for i, table in enumerate(join_tables[1:], 2):
            alias = f"t{i}"
            

            common_fields = self._find_common_fields(main_table, table)
            if common_fields:
                join_field = common_fields[0]
                join_clauses.append(f"LEFT JOIN {table} {alias} ON {main_alias}.{join_field} = {alias}.{join_field}")
            else:

                join_clauses.append(f"CROSS JOIN {table} {alias}")
        

        select_fields = []
        field_mapping = {}
        
        for field in insertion_fields:

            select_fields.append(f"{main_alias}.{field} AS {field}")
            field_mapping[field] = f"{main_alias}.{field}"
        

        where_clause, params = self._build_where_clause(filtering_fields, conditions, field_mapping)
        

        operation_type = self._determine_operation_type(planner_info)
        

        if view_queries:
            views_sql = "\n".join(view_queries)
            
            if operation_type == "INSERT":

                query = f"""
                {views_sql}
                
                INSERT INTO {target_table} ({', '.join(target_fields)})
                SELECT {', '.join(select_fields)}
                FROM {main_table} {main_alias}
                {' '.join(join_clauses)}
                {where_clause}
                """
            else:

                query = f"""
                {views_sql}
                
                WITH joined_data AS (
                    SELECT {', '.join(select_fields)}
                    FROM {main_table} {main_alias}
                    {' '.join(join_clauses)}
                    {where_clause}
                )
                """
                

                key_field = self._get_key_field(planner_info, target_fields, insertion_fields)
                
                if key_field:

                    set_clauses = []
                    for field in target_fields:
                        if field != key_field and field in insertion_fields:
                            set_clauses.append(f"{field} = joined_data.{field}")
                    
                    if set_clauses:
                        query += f"""
                        UPDATE {target_table}
                        SET {', '.join(set_clauses)}
                        FROM joined_data
                        WHERE {target_table}.{key_field} = joined_data.{key_field}
                        """
                    else:

                        query = f"""
                        {views_sql}
                        
                        SELECT {', '.join(select_fields)}
                        FROM {main_table} {main_alias}
                        {' '.join(join_clauses)}
                        {where_clause}
                        """
                else:

                    query = f"""
                    {views_sql}
                    
                    SELECT {', '.join(select_fields)}
                    FROM {main_table} {main_alias}
                    {' '.join(join_clauses)}
                    {where_clause}
                    """
        else:

            if operation_type == "INSERT":
                query = self.sql_templates["insert"].format(
                    target_table=target_table,
                    target_fields=", ".join(target_fields),
                    source_fields=", ".join(select_fields),
                    source_table=f"{main_table} {main_alias} {' '.join(join_clauses)}",
                    where_clause=where_clause
                )
            else:

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

        source_tables = planner_info.get("source_table_name", [])
        insertion_fields = planner_info.get("insertion_fields", [])
        target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
        target_fields = planner_info.get("target_sap_fields", [])
        validation_rules = planner_info.get("validation_rules", [])
        

        if not source_tables or not target_table or not validation_rules:
            logger.error("Missing essential information for VALIDATION operation")
            return "", {}
        

        source_table = source_tables[0]
        

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
        

        select_fields = []
        

        for i, case_expr in enumerate(case_expressions):
            select_fields.append(f"{case_expr} AS validation_result_{i+1}")
        

        if insertion_fields:
            select_fields.extend([f"{field}" for field in insertion_fields])
        else:

            validation_fields = [rule.get("field") for rule in validation_rules if rule.get("field")]
            select_fields.extend([f"{field}" for field in validation_fields if field])
        

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

        source_tables = planner_info.get("source_table_name", [])
        insertion_fields = planner_info.get("insertion_fields", [])
        target_table = planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None
        target_fields = planner_info.get("target_sap_fields", [])
        group_by_fields = planner_info.get("group_by_fields", [])
        aggregation_functions = planner_info.get("aggregation_functions", [])
        filtering_fields = planner_info.get("filtering_fields", [])
        conditions = planner_info.get("extracted_conditions", {})
        

        if not source_tables or not aggregation_functions:
            logger.error("Missing essential information for AGGREGATION operation")
            return "", {}
        

        source_table = source_tables[0]
        

        agg_expressions = []
        
        for agg in aggregation_functions:
            field = agg.get("field")
            function = agg.get("function", "").lower()
            alias = agg.get("alias")
            
            if not field or not function:
                continue
                

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

                agg_expr = f"{function}({field})"
            

            if alias:
                agg_expr += f" AS {alias}"
            else:
                agg_expr += f" AS {function}_{field}"
            
            agg_expressions.append(agg_expr)
        

        where_clause, params = self._build_where_clause(filtering_fields, conditions)
        

        if group_by_fields:

            query = self.sql_templates["aggregation"].format(
                group_fields=", ".join(group_by_fields),
                agg_separator=", " if group_by_fields and agg_expressions else "",
                agg_functions=", ".join(agg_expressions),
                table=source_table,
                where_clause=where_clause
            )
        else:

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

        target_data_samples = planner_info.get("target_data_samples", {})
        
        if isinstance(target_data_samples, pd.DataFrame) and target_data_samples.empty:
            return "INSERT"
        

        query_text = planner_info.get("restructured_query", "").lower()
        
        if any(term in query_text for term in ["insert", "add", "create", "new"]):
            return "INSERT"
        

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
                           insertion_fields: List[str],
                           target_fields: List[str],
                           filtering_fields: List[str],
                           conditions: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Build an INSERT query
        
        Parameters:
        source_table (str): Source table name
        target_table (str): Target table name
        insertion_fields (List[str]): Fields to insert (NOT all source fields)
        target_fields (List[str]): Fields to insert into target
        filtering_fields (List[str]): Fields used for filtering
        conditions (Dict[str, Any]): Conditions to apply
        
        Returns:
        Tuple[str, Dict[str, Any]]: INSERT query and parameters
        """

        field_mapping = {}
        select_fields = []
        
        for target_field in target_fields:

            source_field = target_field
            

            if target_field in insertion_fields:
                field_mapping[target_field] = target_field
                select_fields.append(target_field)
            else:

                potential_matches = [f for f in insertion_fields if 
                                     f.lower() == target_field.lower() or
                                     target_field.lower() in f.lower() or
                                     f.lower() in target_field.lower()]
                
                if potential_matches:
                    source_field = potential_matches[0]
                    field_mapping[target_field] = source_field
                    select_fields.append(f"{source_field} AS {target_field}")
                else:

                    select_fields.append(f"NULL AS {target_field}")
        

        where_clause, params = self._build_where_clause(filtering_fields, conditions)
        

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
                           insertion_fields: List[str],
                           target_fields: List[str],
                           filtering_fields: List[str],
                           conditions: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Build an UPDATE query
        
        Parameters:
        source_table (str): Source table name
        target_table (str): Target table name
        insertion_fields (List[str]): Fields to update (NOT all source fields)
        target_fields (List[str]): Fields to update in target
        filtering_fields (List[str]): Fields used for filtering
        conditions (Dict[str, Any]): Conditions to apply
        
        Returns:
        Tuple[str, Dict[str, Any]]: UPDATE query and parameters
        """

        key_field = None
        

        for field in insertion_fields:
            if field in target_fields:
                key_field = field
                break
        
        if not key_field:
            logger.warning("No key field found for UPDATE operation, using first target field")
            key_field = target_fields[0] if target_fields else None
        
        if not key_field:
            logger.error("Cannot build UPDATE query without key field")
            return "", {}
        

        set_clauses = []
        for field in target_fields:
            if field != key_field and field in insertion_fields:
                set_clauses.append(f"{field} = subquery.{field}")
        
        if not set_clauses:
            logger.warning("No fields to update in UPDATE query")
            return "", {}
        

        where_clause, params = self._build_where_clause(filtering_fields, conditions)
        

        query = f"""
        UPDATE {target_table}
        SET {', '.join(set_clauses)}
        FROM (
            SELECT {', '.join(insertion_fields)}
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

        if rule_type == "not_null":
            return f"CASE WHEN {field} IS NULL THEN 'Invalid: Null value' ELSE 'Valid' END"
            
        elif rule_type == "unique":


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

                if self.db_dialect == "sqlite":
                    return f"CASE WHEN {field} NOT REGEXP :{pattern_param} THEN 'Invalid: Pattern mismatch' ELSE 'Valid' END"
                else:

                    return f"CASE WHEN {field} NOT LIKE :{pattern_param} THEN 'Invalid: Pattern mismatch' ELSE 'Valid' END"
            else:
                return f"'Valid'"
                
        elif rule_type == "exists_in":
            ref_table = parameters.get("reference_table")
            ref_field = parameters.get("reference_field")
            
            if ref_table and ref_field:

                return f"""CASE WHEN NOT EXISTS (
                    SELECT 1 FROM {ref_table} 
                    WHERE {ref_field} = {field}
                ) THEN 'Invalid: Reference not found' ELSE 'Valid' END"""
            else:
                return f"'Valid'"
        

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


        

        common_key_fields = [
            "MATNR",
            "MANDT",
            "KUNNR",
            "LIFNR",
            "WERKS",
            "LGORT",
            "BUKRS",
        ]
        

        if table1 == "MARA" and table2 == "MAKT":
            return ["MATNR", "MANDT"]
        elif table1 == "MARA" and table2 == "MARC":
            return ["MATNR", "MANDT"]
        elif table1 == "MAKT" and table2 == "MARA":
            return ["MATNR", "MANDT"]
        elif table1 == "MARC" and table2 == "MARA":
            return ["MATNR", "MANDT"]
        

        return common_key_fields
    
    def _get_key_field(self, planner_info: Dict[str, Any], target_fields: List[str], insertion_fields: List[str]) -> Optional[str]:
        """
        Get the key field for the operation
        
        Parameters:
        planner_info (Dict): Information extracted by the planner
        target_fields (List[str]): Target fields
        insertion_fields (List[str]): Insertion fields (NOT all source fields)
        
        Returns:
        Optional[str]: Key field or None if not found
        """

        key_mapping = planner_info.get("key_mapping", [])
        if key_mapping and isinstance(key_mapping, list):
            for mapping in key_mapping:
                if isinstance(mapping, dict) and "target_col" in mapping:
                    return mapping["target_col"]
                elif isinstance(mapping, str):
                    return mapping
        

        common_fields = [f for f in target_fields if f in insertion_fields]
        if common_fields:

            key_indicators = ["MATNR", "ID", "KEY", "NR", "CODE", "NUM"]
            for indicator in key_indicators:
                for field in common_fields:
                    if indicator in field.upper():
                        return field
            

            return common_fields[0]
        
        return None

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

            if self._is_valid_sqlite_query(sql_query):
                return sql_query, sql_params, True
                
            best_query = sql_query
            best_params = sql_params
            

            for attempt in range(max_attempts):
                fixed_query, fixed_params = self._fix_sqlite_query(
                    best_query, 
                    sql_params, 
                    planner_info, 
                    attempt
                )
                

                if self._is_valid_sqlite_query(fixed_query):
                    return fixed_query, fixed_params, True
                    

                if self._compare_query_quality(fixed_query, best_query, planner_info):
                    best_query = fixed_query
                    best_params = fixed_params
                    

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

            prompt = f"""
    You are an expert SQLite database engineer. Analyze the following SQL query for SQLite compatibility issues and other problems.

    SQL QUERY:
    {sql_query}

    CONTEXT INFORMATION:
    - Query Type: {planner_info.get("query_type", "SIMPLE_TRANSFORMATION")}
    - Source Tables: {planner_info.get("source_table_name", [])}
    - Insertion Fields: {planner_info.get("insertion_fields", [])}
    - Target Table: {planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None}
    - Target Fields: {planner_info.get("target_sap_fields", [])}
    - Filtering Fields: {planner_info.get("filtering_fields", [])}

    INSTRUCTIONS:
    1. Analyze for SQLite compatibility issues
    2. Check for syntax errors
    3. Check for logical errors
    4. Check for potential performance issues
    5. Verify table and column references
    6. Verify join conditions if present
    7. Verify subqueries if present
    8. Check for proper handling of NULL values
    9. Verify that filtering fields are only used in WHERE clauses, not in INSERT/UPDATE field lists
    10. Verify that only insertion fields are used for data manipulation operations

    Your analysis should be in a structured format with clear categories of issues.
    """


            llm= LLMManager(
                provider="google",
                model="gemini/gemini-2.5-flash",
                api_key=os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
            )
            response = llm.generate(prompt)

            if response:
                return response.strip()
            else:
                return "Failed to analyze query"
                
        except Exception as e:
            logger.error(f"Error in _analyze_sqlite_query: {e}")
            return f"Error analyzing query: {e}"
            
    def _fix_sqlite_query(self, sql_query, sql_params, planner_info, attempt_number):
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

            prompt = f"""
    You are an expert SQLite database engineer. Fix the following SQL query along with the issue that is happening

    ORIGINAL SQL QUERY:
    {sql_query}

    FIX ATTEMPT: {attempt_number + 1}

    CONTEXT INFORMATION:
    - Query Type: {planner_info.get("query_type", "SIMPLE_TRANSFORMATION")}
    - Source Tables: {planner_info.get("source_table_name", [])}
    - Insertion Fields: {planner_info.get("insertion_fields", [])}
    - Target Table: {planner_info.get("target_table_name", [])[0] if planner_info.get("target_table_name") else None}
    - Target Fields: {planner_info.get("target_sap_fields", [])}
    - Filtering Fields: {planner_info.get("filtering_fields", [])}
    - Filtering Conditions: {json.dumps(planner_info.get("extracted_conditions", {}), indent=2)}

    CRITICAL FIELD USAGE RULES:
    1. Insertion Fields ({planner_info.get("insertion_fields", [])}) should ONLY appear in:
       - INSERT INTO field lists
       - UPDATE SET clauses
       - SELECT clauses for data retrieval
    2. Filtering Fields ({planner_info.get("filtering_fields", [])}) should ONLY appear in:
       - WHERE clauses
       - JOIN conditions for filtering
       - HAVING clauses
    3. NEVER include filtering fields in INSERT/UPDATE target field lists

    INSTRUCTIONS:
    1. IMPORTANT: Only generate standard SQLite SQL syntax
    2. Fix all identified issues in the analysis
    3. Maintain the original query intent
    4. Ensure proper table and column references
    5. Ensure proper join syntax if needed
    6. Ensure proper handling of parameters
    7. Pay special attention to SQLite-specific syntax (different from other SQL dialects)
    8. Target Table Has Data: {isinstance(planner_info.get("target_data_samples", {}), pd.DataFrame) and not planner_info.get("target_data_samples", {}).empty}
    9. Ensure filtering fields are not included in INSERT/UPDATE operations

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


            llm= LLMManager(
                provider="google",
                model="gemini/gemini-2.5-flash",
                api_key=os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
            )
            response = llm.generate(prompt)
            

            if response :
                fixed_query = response.strip()
                

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

            if not sql_query or len(sql_query) < 10:
                return False
                

            sql_upper = sql_query.upper()
            has_select = "SELECT" in sql_upper
            has_insert = "INSERT" in sql_upper
            has_update = "UPDATE" in sql_upper
            has_create = "CREATE" in sql_upper
            
            if not (has_select or has_insert or has_update or has_create):
                return False
                

            if "RIGHT JOIN" in sql_upper or "FULL JOIN" in sql_upper:
                return False
                
            if "ISNULL" in sql_upper:
                return False
                

            
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

            if not new_query:
                return False
            
            if not old_query:
                return True
                

            if not self._is_valid_sqlite_query(old_query) and self._is_valid_sqlite_query(new_query):
                return True
                

            insertion_fields = planner_info.get("insertion_fields", [])
            target_fields = planner_info.get("target_sap_fields", [])
            filtering_fields = planner_info.get("filtering_fields", [])
            
            new_insertion_count = sum(1 for field in insertion_fields if field in new_query)
            old_insertion_count = sum(1 for field in insertion_fields if field in old_query)
            
            new_target_count = sum(1 for field in target_fields if field in new_query)
            old_target_count = sum(1 for field in target_fields if field in old_query)
            


            new_has_filter_in_insert = any(
                f"INSERT INTO" in new_query.upper() and field in new_query 
                for field in filtering_fields
            )
            old_has_filter_in_insert = any(
                f"INSERT INTO" in old_query.upper() and field in old_query 
                for field in filtering_fields
            )
            
            new_has_filter_in_update = any(
                f"UPDATE" in new_query.upper() and f"SET" in new_query.upper() and field in new_query.split("WHERE")[0] if "WHERE" in new_query else new_query
                for field in filtering_fields
            )
            old_has_filter_in_update = any(
                f"UPDATE" in old_query.upper() and f"SET" in old_query.upper() and field in old_query.split("WHERE")[0] if "WHERE" in old_query else old_query
                for field in filtering_fields
            )
            

            if not (new_has_filter_in_insert or new_has_filter_in_update) and (old_has_filter_in_insert or old_has_filter_in_update):
                return True
                

            if new_insertion_count > old_insertion_count:
                return True
                

            if new_target_count > old_target_count:
                return True
                

            sqlite_patterns = ["IFNULL", "CASE WHEN", "COALESCE", "GROUP BY", "LEFT JOIN"]
            sqlite_score_new = sum(1 for pattern in sqlite_patterns if pattern.upper() in new_query.upper())
            sqlite_score_old = sum(1 for pattern in sqlite_patterns if pattern.upper() in old_query.upper())
            
            if sqlite_score_new > sqlite_score_old:
                return True
                

            return False
        except Exception as e:
            logger.error(f"Error in _compare_query_quality: {e}")
            return False