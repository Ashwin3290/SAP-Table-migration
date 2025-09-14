import logging
import json
import re
import sqlite3
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import pandas as pd

from DMtool.llm_config import LLMManager

logger = logging.getLogger(__name__)

class SQLiteQueryAnalyzer:
    """Analyzes and fixes SQLite queries to ensure compatibility and correctness"""
    
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
                
            

            analysis = self._analyze_sqlite_query(sql_query, planner_info)
            
            best_query = sql_query
            best_params = sql_params
            

            for attempt in range(max_attempts):
                

                fixed_query, fixed_params = self._fix_sqlite_query(
                    best_query, 
                    sql_params, 
                    planner_info, 
                    analysis, 
                    attempt
                )
                

                if self._is_valid_sqlite_query(fixed_query):
                    return fixed_query, fixed_params, True
                    

                if self._compare_query_quality(fixed_query, best_query, planner_info):
                    best_query = fixed_query
                    best_params = fixed_params
                    

                if attempt < max_attempts - 1:
                    analysis = self._analyze_sqlite_query(fixed_query, planner_info)
                    

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


            llm = LLMManager(
                provider="google",
                model="gemini/gemini-2.5-flash",
                api_key=os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
            )
            response = llm.generate(prompt, temperature=0.1, max_tokens=500)
            

            if response :
                return response.strip()
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


            llm = LLMManager(
                provider="google",
                model="gemini/gemini-2.5-flash",
                api_key=os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
            )
            response = llm.generate(prompt, temperature=0.1, max_tokens=500)
            

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
            has_drop = "DROP" in sql_upper
            has_alter = "ALTER" in sql_upper
            
            if not (has_select or has_insert or has_update or has_create, has_drop or has_alter):
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
                

            source_fields = planner_info.get("source_field_names", [])
            target_fields = planner_info.get("target_sap_fields", [])
            
            new_field_count = sum(1 for field in source_fields if field in new_query)
            old_field_count = sum(1 for field in source_fields if field in old_query)
            
            new_target_count = sum(1 for field in target_fields if field in new_query)
            old_target_count = sum(1 for field in target_fields if field in old_query)
            

            if new_field_count > old_field_count:
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