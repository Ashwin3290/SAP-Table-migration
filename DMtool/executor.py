import logging
import sqlite3
import re
import pandas as pd
import uuid
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from DMtool.sqlite_utils import add_sqlite_functions

load_dotenv()


logger = logging.getLogger(__name__)

class SQLExecutor:
    """Executes SQL queries against the database"""
    
    def __init__(self, db_path=os.environ.get('DB_PATH')):
        """Initialize the SQL executor
        
        Parameters:
        db_path (str): Path to the SQLite database
        """
        self.db_path = db_path
    
    def execute_query(self, 
                     query: str, 
                     params: Optional[Dict[str, Any]] = None, 
                     fetch_results: bool = True,
                     commit: bool = True) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute an SQL query with parameter binding
        
        Parameters:
        query (str): SQL query to execute
        params (Optional[Dict[str, Any]]): Parameters to bind to the query
        fetch_results (bool): Whether to fetch and return results
        commit (bool): Whether to commit changes
        
        Returns:
        Union[List[Dict[str, Any]], Dict[str, Any]]: 
            Query results as a list of dictionaries, or error information
        """
        conn = None
        try:

            conn = sqlite3.connect(self.db_path)
            add_sqlite_functions(conn)
            

            conn.row_factory = sqlite3.Row
            
            cursor = conn.cursor()
            

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                

            if commit:
                conn.commit()
            

            if fetch_results:

                rows = cursor.fetchall()
                result = [dict(row) for row in rows]
                return result
            else:

                return {"rowcount": cursor.rowcount}
                
        except sqlite3.Error as e:

            if conn and commit:
                conn.rollback()
                
            logger.error(f"SQLite error: {e}")
            return {
                "error_type": "SQLiteError", 
                "error_message": str(e),
                "query": query
            }
        except Exception as e:

            if conn and commit:
                conn.rollback()
                
            logger.error(f"Error executing query: {e}")
            return {
                "error_type": "ExecutionError", 
                "error_message": str(e),
                "query": query
            }
        finally:

            if conn:
                conn.close()
    

    def execute_and_fetch_df(self, query: str, params: Optional[Dict[str, Any]] = None) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Execute a query and return results as a pandas DataFrame
        
        Parameters:
        query (str): SQL query to execute
        params (Optional[Dict[str, Any]]): Parameters to bind to the query
        
        Returns:
        Union[pd.DataFrame, Dict[str, Any]]: DataFrame with results or error information
        """
        conn = None
        try:

            conn = sqlite3.connect(self.db_path)
            

            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
                
            return df
                
        except sqlite3.Error as e:

            logger.error(f"SQLite error in execute_and_fetch_df: {e}")
            return {
                "error_type": "SQLiteError", 
                "error_message": str(e),
                "query": query
            }
        except Exception as e:

            logger.error(f"Error in execute_and_fetch_df: {e}")
            return {
                "error_type": "ExecutionError", 
                "error_message": str(e),
                "query": query
            }
        finally:

            if conn:
                conn.close()
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database
        
        Parameters:
        table_name (str): Name of the table to check
        
        Returns:
        bool: True if the table exists, False otherwise
        """
        query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=? Limit 1
        """
        
        result = self.execute_query(query, {"table_name": table_name})
        
        if isinstance(result, list):
            return len(result) > 0
        
        return False
    
    def get_table_schema(self, table_name: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Get the schema information for a table
        
        Parameters:
        table_name (str): Name of the table
        
        Returns:
        Union[List[Dict[str, Any]], Dict[str, Any]]: Schema information or error
        """
        query = f"PRAGMA table_info({table_name})"
        
        return self.execute_query(query)
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Get a sample of rows from a table as a DataFrame
        
        Parameters:
        table_name (str): Name of the table
        limit (int): Maximum number of rows to return
        
        Returns:
        Union[pd.DataFrame, Dict[str, Any]]: DataFrame with sample rows or error
        """
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        
        return self.execute_and_fetch_df(query)
    
    def backup_table(self, table_name: str) -> Tuple[bool, str]:
        """
        Create a backup of a table
        
        Parameters:
        table_name (str): Name of the table to backup
        
        Returns:
        Tuple[bool, str]: Success status and backup table name
        """

        backup_name = f"{table_name}_backup_{uuid.uuid4().hex[:8]}"
        
        query = f"CREATE TABLE {backup_name} AS SELECT * FROM {table_name}"
        
        result = self.execute_query(query, fetch_results=False)
        
        if isinstance(result, dict) and "error_type" in result:
            return False, ""
        
        return True, backup_name
    
    def execute_multi_statement_query(self, multi_sql, sql_params,context_manager=None, session_id=None):
        """Execute multiple SQL statements with recovery support"""
        try:            

            statements = self.split_sql_statements(multi_sql)
            

            if session_id:
                execution_state = context_manager.load_multi_query_state(session_id)
                if execution_state:
                    return self.resume_execution(execution_state, statements, sql_params, context_manager, session_id)
            

            return self.execute_with_recovery(statements, sql_params, session_id, context_manager)
            
        except Exception as e:
            logger.error(f"Error in execute_multi_statement_query: {e}")
            return {"error_type": "MultiQueryError", "error_message": str(e)}
        
    def resume_execution(self, execution_state, statements, sql_params, context_manager, session_id):
        """Resume execution from a previously failed multi-query operation"""
        try:
            logger.info(f"Resuming multi-query execution from session {session_id}")
            

            completed_count = execution_state.get("completed_count", 0)
            failed_count = execution_state.get("failed_count", 0)
            previous_statements = execution_state.get("statements", [])
            

            if len(statements) != len(previous_statements):
                logger.warning("Statement count mismatch during resume, starting fresh execution")
                return self.execute_with_recovery(statements, sql_params, session_id, context_manager)
            

            results = []
            

            for prev_statement in previous_statements:
                if prev_statement.get("status") == "completed":
                    results.append(prev_statement.get("result", {}))
            

            resume_index = completed_count
            

            execution_state["resumed_at"] = datetime.now().isoformat() if 'datetime' in globals() else "resumed"
            execution_state["resume_attempt"] = execution_state.get("resume_attempt", 0) + 1
            
            logger.info(f"Resuming from statement {resume_index + 1}/{len(statements)}")
            

            for i in range(resume_index, len(statements)):
                statement = statements[i]
                
                try:
                    logger.info(f"Executing statement {i+1}/{len(statements)} (resumed): {statement[:100]}...")
                    

                    result = self.execute_query(statement, sql_params, fetch_results=False)
                    
                    if isinstance(result, dict) and "error_type" in result:

                        statement_result = {
                            "statement": statement,
                            "index": i,
                            "status": "failed", 
                            "error": result,
                            "can_retry": self._can_retry_statement(statement, result),
                            "resume_attempt": execution_state.get("resume_attempt", 1)
                        }
                        execution_state["failed_count"] += 1
                        

                        if i < len(execution_state["statements"]):
                            execution_state["statements"][i] = statement_result
                        else:
                            execution_state["statements"].append(statement_result)
                        

                        context_manager.save_multi_query_state(session_id, execution_state)
                        

                        return {
                            "multi_query_result": True,
                            "completed_statements": execution_state["completed_count"],
                            "failed_statement_index": i,
                            "failed_statement": statement,
                            "error": result,
                            "can_resume": True,
                            "session_id": session_id,
                            "all_results": results,
                            "is_resumed_execution": True,
                            "resume_attempt": execution_state.get("resume_attempt", 1)
                        }
                    else:

                        statement_result = {
                            "statement": statement,
                            "index": i,
                            "status": "completed",
                            "result": result,
                            "resumed_execution": True
                        }
                        execution_state["completed_count"] += 1
                        results.append(result)
                        

                        if i < len(execution_state["statements"]):
                            execution_state["statements"][i] = statement_result
                        else:
                            execution_state["statements"].append(statement_result)
                        

                        prev_statement = next((s for s in previous_statements if s.get("index") == i), None)
                        if prev_statement and prev_statement.get("status") == "failed":
                            execution_state["failed_count"] = max(0, execution_state["failed_count"] - 1)
                    
                except Exception as e:

                    logger.error(f"Unexpected error during resume at statement {i}: {e}")
                    return {
                        "error_type": "ResumeExecutionError",
                        "error_message": f"Resume failed at statement {i+1}: {str(e)}",
                        "completed_statements": execution_state["completed_count"],
                        "failed_statement": statement,
                        "session_id": session_id,
                        "is_resumed_execution": True
                    }
            

            context_manager.cleanup_multi_query_state(session_id)
            
            logger.info(f"Multi-query execution resumed and completed successfully. Total statements: {len(statements)}")
            
            return {
                "multi_query_result": True, 
                "completed_statements": len(statements),
                "all_results": results,
                "success": True,
                "is_resumed_execution": True,
                "resume_attempt": execution_state.get("resume_attempt", 1)
            }
            
        except Exception as e:
            logger.error(f"Error in resume_execution: {e}")
            return {
                "error_type": "ResumeExecutionError",
                "error_message": f"Failed to resume execution: {str(e)}",
                "session_id": session_id
            }

    def split_sql_statements(self, multi_sql):
        """Split multi-SQL into individual statements, handling edge cases"""
        statements = []
        current_statement = ""
        in_string = False
        string_char = None
        
        for char in multi_sql:
            if char in ("'", '"') and not in_string:
                in_string = True
                string_char = char
            elif char == string_char and in_string:
                in_string = False
                string_char = None
            elif char == ';' and not in_string:
                if current_statement.strip():
                    statements.append(current_statement.strip())
                current_statement = ""
                continue
            
            current_statement += char
        

        if current_statement.strip():
            statements.append(current_statement.strip())
        
        return statements

    def execute_with_recovery(self, statements, sql_params, session_id, context_manager):
        """Execute statements one by one with failure tracking"""
        results = []
        execution_state = {
            "statements": [],
            "completed_count": 0,
            "failed_count": 0,
            "session_id": session_id,
            "original_sql": "; ".join(statements)
        }
        
        for i, statement in enumerate(statements):
            try:
                logger.info(f"Executing statement {i+1}/{len(statements)}: {statement[:100]}...")
                

                result = self.execute_query(statement, sql_params, fetch_results=False)
                
                if isinstance(result, dict) and "error_type" in result:

                    statement_result = {
                        "statement": statement,
                        "index": i,
                        "status": "failed", 
                        "error": result,
                        "can_retry": self._can_retry_statement(statement, result)
                    }
                    execution_state["failed_count"] += 1
                    execution_state["statements"].append(statement_result)
                    

                    if session_id:
                        context_manager.save_multi_query_state(session_id, execution_state)
                    

                    return {
                        "multi_query_result": True,
                        "completed_statements": execution_state["completed_count"],
                        "failed_statement_index": i,
                        "failed_statement": statement,
                        "error": result,
                        "can_resume": True,
                        "session_id": session_id,
                        "all_results": results
                    }
                else:

                    statement_result = {
                        "statement": statement,
                        "index": i,
                        "status": "completed",
                        "result": result
                    }
                    execution_state["completed_count"] += 1
                    results.append(result)
                    
                execution_state["statements"].append(statement_result)
                
            except Exception as e:

                logger.error(f"Unexpected error executing statement {i}: {e}")
                return {
                    "error_type": "StatementExecutionError",
                    "error_message": f"Statement {i+1} failed: {str(e)}",
                    "completed_statements": execution_state["completed_count"],
                    "failed_statement": statement
                }
        

        if session_id:
            context_manager.cleanup_multi_query_state(session_id)
        
        return {
            "multi_query_result": True, 
            "completed_statements": len(statements),
            "all_results": results,
            "success": True
        }

    def _can_retry_statement(self, statement, error):
        """Determine if a failed statement can be safely retried"""
        error_msg = error.get("error_message", "").lower()
        

        if "alter table" in statement.lower() and "add column" in statement.lower():
            if "already exists" in error_msg or "duplicate column" in error_msg:
                return False
        

        if any(keyword in statement.lower() for keyword in ["update", "insert", "select"]):
            return True
            
        return True