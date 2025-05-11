"""
SAP Query Processor for TableLLM
This is the main orchestrator for SAP query processing
"""
import os
import json
import sqlite3
import traceback
import pandas as pd
from datetime import datetime
from google import genai
from google.genai import types

from agents.intent_agent import IntentAgent
from agents.table_agent import TableAgent
from agents.segment_agent import SegmentAgent
from agents.column_agent import ColumnAgent

from core.workspace_manager import WorkspaceManager
from core.schema_manager import SchemaManager
from core.session_manager import SessionManager
from core.pattern_library import PatternLibrary
from core.code_generator import CodeGenerator

from utils.logging_utils import main_logger as logger
from utils.token_utils import get_token_usage_stats
import config

from code_exec import create_code_file, execute_code

class SAPQueryProcessor:
    """
    Main orchestrator for SAP query processing
    """
    
    def __init__(self):
        """
        Initialize the SAP query processor
        """
        try:
            # Initialize LLM client
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not found in environment variables")
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            self.client = genai.Client(api_key=api_key)
            
            # Initialize database connection
            try:
                self.conn = sqlite3.connect(config.DATABASE_PATH)
                logger.info(f"Connected to database: {config.DATABASE_PATH}")
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                self.conn = None
            
            # Initialize core components
            self.workspace_manager = WorkspaceManager(self.conn)
            self.schema_manager = SchemaManager(self.conn)
            self.session_manager = SessionManager()
            self.pattern_library = PatternLibrary(self.client)
            self.code_generator = CodeGenerator(self.client, self.pattern_library)
            
            # Initialize agents
            self.intent_agent = IntentAgent(self.client)
            self.table_agent = TableAgent(self.client, config.DATABASE_PATH)
            self.segment_agent = SegmentAgent(self.client)
            self.column_agent = ColumnAgent(self.client, config.DATABASE_PATH)
            
            logger.info("Initialized SAPQueryProcessor with all components")
        except Exception as e:
            logger.error(f"Error initializing SAPQueryProcessor: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def process_query(self, query, session_id=None, object_id=29, segment_id=336, project_id=24, target_sap_fields=None):
        """
        Process a query end-to-end
        
        Parameters:
        query (str): Natural language query
        session_id (str, optional): Session ID, creates new session if None
        object_id (int): Object ID for mapping
        segment_id (int): Segment ID for mapping
        project_id (int): Project ID for mapping
        target_sap_fields (str, optional): Optional target SAP field
        
        Returns:
        tuple: (code, result, session_id)
        """
        try:
            logger.info(f"Processing query: {query}")
            start_time = datetime.now()
            
            # Create session if needed
            if not session_id:
                session_id = self.session_manager.create_session()
                logger.info(f"Created new session: {session_id}")
            
            # 1. Intent detection
            intent_info = self.intent_agent.process(query)
            logger.info(f"Intent detected: {intent_info.get('intent_type')}")
            
            # 2. Get workspace
            workspace_name = intent_info.get("workspace")
            workspace_info = self.workspace_manager.get_workspace(workspace_name) if workspace_name else None
            
            # 3. Table selection
            table_info = self.table_agent.process(query, intent_info, workspace_name)
            logger.info(f"Selected tables: {', '.join(table_info.get('source_tables', []))}")
            
            # 4. Segment identification
            segment_info = self.segment_agent.process(query, table_info, intent_info)
            logger.info(f"Primary segment: {segment_info.get('primary_segment')}")
            
            # 5. Column selection
            column_info = self.column_agent.process(query, table_info, segment_info, intent_info)
            
            # 6. Combine all extracted information
            extracted_info = {
                "intent_info": intent_info,
                "workspace_info": workspace_info,
                "table_info": table_info,
                "segment_info": segment_info,
                "column_info": column_info,
                "session_id": session_id,
                "target_sap_fields": target_sap_fields
            }
            
            # 7. Generate transformation plan
            logger.info("Generating transformation plan")
            plan = self.code_generator.generate_plan(query, extracted_info)
            
            # 8. Generate code
            logger.info("Generating code")
            code_content = self.code_generator.generate_code(query, extracted_info)
            if not code_content:
                logger.error("Failed to generate code")
                return None, "Failed to generate code", session_id
            
            # 9. Get source and target dataframes
            source_dfs = self._get_source_dataframes(table_info, column_info)
            target_df = self.session_manager.get_or_create_session_target_df(
                session_id, 
                table_info.get("target_table"),
                self.conn
            )
            
            # 10. Execute code
            logger.info("Executing generated code")
            code_file = create_code_file(code_content, query, is_double=True)
            result = execute_code(code_file, source_dfs, target_df, target_sap_fields)
            
            # 11. Handle execution errors
            if isinstance(result, dict) and "error_type" in result:
                logger.error(f"Code execution error: {result['error_message']}")
                
                # Try to fix the code
                fixed_code = None
                for attempt in range(1, config.MAX_RETRY_ATTEMPTS + 1):
                    logger.info(f"Attempting to fix code (attempt {attempt}/{config.MAX_RETRY_ATTEMPTS})")
                    
                    fixed_code = self.code_generator.fix_code(
                        code_content, 
                        result, 
                        extracted_info,
                        attempt=attempt,
                        max_attempts=config.MAX_RETRY_ATTEMPTS
                    )
                    
                    if not fixed_code:
                        break
                    
                    # Try executing the fixed code
                    fixed_code_file = create_code_file(
                        fixed_code, 
                        f"{query} (fixed attempt {attempt})",
                        is_double=True
                    )
                    
                    fixed_result = execute_code(fixed_code_file, source_dfs, target_df, target_sap_fields)
                    
                    # If fixed code worked, use it
                    if not isinstance(fixed_result, dict) or "error_type" not in fixed_result:
                        logger.info(f"Successfully fixed code on attempt {attempt}")
                        code_content = fixed_code
                        result = fixed_result
                        break
                    else:
                        # Update error for next attempt
                        result = fixed_result
                
                # If still has error after all attempts
                if isinstance(result, dict) and "error_type" in result:
                    logger.error(f"Failed to fix code after {config.MAX_RETRY_ATTEMPTS} attempts")
                    return code_content, f"Failed to execute code: {result['error_message']}", session_id
            
            # 12. Update session with transformation information
            self._update_session_with_transformation(
                session_id, 
                query,
                extracted_info,
                plan,
                code_content
            )
            
            # 13. Save target dataframe
            if isinstance(result, pd.DataFrame):
                self.session_manager.save_session_target_df(session_id, result)
            
            # 14. Log token usage
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            token_stats = get_token_usage_stats()
            logger.info(f"Query processed in {duration:.2f} seconds with {token_stats['total_tokens']} total tokens")
            
            return code_content, result, session_id
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(traceback.format_exc())
            return None, f"An error occurred: {str(e)}", session_id
    
    def _get_source_dataframes(self, table_info, column_info):
        """
        Get source dataframes with selected columns
        
        Parameters:
        table_info (dict): Table information
        column_info (dict): Column information
        
        Returns:
        dict: Dictionary mapping table names to dataframes
        """
        try:
            source_dfs = {}
            source_tables = table_info.get("source_tables", [])
            
            # Return empty dict if no source tables or no database connection
            if not source_tables or not self.conn:
                return source_dfs
            
            # Get dataframe for each source table
            for table in source_tables:
                try:
                    # Get required and optional columns for this table
                    required_columns = []
                    optional_columns = []
                    
                    if table in column_info:
                        table_columns = column_info[table]
                        if isinstance(table_columns, dict):
                            if "required" in table_columns:
                                required_columns = table_columns["required"]
                            if "optional" in table_columns:
                                optional_columns = table_columns["optional"]
                    
                    # Combine required and optional columns
                    columns = list(set(required_columns + optional_columns))
                    
                    # If no columns specified, get all columns
                    query = f"SELECT * FROM {table}"
                    if columns:
                        # Include only specified columns
                        columns_str = ", ".join(columns)
                        query = f"SELECT {columns_str} FROM {table}"
                    
                    # Execute query
                    source_dfs[table] = pd.read_sql_query(query, self.conn)
                    logger.info(f"Loaded {len(source_dfs[table])} rows from {table}")
                except Exception as e:
                    logger.error(f"Error loading source table {table}: {e}")
                    # Create empty dataframe as fallback
                    source_dfs[table] = pd.DataFrame()
            
            return source_dfs
        except Exception as e:
            logger.error(f"Error getting source dataframes: {e}")
            return {}
    
    def _update_session_with_transformation(self, session_id, query, extracted_info, plan, code):
        """
        Update session with transformation information
        
        Parameters:
        session_id (str): Session ID
        query (str): Original query
        extracted_info (dict): Extracted information
        plan (str): Transformation plan
        code (str): Generated code
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            # Create transformation data
            transformation_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "intent_type": extracted_info.get("intent_info", {}).get("intent_type"),
                "source_tables": extracted_info.get("table_info", {}).get("source_tables", []),
                "target_table": extracted_info.get("table_info", {}).get("target_table"),
                "segment": extracted_info.get("segment_info", {}).get("primary_segment"),
                "plan_summary": plan.split("\n")[0] if plan else "No plan available",
                "code_length": len(code) if code else 0
            }
            
            # Add to transformation history
            success = self.session_manager.add_transformation_history(session_id, transformation_data)
            
            # Update segment status
            segment_name = extracted_info.get("segment_info", {}).get("primary_segment")
            if segment_name:
                self.session_manager.update_segment_status(session_id, segment_name, "completed")
            
            return success
        except Exception as e:
            logger.error(f"Error updating session with transformation: {e}")
            return False
    
    def get_session_info(self, session_id):
        """
        Get comprehensive information about a session
        
        Parameters:
        session_id (str): Session ID
        
        Returns:
        dict: Session information
        """
        try:
            # Get basic session context
            context = self.session_manager.get_context(session_id)
            if not context:
                logger.warning(f"Session not found: {session_id}")
                return None
            
            # Get transformation history
            history = self.session_manager.get_transformation_history(session_id)
            
            # Get segment status
            segments = {}
            if "processed_segments" in context:
                segments = context["processed_segments"]
            
            # Get target dataframe information
            target_info = {
                "rows": 0,
                "columns": []
            }
            try:
                session_path = os.path.join(config.SESSIONS_DIR, session_id)
                target_path = os.path.join(session_path, "target_latest.csv")
                if os.path.exists(target_path):
                    target_df = pd.read_csv(target_path)
                    target_info["rows"] = len(target_df)
                    target_info["columns"] = target_df.columns.tolist()
            except Exception as e:
                logger.warning(f"Error getting target dataframe info: {e}")
            
            # Compile comprehensive session information
            return {
                "session_id": session_id,
                "created_at": context.get("created_at"),
                "updated_at": context.get("updated_at"),
                "transformations": history,
                "segments": segments,
                "target": target_info,
                "key_mappings": context.get("key_mappings", {})
            }
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return None
    
    def list_sessions(self):
        """
        List all available sessions
        
        Returns:
        list: List of session info dictionaries
        """
        try:
            return self.session_manager.list_sessions()
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []
    
    def delete_session(self, session_id):
        """
        Delete a session
        
        Parameters:
        session_id (str): Session ID
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            return self.session_manager.delete_session(session_id)
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False
    
    def get_workspaces(self):
        """
        Get all available workspaces
        
        Returns:
        dict: Dictionary of workspaces
        """
        try:
            return self.workspace_manager.get_all_workspaces()
        except Exception as e:
            logger.error(f"Error getting workspaces: {e}")
            return {}
    
    def create_workspace(self, name, tables, description=""):
        """
        Create a custom workspace
        
        Parameters:
        name (str): Workspace name
        tables (list): List of table names
        description (str, optional): Workspace description
        
        Returns:
        dict: Created workspace or None if failed
        """
        try:
            return self.workspace_manager.create_custom_workspace(name, tables, description)
        except Exception as e:
            logger.error(f"Error creating workspace: {e}")
            return None
    
    def get_patterns(self):
        """
        Get all available transformation patterns
        
        Returns:
        dict: Dictionary of patterns
        """
        try:
            return self.pattern_library.get_all_patterns()
        except Exception as e:
            logger.error(f"Error getting patterns: {e}")
            return {}
    
    def get_token_usage(self):
        """
        Get token usage statistics
        
        Returns:
        dict: Token usage statistics
        """
        try:
            return get_token_usage_stats()
        except Exception as e:
            logger.error(f"Error getting token usage: {e}")
            return {}
    
    def close(self):
        """
        Close all connections and resources
        """
        try:
            if self.conn:
                self.conn.close()
            
            self.workspace_manager.close()
            self.schema_manager.close()
            
            logger.info("SAPQueryProcessor closed successfully")
        except Exception as e:
            logger.error(f"Error closing SAPQueryProcessor: {e}")


# Example usage
if __name__ == "__main__":
    try:
        # Initialize processor
        processor = SAPQueryProcessor()
        
        # Example query
        query = "Extract material type where material type equals ROH from MARA table"
        
        # Process query
        code, result, session_id = processor.process_query(query)
        
        # Print result
        if isinstance(result, pd.DataFrame):
            print(f"Processed query successfully. Result has {len(result)} rows.")
            print(result.head())
        else:
            print(f"Error processing query: {result}")
        
        # Close processor
        processor.close()
    except Exception as e:
        print(f"Error in example: {e}")
        traceback.print_exc()
