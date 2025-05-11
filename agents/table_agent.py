"""
Table Agent for TableLLM
This agent is responsible for selecting relevant tables for the transformation
"""
import json
import sqlite3
import pandas as pd
from utils.logging_utils import agent_logger as logger
from utils.sql_utils import get_all_tables, get_table_schema, validate_sql_identifier
from agents.base_agent import BaseAgent
import config

class TableAgent(BaseAgent):
    """
    Agent for selecting relevant tables for transformations
    """
    
    def __init__(self, client=None, db_path=None):
        """
        Initialize the table agent
        
        Parameters:
        client (genai.Client, optional): LLM client, creates new one if None
        db_path (str, optional): Path to SQLite database, defaults to config.DATABASE_PATH
        """
        super().__init__(client)
        
        # Set database path
        self.db_path = db_path if db_path else config.DATABASE_PATH
        
        # Load workspace information
        self.workspaces = config.DEFAULT_WORKSPACES
    
    def process(self, query, intent_info, workspace=None):
        """
        Select relevant tables for a query based on intent
        
        Parameters:
        query (str): The natural language query
        intent_info (dict): Intent information from intent agent
        workspace (str, optional): Workspace name, uses intent_info['workspace'] if None
        
        Returns:
        dict: {
            "source_tables": List of source table names
            "target_table": Target table name
            "confidence": Confidence score
        }
        """
        logger.info(f"Selecting tables for query: {query[:100]}...")
        
        # Get workspace
        if workspace is None and intent_info and "workspace" in intent_info:
            workspace = intent_info["workspace"]
        
        # Get workspace tables
        available_tables = self._get_workspace_tables(workspace)
        
        # Get table schemas for each available table
        tables_with_schemas = self._get_table_schemas(available_tables)
        
        # Create table selection prompt
        prompt = f"""
You are an expert SAP data transformation analyst. Your task is to select the most relevant source and target tables for the query.

QUERY: {query}

INTENT INFORMATION:
{json.dumps(intent_info, indent=2)}

AVAILABLE TABLES:
{json.dumps(tables_with_schemas, indent=2)}

Based on the query and intent, identify:

1. Source Tables - The tables that contain the source data for this transformation. There may be multiple source tables.
2. Target Table - The table where the transformed data should be placed
3. Confidence - Your confidence level in this selection (0.0-1.0)

Consider:
- SAP table naming conventions (e.g., MARA, MAKT, KNA1)
- The intent type (e.g., JOIN operations will need multiple source tables)
- The specific fields mentioned in the query
- Common relationships between SAP tables

Return ONLY a JSON object with these fields and no additional explanation:
{{
  "source_tables": ["TABLE1", "TABLE2", ...],
  "target_table": "TARGET_TABLE_NAME",
  "confidence": CONFIDENCE_SCORE
}}
"""
        
        # Call the LLM
        response_text = self._call_llm(prompt)
        if not response_text:
            logger.warning("Failed to get response from LLM for table selection")
            return self._get_default_tables(workspace, intent_info)
        
        # Parse the response
        table_data = self._parse_json_response(response_text)
        
        # Validate the response
        required_keys = ["source_tables", "target_table", "confidence"]
        if not self._validate_response(table_data, required_keys):
            logger.warning(f"Invalid table selection response: {table_data}")
            return self._get_default_tables(workspace, intent_info)
        
        # Validate that the selected tables exist
        validated_data = self._validate_table_selection(table_data, available_tables)
        
        # Log the result
        src_tables = ", ".join(validated_data["source_tables"])
        logger.info(f"Selected tables - Source: {src_tables}, Target: {validated_data['target_table']}")
        
        return validated_data
    
    def _get_workspace_tables(self, workspace):
        """
        Get tables for a specific workspace
        
        Parameters:
        workspace (str): Workspace name
        
        Returns:
        list: List of table names in the workspace
        """
        if not workspace or workspace not in self.workspaces:
            # If workspace is invalid, get all available tables from database
            try:
                conn = sqlite3.connect(self.db_path)
                tables = get_all_tables(conn)
                conn.close()
                return tables
            except Exception as e:
                logger.error(f"Error getting all tables: {e}")
                return []
        
        # Get tables for the specified workspace
        return self.workspaces[workspace].get("tables", [])
    
    def _get_table_schemas(self, tables):
        """
        Get schemas for the specified tables
        
        Parameters:
        tables (list): List of table names
        
        Returns:
        dict: Dictionary mapping table names to their schemas
        """
        result = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            for table in tables:
                try:
                    # Validate table name
                    safe_table = validate_sql_identifier(table)
                    
                    # Get schema
                    schema_df = get_table_schema(conn, safe_table)
                    
                    # Get column names and types
                    columns = []
                    for _, row in schema_df.iterrows():
                        columns.append({
                            "name": row["name"],
                            "type": row["type"]
                        })
                    
                    # Add to result
                    result[table] = {
                        "columns": columns
                    }
                    
                    # Get sample data (first row only)
                    try:
                        sample_query = f"SELECT * FROM {safe_table} LIMIT 1"
                        sample_df = pd.read_sql_query(sample_query, conn)
                        if not sample_df.empty:
                            result[table]["sample"] = sample_df.iloc[0].to_dict()
                    except Exception as e:
                        logger.warning(f"Error getting sample data for {table}: {e}")
                except Exception as e:
                    logger.warning(f"Error getting schema for {table}: {e}")
            
            conn.close()
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
        
        return result
    
    def _validate_table_selection(self, table_data, available_tables):
        """
        Validate that the selected tables exist
        
        Parameters:
        table_data (dict): Table selection data
        available_tables (list): List of available table names
        
        Returns:
        dict: Validated table selection data
        """
        # Create a copy of the data
        validated = table_data.copy()
        
        # Validate source tables
        if "source_tables" in validated and isinstance(validated["source_tables"], list):
            validated["source_tables"] = [table for table in validated["source_tables"] if table in available_tables]
            
            # If no valid source tables, use default
            if not validated["source_tables"]:
                validated["source_tables"] = [available_tables[0]] if available_tables else []
        else:
            validated["source_tables"] = [available_tables[0]] if available_tables else []
        
        # Validate target table
        if "target_table" not in validated or validated["target_table"] not in available_tables:
            # For target, we might need to create it, so just validate it's a valid identifier
            try:
                if "target_table" in validated:
                    validated["target_table"] = validate_sql_identifier(validated["target_table"])
                else:
                    validated["target_table"] = available_tables[0] if available_tables else "output_table"
            except Exception:
                validated["target_table"] = available_tables[0] if available_tables else "output_table"
        
        return validated
    
    def _get_default_tables(self, workspace, intent_info):
        """
        Get default tables when selection fails
        
        Parameters:
        workspace (str): Workspace name
        intent_info (dict): Intent information
        
        Returns:
        dict: Default table selection
        """
        # Get workspace tables
        available_tables = self._get_workspace_tables(workspace)
        
        # Default to first available table as source and target
        source_table = available_tables[0] if available_tables else "unknown_table"
        target_table = source_table
        
        # Try to infer better defaults based on intent
        if intent_info and "operation" in intent_info:
            operation = intent_info["operation"].lower()
            
            # Check for mentions of specific tables in the operation
            for table in available_tables:
                if table.lower() in operation:
                    source_table = table
                    break
        
        return {
            "source_tables": [source_table],
            "target_table": target_table,
            "confidence": 0.5
        }
