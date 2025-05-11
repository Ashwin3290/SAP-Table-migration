"""
Column Agent for TableLLM
This agent is responsible for selecting and pruning relevant columns for transformations
"""
import json
import sqlite3
import pandas as pd
from utils.logging_utils import agent_logger as logger
from utils.sql_utils import get_table_schema, validate_sql_identifier
from agents.base_agent import BaseAgent
import config

class ColumnAgent(BaseAgent):
    """
    Agent for selecting and pruning relevant columns
    """
    
    def __init__(self, client=None, db_path=None):
        """
        Initialize the column agent
        
        Parameters:
        client (genai.Client, optional): LLM client, creates new one if None
        db_path (str, optional): Path to SQLite database, defaults to config.DATABASE_PATH
        """
        super().__init__(client)
        
        # Set database path
        self.db_path = db_path if db_path else config.DATABASE_PATH
    
    def process(self, query, table_info, segment_info=None, intent_info=None):
        """
        Select relevant columns for each table
        
        Parameters:
        query (str): The natural language query
        table_info (dict): Table selection information
        segment_info (dict, optional): Segment information
        intent_info (dict, optional): Intent information
        
        Returns:
        dict: {
            "table_name": {
                "required": List of required column names
                "optional": List of optional column names
            },
            ...
        }
        """
        logger.info(f"Selecting columns for query: {query[:100]}...")
        
        # Get all columns for each table
        all_columns = self._get_all_columns(table_info)
        if not all_columns:
            logger.warning("Failed to get columns for tables")
            return self._get_default_columns(table_info)
        
        # Create column selection prompt
        prompt = f"""
You are an expert SAP data transformation analyst. Your task is to select only the columns needed for this transformation.

QUERY: {query}

TABLE INFORMATION:
{json.dumps(table_info, indent=2)}

SEGMENT INFORMATION:
{json.dumps(segment_info, indent=2) if segment_info else "Not available"}

INTENT INFORMATION:
{json.dumps(intent_info, indent=2) if intent_info else "Not available"}

ALL COLUMNS:
{json.dumps(all_columns, indent=2)}

Based on the query and tables, identify:

1. Required Columns - The columns that are essential for this transformation
2. Optional Columns - The columns that may be useful but are not essential

For each source and target table, specify the required and optional columns.

Consider:
- Primary key fields are almost always required (e.g., MATNR, KUNNR)
- Only include columns that are directly referenced or needed for the transformation
- Don't include all columns, as this will increase token usage unnecessarily
- Look for clues in the query about which fields to extract or filter

Return ONLY a JSON object with these fields and no additional explanation:
{{
  "table_name_1": {{
    "required": ["COLUMN1", "COLUMN2", ...],
    "optional": ["COLUMN3", "COLUMN4", ...]
  }},
  "table_name_2": {{
    "required": ["COLUMN1", "COLUMN2", ...],
    "optional": ["COLUMN3", "COLUMN4", ...]
  }},
  ...
}}
"""
        
        # Call the LLM
        response_text = self._call_llm(prompt)
        if not response_text:
            logger.warning("Failed to get response from LLM for column selection")
            return self._get_default_columns(table_info)
        
        # Parse the response
        column_data = self._parse_json_response(response_text)
        
        # Validate the response
        if not column_data or not isinstance(column_data, dict):
            logger.warning(f"Invalid column selection response: {column_data}")
            return self._get_default_columns(table_info)
        
        # Validate that the columns exist in their respective tables
        validated_data = self._validate_column_selection(column_data, all_columns)
        
        # Log the result
        for table, columns in validated_data.items():
            required = ", ".join(columns["required"][:5]) + ("..." if len(columns["required"]) > 5 else "")
            logger.info(f"Selected columns for {table} - Required: {required}")
        
        return validated_data
    
    def _get_all_columns(self, table_info):
        """
        Get all columns for the tables in table_info
        
        Parameters:
        table_info (dict): Table selection information
        
        Returns:
        dict: Dictionary mapping table names to column lists
        """
        result = {}
        
        # Get source tables
        source_tables = table_info.get("source_tables", [])
        
        # Add target table
        target_table = table_info.get("target_table")
        if target_table:
            all_tables = source_tables + [target_table]
        else:
            all_tables = source_tables
        
        # Get columns for each table
        try:
            conn = sqlite3.connect(self.db_path)
            
            for table in all_tables:
                try:
                    # Validate table name
                    safe_table = validate_sql_identifier(table)
                    
                    # Get schema
                    schema_df = get_table_schema(conn, safe_table)
                    
                    # Get column information
                    columns = []
                    key_columns = []
                    for _, row in schema_df.iterrows():
                        column_name = row["name"]
                        columns.append(column_name)
                        
                        # Check if it's a key column
                        if "pk" in schema_df.columns and row["pk"] > 0:
                            key_columns.append(column_name)
                    
                    # Add to result
                    result[table] = {
                        "columns": columns,
                        "key_columns": key_columns
                    }
                except Exception as e:
                    logger.warning(f"Error getting columns for {table}: {e}")
            
            conn.close()
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
        
        return result
    
    def _validate_column_selection(self, column_data, all_columns):
        """
        Validate that the selected columns exist in their respective tables
        
        Parameters:
        column_data (dict): Column selection data
        all_columns (dict): Dictionary mapping table names to column lists
        
        Returns:
        dict: Validated column selection data
        """
        validated = {}
        
        for table, columns in column_data.items():
            if table not in all_columns:
                continue
            
            validated[table] = {
                "required": [],
                "optional": []
            }
            
            # Get valid columns for this table
            valid_columns = all_columns[table]["columns"]
            
            # Validate required columns
            if "required" in columns and isinstance(columns["required"], list):
                validated[table]["required"] = [col for col in columns["required"] if col in valid_columns]
                
                # Always include key columns
                for key_col in all_columns[table].get("key_columns", []):
                    if key_col not in validated[table]["required"]:
                        validated[table]["required"].append(key_col)
            else:
                # Default to key columns if available
                validated[table]["required"] = all_columns[table].get("key_columns", [])
            
            # Validate optional columns
            if "optional" in columns and isinstance(columns["optional"], list):
                # Only include valid columns that aren't already in required
                validated[table]["optional"] = [
                    col for col in columns["optional"]
                    if col in valid_columns and col not in validated[table]["required"]
                ]
            else:
                validated[table]["optional"] = []
            
            # If no required columns were found, use first column as fallback
            if not validated[table]["required"] and valid_columns:
                validated[table]["required"] = [valid_columns[0]]
        
        # Make sure all tables from all_columns are included
        for table in all_columns:
            if table not in validated:
                validated[table] = {
                    "required": all_columns[table].get("key_columns", []),
                    "optional": []
                }
                
                # If no key columns, use first column
                if not validated[table]["required"] and all_columns[table]["columns"]:
                    validated[table]["required"] = [all_columns[table]["columns"][0]]
        
        return validated
    
    def _get_default_columns(self, table_info):
        """
        Get default columns when selection fails
        
        Parameters:
        table_info (dict): Table selection information
        
        Returns:
        dict: Default column selection
        """
        default_columns = {}
        
        # Try to get actual columns
        all_columns = self._get_all_columns(table_info)
        
        if all_columns:
            # Use actual columns with key columns as required
            for table, columns in all_columns.items():
                default_columns[table] = {
                    "required": columns.get("key_columns", []),
                    "optional": []
                }
                
                # If no key columns, use first column
                if not default_columns[table]["required"] and columns["columns"]:
                    default_columns[table]["required"] = [columns["columns"][0]]
                
                # Add some common SAP fields as optional if they exist
                common_fields = ["ERDAT", "AEDAT", "ERNAM", "AENAM", "SPRAS", "VKORG", "VTWEG", "SPART"]
                default_columns[table]["optional"] = [
                    col for col in common_fields
                    if col in columns["columns"] and col not in default_columns[table]["required"]
                ]
        else:
            # No column information available, use SAP defaults
            source_tables = table_info.get("source_tables", [])
            target_table = table_info.get("target_table")
            
            # Default columns for source tables
            for table in source_tables:
                if "MARA" in table:
                    default_columns[table] = {
                        "required": ["MATNR", "MTART"],
                        "optional": ["MBRSH", "MEINS"]
                    }
                elif "MAKT" in table:
                    default_columns[table] = {
                        "required": ["MATNR", "SPRAS", "MAKTX"],
                        "optional": []
                    }
                elif "KNA1" in table:
                    default_columns[table] = {
                        "required": ["KUNNR", "NAME1"],
                        "optional": ["ORT01", "LAND1"]
                    }
                else:
                    # Generic default
                    default_columns[table] = {
                        "required": ["ID"],
                        "optional": []
                    }
            
            # Default columns for target table
            if target_table:
                if target_table in source_tables:
                    # Target is one of the source tables, reuse its columns
                    default_columns[target_table] = default_columns[source_tables[0]].copy()
                else:
                    # Generic target columns
                    default_columns[target_table] = {
                        "required": ["ID"],
                        "optional": []
                    }
        
        return default_columns
