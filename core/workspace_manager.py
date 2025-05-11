"""
Workspace Manager for TableLLM
This module manages workspaces for SAP data transformations
"""
import os
import json
import sqlite3
import pandas as pd
from utils.logging_utils import main_logger as logger
from utils.sql_utils import get_all_tables, table_exists, get_table_schema
import config

class WorkspaceManager:
    """
    Manages SAP-specific workspaces with tables, fields, and samples
    """
    
    def __init__(self, db_connection=None, workspaces_path=None):
        """
        Initialize the workspace manager
        
        Parameters:
        db_connection (sqlite3.Connection, optional): Database connection
        workspaces_path (str, optional): Path to workspace storage directory
        """
        # Set workspaces path
        self.workspaces_path = workspaces_path if workspaces_path else config.WORKSPACE_DIR
        
        # Create workspaces directory if it doesn't exist
        os.makedirs(self.workspaces_path, exist_ok=True)
        
        # Initialize database connection
        self.conn = db_connection
        if not self.conn:
            try:
                self.conn = sqlite3.connect(config.DATABASE_PATH)
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                self.conn = None
        
        # Load workspaces
        self.system_workspaces = self._initialize_system_workspaces()
        self.custom_workspaces = self._load_custom_workspaces()
        
        logger.info(f"Initialized WorkspaceManager with {len(self.system_workspaces)} system workspaces and {len(self.custom_workspaces)} custom workspaces")
    
    def _initialize_system_workspaces(self):
        """
        Initialize predefined SAP workspaces
        
        Returns:
        dict: Dictionary of system workspaces
        """
        workspaces = config.DEFAULT_WORKSPACES.copy()
        
        # Add table metadata for each workspace
        for workspace_name, workspace_data in workspaces.items():
            # Add metadata object if not exists
            if "metadata" not in workspace_data:
                workspace_data["metadata"] = {}
            
            # Add sample transformations if not exists
            if "sample_transformations" not in workspace_data:
                workspace_data["sample_transformations"] = []
            
            # Add default sample transformations based on workspace type
            if workspace_name == "Material_Management" and not workspace_data["sample_transformations"]:
                workspace_data["sample_transformations"] = [
                    "Extract material type from MARA",
                    "Map material group to custom values",
                    "Get material description from MAKT where language is EN"
                ]
            elif workspace_name == "Customer_Management" and not workspace_data["sample_transformations"]:
                workspace_data["sample_transformations"] = [
                    "Extract customer details from KNA1",
                    "Map payment methods based on country",
                    "Get customer sales data for a specific region"
                ]
            
            # Get table metadata for each table in the workspace
            for table in workspace_data["tables"]:
                workspace_data["metadata"][table] = self._get_table_metadata(table)
        
        return workspaces
    
    def _load_custom_workspaces(self):
        """
        Load custom workspaces from workspace directory
        
        Returns:
        dict: Dictionary of custom workspaces
        """
        custom_workspaces = {}
        
        # Check workspaces directory
        try:
            workspace_files = [f for f in os.listdir(self.workspaces_path) if f.endswith(".json")]
            
            for file in workspace_files:
                try:
                    file_path = os.path.join(self.workspaces_path, file)
                    with open(file_path, "r") as f:
                        workspace_data = json.load(f)
                    
                    # Validate workspace data
                    if not isinstance(workspace_data, dict) or "name" not in workspace_data or "tables" not in workspace_data:
                        logger.warning(f"Invalid workspace data in {file}")
                        continue
                    
                    # Use name as key
                    workspace_name = workspace_data.pop("name")
                    
                    # Get table metadata for each table
                    if "metadata" not in workspace_data:
                        workspace_data["metadata"] = {}
                    
                    for table in workspace_data["tables"]:
                        if table not in workspace_data["metadata"]:
                            workspace_data["metadata"][table] = self._get_table_metadata(table)
                    
                    custom_workspaces[workspace_name] = workspace_data
                except Exception as e:
                    logger.error(f"Error loading workspace from {file}: {e}")
            
            return custom_workspaces
        except Exception as e:
            logger.error(f"Error loading custom workspaces: {e}")
            return {}
    
    def _get_table_metadata(self, table_name):
        """
        Get metadata for a table
        
        Parameters:
        table_name (str): Table name
        
        Returns:
        dict: Table metadata
        """
        if not self.conn:
            return {"columns": []}
        
        try:
            # Check if table exists
            if not table_exists(self.conn, table_name):
                return {"columns": []}
            
            # Get schema
            schema_df = get_table_schema(self.conn, table_name)
            
            # Get column information
            columns = []
            key_columns = []
            for _, row in schema_df.iterrows():
                column_info = {
                    "name": row["name"],
                    "type": row["type"],
                    "nullable": row["notnull"] == 0
                }
                
                # Check if it's a key column
                if "pk" in schema_df.columns and row["pk"] > 0:
                    column_info["is_key"] = True
                    key_columns.append(row["name"])
                else:
                    column_info["is_key"] = False
                
                columns.append(column_info)
            
            # Get row count
            try:
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                count_df = pd.read_sql_query(count_query, self.conn)
                row_count = count_df["count"].iloc[0] if not count_df.empty else 0
            except Exception as e:
                logger.warning(f"Error getting row count for {table_name}: {e}")
                row_count = 0
            
            # Get sample data
            try:
                sample_query = f"SELECT * FROM {table_name} LIMIT 3"
                sample_df = pd.read_sql_query(sample_query, self.conn)
                sample_data = sample_df.to_dict("records") if not sample_df.empty else []
            except Exception as e:
                logger.warning(f"Error getting sample data for {table_name}: {e}")
                sample_data = []
            
            return {
                "columns": columns,
                "key_columns": key_columns,
                "row_count": row_count,
                "sample_data": sample_data
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {table_name}: {e}")
            return {"columns": []}
    
    def get_workspace(self, workspace_name):
        """
        Get a workspace by name
        
        Parameters:
        workspace_name (str): Workspace name
        
        Returns:
        dict: Workspace data or None if not found
        """
        # Check system workspaces first
        if workspace_name in self.system_workspaces:
            return self.system_workspaces[workspace_name]
        
        # Then check custom workspaces
        if workspace_name in self.custom_workspaces:
            return self.custom_workspaces[workspace_name]
        
        return None
    
    def get_all_workspaces(self):
        """
        Get all available workspaces
        
        Returns:
        dict: Dictionary of all workspaces
        """
        # Combine system and custom workspaces
        all_workspaces = self.system_workspaces.copy()
        all_workspaces.update(self.custom_workspaces)
        return all_workspaces
    
    def create_custom_workspace(self, name, tables, description=""):
        """
        Create a custom workspace
        
        Parameters:
        name (str): Workspace name
        tables (list): List of table names
        description (str, optional): Workspace description
        
        Returns:
        dict: Created workspace data
        """
        # Validate tables
        valid_tables = []
        if self.conn:
            for table in tables:
                if table_exists(self.conn, table):
                    valid_tables.append(table)
                else:
                    logger.warning(f"Table {table} does not exist, skipping")
        else:
            valid_tables = tables
        
        # Create workspace data
        workspace_data = {
            "tables": valid_tables,
            "description": description,
            "metadata": {},
            "sample_transformations": []
        }
        
        # Get table metadata for each table
        for table in valid_tables:
            workspace_data["metadata"][table] = self._get_table_metadata(table)
        
        # Save to custom workspaces
        self.custom_workspaces[name] = workspace_data
        
        # Save to file
        self._save_custom_workspace(name, workspace_data)
        
        logger.info(f"Created custom workspace '{name}' with {len(valid_tables)} tables")
        
        return workspace_data
    
    def _save_custom_workspace(self, name, workspace_data):
        """
        Save a custom workspace to file
        
        Parameters:
        name (str): Workspace name
        workspace_data (dict): Workspace data
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            # Prepare data for saving
            save_data = workspace_data.copy()
            save_data["name"] = name
            
            # Save to file
            file_path = os.path.join(self.workspaces_path, f"{name}.json")
            with open(file_path, "w") as f:
                json.dump(save_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving custom workspace '{name}': {e}")
            return False
    
    def add_sample_transformation(self, workspace_name, transformation):
        """
        Add a sample transformation to a workspace
        
        Parameters:
        workspace_name (str): Workspace name
        transformation (str): Sample transformation description
        
        Returns:
        bool: True if successful, False otherwise
        """
        # Check if workspace exists
        workspace = self.get_workspace(workspace_name)
        if not workspace:
            logger.warning(f"Workspace '{workspace_name}' not found")
            return False
        
        # Add transformation if it doesn't already exist
        if "sample_transformations" not in workspace:
            workspace["sample_transformations"] = []
        
        if transformation not in workspace["sample_transformations"]:
            workspace["sample_transformations"].append(transformation)
            
            # Save if it's a custom workspace
            if workspace_name in self.custom_workspaces:
                self._save_custom_workspace(workspace_name, workspace)
            
            logger.info(f"Added sample transformation to workspace '{workspace_name}'")
            return True
        
        return False
    
    def get_workspace_tables(self, workspace_name):
        """
        Get tables for a specific workspace
        
        Parameters:
        workspace_name (str): Workspace name
        
        Returns:
        list: List of table names
        """
        workspace = self.get_workspace(workspace_name)
        if not workspace:
            logger.warning(f"Workspace '{workspace_name}' not found")
            return []
        
        return workspace.get("tables", [])
    
    def close(self):
        """
        Close the database connection
        """
        if self.conn:
            self.conn.close()
            self.conn = None
