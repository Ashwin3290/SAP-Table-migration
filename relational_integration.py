"""
Relational Integration Module

This module provides a high-level interface for integrating the relational model
with the existing TableLLM system. It serves as a bridge between the old and new approaches.
"""

import logging
from dmtool_relational import RelationalDMTool
from relational_session import RelationalSessionManager
from relational_functions import (
    get_or_create_segment_target_df,
    save_segment_target_df,
    handle_segment_switch,
    generate_session_diagram,
    get_segment_relationships
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TableLLMRelational:
    """
    Main interface for using TableLLM with the relational model.
    This class provides methods for working with multi-segment operations.
    """
    
    def __init__(self):
        """Initialize the TableLLMRelational instance"""
        self.dm_tool = RelationalDMTool()
        self.session_manager = RelationalSessionManager()
    
    def process_query(self, query, object_id, segment_id, project_id, session_id=None, target_sap_fields=None):
        """
        Process a query using the relational model
        
        Args:
            query (str): The user's query
            object_id (int): Object ID for mapping
            segment_id (int): Segment ID for mapping
            project_id (int): Project ID for mapping
            session_id (str): Optional session ID
            target_sap_fields (str/list): Optional target fields
            
        Returns:
            tuple: (code, result, session_id)
        """
        return self.dm_tool.process_sequential_query(
            query, object_id, segment_id, project_id, session_id
        )
    
    def get_session_info(self, session_id):
        """
        Get information about a session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            dict: Session information
        """
        if not session_id:
            return {"error": "No session ID provided"}
            
        try:
            # Get schema
            schema = self.session_manager.get_schema(session_id)
            if not schema:
                # Try to migrate legacy session
                migrated = self.session_manager.migrate_legacy_session(session_id)
                if migrated:
                    schema = self.session_manager.get_schema(session_id)
                else:
                    return {"error": "Session not found"}
            
            # Get number of tables and segments
            tables = schema.get("tables", {})
            segments = set(table_info.get("segment_id") for table_info in tables.values())
            
            # Get relationships
            relationships = self.session_manager.get_relationships(session_id)
            
            return {
                "session_id": session_id,
                "created_at": schema.get("created_at"),
                "active_segment": schema.get("active_segment"),
                "table_count": len(tables),
                "segment_count": len(segments),
                "relationship_count": len(relationships),
                "segments": list(segments),
                "tables": list(tables.keys()),
                "execution_counter": schema.get("execution_counter", 0)
            }
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {"error": str(e)}
    
    def get_segment_tables(self, session_id, segment_id):
        """
        Get all tables for a specific segment
        
        Args:
            session_id (str): Session ID
            segment_id (int): Segment ID
            
        Returns:
            dict: Dictionary of table names and their information
        """
        return self.session_manager.get_segment_tables(session_id, segment_id)
    
    def get_table_data(self, session_id, table_name):
        """
        Get data for a specific table
        
        Args:
            session_id (str): Session ID
            table_name (str): Table name
            
        Returns:
            DataFrame: The table data
        """
        return self.session_manager.get_table_data(session_id, table_name)
    
    def get_session_diagram(self, session_id):
        """
        Generate an ER diagram for the session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            str: Mermaid diagram code
        """
        return self.dm_tool.get_session_diagram(session_id)
    
    def validate_session_integrity(self, session_id):
        """
        Validate the integrity of all tables in a session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            dict: Validation results
        """
        return self.dm_tool.validate_session_integrity(session_id)
    
    def get_session_relationships(self, session_id):
        """
        Get all relationships for a session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            list: List of relationship dictionaries
        """
        return self.dm_tool.get_session_relationships(session_id)
    
    def migrate_legacy_session(self, session_id):
        """
        Migrate a legacy session to the relational model
        
        Args:
            session_id (str): Session ID
            
        Returns:
            bool: True if successful
        """
        return self.session_manager.migrate_legacy_session(session_id)
