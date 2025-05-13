"""
Relational Functions for TableLLM

This module provides functions to integrate the relational model with the existing TableLLM system.
It includes functions for managing segments, tables, and handling the transition between segments.
"""

import logging
import sqlite3
import pandas as pd
from relational_session import RelationalSessionManager
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_or_create_segment_target_df(session_id, segment_id, target_table, conn=None):
    """
    Get or create a target dataframe for a specific segment
    
    Args:
        session_id (str): Session ID
        segment_id (int): Segment ID
        target_table (str): Target table name
        conn (sqlite3.Connection): Optional database connection
        
    Returns:
        DataFrame: The target dataframe
    """
    try:
        if not session_id or not segment_id or not target_table:
            logger.warning("Missing required parameters in get_or_create_segment_target_df")
            return pd.DataFrame()
            
        # Initialize relational session manager
        rsm = RelationalSessionManager()
        
        # Find or create table for this segment
        table_name, df = rsm.get_or_create_table_for_segment(
            session_id, segment_id, target_table, conn
        )
        
        if df is None or df.empty:
            # If we got an empty dataframe, try to get schema from DB
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute(f"PRAGMA table_info({target_table})")
                    columns = [row[1] for row in cursor.fetchall()]
                    df = pd.DataFrame(columns=columns)
                except Exception as e:
                    logger.warning(f"Error getting table schema: {e}")
                    df = pd.DataFrame()
        
        return df
    except Exception as e:
        logger.error(f"Error in get_or_create_segment_target_df: {e}")
        return pd.DataFrame()

def save_segment_target_df(session_id, segment_id, target_table, dataframe):
    """
    Save target dataframe for a specific segment
    
    Args:
        session_id (str): Session ID
        segment_id (int): Segment ID
        target_table (str): Target table name
        dataframe (DataFrame): The dataframe to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not session_id or not segment_id or not target_table:
            logger.warning("Missing required parameters in save_segment_target_df")
            return False
            
        if dataframe is None or not isinstance(dataframe, pd.DataFrame):
            logger.warning(f"Invalid dataframe: {type(dataframe)}")
            return False
            
        # Initialize relational session manager
        rsm = RelationalSessionManager()
        
        # Find existing tables for this segment
        segment_tables = rsm.get_segment_tables(session_id, segment_id)
        
        if segment_tables:
            # Use the first table found
            table_name = next(iter(segment_tables.keys()))
            
            # Save to the existing table
            return rsm.save_table_data(session_id, table_name, dataframe)
        else:
            # Create a new table for this segment
            table_name = f"{target_table}_{segment_id}"
            
            # Add table to schema
            rsm.add_table(
                session_id, 
                segment_id, 
                table_name, 
                fields=dataframe.columns.tolist(),
                primary_keys=[]  # No way to detect primary keys reliably
            )
            
            # Save data
            return rsm.save_table_data(session_id, table_name, dataframe)
    except Exception as e:
        logger.error(f"Error in save_segment_target_df: {e}")
        return False

def detect_relationship_from_results(session_id, segment_id, previous_segment_id, resolved_data, code_result):
    """
    Detect and create a relationship between segment tables based on query results
    
    Args:
        session_id (str): Session ID
        segment_id (int): Current segment ID
        previous_segment_id (int): Previous segment ID
        resolved_data (dict): Resolved data from planner
        code_result (DataFrame): Result of code execution
        
    Returns:
        dict: Detected relationship or None
    """
    try:
        # Skip if the segments are the same
        if segment_id == previous_segment_id:
            return None
            
        # Initialize relational session manager
        rsm = RelationalSessionManager()
        
        # Get tables for current and previous segments
        current_tables = rsm.get_segment_tables(session_id, segment_id)
        previous_tables = rsm.get_segment_tables(session_id, previous_segment_id)
        
        if not current_tables or not previous_tables:
            logger.warning("Missing current or previous tables")
            return None
            
        # Get the first table from each segment
        current_table_name = next(iter(current_tables.keys()))
        previous_table_name = next(iter(previous_tables.keys()))
        
        # Get field data
        current_fields = current_tables[current_table_name].get("fields", [])
        previous_fields = previous_tables[previous_table_name].get("fields", [])
        
        # Try to detect relationship from the query
        query = resolved_data.get("restructured_query", "")
        source_fields = resolved_data.get("source_field_names", [])
        target_fields = []
        if isinstance(resolved_data.get("target_sap_fields"), str):
            target_fields = [resolved_data["target_sap_fields"]]
        elif isinstance(resolved_data.get("target_sap_fields"), list):
            target_fields = resolved_data["target_sap_fields"]
            
        # See if there are key mappings
        key_mapping = resolved_data.get("key_mapping", [])
        mapping = {}
        
        if key_mapping and isinstance(key_mapping, list):
            for item in key_mapping:
                if isinstance(item, dict) and "target_col" in item and "source_col" in item:
                    mapping[item["source_col"]] = item["target_col"]
        
        # Try to detect relationship from the query
        relationship = rsm.detect_relationship_from_query(
            session_id,
            query,
            previous_table_name,  # Previous segment table is the source
            current_table_name,   # Current segment table is the target
            previous_fields,      # Fields in previous table
            current_fields        # Fields in current table
        )
        
        # If relationship detected, add it
        if relationship:
            return rsm.add_relationship(
                session_id,
                relationship["from_table"],
                relationship["to_table"],
                relationship["mapping"],
                relationship["type"]
            )
            
        # If no relationship detected from query, try to infer from common fields
        common_fields = set(previous_fields).intersection(set(current_fields))
        if common_fields:
            # Use common fields as a basis for relationship
            field_mapping = {field: field for field in common_fields}
            
            return rsm.add_relationship(
                session_id,
                previous_table_name,
                current_table_name,
                field_mapping,
                "inferred_relation"  # Mark as inferred
            )
            
        return None
    except Exception as e:
        logger.error(f"Error in detect_relationship_from_results: {e}")
        return None

def handle_segment_switch(session_id, segment_id, object_id, project_id, target_table, conn=None):
    """
    Handle switching between segments in a session
    
    Args:
        session_id (str): Session ID
        segment_id (int): New segment ID to switch to
        object_id (int): Object ID
        project_id (int): Project ID
        target_table (str): Target table name
        conn (sqlite3.Connection): Optional database connection
        
    Returns:
        tuple: (previous_segment_id, dataframe)
    """
    try:
        # Initialize relational session manager
        rsm = RelationalSessionManager()
        
        # Get current active segment
        schema = rsm.get_schema(session_id)
        if not schema:
            # No schema, try to migrate legacy session
            migrated = rsm.migrate_legacy_session(session_id)
            if migrated:
                schema = rsm.get_schema(session_id)
            else:
                # Create new session
                rsm.create_session()
                schema = rsm.get_schema(session_id)
        
        previous_segment_id = schema.get("active_segment")
        
        # If switching to the same segment, just return current dataframe
        if previous_segment_id == segment_id:
            # Get current segment tables
            segment_tables = rsm.get_segment_tables(session_id, segment_id)
            if segment_tables:
                table_name = next(iter(segment_tables.keys()))
                df = rsm.get_table_data(session_id, table_name)
                return previous_segment_id, df
            
        # Switching to a different segment
        # Set new active segment
        rsm.set_active_segment(session_id, segment_id)
        
        # Get or create target dataframe for the new segment
        df = get_or_create_segment_target_df(session_id, segment_id, target_table, conn)
        
        return previous_segment_id, df
    except Exception as e:
        logger.error(f"Error in handle_segment_switch: {e}")
        return None, pd.DataFrame()

def replace_get_or_create_target_df(session_id, segment_id, target_table, conn):
    """
    Replacement for the legacy get_or_create_session_target_df function
    
    Args:
        session_id (str): Session ID
        segment_id (int): Segment ID
        target_table (str): Target table name
        conn (sqlite3.Connection): Database connection
        
    Returns:
        DataFrame: The target dataframe
    """
    return get_or_create_segment_target_df(session_id, segment_id, target_table, conn)

def replace_save_target_df(session_id, segment_id, target_table, dataframe):
    """
    Replacement for the legacy save_session_target_df function
    
    Args:
        session_id (str): Session ID
        segment_id (int): Segment ID
        target_table (str): Target table name
        dataframe (DataFrame): The dataframe to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    return save_segment_target_df(session_id, segment_id, target_table, dataframe)

def get_segment_relationships(session_id, segment_id):
    """
    Get relationships for tables in a segment
    
    Args:
        session_id (str): Session ID
        segment_id (int): Segment ID
        
    Returns:
        list: List of relationship dictionaries
    """
    try:
        # Initialize relational session manager
        rsm = RelationalSessionManager()
        
        # Get tables for the segment
        segment_tables = rsm.get_segment_tables(session_id, segment_id)
        if not segment_tables:
            return []
            
        # Get table names
        table_names = list(segment_tables.keys())
        
        # Get relationships where any of these tables participate
        all_relationships = []
        for table_name in table_names:
            relationships = rsm.get_relationships(session_id, table_name)
            all_relationships.extend(relationships)
            
        # Remove duplicates
        unique_relationships = []
        seen_ids = set()
        for rel in all_relationships:
            if rel["id"] not in seen_ids:
                unique_relationships.append(rel)
                seen_ids.add(rel["id"])
                
        return unique_relationships
    except Exception as e:
        logger.error(f"Error in get_segment_relationships: {e}")
        return []

def generate_session_diagram(session_id):
    """
    Generate an ER diagram for the entire session
    
    Args:
        session_id (str): Session ID
        
    Returns:
        str: Mermaid diagram code
    """
    try:
        # Initialize relational session manager
        rsm = RelationalSessionManager()
        
        # Generate diagram
        return rsm.generate_er_diagram(session_id)
    except Exception as e:
        logger.error(f"Error in generate_session_diagram: {e}")
        return "```mermaid\nerDiagram\n  %% Error generating diagram\n```"

def merge_segment_tables(session_id, from_segment_id, to_segment_id, mapping, key_mapping):
    """
    Merge data from one segment's table to another segment's table
    
    Args:
        session_id (str): Session ID
        from_segment_id (int): Source segment ID
        to_segment_id (int): Target segment ID
        mapping (dict): Field mapping
        key_mapping (dict): Key field mapping
        
    Returns:
        DataFrame: The merged dataframe
    """
    try:
        # Initialize relational session manager
        rsm = RelationalSessionManager()
        
        # Get tables for both segments
        from_tables = rsm.get_segment_tables(session_id, from_segment_id)
        to_tables = rsm.get_segment_tables(session_id, to_segment_id)
        
        if not from_tables or not to_tables:
            logger.warning("Missing source or target tables")
            return pd.DataFrame()
            
        # Get the first table from each segment
        from_table_name = next(iter(from_tables.keys()))
        to_table_name = next(iter(to_tables.keys()))
        
        # Merge tables
        return rsm.merge_tables(session_id, from_table_name, to_table_name, mapping, key_mapping)
    except Exception as e:
        logger.error(f"Error in merge_segment_tables: {e}")
        return pd.DataFrame()
