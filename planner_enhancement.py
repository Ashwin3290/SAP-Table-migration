"""
Enhancement module for planner.py.
This file contains new classes and functions to enhance the planner module.
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd
import sqlite3
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import necessary classes and functions from planner
from planner import SessionError

class SegmentManager:
    """Manages segments and their relationships"""
    
    def __init__(self, storage_path="sessions"):
        self.storage_path = storage_path
        try:
            os.makedirs(storage_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create segment storage directory: {e}")
            raise SessionError(f"Failed to create segment storage: {e}")
    
    def get_available_segments(self, conn, project_id=None):
        """Get all available segments, optionally filtered by project"""
        try:
            query = "SELECT segment_id, obj_id_id, project_id_id, table_name FROM connection_segments"
            params = []
            
            if project_id:
                query += " WHERE project_id_id = ?"
                params.append(project_id)
                
            segments_df = pd.read_sql_query(query, conn, params=params)
            return segments_df
        except Exception as e:
            logger.error(f"Error getting available segments: {e}")
            return pd.DataFrame()
    
    def track_segment_change(self, session_id, current_segment, new_segment, parent_segment=None):
        """Track a segment change in the session"""
        try:
            segment_file = f"{self.storage_path}/{session_id}/segment_ledger.json"
            
            # Create or load existing ledger
            if os.path.exists(segment_file):
                with open(segment_file, 'r') as f:
                    ledger = json.load(f)
            else:
                ledger = {
                    "current_segment": current_segment,
                    "segment_history": [],
                    "segment_tree": {}
                }
            
            # Update ledger
            ledger["segment_history"].append({
                "timestamp": datetime.now().isoformat(),
                "from_segment": current_segment,
                "to_segment": new_segment,
                "parent_segment": parent_segment
            })
            
            ledger["current_segment"] = new_segment
            
            # Update segment tree
            if parent_segment:
                if parent_segment not in ledger["segment_tree"]:
                    ledger["segment_tree"][parent_segment] = []
                if new_segment not in ledger["segment_tree"][parent_segment]:
                    ledger["segment_tree"][parent_segment].append(new_segment)
            else:
                # This is a root segment
                if "root_segments" not in ledger:
                    ledger["root_segments"] = []
                if new_segment not in ledger["root_segments"]:
                    ledger["root_segments"].append(new_segment)
            
            # Save updated ledger
            with open(segment_file, 'w') as f:
                json.dump(ledger, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error tracking segment change: {e}")
            return False
            
    def get_current_segment(self, session_id):
        """Get the current segment for a session"""
        try:
            segment_file = f"{self.storage_path}/{session_id}/segment_ledger.json"
            
            if not os.path.exists(segment_file):
                return None
                
            with open(segment_file, 'r') as f:
                ledger = json.load(f)
                
            return ledger.get("current_segment")
        except Exception as e:
            logger.error(f"Error getting current segment: {e}")
            return None
            
    def get_segment_history(self, session_id):
        """Get the segment history for a session"""
        try:
            segment_file = f"{self.storage_path}/{session_id}/segment_ledger.json"
            
            if not os.path.exists(segment_file):
                return []
                
            with open(segment_file, 'r') as f:
                ledger = json.load(f)
                
            return ledger.get("segment_history", [])
        except Exception as e:
            logger.error(f"Error getting segment history: {e}")
            return []
            
    def get_segment_tree(self, session_id):
        """Get the segment tree for a session"""
        try:
            segment_file = f"{self.storage_path}/{session_id}/segment_ledger.json"
            
            if not os.path.exists(segment_file):
                return {}
                
            with open(segment_file, 'r') as f:
                ledger = json.load(f)
                
            return ledger.get("segment_tree", {})
        except Exception as e:
            logger.error(f"Error getting segment tree: {e}")
            return {}


class TableRelationshipManager:
    """Manages parent-child relationships between tables"""
    
    def __init__(self, storage_path="sessions"):
        self.storage_path = storage_path
        try:
            os.makedirs(storage_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create relationship storage directory: {e}")
            raise SessionError(f"Failed to create relationship storage: {e}")
    
    def establish_parent_child_relationship(self, session_id, parent_table, child_table, key_columns):
        """Establish a parent-child relationship between tables"""
        try:
            relationship_file = f"{self.storage_path}/{session_id}/table_relationships.json"
            
            # Create or load existing relationships
            if os.path.exists(relationship_file):
                with open(relationship_file, 'r') as f:
                    relationships = json.load(f)
            else:
                relationships = {
                    "root_table": parent_table,
                    "relationships": [],
                    "table_tree": {}
                }
            
            # Add new relationship
            relationship = {
                "parent_table": parent_table,
                "child_table": child_table,
                "key_columns": key_columns,
                "timestamp": datetime.now().isoformat()
            }
            relationships["relationships"].append(relationship)
            
            # Update table tree
            if parent_table not in relationships["table_tree"]:
                relationships["table_tree"][parent_table] = []
            if child_table not in relationships["table_tree"][parent_table]:
                relationships["table_tree"][parent_table].append(child_table)
            
            # Save updated relationships
            with open(relationship_file, 'w') as f:
                json.dump(relationships, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error establishing parent-child relationship: {e}")
            return False
            
    def get_table_relationships(self, session_id):
        """Get all table relationships for a session"""
        try:
            relationship_file = f"{self.storage_path}/{session_id}/table_relationships.json"
            
            if not os.path.exists(relationship_file):
                return {"root_table": None, "relationships": [], "table_tree": {}}
                
            with open(relationship_file, 'r') as f:
                relationships = json.load(f)
                
            return relationships
        except Exception as e:
            logger.error(f"Error getting table relationships: {e}")
            return {"root_table": None, "relationships": [], "table_tree": {}}
            
    def get_parent_tables(self, session_id, child_table):
        """Get all parent tables for a child table"""
        try:
            relationships = self.get_table_relationships(session_id)
            
            parent_tables = []
            for relationship in relationships.get("relationships", []):
                if relationship.get("child_table") == child_table:
                    parent_tables.append(relationship.get("parent_table"))
                    
            return parent_tables
        except Exception as e:
            logger.error(f"Error getting parent tables: {e}")
            return []
            
    def get_child_tables(self, session_id, parent_table):
        """Get all child tables for a parent table"""
        try:
            relationships = self.get_table_relationships(session_id)
            
            child_tables = []
            for relationship in relationships.get("relationships", []):
                if relationship.get("parent_table") == parent_table:
                    child_tables.append(relationship.get("child_table"))
                    
            return child_tables
        except Exception as e:
            logger.error(f"Error getting child tables: {e}")
            return []


def validate_entity_existence(conn, entity_type, entity_name):
    """Validate that a table, column, or segment exists in the database"""
    try:
        if entity_type == "table":
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (entity_name,))
            exists = cursor.fetchone() is not None
            if not exists:
                return False, f"Table '{entity_name}' does not exist in the database"
            
        elif entity_type == "column":
            # Parse table.column format
            if "." in entity_name:
                table_name, column_name = entity_name.split(".")
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [row[1] for row in cursor.fetchall()]
                if column_name not in columns:
                    return False, f"Column '{column_name}' does not exist in table '{table_name}'"
                
        elif entity_type == "segment":
            # Check if segment exists
            cursor = conn.cursor()
            cursor.execute("SELECT segment_id FROM connection_segments WHERE segment_id=?", (entity_name,))
            exists = cursor.fetchone() is not None
            if not exists:
                return False, f"Segment '{entity_name}' does not exist"
                
        return True, f"{entity_type.capitalize()} '{entity_name}' exists"
        
    except Exception as e:
        return False, f"Error validating {entity_type} '{entity_name}': {str(e)}"

# Additional functions for enhanced planner capabilities can be added here
