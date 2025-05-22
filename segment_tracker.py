import re
import logging
from sql_executor import SQLExecutor
from typing import Dict, List, Any, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SegmentTracker:
    """Tracks segments and their associated target tables"""
    
    def __init__(self, db_path="db.sqlite3"):
        self.db_path = db_path
        self.sql_executor = SQLExecutor(db_path)
        self.segment_cache = {}  # Cache for segment info
    
    def get_segment_name(self, segment_id):
        """Get the segment name for a given segment ID"""
        if segment_id in self.segment_cache:
            return self.segment_cache[segment_id]
            
        try:
            query = "SELECT segement_name FROM connection_segments WHERE segment_id = ?"
            result = self.sql_executor.execute_query(query, {"segment_id": segment_id})
            
            if isinstance(result, list) and result:
                segment_name = result[0].get("segement_name")
                if segment_name:
                    # Cache the result
                    self.segment_cache[segment_id] = segment_name
                    return segment_name
            
            # If we get here, segment not found
            return f"Unknown_Segment_{segment_id}"
        except Exception as e:
            logger.error(f"Error getting segment name: {e}")
            return f"Error_Segment_{segment_id}"
    
    def register_segment_target_table(self, session_id, segment_id, target_table):
        """Register a target table for a segment in the current session"""
        try:
            # Get the segment name
            segment_name = self.get_segment_name(segment_id)
            
            # Load current session context
            from planner_sql import ContextualSessionManager
            context_manager = ContextualSessionManager()
            context = context_manager.get_context(session_id)
            
            if not context:
                context = {
                    "session_id": session_id,
                    "segments_visited": {},
                    "segment_target_tables": {}
                }
            
            # Initialize segment_target_tables if it doesn't exist
            if "segment_target_tables" not in context:
                context["segment_target_tables"] = {}
                
            # Normalize segment name for consistent lookup
            normalized_segment_name = segment_name.lower().replace(" ", "_")
            
            # Register the target table for this segment
            context["segment_target_tables"][normalized_segment_name] = target_table
            context["segment_target_tables"][str(segment_id)] = target_table
            
            # Also store common variations to help with matching
            segment_shortname = segment_name.split("_")[0].lower() if "_" in segment_name else segment_name.lower()
            context["segment_target_tables"][segment_shortname] = target_table
            
            # Update the context
            context_manager.update_context(session_id, context)
            
            logger.info(f"Registered target table {target_table} for segment {segment_name} (ID: {segment_id})")
            return True
        except Exception as e:
            logger.error(f"Error registering segment target table: {e}")
            return False
    
    def get_target_table_for_segment(self, session_id, segment_reference):
        """
        Get the target table for a segment reference
        
        Parameters:
        session_id (str): Session ID
        segment_reference (str): Reference to a segment (name, ID, or partial name)
        
        Returns:
        str: Target table name or None if not found
        """
        try:
            # Load current session context
            from planner_sql import ContextualSessionManager
            context_manager = ContextualSessionManager()
            context = context_manager.get_context(session_id)
            
            if not context or "segment_target_tables" not in context:
                return None
                
            segment_target_tables = context["segment_target_tables"]
            
            # Direct lookup
            if segment_reference in segment_target_tables:
                return segment_target_tables[segment_reference]
                
            # Try normalized form
            normalized_reference = segment_reference.lower().replace(" ", "_")
            if normalized_reference in segment_target_tables:
                return segment_target_tables[normalized_reference]
                
            # Try variations
            segment_reference_lower = segment_reference.lower()
            for segment_key, table in segment_target_tables.items():
                # Check for partial matches
                if (segment_reference_lower in segment_key.lower() or 
                    segment_key.lower() in segment_reference_lower):
                    return table
            
            return None
        except Exception as e:
            logger.error(f"Error getting target table for segment: {e}")
            return None
    
    def find_segment_references(self, query):
        """
        Find references to segments in a query
        
        Parameters:
        query (str): The query text
        
        Returns:
        list: List of potential segment references
        """
        try:
            segment_references = []
            
            # Common segment patterns
            segment_patterns = [
                r"(?i)([a-z_]+)\s+segment",  # Basic segment, Material segment, etc.
                r"(?i)segment\s+([a-z_]+)",  # Segment Basic, Segment Material, etc.
                r"(?i)transformation\s+(\d+)", # Transformation 1, Transformation 2, etc.
                r"(?i)output\s+of\s+transformation\s+([0-9,\s]+)" # Output of Transformation 1,2,3
            ]
            
            for pattern in segment_patterns:
                matches = re.findall(pattern, query)
                segment_references.extend(matches)
                
            # Remove duplicates and clean up
            segment_references = [ref.strip() for ref in segment_references if ref.strip()]
            return segment_references
        except Exception as e:
            logger.error(f"Error finding segment references: {e}")
            return []