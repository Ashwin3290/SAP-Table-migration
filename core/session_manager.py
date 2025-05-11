"""
Session Manager for TableLLM
This module manages sessions with segment awareness
"""
import os
import json
import uuid
import pandas as pd
from datetime import datetime
from utils.logging_utils import main_logger as logger
import config

class SessionManager:
    """
    Manages sessions with segment awareness
    """
    
    def __init__(self, storage_path=None):
        """
        Initialize the session manager
        
        Parameters:
        storage_path (str, optional): Path to session storage directory
        """
        # Set storage path
        self.storage_path = storage_path if storage_path else config.SESSIONS_DIR
        
        # Create directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        logger.info(f"Initialized SessionManager with storage path: {self.storage_path}")
    
    def create_session(self):
        """
        Create a new session and return its ID
        
        Returns:
        str: Session ID
        """
        try:
            # Generate a unique session ID
            session_id = str(uuid.uuid4())
            
            # Create session directory
            session_path = os.path.join(self.storage_path, session_id)
            os.makedirs(session_path, exist_ok=True)
            
            # Initialize session context
            context = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "transformation_history": [],
                "processed_segments": {},
                "segment_dependencies": {},
                "key_mappings": {}
            }
            
            # Save context
            self._save_context(session_id, context)
            
            logger.info(f"Created session: {session_id}")
            
            return session_id
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None
    
    def get_context(self, session_id):
        """
        Get the current context for a session
        
        Parameters:
        session_id (str): Session ID
        
        Returns:
        dict: Session context
        """
        try:
            if not session_id:
                logger.warning("No session ID provided for get_context")
                return None
            
            # Check if session exists
            context_path = os.path.join(self.storage_path, session_id, "context.json")
            if not os.path.exists(context_path):
                logger.warning(f"Session not found: {session_id}")
                return None
            
            # Load context
            with open(context_path, "r") as f:
                context = json.load(f)
            
            return context
        except Exception as e:
            logger.error(f"Error getting context for session {session_id}: {e}")
            return None
    
    def _save_context(self, session_id, context):
        """
        Save context for a session
        
        Parameters:
        session_id (str): Session ID
        context (dict): Session context
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            if not session_id:
                logger.warning("No session ID provided for _save_context")
                return False
            
            # Update timestamp
            context["updated_at"] = datetime.now().isoformat()
            
            # Create session directory if it doesn't exist
            session_path = os.path.join(self.storage_path, session_id)
            os.makedirs(session_path, exist_ok=True)
            
            # Save context
            context_path = os.path.join(session_path, "context.json")
            with open(context_path, "w") as f:
                json.dump(context, f, indent=2)
            
            # Also save as a versioned snapshot for history
            snapshot_path = os.path.join(
                session_path,
                f"context_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            )
            with open(snapshot_path, "w") as f:
                json.dump(context, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving context for session {session_id}: {e}")
            return False
    
    def update_context(self, session_id, updates):
        """
        Update context for a session
        
        Parameters:
        session_id (str): Session ID
        updates (dict): Updates to apply to the context
        
        Returns:
        dict: Updated context or None if failed
        """
        try:
            # Get current context
            context = self.get_context(session_id)
            if not context:
                logger.warning(f"Cannot update context for non-existent session: {session_id}")
                return None
            
            # Apply updates
            for key, value in updates.items():
                if isinstance(value, dict) and key in context and isinstance(context[key], dict):
                    # Deep update for dictionaries
                    context[key].update(value)
                else:
                    # Simple update for other types
                    context[key] = value
            
            # Save updated context
            success = self._save_context(session_id, context)
            if not success:
                logger.warning(f"Failed to save updated context for session {session_id}")
                return None
            
            return context
        except Exception as e:
            logger.error(f"Error updating context for session {session_id}: {e}")
            return None
    
    def update_segment_status(self, session_id, segment_name, status):
        """
        Update the status of a segment in the session
        
        Parameters:
        session_id (str): Session ID
        segment_name (str): Segment name
        status (str): Status to set ("not_started", "in_progress", "completed", "failed")
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            # Get current context
            context = self.get_context(session_id)
            if not context:
                logger.warning(f"Cannot update segment status for non-existent session: {session_id}")
                return False
            
            # Initialize processed_segments if it doesn't exist
            if "processed_segments" not in context:
                context["processed_segments"] = {}
            
            # Update segment status
            context["processed_segments"][segment_name] = {
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
            
            # Save updated context
            success = self._save_context(session_id, context)
            return success
        except Exception as e:
            logger.error(f"Error updating segment status for session {session_id}: {e}")
            return False
    
    def get_segment_status(self, session_id, segment_name):
        """
        Get the status of a segment in the session
        
        Parameters:
        session_id (str): Session ID
        segment_name (str): Segment name
        
        Returns:
        dict: Segment status or None if not found
        """
        try:
            # Get current context
            context = self.get_context(session_id)
            if not context:
                logger.warning(f"Cannot get segment status for non-existent session: {session_id}")
                return None
            
            # Check if processed_segments exists
            if "processed_segments" not in context:
                logger.warning(f"No processed segments found for session {session_id}")
                return None
            
            # Get segment status
            return context["processed_segments"].get(segment_name)
        except Exception as e:
            logger.error(f"Error getting segment status for session {session_id}: {e}")
            return None
    
    def add_transformation_history(self, session_id, transformation_data):
        """
        Add transformation data to the session history
        
        Parameters:
        session_id (str): Session ID
        transformation_data (dict): Transformation data to add
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            # Get current context
            context = self.get_context(session_id)
            if not context:
                logger.warning(f"Cannot add transformation history for non-existent session: {session_id}")
                return False
            
            # Initialize transformation_history if it doesn't exist
            if "transformation_history" not in context:
                context["transformation_history"] = []
            
            # Add timestamp to transformation data
            transformation_data["timestamp"] = datetime.now().isoformat()
            
            # Add to history
            context["transformation_history"].append(transformation_data)
            
            # Save updated context
            success = self._save_context(session_id, context)
            return success
        except Exception as e:
            logger.error(f"Error adding transformation history for session {session_id}: {e}")
            return False
    
    def get_transformation_history(self, session_id):
        """
        Get the transformation history for a session
        
        Parameters:
        session_id (str): Session ID
        
        Returns:
        list: Transformation history
        """
        try:
            # Get current context
            context = self.get_context(session_id)
            if not context:
                logger.warning(f"Cannot get transformation history for non-existent session: {session_id}")
                return []
            
            # Return transformation history or empty list if not found
            return context.get("transformation_history", [])
        except Exception as e:
            logger.error(f"Error getting transformation history for session {session_id}: {e}")
            return []
    
    def add_key_mapping(self, session_id, target_column, source_column):
        """
        Add a key mapping to the session
        
        Parameters:
        session_id (str): Session ID
        target_column (str): Target column name
        source_column (str): Source column name
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            # Get current context
            context = self.get_context(session_id)
            if not context:
                logger.warning(f"Cannot add key mapping for non-existent session: {session_id}")
                return False
            
            # Initialize key_mappings if it doesn't exist
            if "key_mappings" not in context:
                context["key_mappings"] = {}
            
            # Add mapping
            context["key_mappings"][target_column] = source_column
            
            # Save updated context
            success = self._save_context(session_id, context)
            return success
        except Exception as e:
            logger.error(f"Error adding key mapping for session {session_id}: {e}")
            return False
    
    def get_key_mappings(self, session_id):
        """
        Get all key mappings for a session
        
        Parameters:
        session_id (str): Session ID
        
        Returns:
        dict: Key mappings
        """
        try:
            # Get current context
            context = self.get_context(session_id)
            if not context:
                logger.warning(f"Cannot get key mappings for non-existent session: {session_id}")
                return {}
            
            # Return key mappings or empty dict if not found
            return context.get("key_mappings", {})
        except Exception as e:
            logger.error(f"Error getting key mappings for session {session_id}: {e}")
            return {}
    
    def get_or_create_session_target_df(self, session_id, target_table, conn=None):
        """
        Get the current target dataframe for a session, or create a new one
        
        Parameters:
        session_id (str): Session ID
        target_table (str): Target table name
        conn (sqlite3.Connection, optional): Database connection
        
        Returns:
        pandas.DataFrame: Target dataframe
        """
        try:
            # Check if session exists
            session_path = os.path.join(self.storage_path, session_id)
            target_path = os.path.join(session_path, "target_latest.csv")
            
            # If target file exists, load it
            if os.path.exists(target_path):
                try:
                    target_df = pd.read_csv(target_path)
                    return target_df
                except Exception as e:
                    logger.warning(f"Error reading target file for session {session_id}: {e}")
            
            # Target file doesn't exist or couldn't be loaded, create from database if conn provided
            if conn:
                try:
                    # Get data from database
                    query = f"SELECT * FROM {target_table}"
                    target_df = pd.read_sql_query(query, conn)
                    
                    # Save to file
                    os.makedirs(session_path, exist_ok=True)
                    target_df.to_csv(target_path, index=False)
                    
                    return target_df
                except Exception as e:
                    logger.warning(f"Error getting target data from database for session {session_id}: {e}")
            
            # Create empty dataframe
            target_df = pd.DataFrame()
            
            # Save to file
            os.makedirs(session_path, exist_ok=True)
            target_df.to_csv(target_path, index=False)
            
            return target_df
        except Exception as e:
            logger.error(f"Error getting target dataframe for session {session_id}: {e}")
            return pd.DataFrame()
    
    def save_session_target_df(self, session_id, target_df):
        """
        Save the target dataframe for a session
        
        Parameters:
        session_id (str): Session ID
        target_df (pandas.DataFrame): Target dataframe
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            # Create session directory if it doesn't exist
            session_path = os.path.join(self.storage_path, session_id)
            os.makedirs(session_path, exist_ok=True)
            
            # Save to latest file
            target_path = os.path.join(session_path, "target_latest.csv")
            target_df.to_csv(target_path, index=False)
            
            # Also save timestamped version
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            history_path = os.path.join(session_path, f"target_{timestamp}.csv")
            target_df.to_csv(history_path, index=False)
            
            return True
        except Exception as e:
            logger.error(f"Error saving target dataframe for session {session_id}: {e}")
            return False
    
    def list_sessions(self):
        """
        List all sessions
        
        Returns:
        list: List of session information dictionaries
        """
        try:
            # Get all session directories
            sessions = []
            
            for session_id in os.listdir(self.storage_path):
                session_path = os.path.join(self.storage_path, session_id)
                
                # Skip if not a directory
                if not os.path.isdir(session_path):
                    continue
                
                # Get context file
                context_path = os.path.join(session_path, "context.json")
                if not os.path.exists(context_path):
                    continue
                
                try:
                    # Load context
                    with open(context_path, "r") as f:
                        context = json.load(f)
                    
                    # Get basic session info
                    session_info = {
                        "session_id": session_id,
                        "created_at": context.get("created_at"),
                        "updated_at": context.get("updated_at"),
                        "num_transformations": len(context.get("transformation_history", [])),
                        "processed_segments": list(context.get("processed_segments", {}).keys())
                    }
                    
                    sessions.append(session_info)
                except Exception as e:
                    logger.warning(f"Error loading context for session {session_id}: {e}")
            
            return sessions
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
            # Check if session exists
            session_path = os.path.join(self.storage_path, session_id)
            if not os.path.exists(session_path) or not os.path.isdir(session_path):
                logger.warning(f"Session not found: {session_id}")
                return False
            
            # Delete all files in the session directory
            for file in os.listdir(session_path):
                file_path = os.path.join(session_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.warning(f"Error deleting file {file_path}: {e}")
            
            # Delete the session directory
            os.rmdir(session_path)
            
            logger.info(f"Deleted session: {session_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    def set_segment_dependency(self, session_id, segment_name, dependency_name):
        """
        Set a dependency between segments
        
        Parameters:
        session_id (str): Session ID
        segment_name (str): Segment name
        dependency_name (str): Dependency segment name
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            # Get current context
            context = self.get_context(session_id)
            if not context:
                logger.warning(f"Cannot set segment dependency for non-existent session: {session_id}")
                return False
            
            # Initialize segment_dependencies if it doesn't exist
            if "segment_dependencies" not in context:
                context["segment_dependencies"] = {}
            
            # Initialize dependencies for this segment if it doesn't exist
            if segment_name not in context["segment_dependencies"]:
                context["segment_dependencies"][segment_name] = []
            
            # Add dependency if it doesn't already exist
            if dependency_name not in context["segment_dependencies"][segment_name]:
                context["segment_dependencies"][segment_name].append(dependency_name)
            
            # Save updated context
            success = self._save_context(session_id, context)
            return success
        except Exception as e:
            logger.error(f"Error setting segment dependency for session {session_id}: {e}")
            return False
    
    def get_segment_dependencies(self, session_id, segment_name):
        """
        Get dependencies for a segment
        
        Parameters:
        session_id (str): Session ID
        segment_name (str): Segment name
        
        Returns:
        list: List of dependency segment names
        """
        try:
            # Get current context
            context = self.get_context(session_id)
            if not context:
                logger.warning(f"Cannot get segment dependencies for non-existent session: {session_id}")
                return []
            
            # Check if segment_dependencies exists
            if "segment_dependencies" not in context:
                return []
            
            # Check if this segment has dependencies
            if segment_name not in context["segment_dependencies"]:
                return []
            
            # Return dependencies
            return context["segment_dependencies"][segment_name]
        except Exception as e:
            logger.error(f"Error getting segment dependencies for session {session_id}: {e}")
            return []
