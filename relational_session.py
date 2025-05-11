"""
Relational Session Manager for TableLLM

This module provides a relational data model approach to session management,
supporting multiple tables per session with relationships between them.
It maintains an ER-like structure of tables and their relationships.
"""

import os
import json
import uuid
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import sqlite3
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class RelationalError(Exception):
    """Exception raised for errors in relational operations."""
    pass

class RelationalSessionManager:
    """
    Manages a relational model of tables and their relationships within a session
    """

    def __init__(self, storage_path="sessions"):
        """
        Initialize a RelationalSessionManager
        
        Args:
            storage_path (str): Base directory for session storage
        """
        self.storage_path = storage_path
        try:
            os.makedirs(storage_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create session storage directory: {e}")
            raise RelationalError(f"Failed to create session storage: {e}")

    def create_session(self):
        """
        Create a new session with relational schema support
        
        Returns:
            str: New session ID
        """
        try:
            session_id = str(uuid.uuid4())
            session_path = f"{self.storage_path}/{session_id}"
            
            # Create session directory structure
            os.makedirs(session_path, exist_ok=True)
            os.makedirs(f"{session_path}/tables", exist_ok=True)
            os.makedirs(f"{session_path}/relationships", exist_ok=True)
            
            # Initialize session schema
            schema = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "active_segment": None,
                "execution_counter": 0,
                "tables": {},
                "relationships": [],
                "execution_history": []
            }
            
            # Save schema
            with open(f"{session_path}/schema.json", "w") as f:
                json.dump(schema, f, indent=2)
                
            # Legacy support: Create empty context.json for backward compatibility
            legacy_context = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "transformation_history": [],
                "target_table_state": {
                    "populated_fields": [],
                    "remaining_mandatory_fields": [],
                    "total_rows": 0,
                    "rows_with_data": 0,
                },
            }
            
            with open(f"{session_path}/context.json", "w") as f:
                json.dump(legacy_context, f, indent=2)
                
            logger.info(f"Created new relational session: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to create relational session: {e}")
            raise RelationalError(f"Failed to create session: {e}")

    def get_schema(self, session_id):
        """
        Get the current schema for a session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            dict: The session schema or None if not found
        """
        try:
            if not session_id:
                logger.warning("No session ID provided for get_schema")
                return None

            schema_path = f"{self.storage_path}/{session_id}/schema.json"
            if not os.path.exists(schema_path):
                logger.warning(f"Schema file not found for session {session_id}")
                return None

            with open(schema_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in schema file for session {session_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get schema for session {session_id}: {e}")
            return None

    def update_schema(self, session_id, schema):
        """
        Update the schema for a session
        
        Args:
            session_id (str): Session ID
            schema (dict): The updated schema
            
        Returns:
            dict: The updated schema or None on failure
        """
        try:
            if not session_id:
                logger.warning("No session ID provided for update_schema")
                raise RelationalError("No session ID provided")

            if not schema:
                logger.warning("No schema provided for update_schema")
                raise RelationalError("No schema provided")

            session_path = f"{self.storage_path}/{session_id}"
            if not os.path.exists(session_path):
                logger.warning(f"Session directory not found: {session_path}")
                os.makedirs(session_path, exist_ok=True)

            schema_path = f"{session_path}/schema.json"

            # Save the updated schema
            with open(schema_path, "w") as f:
                json.dump(schema, f, indent=2)

            # Also save as a versioned snapshot for history
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            snapshot_path = f"{session_path}/schema_{timestamp}.json"
            with open(snapshot_path, "w") as f:
                json.dump(schema, f, indent=2)

            return schema
        except Exception as e:
            logger.error(f"Failed to update schema for session {session_id}: {e}")
            raise RelationalError(f"Failed to update schema: {e}")

    def get_active_segment(self, session_id):
        """
        Get the active segment for a session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            int: The active segment ID or None if not set
        """
        try:
            schema = self.get_schema(session_id)
            if not schema:
                return None
                
            return schema.get("active_segment")
        except Exception as e:
            logger.error(f"Failed to get active segment for session {session_id}: {e}")
            return None

    def set_active_segment(self, session_id, segment_id):
        """
        Set the active segment for a session
        
        Args:
            session_id (str): Session ID
            segment_id (int): Segment ID to set as active
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            schema = self.get_schema(session_id)
            if not schema:
                logger.error(f"No schema found for session {session_id}")
                return False
                
            # Update the active segment
            schema["active_segment"] = segment_id
            
            # Add to execution history
            schema["execution_history"].append({
                "action": "switch_segment",
                "segment_id": segment_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save the updated schema
            self.update_schema(session_id, schema)
            logger.info(f"Set active segment {segment_id} for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set active segment for session {session_id}: {e}")
            return False

    def add_table(self, session_id, segment_id, table_name, fields=None, primary_keys=None):
        """
        Add a new table to the session schema
        
        Args:
            session_id (str): Session ID
            segment_id (int): Segment ID associated with the table
            table_name (str): Name of the table
            fields (list): List of field names in the table
            primary_keys (list): List of primary key field names
            
        Returns:
            dict: The updated table information or None on failure
        """
        try:
            if not session_id or not segment_id or not table_name:
                logger.warning(f"Missing required parameters for add_table: session_id={session_id}, segment_id={segment_id}, table_name={table_name}")
                return None
                
            schema = self.get_schema(session_id)
            if not schema:
                logger.error(f"No schema found for session {session_id}")
                return None
                
            # Initialize fields and primary keys if not provided
            fields = fields or []
            primary_keys = primary_keys or []
            
            # Increment execution counter
            schema["execution_counter"] += 1
            
            # Create table entry in schema
            table_info = {
                "segment_id": segment_id,
                "execution_order": schema["execution_counter"],
                "created_at": datetime.now().isoformat(),
                "fields": fields,
                "primary_keys": primary_keys,
                "last_updated": datetime.now().isoformat()
            }
            
            # Add to schema
            schema["tables"][table_name] = table_info
            
            # Set as active segment
            schema["active_segment"] = segment_id
            
            # Add to execution history
            schema["execution_history"].append({
                "action": "create_table",
                "segment_id": segment_id,
                "table_name": table_name,
                "execution_order": schema["execution_counter"],
                "timestamp": datetime.now().isoformat()
            })
            
            # Create directory for table data
            table_dir = f"{self.storage_path}/{session_id}/tables/{table_name}"
            os.makedirs(table_dir, exist_ok=True)
            
            # Save table metadata
            with open(f"{table_dir}/metadata.json", "w") as f:
                json.dump(table_info, f, indent=2)
            
            # Save the updated schema
            self.update_schema(session_id, schema)
            
            logger.info(f"Added table {table_name} for segment {segment_id} in session {session_id}")
            return table_info
        except Exception as e:
            logger.error(f"Failed to add table for session {session_id}: {e}")
            return None

    def update_table(self, session_id, table_name, fields=None, primary_keys=None):
        """
        Update an existing table's metadata
        
        Args:
            session_id (str): Session ID
            table_name (str): Name of the table to update
            fields (list): Updated list of field names
            primary_keys (list): Updated list of primary key field names
            
        Returns:
            dict: The updated table information or None on failure
        """
        try:
            if not session_id or not table_name:
                logger.warning(f"Missing required parameters for update_table: session_id={session_id}, table_name={table_name}")
                return None
                
            schema = self.get_schema(session_id)
            if not schema:
                logger.error(f"No schema found for session {session_id}")
                return None
                
            # Check if table exists
            if table_name not in schema["tables"]:
                logger.error(f"Table {table_name} not found in session {session_id}")
                return None
                
            # Get existing table info
            table_info = schema["tables"][table_name]
            
            # Update fields if provided
            if fields is not None:
                table_info["fields"] = fields
                
            # Update primary keys if provided
            if primary_keys is not None:
                table_info["primary_keys"] = primary_keys
                
            # Update last_updated timestamp
            table_info["last_updated"] = datetime.now().isoformat()
            
            # Update in schema
            schema["tables"][table_name] = table_info
            
            # Add to execution history
            schema["execution_history"].append({
                "action": "update_table",
                "table_name": table_name,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update table metadata
            table_dir = f"{self.storage_path}/{session_id}/tables/{table_name}"
            with open(f"{table_dir}/metadata.json", "w") as f:
                json.dump(table_info, f, indent=2)
            
            # Save the updated schema
            self.update_schema(session_id, schema)
            
            logger.info(f"Updated table {table_name} in session {session_id}")
            return table_info
        except Exception as e:
            logger.error(f"Failed to update table for session {session_id}: {e}")
            return None

    def get_table_info(self, session_id, table_name):
        """
        Get information about a specific table
        
        Args:
            session_id (str): Session ID
            table_name (str): Name of the table
            
        Returns:
            dict: Table information or None if not found
        """
        try:
            if not session_id or not table_name:
                logger.warning(f"Missing required parameters for get_table_info: session_id={session_id}, table_name={table_name}")
                return None
                
            schema = self.get_schema(session_id)
            if not schema:
                logger.error(f"No schema found for session {session_id}")
                return None
                
            # Return table info if exists
            return schema["tables"].get(table_name)
        except Exception as e:
            logger.error(f"Failed to get table info for session {session_id}: {e}")
            return None

    def add_relationship(self, session_id, from_table, to_table, mapping, relationship_type="many_to_one"):
        """
        Add a relationship between two tables
        
        Args:
            session_id (str): Session ID
            from_table (str): Name of the source table
            to_table (str): Name of the target table
            mapping (dict): Dictionary mapping source fields to target fields
            relationship_type (str): Type of relationship (one_to_one, one_to_many, many_to_one, many_to_many)
            
        Returns:
            dict: The created relationship or None on failure
        """
        try:
            if not session_id or not from_table or not to_table or not mapping:
                logger.warning(f"Missing required parameters for add_relationship")
                return None
                
            schema = self.get_schema(session_id)
            if not schema:
                logger.error(f"No schema found for session {session_id}")
                return None
                
            # Validate that both tables exist
            if from_table not in schema["tables"]:
                logger.error(f"Source table {from_table} not found in session {session_id}")
                return None
                
            if to_table not in schema["tables"]:
                logger.error(f"Target table {to_table} not found in session {session_id}")
                return None
                
            # Create relationship entry
            relationship = {
                "id": str(uuid.uuid4()),
                "from_table": from_table,
                "to_table": to_table,
                "mapping": mapping,
                "type": relationship_type,
                "created_at": datetime.now().isoformat()
            }
            
            # Add to schema
            schema["relationships"].append(relationship)
            
            # Add to execution history
            schema["execution_history"].append({
                "action": "create_relationship",
                "from_table": from_table,
                "to_table": to_table,
                "relationship_type": relationship_type,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save relationship metadata
            rel_dir = f"{self.storage_path}/{session_id}/relationships"
            with open(f"{rel_dir}/{relationship['id']}.json", "w") as f:
                json.dump(relationship, f, indent=2)
            
            # Save the updated schema
            self.update_schema(session_id, schema)
            
            logger.info(f"Added relationship between {from_table} and {to_table} in session {session_id}")
            return relationship
        except Exception as e:
            logger.error(f"Failed to add relationship for session {session_id}: {e}")
            return None

    def get_relationships(self, session_id, table_name=None):
        """
        Get relationships for a session, optionally filtered by table
        
        Args:
            session_id (str): Session ID
            table_name (str): Optional table name to filter relationships
            
        Returns:
            list: List of relationship dictionaries
        """
        try:
            if not session_id:
                logger.warning("No session ID provided for get_relationships")
                return []
                
            schema = self.get_schema(session_id)
            if not schema:
                logger.error(f"No schema found for session {session_id}")
                return []
                
            relationships = schema.get("relationships", [])
            
            # Filter by table if specified
            if table_name:
                relationships = [
                    rel for rel in relationships 
                    if rel["from_table"] == table_name or rel["to_table"] == table_name
                ]
                
            return relationships
        except Exception as e:
            logger.error(f"Failed to get relationships for session {session_id}: {e}")
            return []

    def save_table_data(self, session_id, table_name, dataframe):
        """
        Save table data for a session
        
        Args:
            session_id (str): Session ID
            table_name (str): Name of the table
            dataframe (pandas.DataFrame): The table data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not session_id or not table_name:
                logger.warning(f"Missing required parameters for save_table_data: session_id={session_id}, table_name={table_name}")
                return False
                
            if dataframe is None or not isinstance(dataframe, pd.DataFrame):
                logger.warning(f"Invalid dataframe for save_table_data: {type(dataframe)}")
                return False
                
            # Get schema to ensure table exists
            schema = self.get_schema(session_id)
            if not schema or table_name not in schema["tables"]:
                logger.error(f"Table {table_name} not found in session {session_id}")
                return False
                
            # Create paths for table data
            table_dir = f"{self.storage_path}/{session_id}/tables/{table_name}"
            os.makedirs(table_dir, exist_ok=True)
            
            latest_path = f"{table_dir}/latest.csv"
            
            # Save timestamped version for history
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            history_path = f"{table_dir}/{timestamp}.csv"
            
            # Save the dataframe
            dataframe.to_csv(latest_path, index=False)
            dataframe.to_csv(history_path, index=False)
            
            # Update table metadata with field count
            table_info = schema["tables"][table_name]
            table_info["last_updated"] = datetime.now().isoformat()
            table_info["row_count"] = len(dataframe)
            table_info["fields"] = dataframe.columns.tolist()
            
            # Update schema
            schema["tables"][table_name] = table_info
            self.update_schema(session_id, schema)
            
            # Update table metadata file
            with open(f"{table_dir}/metadata.json", "w") as f:
                json.dump(table_info, f, indent=2)
            
            logger.info(f"Saved table data for {table_name} in session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save table data for session {session_id}: {e}")
            return False

    def get_table_data(self, session_id, table_name):
        """
        Get table data for a session
        
        Args:
            session_id (str): Session ID
            table_name (str): Name of the table
            
        Returns:
            pandas.DataFrame: The table data or empty DataFrame if not found
        """
        try:
            if not session_id or not table_name:
                logger.warning(f"Missing required parameters for get_table_data: session_id={session_id}, table_name={table_name}")
                return pd.DataFrame()
                
            # Get schema to ensure table exists
            schema = self.get_schema(session_id)
            if not schema or table_name not in schema["tables"]:
                logger.error(f"Table {table_name} not found in session {session_id}")
                return pd.DataFrame()
                
            # Check if table data exists
            table_path = f"{self.storage_path}/{session_id}/tables/{table_name}/latest.csv"
            if not os.path.exists(table_path):
                logger.warning(f"No data file found for table {table_name} in session {session_id}")
                return pd.DataFrame()
                
            # Load and return dataframe
            try:
                df = pd.read_csv(table_path)
                return df
            except Exception as e:
                logger.error(f"Error reading table data for {table_name}: {e}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to get table data for session {session_id}: {e}")
            return pd.DataFrame()

    def get_segment_tables(self, session_id, segment_id):
        """
        Get all tables for a specific segment
        
        Args:
            session_id (str): Session ID
            segment_id (int): Segment ID
            
        Returns:
            dict: Dictionary of table names and their information
        """
        try:
            if not session_id or not segment_id:
                logger.warning(f"Missing required parameters for get_segment_tables: session_id={session_id}, segment_id={segment_id}")
                return {}
                
            schema = self.get_schema(session_id)
            if not schema:
                logger.error(f"No schema found for session {session_id}")
                return {}
                
            # Filter tables by segment_id
            segment_tables = {
                table_name: table_info
                for table_name, table_info in schema["tables"].items()
                if table_info["segment_id"] == segment_id
            }
            
            return segment_tables
        except Exception as e:
            logger.error(f"Failed to get segment tables for session {session_id}: {e}")
            return {}

    def get_tables_by_execution_order(self, session_id):
        """
        Get all tables in execution order
        
        Args:
            session_id (str): Session ID
            
        Returns:
            list: List of (table_name, table_info) tuples sorted by execution_order
        """
        try:
            if not session_id:
                logger.warning("No session ID provided for get_tables_by_execution_order")
                return []
                
            schema = self.get_schema(session_id)
            if not schema:
                logger.error(f"No schema found for session {session_id}")
                return []
                
            # Get tables and sort by execution_order
            tables = list(schema["tables"].items())
            tables.sort(key=lambda x: x[1]["execution_order"])
            
            return tables
        except Exception as e:
            logger.error(f"Failed to get tables by execution order for session {session_id}: {e}")
            return []

    def find_or_create_table_for_segment(self, session_id, segment_id, target_table_name, conn=None):
        """
        Find an existing table for a segment or create a new one
        
        Args:
            session_id (str): Session ID
            segment_id (int): Segment ID
            target_table_name (str): Target table name to use when creating
            conn (sqlite3.Connection): Optional database connection
            
        Returns:
            tuple: (table_name, DataFrame, is_new_table)
        """
        try:
            if not session_id or not segment_id:
                logger.warning(f"Missing required parameters for find_or_create_table_for_segment")
                return None, pd.DataFrame(), False
                
            # Get schema
            schema = self.get_schema(session_id)
            if not schema:
                logger.error(f"No schema found for session {session_id}")
                return None, pd.DataFrame(), False
                
            # Check if this segment already has tables
            segment_tables = self.get_segment_tables(session_id, segment_id)
            
            if segment_tables:
                # Use the first table found for this segment
                table_name = next(iter(segment_tables.keys()))
                table_df = self.get_table_data(session_id, table_name)
                logger.info(f"Found existing table {table_name} for segment {segment_id}")
                return table_name, table_df, False
            
            # No table found, create a new one with a unique name
            unique_table_name = f"{target_table_name}_{segment_id}_{len(schema['tables']) + 1}"
            
            # Get table structure from database if connection provided
            fields = []
            primary_keys = []
            
            if conn:
                try:
                    # Validate table name
                    safe_table = self._validate_sql_identifier(target_table_name)
                    
                    # Get table information
                    cursor = conn.cursor()
                    cursor.execute(f"PRAGMA table_info({safe_table})")
                    table_info = cursor.fetchall()
                    
                    # Extract field names and primary keys
                    fields = [row[1] for row in table_info]
                    primary_keys = [row[1] for row in table_info if row[5] > 0]  # SQLite primary key column
                except Exception as e:
                    logger.warning(f"Error getting table structure from database: {e}")
                    # Continue with empty fields
            
            # Create the new table entry
            table_info = self.add_table(
                session_id, 
                segment_id, 
                unique_table_name, 
                fields=fields,
                primary_keys=primary_keys
            )
            
            if not table_info:
                logger.error(f"Failed to create new table for segment {segment_id}")
                return None, pd.DataFrame(), False
                
            # Set this segment as active
            self.set_active_segment(session_id, segment_id)
            
            # Create empty dataframe with fields if available
            if fields:
                table_df = pd.DataFrame(columns=fields)
            else:
                table_df = pd.DataFrame()
                
            # Save the empty dataframe
            self.save_table_data(session_id, unique_table_name, table_df)
            
            logger.info(f"Created new table {unique_table_name} for segment {segment_id}")
            return unique_table_name, table_df, True
        except Exception as e:
            logger.error(f"Failed to find or create table for segment {segment_id}: {e}")
            return None, pd.DataFrame(), False

    def detect_relationship_from_query(self, session_id, query, source_table, target_table, source_fields, target_fields):
        """
        Attempt to detect a relationship from a query
        
        Args:
            session_id (str): Session ID
            query (str): The query text
            source_table (str): Source table name
            target_table (str): Target table name
            source_fields (list): Source fields used in the query
            target_fields (list): Target fields used in the query
            
        Returns:
            dict: Detected relationship or None
        """
        try:
            if not session_id or not query or not source_table or not target_table:
                logger.warning(f"Missing required parameters for detect_relationship_from_query")
                return None
                
            # Get schema
            schema = self.get_schema(session_id)
            if not schema:
                logger.error(f"No schema found for session {session_id}")
                return None
                
            # Check if source and target tables exist
            if source_table not in schema["tables"] or target_table not in schema["tables"]:
                logger.warning(f"Source or target table not found in schema")
                return None
                
            # Simple pattern matching for common relationship indicators
            join_pattern = r'\b(?:join|connect|link|map|relate)\b'
            on_pattern = r'\bon\b|\bwhere\b|\busing\b|\bmatching\b|\bwith\b'
            equal_pattern = r'(\w+)\s*=\s*(\w+)'
            
            # Check if query indicates a relationship
            if re.search(join_pattern, query.lower()) and re.search(on_pattern, query.lower()):
                # Look for field equality patterns
                equal_matches = re.findall(equal_pattern, query)
                
                if equal_matches:
                    # Extract potential field mappings
                    mapping = {}
                    for left, right in equal_matches:
                        # Clean up field names
                        left = left.strip()
                        right = right.strip()
                        
                        # Check if these fields are in our source and target fields
                        if left in source_fields and right in target_fields:
                            mapping[left] = right
                        elif left in target_fields and right in source_fields:
                            mapping[right] = left
                    
                    if mapping:
                        # Default to many_to_one relationship
                        relationship_type = "many_to_one"
                        
                        # Look for one-to-one indicators
                        if re.search(r'\bone[- ]to[- ]one\b|\bexactly one\b|\bunique\b|\bdistinct\b', query.lower()):
                            relationship_type = "one_to_one"
                        # Look for one-to-many indicators
                        elif re.search(r'\bone[- ]to[- ]many\b|\bmultiple\b', query.lower()):
                            relationship_type = "one_to_many"
                        # Look for many-to-many indicators
                        elif re.search(r'\bmany[- ]to[- ]many\b', query.lower()):
                            relationship_type = "many_to_many"
                        
                        return {
                            "from_table": source_table,
                            "to_table": target_table,
                            "mapping": mapping,
                            "type": relationship_type
                        }
            
            # If no relationship detected from the query, but we have source and target
            # with overlapping fields, suggest a potential relationship
            if source_fields and target_fields:
                # Find common field names that might indicate a relationship
                common_fields = set(source_fields).intersection(set(target_fields))
                if common_fields:
                    mapping = {field: field for field in common_fields}
                    return {
                        "from_table": source_table,
                        "to_table": target_table,
                        "mapping": mapping,
                        "type": "suggested_relation"  # Mark as suggested since we're inferring
                    }
            
            return None
        except Exception as e:
            logger.error(f"Failed to detect relationship from query: {e}")
            return None

    def _validate_sql_identifier(self, identifier):
        """
        Validate that an SQL identifier doesn't contain injection attempts
        
        Args:
            identifier (str): SQL identifier to validate
            
        Returns:
            str: Sanitized identifier
            
        Raises:
            RelationalError: If identifier contains potential SQL injection patterns
        """
        if not identifier:
            raise RelationalError("Empty SQL identifier provided")

        # Check for common SQL injection patterns
        dangerous_patterns = [
            ";", "--", "/*", "*/", "DROP", "DELETE", "INSERT", 
            "UPDATE", "UNION", "EXEC", "EXECUTE",
        ]
        for pattern in dangerous_patterns:
            if pattern.lower() in identifier.lower():
                raise RelationalError(f"Potentially dangerous SQL pattern found: {pattern}")

        # Only allow alphanumeric characters, underscores, and some specific characters
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", identifier):
            raise RelationalError("SQL identifier contains invalid characters")
            
        return identifier

    def get_or_create_table_for_segment(self, session_id, segment_id, target_table_name, conn=None):
        """
        Find or create a table for a segment
        
        Args:
            session_id (str): Session ID
            segment_id (int): Segment ID
            target_table_name (str): Target table name
            conn (sqlite3.Connection): Optional DB connection
            
        Returns:
            tuple: (table_name, dataframe)
        """
        table_name, df, is_new = self.find_or_create_table_for_segment(
            session_id, segment_id, target_table_name, conn
        )
        return table_name, df

    def merge_tables(self, session_id, source_table_name, target_table_name, mapping, key_fields):
        """
        Merge data from one table to another based on key fields
        
        Args:
            session_id (str): Session ID
            source_table_name (str): Source table name
            target_table_name (str): Target table name
            mapping (dict): Mapping of source fields to target fields
            key_fields (dict): Mapping of source key fields to target key fields
            
        Returns:
            pandas.DataFrame: The merged dataframe
        """
        try:
            if not session_id or not source_table_name or not target_table_name or not mapping or not key_fields:
                logger.warning(f"Missing required parameters for merge_tables")
                return pd.DataFrame()
                
            # Get source and target dataframes
            source_df = self.get_table_data(session_id, source_table_name)
            target_df = self.get_table_data(session_id, target_table_name)
            
            if source_df.empty:
                logger.warning(f"Source table {source_table_name} is empty")
                return target_df
                
            # Create a copy of target_df to avoid modifying the original
            result_df = target_df.copy()
            
            # If target is empty, create a new dataframe
            if result_df.empty:
                # Create a new dataframe with all needed columns
                columns = set(mapping.values()) | set(key_fields.values())
                result_df = pd.DataFrame(columns=list(columns))
                
                # Add source data with proper mapping
                for _, source_row in source_df.iterrows():
                    new_row = {}
                    
                    # Map key fields
                    for source_key, target_key in key_fields.items():
                        if source_key in source_df.columns:
                            new_row[target_key] = source_row[source_key]
                    
                    # Map other fields
                    for source_field, target_field in mapping.items():
                        if source_field in source_df.columns:
                            new_row[target_field] = source_row[source_field]
                    
                    # Add the new row
                    result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                # Merge data based on key fields
                for _, source_row in source_df.iterrows():
                    # Create a mask to find matching rows in target
                    mask = pd.Series(True, index=result_df.index)
                    for source_key, target_key in key_fields.items():
                        if source_key in source_df.columns and target_key in result_df.columns:
                            mask &= (result_df[target_key] == source_row[source_key])
                    
                    # If matching rows found, update them
                    if mask.any():
                        for source_field, target_field in mapping.items():
                            if source_field in source_df.columns and target_field in result_df.columns:
                                result_df.loc[mask, target_field] = source_row[source_field]
                    else:
                        # No match found, add a new row
                        new_row = {}
                        
                        # Map key fields
                        for source_key, target_key in key_fields.items():
                            if source_key in source_df.columns:
                                new_row[target_key] = source_row[source_key]
                        
                        # Map other fields
                        for source_field, target_field in mapping.items():
                            if source_field in source_df.columns:
                                new_row[target_field] = source_row[source_field]
                        
                        # Add the new row
                        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save the merged dataframe
            self.save_table_data(session_id, target_table_name, result_df)
            
            # Add relationship between tables if it doesn't exist
            relationship = {source_key: target_key for source_key, target_key in key_fields.items()}
            relationship.update({source_field: target_field for source_field, target_field in mapping.items()})
            
            # Check if relationship already exists
            existing_relationships = self.get_relationships(session_id)
            has_relationship = False
            for rel in existing_relationships:
                if rel["from_table"] == source_table_name and rel["to_table"] == target_table_name:
                    has_relationship = True
                    break
            
            # Add relationship if it doesn't exist
            if not has_relationship:
                self.add_relationship(
                    session_id,
                    source_table_name,
                    target_table_name,
                    relationship,
                    "many_to_one"  # Default type for merges
                )
            
            return result_df
        except Exception as e:
            logger.error(f"Failed to merge tables: {e}")
            return pd.DataFrame()

    def migrate_legacy_session(self, session_id):
        """
        Migrate a legacy session to the relational model
        
        Args:
            session_id (str): Session ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not session_id:
                logger.warning("No session ID provided for migrate_legacy_session")
                return False
                
            session_path = f"{self.storage_path}/{session_id}"
            
            # Check if it's a legacy session (has context.json but no schema.json)
            if not os.path.exists(f"{session_path}/context.json"):
                logger.warning(f"No context.json found for session {session_id}")
                return False
                
            if os.path.exists(f"{session_path}/schema.json"):
                logger.info(f"Session {session_id} already has schema.json, no migration needed")
                return True
                
            # Create directory structure
            os.makedirs(f"{session_path}/tables", exist_ok=True)
            os.makedirs(f"{session_path}/relationships", exist_ok=True)
            
            # Load legacy context
            with open(f"{session_path}/context.json", "r") as f:
                legacy_context = json.load(f)
                
            # Check if legacy target file exists
            legacy_target_path = f"{session_path}/target_latest.csv"
            if os.path.exists(legacy_target_path):
                # Load legacy target dataframe
                legacy_df = pd.read_csv(legacy_target_path)
                
                # Get legacy segment ID
                legacy_segment_id = legacy_context.get("active_segment")
                
                # If no segment ID, use a default
                if not legacy_segment_id:
                    # Try to extract from metadata
                    if "context" in legacy_context and "segment_id" in legacy_context["context"]:
                        legacy_segment_id = legacy_context["context"]["segment_id"]
                    else:
                        legacy_segment_id = 0
                
                # Create a schema
                schema = {
                    "session_id": session_id,
                    "created_at": legacy_context.get("created_at", datetime.now().isoformat()),
                    "active_segment": legacy_segment_id,
                    "execution_counter": 1,
                    "tables": {},
                    "relationships": [],
                    "execution_history": []
                }
                
                # Add legacy table
                table_name = f"legacy_table_{legacy_segment_id}"
                
                # Add table to schema
                schema["tables"][table_name] = {
                    "segment_id": legacy_segment_id,
                    "execution_order": 1,
                    "created_at": legacy_context.get("created_at", datetime.now().isoformat()),
                    "fields": legacy_df.columns.tolist(),
                    "primary_keys": [],  # No way to detect primary keys reliably
                    "last_updated": datetime.now().isoformat()
                }
                
                # Add to execution history
                schema["execution_history"].append({
                    "action": "migrate_legacy",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Save schema
                with open(f"{session_path}/schema.json", "w") as f:
                    json.dump(schema, f, indent=2)
                    
                # Create table directory and copy data
                table_dir = f"{session_path}/tables/{table_name}"
                os.makedirs(table_dir, exist_ok=True)
                
                # Copy legacy data
                import shutil
                shutil.copy(legacy_target_path, f"{table_dir}/latest.csv")
                
                # Also save table metadata
                with open(f"{table_dir}/metadata.json", "w") as f:
                    json.dump(schema["tables"][table_name], f, indent=2)
                
                logger.info(f"Successfully migrated legacy session {session_id}")
                return True
            else:
                # No legacy target file, just create an empty schema
                schema = {
                    "session_id": session_id,
                    "created_at": legacy_context.get("created_at", datetime.now().isoformat()),
                    "active_segment": None,
                    "execution_counter": 0,
                    "tables": {},
                    "relationships": [],
                    "execution_history": [{
                        "action": "migrate_legacy_empty",
                        "timestamp": datetime.now().isoformat()
                    }]
                }
                
                # Save schema
                with open(f"{session_path}/schema.json", "w") as f:
                    json.dump(schema, f, indent=2)
                
                logger.info(f"Created empty schema for legacy session {session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to migrate legacy session {session_id}: {e}")
            return False

    def to_legacy_context(self, session_id):
        """
        Convert relational session data to legacy context format for backward compatibility
        
        Args:
            session_id (str): Session ID
            
        Returns:
            dict: Legacy context data
        """
        try:
            if not session_id:
                logger.warning("No session ID provided for to_legacy_context")
                return None
                
            # Get schema
            schema = self.get_schema(session_id)
            if not schema:
                logger.error(f"No schema found for session {session_id}")
                return None
                
            # Get active segment
            active_segment = schema.get("active_segment")
            
            # Get tables for active segment
            segment_tables = {}
            if active_segment:
                segment_tables = self.get_segment_tables(session_id, active_segment)
            
            # If no active segment or no tables, get the most recently created table
            if not segment_tables:
                tables = self.get_tables_by_execution_order(session_id)
                if tables:
                    # Get the latest table (highest execution order)
                    latest_table = tables[-1]
                    table_name = latest_table[0]
                    table_info = latest_table[1]
                    active_segment = table_info["segment_id"]
                    segment_tables = {table_name: table_info}
            
            # Build legacy context
            legacy_context = {
                "session_id": session_id,
                "created_at": schema.get("created_at", datetime.now().isoformat()),
                "context": {
                    "segment_id": active_segment,
                    "transformation_history": schema.get("execution_history", []),
                },
                "target_table_state": {
                    "populated_fields": [],
                    "remaining_mandatory_fields": [],
                    "total_rows": 0,
                    "rows_with_data": 0,
                }
            }
            
            # If we have an active table, populate target_table_state
            if segment_tables:
                table_name = next(iter(segment_tables.keys()))
                table_info = segment_tables[table_name]
                table_df = self.get_table_data(session_id, table_name)
                
                # Update target_table_state
                legacy_context["target_table_state"]["populated_fields"] = table_info.get("fields", [])
                legacy_context["target_table_state"]["total_rows"] = len(table_df) if not table_df.empty else 0
                
                # Count rows with data
                if not table_df.empty:
                    rows_with_data = (table_df.count(axis=1) > 0).sum()
                    legacy_context["target_table_state"]["rows_with_data"] = rows_with_data
            
            return legacy_context
        except Exception as e:
            logger.error(f"Failed to convert to legacy context for session {session_id}: {e}")
            return None

    def validate_table_integrity(self, session_id, table_name):
        """
        Validate the integrity of a table based on relationships
        
        Args:
            session_id (str): Session ID
            table_name (str): Table name to validate
            
        Returns:
            dict: Validation results with any issues found
        """
        try:
            if not session_id or not table_name:
                logger.warning(f"Missing required parameters for validate_table_integrity")
                return {"valid": False, "issues": ["Missing required parameters"]}
                
            # Get table data
            table_df = self.get_table_data(session_id, table_name)
            if table_df.empty:
                return {"valid": True, "issues": []}  # Empty table has no integrity issues
                
            # Get table info
            table_info = self.get_table_info(session_id, table_name)
            if not table_info:
                return {"valid": False, "issues": [f"Table {table_name} not found"]}
                
            # Get relationships where this table is the target
            relationships = self.get_relationships(session_id, table_name)
            target_relationships = [r for r in relationships if r["to_table"] == table_name]
            
            issues = []
            
            # Check primary key integrity
            primary_keys = table_info.get("primary_keys", [])
            if primary_keys:
                for key in primary_keys:
                    # Check for nulls
                    if key in table_df.columns and table_df[key].isna().any():
                        issues.append(f"Primary key {key} contains NULL values")
                    
                    # Check for duplicates
                    if key in table_df.columns and table_df[key].duplicated().any():
                        issues.append(f"Primary key {key} contains duplicate values")
            
            # Check referential integrity for relationships
            for rel in target_relationships:
                source_table = rel["from_table"]
                source_df = self.get_table_data(session_id, source_table)
                
                if source_df.empty:
                    continue  # No data to check
                
                # Check each mapping
                for source_field, target_field in rel["mapping"].items():
                    if source_field not in source_df.columns:
                        issues.append(f"Source field {source_field} not found in table {source_table}")
                        continue
                        
                    if target_field not in table_df.columns:
                        issues.append(f"Target field {target_field} not found in table {table_name}")
                        continue
                    
                    # Check for orphaned records (reference integrity)
                    if rel["type"] in ["many_to_one", "one_to_one"]:
                        # Get unique values from source field
                        source_values = source_df[source_field].dropna().unique()
                        
                        # Check if all source values exist in target
                        target_values = table_df[target_field].dropna().unique()
                        missing_values = set(source_values) - set(target_values)
                        
                        if missing_values:
                            issues.append(f"Relationship integrity issue: {len(missing_values)} values in {source_table}.{source_field} not found in {table_name}.{target_field}")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues
            }
        except Exception as e:
            logger.error(f"Failed to validate table integrity: {e}")
            return {"valid": False, "issues": [f"Validation error: {str(e)}"]}

    def generate_er_diagram(self, session_id):
        """
        Generate an ER diagram representation of the session's tables and relationships
        
        Args:
            session_id (str): Session ID
            
        Returns:
            str: Mermaid diagram code for the ER diagram
        """
        try:
            if not session_id:
                logger.warning("No session ID provided for generate_er_diagram")
                return "```mermaid\nerDiagram\n  %% No session ID provided\n```"
                
            # Get schema
            schema = self.get_schema(session_id)
            if not schema:
                logger.error(f"No schema found for session {session_id}")
                return "```mermaid\nerDiagram\n  %% No schema found\n```"
                
            # Generate Mermaid ER diagram
            mermaid = "```mermaid\nerDiagram\n"
            
            # Add tables
            for table_name, table_info in schema["tables"].items():
                # Clean table name for Mermaid (no spaces or special chars)
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
                
                # Add table definition
                mermaid += f"  {clean_name} {{\n"
                
                # Add fields
                for field in table_info.get("fields", []):
                    # Mark primary keys
                    if field in table_info.get("primary_keys", []):
                        mermaid += f"    {field} PK\n"
                    else:
                        mermaid += f"    {field}\n"
                
                mermaid += "  }\n"
            
            # Add relationships
            rel_count = 0
            for rel in schema.get("relationships", []):
                from_table = re.sub(r'[^a-zA-Z0-9_]', '_', rel["from_table"])
                to_table = re.sub(r'[^a-zA-Z0-9_]', '_', rel["to_table"])
                
                # Map relationship type to Mermaid notation
                if rel["type"] == "one_to_one":
                    cardinality = "||--||"
                elif rel["type"] == "one_to_many":
                    cardinality = "||--o{"
                elif rel["type"] == "many_to_one":
                    cardinality = "}o--||"
                elif rel["type"] == "many_to_many":
                    cardinality = "}o--o{"
                else:
                    cardinality = "--"
                
                # Add relationship with label
                rel_count += 1
                mermaid += f"  {from_table} {cardinality} {to_table} : rel{rel_count}\n"
            
            mermaid += "```"
            return mermaid
        except Exception as e:
            logger.error(f"Failed to generate ER diagram: {e}")
            return "```mermaid\nerDiagram\n  %% Error generating diagram\n```"