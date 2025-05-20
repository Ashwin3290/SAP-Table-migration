import sqlite3
import pandas as pd
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Fix for the SQLite thread issue in workspace_db.py
# Add this at the beginning of the WorkspaceDB class:

import threading

class WorkspaceDB:
    """Manages temporary workspace tables for cross-segment operations"""
    


class WorkspaceDB:
    """Manages temporary workspace tables for cross-segment operations"""
    
    def __init__(self, db_path="workspace.db"):
        self.db_path = db_path
        self._local = threading.local()  # Thread-local storage for connections
        self.initialize_db()
        
    def initialize_db(self):
        """Initialize workspace database with thread-safe connection"""
        try:
            # Create thread-local connection if it doesn't exist
            if not hasattr(self._local, 'conn') or self._local.conn is None:
                # Create a new connection for this thread
                self._local.conn = sqlite3.connect(self.db_path)
                
                # Create metadata table to track tables by session
                self._local.conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_metadata (
                    session_id TEXT,
                    table_name TEXT,
                    segment_id INTEGER,
                    segment_name TEXT,
                    created_at TEXT,
                    PRIMARY KEY (session_id, table_name)
                )
                """)
                self._local.conn.commit()
                logger.info(f"Initialized workspace database at {self.db_path} for thread {threading.current_thread().name}")
        except Exception as e:
            logger.error(f"Error initializing workspace database: {e}")
            raise
            
    @property
    def conn(self):
        """Get the connection for the current thread"""
        # Make sure this thread has its own connection
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self.initialize_db()
        return self._local.conn
    
    def close(self):
        """Close the connection for the current thread only"""
        try:
            if hasattr(self._local, 'conn') and self._local.conn is not None:
                self._local.conn.close()
                self._local.conn = None
                logger.info(f"Closed workspace connection for thread {threading.current_thread().name}")
        except Exception as e:
            logger.error(f"Error closing workspace connection: {e}")

    def save_segment_table(self, session_id, segment_id, segment_name, df):
        """Save a dataframe as a segment table in the workspace"""
        try:
            if self.conn is None:
                self.initialize_db()
                
            # Sanitize segment name for table name - remove special chars and spaces
            sanitized_name = ''.join(c if c.isalnum() else '_' for c in segment_name.lower())
            table_name = f"{sanitized_name}_{session_id[-8:]}"
            
            # Save dataframe to SQLite
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)
            
            # Update metadata
            self.conn.execute("""
            INSERT OR REPLACE INTO workspace_metadata 
            (session_id, table_name, segment_id, segment_name, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """, (session_id, table_name, segment_id, segment_name))
            
            self.conn.commit()
            logger.info(f"Saved segment table {table_name} for session {session_id}")
            return table_name
        except Exception as e:
            logger.error(f"Error saving segment table: {e}")
            return None
            
    def get_segment_table(self, session_id, segment_id=None, segment_name=None):
        """Get a segment table from the workspace"""
        try:
            if self.conn is None:
                self.initialize_db()
                
            # Query for the table name
            cursor = self.conn.cursor()
            query = None
            params = None
            
            if segment_id is not None:
                query = """
                SELECT table_name FROM workspace_metadata 
                WHERE session_id = ? AND segment_id = ?
                ORDER BY created_at DESC LIMIT 1
                """
                params = (session_id, segment_id)
            elif segment_name is not None:
                # Search for partial match of segment name (case insensitive)
                query = """
                SELECT table_name FROM workspace_metadata 
                WHERE session_id = ? AND lower(segment_name) LIKE lower(?)
                ORDER BY created_at DESC LIMIT 1
                """
                params = (session_id, f"%{segment_name}%")
            else:
                return None
                
            cursor.execute(query, params)
            result = cursor.fetchone()
            if not result:
                return None
                
            table_name = result[0]
            
            # Read dataframe from table
            df = pd.read_sql_query(f"SELECT * FROM '{table_name}'", self.conn)
            return df
        except Exception as e:
            logger.error(f"Error getting segment table: {e}")
            return None
    
    def get_segment_table_name(self, session_id, segment_id=None, segment_name=None):
        """Get the table name for a segment without loading the data"""
        try:
            if self.conn is None:
                self.initialize_db()
                
            # Query for the table name
            cursor = self.conn.cursor()
            query = None
            params = None
            
            if segment_id is not None:
                query = """
                SELECT table_name FROM workspace_metadata 
                WHERE session_id = ? AND segment_id = ?
                ORDER BY created_at DESC LIMIT 1
                """
                params = (session_id, segment_id)
            elif segment_name is not None:
                # Search for partial match of segment name (case insensitive)
                query = """
                SELECT table_name FROM workspace_metadata 
                WHERE session_id = ? AND lower(segment_name) LIKE lower(?)
                ORDER BY created_at DESC LIMIT 1
                """
                params = (session_id, f"%{segment_name}%")
            else:
                return None
                
            cursor.execute(query, params)
            result = cursor.fetchone()
            if not result:
                return None
                
            return result[0]
        except Exception as e:
            logger.error(f"Error getting segment table name: {e}")
            return None
            
    def list_session_tables(self, session_id):
        """List all tables for a session"""
        try:
            if self.conn is None:
                self.initialize_db()
                
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT table_name, segment_id, segment_name, created_at 
            FROM workspace_metadata 
            WHERE session_id = ?
            ORDER BY created_at
            """, (session_id,))
            
            results = cursor.fetchall()
            return [{"table_name": r[0], "segment_id": r[1], 
                    "segment_name": r[2], "created_at": r[3]} 
                    for r in results]
        except Exception as e:
            logger.error(f"Error listing session tables: {e}")
            return []
            
    def clear_session_tables(self, session_id):
        """Clear all tables for a session"""
        try:
            if self.conn is None:
                self.initialize_db()
                
            # Get all tables for the session
            tables = self.list_session_tables(session_id)
            
            # Drop each table
            for table in tables:
                self.conn.execute(f"DROP TABLE IF EXISTS '{table['table_name']}'")
                
            # Remove from metadata
            self.conn.execute("""
            DELETE FROM workspace_metadata WHERE session_id = ?
            """, (session_id,))
            
            self.conn.commit()
            logger.info(f"Cleared {len(tables)} tables for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing session tables: {e}")
            return False
            