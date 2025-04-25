import os
import json
import pickle
import logging
import uuid
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TableOperation:
    """Represents a single table operation"""
    
    def __init__(self, operation_id=None, table_name=None, segment_id=None, operation_type=None, parent_id=None):
        self.operation_id = operation_id or str(uuid.uuid4())
        self.table_name = table_name
        self.segment_id = segment_id
        self.operation_type = operation_type  # 'query', 'update', 'join', etc.
        self.parent_id = parent_id
        self.children = []
        self.timestamp = datetime.now().isoformat()
        self.additional_info = {}
        
    def add_child(self, child_operation):
        """Add a child operation"""
        self.children.append(child_operation)
        
    def to_dict(self):
        """Convert operation to dictionary"""
        return {
            "operation_id": self.operation_id,
            "table_name": self.table_name,
            "segment_id": self.segment_id,
            "operation_type": self.operation_type,
            "parent_id": self.parent_id,
            "children": [child.to_dict() for child in self.children],
            "timestamp": self.timestamp,
            "additional_info": self.additional_info
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create operation from dictionary"""
        operation = cls(
            operation_id=data.get("operation_id"),
            table_name=data.get("table_name"),
            segment_id=data.get("segment_id"),
            operation_type=data.get("operation_type"),
            parent_id=data.get("parent_id")
        )
        operation.timestamp = data.get("timestamp")
        operation.additional_info = data.get("additional_info", {})
        
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data)
            operation.add_child(child)
            
        return operation

class OperationTracker:
    """Tracks and manages table operations"""
    
    def __init__(self, storage_path="sessions"):
        self.storage_path = storage_path
        try:
            os.makedirs(storage_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create operation tracker storage directory: {e}")
            raise Exception(f"Failed to create operation tracker storage: {e}")
            
    def create_root_operation(self, session_id, table_name, segment_id, operation_type="query"):
        """Create a root operation for a session"""
        try:
            operation = TableOperation(
                table_name=table_name,
                segment_id=segment_id,
                operation_type=operation_type
            )
            
            # Save operation
            return self.save_operation_tree(session_id, operation)
        except Exception as e:
            logger.error(f"Error creating root operation: {e}")
            return None
            
    def add_child_operation(self, session_id, parent_id, table_name, segment_id, operation_type="query"):
        """Add a child operation to an existing operation"""
        try:
            # Load current operation tree
            root_operation = self.get_operation_tree(session_id)
            if not root_operation:
                logger.error(f"No root operation found for session {session_id}")
                return None
                
            # Find parent operation
            parent_operation = self._find_operation_by_id(root_operation, parent_id)
            if not parent_operation:
                logger.error(f"Parent operation {parent_id} not found")
                return None
                
            # Create and add child operation
            child_operation = TableOperation(
                table_name=table_name,
                segment_id=segment_id,
                operation_type=operation_type,
                parent_id=parent_id
            )
            
            parent_operation.add_child(child_operation)
            
            # Save updated tree
            self.save_operation_tree(session_id, root_operation)
            
            return child_operation
        except Exception as e:
            logger.error(f"Error adding child operation: {e}")
            return None
            
    def _find_operation_by_id(self, operation, operation_id):
        """Find an operation by ID in the tree"""
        if operation.operation_id == operation_id:
            return operation
            
        for child in operation.children:
            found = self._find_operation_by_id(child, operation_id)
            if found:
                return found
                
        return None
            
    def save_operation_tree(self, session_id, operation):
        """Save operation tree to disk"""
        try:
            session_dir = f"{self.storage_path}/{session_id}"
            os.makedirs(session_dir, exist_ok=True)
            
            # Save as JSON for human readability
            json_path = f"{session_dir}/operations.json"
            with open(json_path, 'w') as f:
                json.dump(operation.to_dict(), f, indent=2)
                
            # Also save as pickle for more reliable deserialization
            pickle_path = f"{session_dir}/operations.pickle"
            with open(pickle_path, 'wb') as f:
                pickle.dump(operation, f)
                
            return operation
        except Exception as e:
            logger.error(f"Error saving operation tree: {e}")
            return None
            
    def get_operation_tree(self, session_id):
        """Get operation tree for a session"""
        try:
            pickle_path = f"{self.storage_path}/{session_id}/operations.pickle"
            
            if not os.path.exists(pickle_path):
                return None
                
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error getting operation tree: {e}")
            
            # Try to rebuild from JSON as fallback
            try:
                json_path = f"{self.storage_path}/{session_id}/operations.json"
                
                if not os.path.exists(json_path):
                    return None
                    
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                return TableOperation.from_dict(data)
            except Exception as json_e:
                logger.error(f"Error rebuilding operation tree from JSON: {json_e}")
                return None
                
    def get_operation_history(self, session_id):
        """Get a flattened history of operations for a session"""
        try:
            root_operation = self.get_operation_tree(session_id)
            if not root_operation:
                return []
                
            operations = []
            self._collect_operations(root_operation, operations)
            
            # Sort by timestamp
            operations.sort(key=lambda op: op.timestamp)
            
            return operations
        except Exception as e:
            logger.error(f"Error getting operation history: {e}")
            return []
            
    def _collect_operations(self, operation, operation_list):
        """Recursively collect operations from the tree"""
        operation_list.append(operation)
        
        for child in operation.children:
            self._collect_operations(child, operation_list)
