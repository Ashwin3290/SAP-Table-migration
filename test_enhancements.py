"""
Test script for TableLLM enhancements.
Run this script to verify the functionality of the implemented enhancements.
"""

import os
import sys
import unittest
import pandas as pd
import sqlite3
import json
import tempfile
import shutil
from datetime import datetime

# Add the path to the project directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import TableLLM classes and functions
try:
    from operation_tracker import OperationTracker, TableOperation
    from template_manager import TemplateManager, Template, generalize_code
    from planner_enhancement import SegmentManager, TableRelationshipManager, validate_entity_existence
    from tablellm_enhancement import validate_key_column_operation, format_conversational_response
    from code_exec_enhancement import enhanced_execute_code, enhanced_validation_handling
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are installed and in the correct path.")
    sys.exit(1)

class TestOperationTracker(unittest.TestCase):
    """Test the operation tracker functionality"""
    
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.tracker = OperationTracker(storage_path=self.test_dir)
        self.session_id = "test_session"
        
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_create_root_operation(self):
        """Test creating a root operation"""
        operation = self.tracker.create_root_operation(
            self.session_id, "test_table", 123, "query"
        )
        
        self.assertIsNotNone(operation)
        self.assertEqual(operation.table_name, "test_table")
        self.assertEqual(operation.segment_id, 123)
        self.assertEqual(operation.operation_type, "query")
        self.assertIsNone(operation.parent_id)
        
    def test_add_child_operation(self):
        """Test adding a child operation"""
        # Create root operation
        root = self.tracker.create_root_operation(
            self.session_id, "parent_table", 123, "query"
        )
        
        # Add child operation
        child = self.tracker.add_child_operation(
            self.session_id, root.operation_id, "child_table", 456, "update"
        )
        
        self.assertIsNotNone(child)
        self.assertEqual(child.table_name, "child_table")
        self.assertEqual(child.segment_id, 456)
        self.assertEqual(child.operation_type, "update")
        self.assertEqual(child.parent_id, root.operation_id)
        
    def test_get_operation_tree(self):
        """Test retrieving the operation tree"""
        # Create root operation
        root = self.tracker.create_root_operation(
            self.session_id, "root_table", 123, "query"
        )
        
        # Add child operations
        child1 = self.tracker.add_child_operation(
            self.session_id, root.operation_id, "child_table1", 456, "update"
        )
        
        child2 = self.tracker.add_child_operation(
            self.session_id, root.operation_id, "child_table2", 789, "join"
        )
        
        # Get the operation tree
        tree = self.tracker.get_operation_tree(self.session_id)
        
        self.assertIsNotNone(tree)
        self.assertEqual(tree.table_name, "root_table")
        self.assertEqual(len(tree.children), 2)


class TestTemplateManager(unittest.TestCase):
    """Test the template manager functionality"""
    
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.manager = TemplateManager(storage_path=self.test_dir)
        
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_create_template(self):
        """Test creating and saving a template"""
        template = self.manager.create_template(
            template_type="code",
            template_content="def analyze_data(source_dfs, target_df):\n    return target_df",
            original_query="Transform the data",
            metadata={"test": "metadata"}
        )
        
        self.assertIsNotNone(template)
        self.assertEqual(template.template_type, "code")
        self.assertIn("def analyze_data", template.template_content)
        self.assertEqual(template.original_query, "Transform the data")
        
    def test_generalize_code(self):
        """Test code generalization"""
        code = """
def analyze_data(source_dfs, target_df):
    # Get source dataframe
    source_df = source_dfs['MARA']
    
    # Filter data
    filtered = source_df[source_df['MTART'] == 'FERT']
    
    # Update target dataframe
    target_df['MATNR'] = filtered['MATNR']
    
    return target_df
"""
        
        generalized = generalize_code(code)
        
        self.assertIn("TABLE_1", generalized)
        self.assertNotIn("MARA", generalized)
        self.assertNotIn("MTART", generalized)


class TestSegmentManager(unittest.TestCase):
    """Test the segment manager functionality"""
    
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.manager = SegmentManager(storage_path=self.test_dir)
        self.session_id = "test_session"
        
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_track_segment_change(self):
        """Test tracking segment changes"""
        # Track a segment change
        result = self.manager.track_segment_change(
            self.session_id, None, 123
        )
        
        self.assertTrue(result)
        
        # Get the current segment
        current = self.manager.get_current_segment(self.session_id)
        self.assertEqual(current, 123)
        
        # Track another change
        result = self.manager.track_segment_change(
            self.session_id, 123, 456, parent_segment=123
        )
        
        self.assertTrue(result)
        
        # Get the current segment again
        current = self.manager.get_current_segment(self.session_id)
        self.assertEqual(current, 456)


class TestTableRelationshipManager(unittest.TestCase):
    """Test the table relationship manager functionality"""
    
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.manager = TableRelationshipManager(storage_path=self.test_dir)
        self.session_id = "test_session"
        
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_establish_relationship(self):
        """Test establishing parent-child relationships"""
        # Establish relationship
        result = self.manager.establish_parent_child_relationship(
            self.session_id, "parent_table", "child_table", ["key1", "key2"]
        )
        
        self.assertTrue(result)
        
        # Get relationships
        relationships = self.manager.get_table_relationships(self.session_id)
        
        self.assertEqual(relationships["root_table"], "parent_table")
        self.assertEqual(len(relationships["relationships"]), 1)
        self.assertEqual(relationships["relationships"][0]["parent_table"], "parent_table")
        self.assertEqual(relationships["relationships"][0]["child_table"], "child_table")
        

class TestKeyColumnValidation(unittest.TestCase):
    """Test key column validation functionality"""
    
    def test_validate_key_column_operation(self):
        """Test key column validation"""
        # Create test data
        planner_info = {
            "key_columns": ["id"]
        }
        
        # Valid data
        source_dfs = {
            "test_table": pd.DataFrame({
                "id": [1, 2, 3],
                "value": ["a", "b", "c"]
            })
        }
        target_df = pd.DataFrame(columns=["id", "value"])
        
        is_valid, message = validate_key_column_operation(planner_info, source_dfs, target_df)
        self.assertTrue(is_valid)
        
        # Invalid data with nulls
        source_dfs = {
            "test_table": pd.DataFrame({
                "id": [1, None, 3],
                "value": ["a", "b", "c"]
            })
        }
        
        is_valid, message = validate_key_column_operation(planner_info, source_dfs, target_df)
        self.assertFalse(is_valid)
        self.assertIn("null values", message)
        
        # Invalid data with duplicates
        source_dfs = {
            "test_table": pd.DataFrame({
                "id": [1, 1, 3],
                "value": ["a", "b", "c"]
            })
        }
        
        is_valid, message = validate_key_column_operation(planner_info, source_dfs, target_df)
        self.assertFalse(is_valid)
        self.assertIn("duplicate values", message)


class TestConversationalFormatting(unittest.TestCase):
    """Test conversational formatting functionality"""
    
    def test_format_conversational_response(self):
        """Test conversational response formatting"""
        # Test with DataFrame
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": ["a", "b", "c"]
        })
        
        response = format_conversational_response(df)
        self.assertIn("I've processed your request", response)
        self.assertIn("3 rows", response)
        
        # Test with error
        error = {
            "error_type": "ValueError",
            "error_message": "Test error message"
        }
        
        response = format_conversational_response(error)
        self.assertIn("I encountered an error", response)
        self.assertIn("Test error message", response)
        
        # Test with string
        response = format_conversational_response("Test message")
        self.assertEqual(response, "Test message")


if __name__ == "__main__":
    print("Running tests for TableLLM enhancements...")
    unittest.main()
