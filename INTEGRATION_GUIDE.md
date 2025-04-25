# TableLLM Enhancement Integration Guide

This guide provides step-by-step instructions for integrating the new enhancements into the existing TableLLM codebase.

## 1. Install Required Dependencies

First, make sure to install the new dependencies:

```bash
pip install sentence-transformers
```

## 2. New Files Added

The following new files have been added to the project:

- `operation_tracker.py` - For tracking hierarchical table operations
- `template_manager.py` - For implementing a RAG-based template system
- `planner_enhancement.py` - For segment management and table relationships
- `tablellm_enhancement.py` - For key column validation and conversational interface
- `code_exec_enhancement.py` - For multi-table operations and enhanced validation

## 3. Integration Steps

### Step 1: Import new modules in TableLLM

Update the imports in `tablellm.py` to include the new modules:

```python
# Add these imports at the top of tablellm.py
from operation_tracker import OperationTracker, TableOperation
from template_manager import TemplateManager, Template, generalize_code
from tablellm_enhancement import (
    validate_key_column_operation, 
    format_conversational_response,
    generate_join_operation_code,
    initialize_join_templates
)
from code_exec_enhancement import enhanced_execute_code
from planner_enhancement import SegmentManager, TableRelationshipManager, validate_entity_existence
```

### Step 2: Update the TableLLM.__init__ method

Modify the `__init__` method to initialize the new components:

```python
def __init__(self):
    """Initialize the TableLLM instance"""
    try:
        # ... (existing code) ...
        
        # Initialize template manager with smaller embedding model
        self.template_manager = TemplateManager()
        
        # Initialize operation tracker
        self.operation_tracker = OperationTracker()
        
        # Initialize segment manager
        self.segment_manager = SegmentManager()
        
        # Initialize table relationship manager
        self.table_relationship_manager = TableRelationshipManager()
        
        # Load code templates with join template
        self.code_templates = self._initialize_templates()
        
        # Add join templates
        self.code_templates.update(initialize_join_templates())
        
        # ... (rest of existing code) ...
    except Exception as e:
        # ... (existing error handling) ...
        raise
```

### Step 3: Update the process_sequential_query method

Modify the `process_sequential_query` method to use the new features:

```python
def process_sequential_query(self, query, object_id=29, segment_id=336, project_id=24, 
                            session_id=None, target_sap_fields=None):
    conn = None
    try:
        # ... (existing validation code) ...

        # Process query with the planner
        logger.info(f"Processing query: {query}")
        resolved_data = planner_process_query(
            object_id, segment_id, project_id, query, session_id, target_sap_fields
        )
        if not resolved_data:
            logger.error("Failed to resolve query with planner")
            return None, "Failed to resolve query", session_id

        # Connect to database
        print(resolved_data["key_mapping"])
        if not len(resolved_data["key_mapping"]) == 0:
            if isinstance(resolved_data["key_mapping"][0], str):
                return None, resolved_data["key_mapping"][0], session_id
        try:
            conn = sqlite3.connect("db.sqlite3")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            return None, f"Database connection error: {e}", session_id

        # ... (existing code up to extracting planner information) ...
        
        # Get current segment or track first segment change
        current_segment = self.segment_manager.get_current_segment(session_id)
        if not current_segment:
            # First operation in the session
            self.segment_manager.track_segment_change(session_id, None, segment_id)
            
            # Create root operation in the operation tracker
            if target_table:
                self.operation_tracker.create_root_operation(
                    session_id, 
                    target_table, 
                    segment_id, 
                    "query"
                )
        elif segment_id != current_segment:
            # Segment has changed, track the change
            self.segment_manager.track_segment_change(
                session_id, current_segment, segment_id, current_segment
            )
            
            # Add child operation to the operation tracker
            operations = self.operation_tracker.get_operation_history(session_id)
            if operations:
                last_operation = operations[-1]
                
                # Add child operation
                if target_table:
                    child_op = self.operation_tracker.add_child_operation(
                        session_id,
                        last_operation.operation_id,
                        target_table,
                        segment_id,
                        "query"
                    )
                    
                    # Establish relationship with parent table
                    self.table_relationship_manager.establish_parent_child_relationship(
                        session_id,
                        last_operation.table_name,
                        target_table,
                        resolved_data.get("key_mapping", [])
                    )

        # Add validation for key column operations
        validation_result, validation_message = validate_key_column_operation(
            planner_info, source_dfs, target_df
        )
        
        if not validation_result:
            logger.warning(f"Key column validation failed: {validation_message}")
            if conn:
                conn.close()
            return None, validation_message, session_id

        # Try to use template from library first
        use_template = True
        template_candidates = self.template_manager.find_similar_templates(
            query=query,
            template_type='code',
            top_k=3
        )
        
        if template_candidates and template_candidates[0]['similarity'] > 0.85:
            # We have a good template match
            best_template = template_candidates[0]['template']
            logger.info(f"Using template {best_template.id} with similarity {template_candidates[0]['similarity']}")
            
            # Use the template with placeholders replaced
            code_content = best_template.template_content
            
            # Replace table placeholders
            for i, table in enumerate(source_tables):
                code_content = code_content.replace(f"TABLE_{i+1}", table)
                
            # Replace field placeholders
            fields = planner_info.get("source_fields", []) + planner_info.get("target_fields", [])
            for i, field in enumerate(fields):
                code_content = code_content.replace(f"FIELD_{i+1}", field)
        else:
            # No good template, generate from scratch
            use_template = False
            
            # ... (existing code for plan and code generation) ...
            

        # ... (existing code for code execution) ...
        # Use enhanced execution
        try:
            code_file = create_code_file(code_content, query, is_double=True)
            result = enhanced_execute_code(code_file, source_dfs, target_df, target_sap_fields)
            
            # ... (existing error handling) ...
            
        # Save the generated code as a template for future use if not using existing template
        if not use_template and isinstance(result, pd.DataFrame):
            self.template_manager.create_template(
                template_type='code',
                template_content=generalize_code(code_content),
                original_query=query,
                metadata={
                    'source_tables': source_tables,
                    'target_table': target_table,
                    'source_fields': planner_info.get('source_fields', []),
                    'target_fields': planner_info.get('target_fields', []),
                    'filtering_fields': planner_info.get('filtering_fields', []),
                }
            )
            
        # Format the result as a conversational response
        if isinstance(result, pd.DataFrame):
            conversational_result = format_conversational_response(result)
        elif isinstance(result, dict) and "error_type" in result:
            conversational_result = format_conversational_response(result)
        else:
            conversational_result = result
        
        # ... (existing code for saving the result) ...
        
        # Return the results with conversational format
        if conn:
            conn.close()
        return code_content, conversational_result, session_id

    except Exception as e:
        # ... (existing error handling) ...
```

### Step 4: Update Imports in Planner

Update the planner.py file to import and use the new classes:

```python
# Add these imports near the top of planner.py
from planner_enhancement import (
    SegmentManager, 
    TableRelationshipManager, 
    validate_entity_existence
)
```

### Step 5: Update process_query in Planner

Modify the process_query function to use the new segment and relationship classes:

```python
def process_query(object_id, segment_id, project_id, query, session_id=None, target_sap_fields=None):
    conn = None
    try:
        # ... (existing validation code) ...
        
        # Initialize context manager
        context_manager = ContextualSessionManager()
        
        # Initialize segment manager and table relationship manager
        segment_manager = SegmentManager()
        relationship_manager = TableRelationshipManager()
        
        # ... (existing code) ...
        
        # Validate segment existence
        is_valid, message = validate_entity_existence(conn, "segment", segment_id)
        if not is_valid:
            logger.error(message)
            if conn:
                conn.close()
            return None
            
        # Validate tables in joined_df
        target_tables = joined_df["table_name"].unique().tolist()
        for table in target_tables:
            is_valid, message = validate_entity_existence(conn, "table", table)
            if not is_valid:
                logger.error(message)
                if conn:
                    conn.close()
                return None
        
        # ... (rest of existing function) ...
```

### Step 6: Update code_exec.py

Replace the execute_code function in code_exec.py with the enhanced version:

```python
# Add import at the top of code_exec.py
from code_exec_enhancement import enhanced_execute_code, enhanced_validation_handling

# Replace the existing function with a wrapper that calls the enhanced version
def execute_code(file_path, source_dfs, target_df, target_sap_fields):
    """Execute a Python file and return the result or detailed error traceback"""
    return enhanced_execute_code(file_path, source_dfs, target_df, target_sap_fields)

# Replace the validation_handling function with the enhanced version
def validation_handling(source_dfs, target_df, result, target_sap_fields):
    """Validate the execution result"""
    return enhanced_validation_handling(source_dfs, target_df, result, target_sap_fields)
```

## 4. Testing After Integration

After integrating all changes, you should test:

1. Basic queries first to ensure core functionality still works
2. Key column validation to prevent data integrity issues
3. Segment changes to verify tracking works
4. Parent-child table relationships to ensure they're established correctly
5. Template-based code generation to verify it reuses patterns
6. Conversational responses to check improved formatting
7. Multi-table operations to verify they work correctly

## 5. Troubleshooting

### Common Issues

1. **Import errors**: Ensure all new modules are in the correct paths
2. **Template loading issues**: Check if the sentence-transformers library is installed correctly
3. **Segment tracking issues**: Verify the session directories are created correctly
4. **Key validation fails**: Make sure source data meets quality requirements

### Debugging Tips

1. Check the logs for detailed error messages
2. Enable more verbose logging if needed
3. Use a test session to step through each phase of the process
4. Validate each sub-component independently before full integration

## 6. Production Deployment Considerations

1. **Data migration**: For existing installations, create a migration script to create relationship and segment ledgers
2. **Backup**: Always backup your data before upgrading
3. **Environment**: Update environment variables if needed
4. **Templates**: Initialize with some basic templates to improve initial performance
5. **User interface**: Consider updating any UI components to display relationship hierarchies

For any issues or questions, please refer to the project documentation or contact the development team.
