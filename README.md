# TableLLM Enhancement Project

This project enhances the TableLLM system with additional features for more complex data transformations, segment flexibility, parent-child relationships, and improved robustness.

## Overview of Enhancements

### 1. Conversational Interface
- Added validation checks for data integrity
- Implemented more user-friendly response formatting
- Prevents non-primary data insertion into key columns

### 2. Enhanced Error Handling
- Added better handling for missing columns, tables, and segments
- Improved error messages with specific details
- Added validation for entity existence

### 3. Segment Flexibility
- Allows segments to change (previously assumed fixed segment IDs)
- Extracts segment information from connection.segments table
- Maintains a ledger of segment changes during a session

### 4. Parent-Child Table Relations
- First table in a query session becomes the "parent" or "root" table
- Operations on different segments become "child" operations
- Uses primary key columns from parent tables as foreign keys in child tables
- Enables operations across related tables

### 5. Hierarchical Table Operation Tracking
- Stores tree structure of table operations (parent-child relationships)
- Implements persistent storage for this hierarchy
- Tracks dependencies between operations

### 6. Multi-Table Operations
- Added support for multiple target tables
- Implemented join operations between parent and child tables
- Added support for multi-column insertions

### 7. RAG-Based Template System
- Creates library of generalized plans and code templates from existing queries
- Replaces specific SAP column names with generic placeholders
- Implements similarity search to find relevant templates for new queries
- Makes system more robust against AI hallucinations

## New Files

1. **operation_tracker.py** - Tracks hierarchical operations and maintains parent-child relationships
2. **template_manager.py** - Manages code templates with similarity-based retrieval
3. **planner_enhancement.py** - Adds segment management and table relationship tracking
4. **tablellm_enhancement.py** - Implements key column validation and conversational formatting
5. **code_exec_enhancement.py** - Adds support for multi-table operations and enhanced validation
6. **INTEGRATION_GUIDE.md** - Detailed instructions for integrating the enhancements
7. **test_enhancements.py** - Test cases to verify the functionality

## Integration

Follow the steps in the [Integration Guide](INTEGRATION_GUIDE.md) to incorporate these enhancements into your existing TableLLM project.

## Testing

Run the test script to verify the functionality:

```bash
python test_enhancements.py
```

## Dependencies

- **sentence-transformers**: Used for embedding queries to find similar templates
- All existing dependencies of TableLLM

## Usage Examples

### Creating a Template

```python
template_manager = TemplateManager()
template = template_manager.create_template(
    template_type="code",
    template_content=generalize_code(code_content),
    original_query="Transform the data",
    metadata={"source_tables": ["TABLE1"], "target_table": "TARGET"}
)
```

### Tracking Parent-Child Relationships

```python
relationship_manager = TableRelationshipManager()
relationship_manager.establish_parent_child_relationship(
    session_id="session1",
    parent_table="PARENT_TABLE",
    child_table="CHILD_TABLE",
    key_columns=["ID"]
)
```

### Executing with Enhanced Validation

```python
result = enhanced_execute_code(
    file_path="generated_code.py",
    source_dfs={"source_table": source_df},
    target_df=target_df,
    target_sap_fields="TARGET_FIELD"
)
```

### Creating Operations Hierarchy

```python
operation_tracker = OperationTracker()
root_op = operation_tracker.create_root_operation(
    session_id="session1",
    table_name="ROOT_TABLE",
    segment_id=123,
    operation_type="query"
)
child_op = operation_tracker.add_child_operation(
    session_id="session1",
    parent_id=root_op.operation_id,
    table_name="CHILD_TABLE",
    segment_id=456,
    operation_type="join"
)
```

## Future Work

- UI improvements to display hierarchical relationships
- Performance optimization for large datasets
- More robust template generalization

## Contributing

Contributions are welcome! Please follow the existing code style and add tests for any new functionality.
