# DMTool Integration Guide

This document explains how to integrate and use the DMTool's data transformation capabilities in your Python application.

## Overview

The DMTool is a context-aware data transformation framework that helps map and transform data between source and target tables using natural language queries. It's particularly useful for SAP data migrations and transformations.

## Prerequisites

- Python 3.7+
- Required libraries: pandas, numpy, sqlite3
- Access to the following modules:
  - dmtool.py
  - code_exec.py
  - planner.py
  - code_generator.py
  - transform_utils.py
  - token_tracker.py

## Basic Usage

### 1. Initialize the DMTool

```python
from dmtool import DMTool

# Initialize the DMTool instance
dmtool = DMTool()
```

### 2. Process a Transformation Query

The main function you'll use is `process_sequential_query()`:

```python
def process_transformation(query, object_id, segment_id, project_id, 
                          session_id=None, target_sap_fields=None):
    """
    Process a data transformation query
    
    Parameters:
    query (str): Natural language query describing the transformation
    object_id (int): The object ID for mapping
    segment_id (int): The segment ID for mapping
    project_id (int): The project ID for mapping
    session_id (str, optional): Session ID for sequential transformations
    target_sap_fields (str/list, optional): Target SAP fields to populate
    
    Returns:
    tuple: (code, result, session_id)
    """
    # Call DMTool's process_sequential_query function
    code, result, session_id = dmtool.process_sequential_query(
        query=query,
        object_id=object_id, 
        segment_id=segment_id, 
        project_id=project_id,
        session_id=session_id,
        target_sap_fields=target_sap_fields
    )
    
    # Process the results
    if code is None:
        print(f"Error: {result}")
        return None, result, session_id
    
    # Handle successful transformation
    if isinstance(result, pd.DataFrame):
        # Filter out any columns with all None values
        filtered_result = result.dropna(axis=1, how='all')
        print(f"Transformation successful. Rows: {len(filtered_result)}")
        return code, filtered_result, session_id
    else:
        return code, result, session_id
```

### 3. Example Implementation

Here's a complete example showing how to use the DMTool for a data transformation:

```python
import pandas as pd
import sqlite3
from dmtool import DMTool

def main():
    # Initialize DMTool
    dmtool = DMTool()
    
    # Connection to your database
    conn = sqlite3.connect('db.sqlite3')
    
    # Set your mapping parameters
    object_id = 41
    segment_id = 577
    project_id = 24
    
    # Your transformation query
    query = "Bring Material Number with Material Type = ROH from MARA Table"
    
    # Target SAP field you want to populate
    target_field = "MATNR"  # Material Number
    
    # Process the query
    code, result, session_id = dmtool.process_sequential_query(
        query=query,
        object_id=object_id,
        segment_id=segment_id,
        project_id=project_id,
        session_id=None,  # Start a new session
        target_sap_fields=target_field
    )
    
    # Handle the results
    if code is None:
        print(f"Error: {result}")
        return
    
    # Display or process the results
    if isinstance(result, pd.DataFrame):
        print(f"Transformation completed successfully.")
        print(f"Records processed: {len(result)}")
        print("\nSample results:")
        print(result.head())
        
        # You can save the result to a CSV file
        result.to_csv('transformation_result.csv', index=False)
        
        # Or process it further as needed
        # ...
    else:
        print(f"Result: {result}")
    
    # For sequential transformations, you'd use the session_id
    print(f"\nSession ID for sequential transformations: {session_id}")
    
    # Close the connection
    conn.close()

if __name__ == "__main__":
    main()
```

## Sequential Transformations

To perform sequential transformations that maintain context and state:

```python
# First transformation
code1, result1, session_id = dmtool.process_sequential_query(
    query="Bring Material Number with Material Type = ROH from MARA Table",
    object_id=41,
    segment_id=577,
    project_id=24,
    session_id=None,  # First query in sequence starts a new session
    target_sap_fields="MATNR"
)

# Second transformation using the same session
code2, result2, session_id = dmtool.process_sequential_query(
    query="Now, add the Material Description from MAKT table where language is EN",
    object_id=41,
    segment_id=577,
    project_id=24,
    session_id=session_id,  # Use the session from the first transformation
    target_sap_fields="MAKTX"
)

# Third transformation
code3, result3, session_id = dmtool.process_sequential_query(
    query="Finally, add the Material Group from MARA",
    object_id=41,
    segment_id=577,
    project_id=24,
    session_id=session_id,  # Continue the same session
    target_sap_fields="MATKL"
)

# The final result will have all three fields populated
final_result = result3
```

## Error Handling

The `process_sequential_query` function returns a tuple of `(code, result, session_id)`:

- If `code` is `None`, an error occurred and `result` contains the error message
- If `code` is not `None`, the transformation was processed and:
  - `result` is a pandas DataFrame if successful
  - `result` is an error message if there was a problem during execution
  - `session_id` is the session ID for sequential transformations

```python
code, result, session_id = dmtool.process_sequential_query(...)

if code is None:
    print(f"Error: {result}")
else:
    if isinstance(result, pd.DataFrame):
        print("Transformation successful")
        # Process the DataFrame
    else:
        print(f"Execution error: {result}")
```

## Advanced Usage

### Multiple Target Fields

You can update multiple target fields at once:

```python
# Update both material number and description
target_fields = ["MATNR", "MAKTX"]

code, result, session_id = dmtool.process_sequential_query(
    query="Get Material Number and Description for all materials with Type = FERT",
    object_id=41,
    segment_id=577,
    project_id=24,
    target_sap_fields=target_fields
)
```

### Session Management

You can manage sessions to create complex transformations:

```python
import uuid

# Create a new session ID
new_session_id = str(uuid.uuid4())

# Use it for a sequence of transformations
code1, result1, session_id = dmtool.process_sequential_query(
    query="First transformation...",
    object_id=41,
    segment_id=577,
    project_id=24,
    session_id=new_session_id,
    target_sap_fields="FIELD1"
)

# Save the session_id for later use
with open('session_id.txt', 'w') as f:
    f.write(session_id)

# Later, load the session_id and continue
with open('session_id.txt', 'r') as f:
    saved_session_id = f.read().strip()

# Continue the transformation
code2, result2, session_id = dmtool.process_sequential_query(
    query="Continue transformation...",
    object_id=41,
    segment_id=577,
    project_id=24,
    session_id=saved_session_id,
    target_sap_fields="FIELD2"
)
```

## Integration Tips

1. **Database Connection**: Ensure your database is properly set up with the required tables before using DMTool.

2. **Error Logging**: Implement proper error logging for production use.

3. **Session Management**: For long-running processes, store session IDs securely.

4. **Validation**: Always validate the output DataFrame before using it in critical processes.

5. **Performance**: For large datasets, consider performance optimization techniques.

## Troubleshooting

### Common Issues

1. **Missing Tables**: Ensure the source and target tables exist in your database.

2. **Invalid Object/Segment/Project IDs**: Verify these IDs exist in your mapping configuration.

3. **Ambiguous Queries**: If the transformation fails, try making your query more specific.

4. **Key Mapping Errors**: Ensure proper key mappings between source and target tables.
