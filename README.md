# TableLLM

TableLLM is an intelligent data transformation system that uses natural language processing to automate and simplify SAP data mapping and transformation tasks. It leverages the Gemini API to understand natural language queries and generate appropriate Python code for complex data transformations.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage](#usage)
- [Utility Functions](#utility-functions)
- [Database Analysis](#database-analysis)
- [Session Management](#session-management)
- [Error Handling](#error-handling)
- [Contributing](#contributing)

## Overview

TableLLM provides a bridge between natural language queries and data transformation operations, particularly focused on SAP data structures. Users can describe their transformation needs in plain English, and the system will interpret their intent, identify the relevant source and target tables/fields, and generate executable Python code to perform the transformations.

Key capabilities include:
- Processing natural language queries to identify data transformation intent
- Mapping queries to appropriate source and target fields in SAP data structures
- Generating and executing Python code for data transformations
- Tracking transformation operations across sessions
- Analyzing database structures and data relationships
- Providing utility functions for common SAP data transformation operations

## Architecture

The system follows a pipeline architecture:

1. **Query Processing**: Natural language queries are processed by the planner to identify intent, source/target tables, fields, and transformation logic.
2. **Information Extraction**: Relevant information is extracted and organized from database schema and existing mappings.
3. **Code Generation**: Based on the extracted information, Python code is generated to perform the transformation.
4. **Code Execution**: The generated code is executed against the source and target tables.
5. **Result Management**: Transformed data is saved and made available for subsequent operations.

## Key Components

### Main Modules

1. **tablellm.py**: Core module that orchestrates the entire process from query processing to code execution.
   - `TableLLM` class manages the flow of information from query to transformation results
   - Handles code generation, execution, and error correction
   - Maintains session context for sequential transformations

2. **planner.py**: Responsible for processing natural language queries and extracting transformation intent.
   - Maps queries to tables, fields, and transformation logic
   - Creates and manages sessions for multi-step transformations
   - Validates data for primary key operations
   - Handles tracking of key field mappings

3. **transform_utils.py**: Library of utility functions for common data transformation operations.
   - Filtering, mapping, joining, and conditional logic
   - SAP-specific transformations (material type mapping, date conversion, etc.)
   - Data aggregation and table manipulation

4. **code_exec.py**: Handles the execution of generated Python code.
   - Creates Python files from generated code
   - Executes code against source and target data
   - Captures detailed error information for debugging

5. **code_generator.py**: Generates boilerplate code for data exploration.
   - Single and double table analysis code templates
   - Visualization code generation
   - Comparative data analysis

6. **database_analysis.py**: Analyzes database structure and content.
   - Schema analysis
   - Table relationship discovery
   - Data profiling and statistics generation
   - Visualization of database insights

### Supporting Files

- **token_tracker.py**: Tracks token usage for the Gemini API.
- **key_mapper.py**: Manages key field mappings between source and target tables.
- **prompt_format.py**: Formats prompts for the Gemini API.
- **streamlit.py**: Provides a web interface for the system.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/tablellm.git
   cd tablellm
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

4. Prepare the database:
   - Ensure your SQLite database is in place or create one
   - The default database path is `db.sqlite3` in the root directory

## Usage

### Basic Usage

```python
from tablellm import TableLLM

# Initialize the system
system = TableLLM()

# Process a natural language query
query = "Get material type from MARA and map it to MTART in the target table"
code, result, session_id = system.process_sequential_query(
    query,
    object_id=29,
    segment_id=336,
    project_id=24
)

# Code contains the generated Python code
# Result contains the transformation result
# session_id can be used for sequential operations
```

### Sequential Transformations

```python
# Continue with a previous session
query2 = "Now add the material description from MAKT where the language is EN"
code2, result2, session_id = system.process_sequential_query(
    query2,
    object_id=29,
    segment_id=336,
    project_id=24,
    session_id=session_id  # Same session ID from previous query
)
```

## Utility Functions

TableLLM provides a rich set of utility functions through the `transform_utils.py` module:

### Data Filtering

```python
from transform_utils import filter_dataframe

# Filter records where MTART equals 'ROH'
filtered_df = filter_dataframe(source_df, 'MTART', 'equals', 'ROH')
```

### Field Mapping

```python
from transform_utils import map_fields

# Map fields from source to target
result_df = map_fields(
    source_df, 
    target_df, 
    {'TARGET_FIELD1': 'SOURCE_FIELD1', 'TARGET_FIELD2': 'SOURCE_FIELD2'},
    ('TARGET_KEY', 'SOURCE_KEY')  # Optional key mapping
)
```

### Conditional Mapping

```python
from transform_utils import conditional_mapping

# Apply conditional logic
target_df['CATEGORY'] = conditional_mapping(
    source_df, 
    'MTART', 
    ["== 'ROH'", "== 'HALB'"], 
    value_map=['Raw', 'Semi'], 
    default='Other'
)
```

### Table Joining

```python
from transform_utils import join_tables

# Join tables on key fields
result_df = join_tables(
    main_df, 
    other_df, 
    'MATNR',  # Key in main table 
    'PRODUCT',  # Key in other table
    ['MAKTX', 'MEINS']  # Fields to add from other table
)
```

### SAP-Specific Functions

```python
from transform_utils import map_material_type, convert_sap_date, handle_sap_leading_zeros

# Map material types to descriptions
result_df['MATERIAL_TYPE_TEXT'] = map_material_type(source_df, 'MTART')

# Convert SAP date format (YYYYMMDD) to standard format
result_df['CREATION_DATE'] = convert_sap_date(source_df, 'ERDAT', 'YYYY-MM-DD')

# Handle leading zeros in SAP material numbers
result_df['MATERIAL_NUMBER'] = handle_sap_leading_zeros(source_df, 'MATNR', 10)
```

## Database Analysis

The `database_analysis.py` module provides tools for analyzing the database structure and content:

```python
from database_analysis import DatabaseAnalyzer

# Initialize the analyzer
analyzer = DatabaseAnalyzer()

# Connect to the database
analyzer.connect()

# Get list of tables
tables = analyzer.get_table_list()

# Analyze a specific table
table_analysis = analyzer.analyze_table('MARA')

# Analyze the entire database
analysis_results = analyzer.analyze_database()

# Export analysis to JSON
json_file = analyzer.export_analysis_to_json()

# Generate summary report
summary = analyzer.generate_summary_report()

# Close the connection
analyzer.close()
```

## Session Management

TableLLM maintains context across multiple transformations using session management:

```python
from planner import get_session_context, get_or_create_session_target_df, save_session_target_df

# Get the current context for a session
context = get_session_context(session_id)

# Get or create the target dataframe for a session
target_df = get_or_create_session_target_df(session_id, target_table, conn)

# Save the updated target dataframe
save_success = save_session_target_df(session_id, updated_df)
```

## Error Handling

TableLLM implements comprehensive error handling with custom exceptions:

```python
from planner import SQLInjectionError, SessionError, APIError, DataProcessingError
from tablellm import CodeGenerationError, ExecutionError

try:
    result = system.process_sequential_query(query, object_id, segment_id, project_id)
except SQLInjectionError as e:
    print(f"SQL Injection attempt detected: {e}")
except SessionError as e:
    print(f"Session error: {e}")
except APIError as e:
    print(f"API error: {e}")
except DataProcessingError as e:
    print(f"Data processing error: {e}")
except CodeGenerationError as e:
    print(f"Code generation error: {e}")
except ExecutionError as e:
    print(f"Code execution error: {e}")
```