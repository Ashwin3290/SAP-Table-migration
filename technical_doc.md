# GenAI Data Migration Tool - Technical Documentation

## Table of Contents
1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Core Components](#3-core-components)
4. [Data Flow](#4-data-flow)
5. [DMTool Core Class](#5-dmtool-core-class)
6. [Query Planning & Processing](#6-query-planning--processing)
7. [Code Generation](#7-code-generation)
8. [Code Execution](#8-code-execution)
9. [Utility Functions](#9-utility-functions)
10. [Database Integration](#10-database-integration)
11. [API Integration](#11-api-integration)
12. [Session Management](#12-session-management)
13. [Error Handling & Recovery](#13-error-handling--recovery)
14. [Security Considerations](#14-security-considerations)
15. [Performance Optimization](#15-performance-optimization)
16. [Development & Extension Guide](#16-development--extension-guide)
17. [Testing Framework](#17-testing-framework)
18. [Deployment Guide](#18-deployment-guide)
19. [Troubleshooting](#19-troubleshooting)
20. [Appendix](#20-appendix)

## 1. Introduction

### 1.1 Purpose
The GenAI Data Migration Tool is a sophisticated system that leverages generative AI to translate natural language queries into executable Python code for SAP data transformation and migration tasks. It enables business users with limited technical expertise to perform complex data migrations using natural language instructions.

### 1.2 Business Value
- Reduces reliance on specialized developers for data transformation
- Accelerates data migration timelines by 70-80%
- Enables business users to directly implement data transformations
- Provides consistency and reliability through standardized code generation
- Maintains audit trail of all transformations

### 1.3 Key Features
- Natural language processing for data transformation requests
- Automatic Python code generation for data transformation
- Context-aware sequential transformations
- SAP-specific data handling
- Comprehensive error handling and correction
- Session state preservation
- SQL injection prevention
- Token usage optimization

### 1.4 Technical Approach
The system combines Large Language Models (specifically Google's Gemini API) with a structured pipeline for query understanding, code generation, and execution. It maintains context across sequential transformations and incorporates specialized knowledge of SAP data structures.

## 2. System Architecture

### 2.1 High-Level Architecture
The system follows a modular architecture with the following key components:

```
[User Query] → [Planner Module] → [Code Generator] → [Code Executor] → [Results]
                      ↑                                      ↓
                      ↑                                      ↓
                [Session Context] ←------------------------ ↓
                      ↑                                      ↓
                      ↑                                      ↓
               [Database] ←-------------------------------- ↓
```

### 2.2 Component Interactions
- **User Query Entry**: Natural language query describing desired transformation
- **Planner Module**: Processes query, identifies tables, fields, and operations
- **Code Generator**: Creates Python code to perform the transformation
- **Code Executor**: Runs the generated code against source and target tables
- **Session Context**: Maintains state between sequential transformations
- **Database**: Stores source and target data, as well as mapping information

### 2.3 Technology Stack
- **Programming Language**: Python 3.7+
- **Database**: SQLite
- **AI Integration**: Google Gemini API
- **Key Libraries**:
  - pandas: Data manipulation
  - numpy: Numerical operations
  - spacy: Natural language processing
  - docx: Document processing
  - google-generativeai: Gemini API client

## 3. Core Components

### 3.1 File Structure
```
.
├── dmtool.py              # Core orchestration class
├── planner.py             # Query understanding and planning
├── code_exec.py           # Code execution and validation
├── code_generator.py      # Code template generation
├── transform_utils.py     # Transformation utility functions
├── token_tracker.py       # API token usage tracking
├── streamlit.py           # Web interface (not covered in this doc)
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

### 3.2 Component Responsibilities
| Component | Primary Responsibility |
|-----------|------------------------|
| DMTool | Orchestrates transformation process end-to-end |
| Planner | Processes natural language, identifies tables/fields |
| Code Generator | Creates Python code templates for transformations |
| Code Executor | Executes generated code, validates results |
| Transform Utilities | Provides standard functions for data manipulation |
| Token Tracker | Monitors and logs API token usage |

### 3.3 Dependencies
All required external dependencies are listed in `requirements.txt`:
- tqdm==4.66.1
- pandas==2.2.0
- transformers==4.37.2
- torch
- streamlit
- pymongo
- requests
- python-docx
- google
- google-generativeai

## 4. Data Flow

### 4.1 Process Flow
1. **Query Input**: User provides natural language description of desired transformation
2. **Query Analysis**: System analyzes query to identify intent, source/target tables, and fields
3. **Planning**: System creates a step-by-step plan for the transformation
4. **Code Generation**: System generates Python code based on the plan
5. **Execution**: Code is executed against source and target tables
6. **Validation**: Results are validated for data integrity
7. **Output**: Transformed data is returned to the user and stored in session context

### 4.2 Data Transformations
The system supports various types of transformations:
- Field mapping between source and target tables
- Filtering based on conditions
- Conditional value mapping
- Table joining
- Data aggregation
- Data type conversion
- SAP-specific transformations

### 4.3 Information Exchange
- Between components, data is passed as Python objects (dataframes, dictionaries)
- Session information is persisted as JSON and CSV files
- Database interactions use SQL queries via SQLite

## 5. DMTool Core Class

### 5.1 Class Structure
The `DMTool` class in `dmtool.py` serves as the central orchestrator for the system.

```python
class DMTool:
    def __init__(self):
        # Initialize Gemini client, code templates, and context
        
    def process_sequential_query(self, query, object_id, segment_id, project_id, session_id=None, target_sap_fields=None):
        # Main entry point for processing a transformation query
        
    def _extract_planner_info(self, resolved_data):
        # Extract and organize information from planner's output
        
    def _classify_query(self, query, planner_info):
        # Classify the type of transformation
        
    def _generate_simple_plan(self, planner_info):
        # Generate a step-by-step plan for the transformation
        
    def _generate_code_from_simple_plan(self, simple_plan, planner_info):
        # Generate Python code from the transformation plan
        
    def _fix_code(self, code_content, error_info, planner_info, attempt=1, max_attempts=3):
        # Attempt to fix code that generated errors
```

### 5.2 Process Sequential Query
The `process_sequential_query` method is the main entry point for the system:

```python
def process_sequential_query(self, query, object_id=29, segment_id=336, project_id=24, session_id=None, target_sap_fields=None):
    """
    Process a query as part of a sequential transformation
    
    Parameters:
    query (str): The user's query
    object_id (int): Object ID for mapping
    segment_id (int): Segment ID for mapping
    project_id (int): Project ID for mapping
    session_id (str): Optional session ID, creates new session if None
    target_sap_fields (list): Optional list of target SAP fields
    
    Returns:
    tuple: (code, result, session_id)
    """
```

The method performs the following steps:
1. Validates input parameters
2. Processes the query with the planner
3. Extracts and organizes relevant information
4. Gets source and target dataframes
5. Generates a transformation plan
6. Generates code from the plan
7. Executes the generated code
8. Saves the updated target dataframe
9. Returns the results

### 5.3 Code Templates
The DMTool uses code templates for common operations:
- **Filter Template**: For filtering data based on conditions
- **Update Template**: For updating target fields from source fields
- **Conditional Mapping Template**: For applying conditional logic to field values

## 6. Query Planning & Processing

### 6.1 Planner Module
The `planner.py` module is responsible for understanding natural language queries and mapping them to database structures.

Key functions:
- `process_query()`: Main entry point for processing a query
- `parse_data_with_context()`: Uses Gemini API to understand the query
- `fetch_data_by_ids()`: Retrieves table and field mapping information
- `process_info()`: Processes the resolved query information

### 6.2 Query Analysis
The system analyzes queries using the following approach:
1. Extract key entities (tables, fields, conditions)
2. Match entities to database schema
3. Identify transformation intent
4. Determine filtering conditions
5. Extract key mappings between source and target

Example query analysis for "Bring Material Number with Material Type = ROH from MARA Table":
- Source Table: MARA
- Source Field: MATNR
- Filter Field: MTART
- Filter Condition: MTART = 'ROH'
- Target Field: MATNR

### 6.3 ContextualSessionManager
The `ContextualSessionManager` class manages session state and context:

```python
class ContextualSessionManager:
    def __init__(self, storage_path="sessions"):
        # Initialize session storage
        
    def create_session(self):
        # Create a new session and return its ID
        
    def get_context(self, session_id):
        # Get the current context for a session
        
    def update_context(self, session_id, resolved_data):
        # Update the context with new resolved data
        
    def get_transformation_history(self, session_id):
        # Get the transformation history for a session
        
    def add_key_mapping(self, session_id, target_col, source_col):
        # Add a key mapping for a session
        
    def get_key_mapping(self, session_id):
        # Get key mappings for a session
```

### 6.4 SQL Injection Prevention
The system includes robust SQL injection prevention through the `validate_sql_identifier()` function:

```python
def validate_sql_identifier(identifier):
    """
    Validate that an SQL identifier doesn't contain injection attempts
    Returns sanitized identifier or raises exception
    """
    # Check for dangerous patterns
    # Only allow alphanumeric characters, underscores, and some specific characters
    # Raise SQLInjectionError if dangerous patterns found
```

## 7. Code Generation

### 7.1 Code Generation Process
The code generation process involves several steps:
1. Query classification to determine transformation type
2. Generation of a simple, step-by-step plan
3. Translation of the plan into executable Python code
4. Code validation and error correction if needed

### 7.2 Query Classification
The system classifies queries into different categories:
- `FILTER_AND_EXTRACT`: Filtering records and extracting specific fields
- `UPDATE_EXISTING`: Updating values in existing target records
- `CONDITIONAL_MAPPING`: Applying if/else logic to determine values
- `EXTRACTION`: Simple extraction without complex filtering
- `TIERED_LOOKUP`: Looking up data in multiple tables in sequence
- `AGGREGATION`: Performing calculations or aggregations

### 7.3 Simple Plan Generation
The system generates a step-by-step plan in natural language. Example:

```
1. Get source data from MARA table
2. Validate that required fields (MATNR, MTART) exist
3. Filter records where MTART equals 'ROH'
4. Check if target dataframe is empty
5. If target is empty, create new dataframe with MATNR field
6. If target is not empty, update existing records with matching key
7. Return the updated target dataframe
```

### 7.4 Code Generation from Plan
The system translates the simple plan into executable Python code:

```python
def analyze_data(source_dfs, target_df):
    # Import required utilities
    import pandas as pd
    import numpy as np
    
    # Get source dataframe
    if 'MARA' not in source_dfs:
        return target_df
    
    source_df = source_dfs['MARA']
    
    # Validate required fields exist
    if 'MATNR' not in source_df.columns or 'MTART' not in source_df.columns:
        return target_df
    
    # Filter records where MTART equals 'ROH'
    filtered_df = source_df[source_df['MTART'] == 'ROH'].copy()
    
    # Check if target dataframe is empty
    if len(target_df) == 0:
        # Create new dataframe with MATNR field
        result_df = pd.DataFrame()
        result_df['MATNR'] = filtered_df['MATNR']
        return result_df
    else:
        # Update existing records with matching key
        # ... (code for updating existing records)
    
    return target_df
```

### 7.5 Exploration Code Generation
The `code_generator.py` module generates code for data exploration:

```python
def generate_exploration_code(df, filename, is_double=False):
    """Generate boilerplate code for exploring a dataframe"""
    # Determine numeric, categorical, and date columns
    # Generate code for the appropriate type (single or double table)
    # Save the code to a file
```

## 8. Code Execution

### 8.1 Code Execution Module
The `code_exec.py` module handles the execution of generated code:

```python
def create_code_file(code_content, query, is_double=False):
    """Create a permanent Python file with the provided code content"""
    # Create a sanitized filename from the query
    # Fill in a template with the necessary imports
    # Write to permanent file
    
def execute_code(file_path, source_dfs, target_df, target_sap_fields):
    """Execute a Python file and return the result or detailed error traceback"""
    # Import the module
    # Execute the code
    # Validate the results
    # Return the result or error information
```

### 8.2 Validation Handling
The `validation_handling()` function validates transformation results:

```python
def validation_handling(source_df, target_df, result, target_sap_fields):
    """Validate the transformation results for data integrity"""
    # Check 1: Length validation - target_df should never be smaller than result
    # Get non-null columns in both dataframes
    # Validate that result has exactly the expected non-null columns
    # Raise ValidationError if validation fails
```

### 8.3 Error Handling
The code execution module includes detailed error handling:
- Capture full traceback information
- Return detailed error information for diagnosis
- Support for error correction attempts

### 8.4 File Operations
The module also includes functions for file operations:
- `save_file()`: Save an uploaded file and return its dataframe
- `docx2tabular()`: Convert a DOCX table to a list of rows

## 9. Utility Functions

### 9.1 Transform Utilities
The `transform_utils.py` module provides a rich set of utility functions for data transformations.

#### 9.1.1 Data Filtering
```python
def filter_dataframe(df, field, condition_type, value):
    """
    Filter a dataframe based on field and condition
    
    Parameters:
    df (pd.DataFrame): Source dataframe
    field (str): Field name to filter on
    condition_type (str): One of: 'equals', 'not_equals', 'contains', etc.
    value (Any): Value to compare against
    
    Returns:
    pd.DataFrame: Filtered dataframe
    """
```

#### 9.1.2 Field Mapping
```python
def map_fields(source_df, target_df, field_mapping, key_field_pair=None):
    """
    Map fields from source to target dataframe
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    target_df (pd.DataFrame): Target dataframe
    field_mapping (Dict[str, str]): Dictionary mapping target fields to source fields
    key_field_pair (Optional[Tuple[str, str]]): Optional tuple containing (target_key, source_key)
    
    Returns:
    pd.DataFrame: Updated target dataframe with mapped fields
    """
```

#### 9.1.3 Conditional Mapping
```python
def conditional_mapping(source_df, condition_field, conditions, value_field=None, value_map=None, default=None):
    """
    Map values based on conditions
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    condition_field (str): Field to apply conditions to
    conditions (List[str]): List of condition expressions as strings
    value_field (Optional[str]): Field to get values from if value_map not provided
    value_map (Optional[List[Any]]): List of values corresponding to conditions
    default (Optional[Any]): Value to use when no conditions match
    
    Returns:
    pd.Series: Series with mapped values
    """
```

### 9.2 SAP-Specific Functions

#### 9.2.1 Material Type Mapping
```python
def map_material_type(source_df, source_field='MTART', target_field='MATERIAL_TYPE_TEXT'):
    """
    Map SAP material type codes to readable text descriptions
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe with material types
    source_field (str): Field containing material type codes
    target_field (str): Field name for the text descriptions
    
    Returns:
    pd.Series: Series with mapped text descriptions
    """
```

#### 9.2.2 SAP Date Conversion
```python
def convert_sap_date(source_df, date_field, output_format='YYYY-MM-DD'):
    """
    Convert SAP date format (YYYYMMDD) to standard date format
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe with SAP dates
    date_field (str): Field containing SAP dates
    output_format (str): Output format specification
    
    Returns:
    pd.Series: Series with formatted dates
    """
```

#### 9.2.3 Leading Zeros Handling
```python
def handle_sap_leading_zeros(source_df, field, length=10):
    """
    Ensure SAP numerical keys have proper leading zeros
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    field (str): Field to process
    length (int): Target field length (default: 10 for material numbers)
    
    Returns:
    pd.Series: Series with properly formatted values
    """
```

### 9.3 Token Tracking
The `token_tracker.py` module provides token usage tracking for the Gemini API:

```python
@track_token_usage()
def your_function_that_calls_gemini(prompt, ...):
    # Function implementation
```

Key functions:
- `track_token_usage()`: Decorator for tracking token usage
- `estimate_tokens()`: Estimate the number of tokens in a text string
- `get_token_usage_stats()`: Get current token usage statistics
- `reset_token_usage_stats()`: Reset token usage statistics

## 10. Database Integration

### 10.1 Database Schema
The system interacts with an SQLite database containing the following tables:

#### 10.1.1 Mapping Tables
- `connection_fields`: Field definitions and metadata
- `connection_rule`: Mapping rules between source and target fields
- `connection_segments`: Table definitions and relationships

#### 10.1.2 SAP Tables
The system is designed to work with standard SAP tables, including:
- `MARA`: Material master data
- `MAKT`: Material descriptions
- Other SAP tables as configured in the mapping

### 10.2 Database Access
The system uses SQLite for database access:

```python
def fetch_data_by_ids(object_id, segment_id, project_id, conn):
    """Fetch data mappings from the database"""
    # Validate parameters
    # Construct and execute SQL query
    # Return results as dataframe
```

### 10.3 Database Operations
Key database operations include:
- Fetching mapping data based on object, segment, and project IDs
- Retrieving source data from SAP tables
- Retrieving target data from SAP tables
- Saving updated target data

### 10.4 Safety Mechanisms
All database interactions include safety mechanisms:
- SQL identifier validation
- Parameterized queries
- Error handling for database operations
- Connection management

## 11. API Integration

### 11.1 Gemini API Integration
The system integrates with Google's Gemini API for natural language processing and code generation:

```python
client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-2.5-flash-thinking-exp-01-21", 
    contents=prompt,
    config=types.GenerateContentConfig(temperature=0.2)
)
```

### 11.2 API Configuration
API configuration is managed through environment variables:
- API key stored in `.env` file
- Model selection in code
- Temperature and other parameters configurable

### 11.3 Prompt Engineering
The system uses carefully engineered prompts for different tasks:
- Query understanding prompts
- Code generation prompts
- Error correction prompts

Example prompt structure:
```
You are a data transformation assistant specializing in SAP data mappings. 
Your task is to analyze a natural language query about data transformations and match it to the appropriate source and target tables and fields.
 
CONTEXT DATA SCHEMA: {table_desc}

CURRENT TARGET TABLE STATE:
{target_df_sample}
 
USER QUERY: {question}

INSTRUCTIONS:
1. Identify key entities in the query:
   - Source table(s)
   - Source field(s)
   - Filtering or transformation conditions
   ...
```

### 11.4 Token Usage Tracking
Token usage is tracked using the `token_tracker.py` module:
- Counts input and output tokens
- Logs usage to file
- Provides usage statistics

## 12. Session Management

### 12.1 Session Structure
Sessions maintain context for sequential transformations:
- Session ID: UUID for identifying the session
- Context: JSON object with session information
- Target State: Current state of the target dataframe
- Transformation History: List of previous transformations
- Key Mappings: Relationships between source and target keys

### 12.2 Session Storage
Session data is stored in the filesystem:
- Base directory: `sessions/`
- Session directory: `sessions/<session_id>/`
- Context file: `sessions/<session_id>/context.json`
- Target data: `sessions/<session_id>/target_latest.csv`
- Historical data: `sessions/<session_id>/target_<timestamp>.csv`

### 12.3 Session Functions
Key functions for session management:
- `get_session_context()`: Get the current context for a session
- `get_or_create_session_target_df()`: Get existing target dataframe or create new one
- `save_session_target_df()`: Save the updated target dataframe

```python
def get_or_create_session_target_df(session_id, target_table, conn):
    """
    Get existing target dataframe for a session or create a new one
    
    Parameters:
    session_id (str): Session ID
    target_table (str): Target table name
    conn (Connection): SQLite connection
    
    Returns:
    DataFrame: The target dataframe
    """
```

## 13. Error Handling & Recovery

### 13.1 Custom Exceptions
The system defines custom exceptions for different error types:

```python
class SQLInjectionError(Exception):
    """Exception raised for potential SQL injection attempts."""
    pass

class SessionError(Exception):
    """Exception raised for session-related errors."""
    pass

class APIError(Exception):
    """Exception raised for API-related errors."""
    pass

class DataProcessingError(Exception):
    """Exception raised for data processing errors."""
    pass

class CodeGenerationError(Exception):
    """Exception raised for code generation errors."""
    pass

class ExecutionError(Exception):
    """Exception raised for code execution errors."""
    pass

class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass
```

### 13.2 Error Handling Strategy
The system implements a comprehensive error handling strategy:
- Validate inputs before processing
- Catch and classify exceptions
- Provide detailed error information
- Log errors for troubleshooting
- Attempt recovery when possible

### 13.3 Code Correction
The system can attempt to fix errors in generated code:

```python
def _fix_code(self, code_content, error_info, planner_info, attempt=1, max_attempts=3):
    """
    Attempt to fix code based on error traceback
    
    Parameters:
    code_content (str): The original code that failed
    error_info (dict): Error information with traceback
    planner_info (dict): Context information from planner
    attempt (int): Current attempt number
    max_attempts (int): Maximum number of attempts to fix the code
    
    Returns:
    str: Fixed code or None if max attempts reached
    """
```

The function:
1. Extracts error information and traceback
2. Creates a prompt for the Gemini API
3. Gets a fixed version of the code
4. Returns the fixed code for execution

### 13.4 Fallback Mechanisms
The system includes fallback mechanisms for critical functions:
- Default code templates if generation fails
- Empty dataframes if database access fails
- Default session ID if session creation fails

## 14. Security Considerations

### 14.1 SQL Injection Prevention
The system prevents SQL injection through several mechanisms:
- Validation of all SQL identifiers (table names, field names)
- Pattern matching to detect malicious input
- Parameterized queries for all database access

```python
def validate_sql_identifier(identifier):
    """
    Validate that an SQL identifier doesn't contain injection attempts
    """
    # Check for common SQL injection patterns
    dangerous_patterns = [
        ";", "--", "/*", "*/", "DROP", "DELETE", "INSERT", "UPDATE", "UNION", "EXEC", "EXECUTE",
    ]
    for pattern in dangerous_patterns:
        if pattern.lower() in identifier.lower():
            raise SQLInjectionError(f"Potentially dangerous SQL pattern found: {pattern}")
            
    # Only allow alphanumeric characters, underscores, and some specific characters
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", identifier):
        raise SQLInjectionError("SQL identifier contains invalid characters")
```

### 14.2 API Security
API security is managed through:
- API keys stored in environment variables
- Error handling for API failures
- Rate limiting through token tracking

### 14.3 Data Privacy
The system includes considerations for data privacy:
- No storage of sensitive data outside the database
- Session data is stored locally, not sent to external services
- Code execution occurs locally, not on remote servers

## 15. Performance Optimization

### 15.1 Database Optimization
The system optimizes database operations:
- Uses parameterized queries for efficiency
- Limits result sets to reduce memory usage
- Reuses database connections where possible

### 15.2 Memory Management
Memory usage is optimized through:
- Creating dataframe copies only when necessary
- Using filtering to reduce dataframe size
- Cleaning up resources after use

### 15.3 Token Optimization
Token usage is optimized through:
- Careful prompt engineering
- Context preservation to reduce redundant API calls
- Using the most appropriate model for each task

```python
# Example of token optimization in prompt engineering
prompt = f"""
Classify this data transformation query into ONE of these categories:
- FILTER_AND_EXTRACT: Filtering records from source and extracting specific fields
- UPDATE_EXISTING: Updating values in existing target records only
- CONDITIONAL_MAPPING: Applying if/else logic to determine values
...

Return ONLY the classification name with no explanation.
"""
```

## 16. Development & Extension Guide

### 16.1 Adding New Transformation Types
To add new transformation types:
1. Update the `_classify_query()` method to include the new type
2. Add a new code template to the `_initialize_templates()` method
3. Update the `_generate_code_from_simple_plan()` method to handle the new type
4. Add appropriate utility functions to `transform_utils.py`

### 16.2 Adding SAP-Specific Functions
To add support for additional SAP fields or tables:
1. Add mapping dictionaries to `transform_utils.py`
2. Create utility functions for the new fields
3. Test the functions with sample data

Example of adding a new SAP mapping function:
```python
def map_sap_storage_location(source_df, field='LGORT'):
    """
    Map SAP storage location codes to descriptions
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe with storage locations
    field (str): Field containing storage location codes
    
    Returns:
    pd.Series: Series with storage location descriptions
    """
    # Implementation details
```

### 16.3 Extending the Planner
To extend the planner's capabilities:
1. Update the prompt in `parse_data_with_context()`
2. Add new parsing logic if needed
3. Update the `process_info()` function to handle new information types

### 16.4 Customizing Code Generation
To customize code generation:
1. Modify the prompt in `_generate_code_from_simple_plan()`
2. Update code templates in `_initialize_templates()`
3. Add new transformation patterns as needed

## 17. Testing Framework

### 17.1 Unit Testing
The system supports unit testing for individual components:
- Utility function testing
- Token estimation testing
- SQL validation testing
- Error handling testing

### 17.2 Integration Testing
Integration testing covers the interaction between components:
- End-to-end query processing
- Database interaction testing
- Session management testing
- Error recovery testing

### 17.3 Validation Testing
Validation testing ensures data integrity:
- Key preservation testing
- Data type validation
- Null value handling
- Edge case testing

## 18. Deployment Guide

### 18.1 Prerequisites
- Python 3.7+
- SQLite database
- Gemini API access
- Required Python packages (`requirements.txt`)

### 18.2 Installation Steps
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add `GEMINI_API_KEY=your_api_key_here`
4. Prepare the database:
   - Ensure SQLite database exists (`db.sqlite3`)
   - Configure table mappings in database

### 18.3 Configuration
Key configuration points:
- Database connection in code
- API key in `.env` file
- Session storage path in `ContextualSessionManager`
- Token tracking options in `track_token_usage()`

### 18.4 Environment Setup
Required directories:
- `sessions/`: For session storage
- `generated_code/`: For generated Python files
- `uploaded_files/`: For uploaded data files

## 19. Troubleshooting

### 19.1 Common Issues and Solutions

#### 19.1.1 API Connection Issues
- **Issue**: "Failed to call Gemini API"
- **Solution**: Check API key validity, network connectivity, and token limits

#### 19.1.2 Database Connection Issues
- **Issue**: "No such table: [table_name]"
- **Solution**: Verify database file path and table existence

#### 19.1.3 Code Generation Issues
- **Issue**: "Failed to generate code"
- **Solution**: Check query complexity, API response, and token usage logs

#### 19.1.4 Execution Errors
- **Issue**: Error in generated code execution
- **Solution**: Review error traceback, check data types, and validate field names

### 19.2 Logging
The system includes comprehensive logging:
- API calls logged in `gemini_planner_usage.log`
- General logs using Python's logging module
- Error tracebacks captured in execution results
- Token usage statistics available through `get_token_usage_stats()`

Example logging configuration:
```python
# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
```

### 19.3 Debugging Tools
Tools available for debugging:
- Code inspection through `generated_code/` directory
- Session inspection through `sessions/` directory
- Token usage logs for API call analysis
- Error traceback information in execution results

### 19.4 Support Process
When issues arise:
1. Check logs for error messages
2. Verify database connectivity and schema
3. Validate API access and token usage
4. Inspect generated code for logical errors
5. Review session context for state inconsistencies
6. Validate input query for ambiguities or unsupported requests

## 20. Appendix

### 20.1 API Reference

#### 20.1.1 DMTool Class
```python
class DMTool:
    def __init__(self):
        """Initialize the DMTool instance"""
        
    def process_sequential_query(self, query, object_id, segment_id, project_id, session_id=None, target_sap_fields=None):
        """Process a query as part of a sequential transformation"""
        
    def _extract_planner_info(self, resolved_data):
        """Extract and organize information from planner's output"""
        
    def _classify_query(self, query, planner_info):
        """Classify the type of transformation"""
        
    def _generate_simple_plan(self, planner_info):
        """Generate a step-by-step plan for the transformation"""
        
    def _generate_code_from_simple_plan(self, simple_plan, planner_info):
        """Generate Python code from the transformation plan"""
        
    def _fix_code(self, code_content, error_info, planner_info, attempt=1, max_attempts=3):
        """Attempt to fix code that generated errors"""
        
    def _initialize_templates(self):
        """Initialize code templates for common operations"""
        
    def post_proccess_result(self, result):
        """Post-process the result DataFrame"""
```

#### 20.1.2 Planner Functions
```python
def process_query(object_id, segment_id, project_id, query, session_id=None, target_sap_fields=None):
    """Process a query with context awareness"""
    
def parse_data_with_context(joined_df, query, session_id=None, previous_context=None, target_table_desc=None):
    """Parse data using Gemini API with token usage tracking and context awareness"""
    
def fetch_data_by_ids(object_id, segment_id, project_id, conn):
    """Fetch data mappings from the database"""
    
def validate_sql_identifier(identifier):
    """Validate that an SQL identifier doesn't contain injection attempts"""
    
def check_distinct_requirement(sentence):
    """Determines if a sentence requires distinct values"""
    
def get_session_context(session_id):
    """Get the current context for a session"""
    
def get_or_create_session_target_df(session_id, target_table, conn):
    """Get existing target dataframe for a session or create a new one"""
    
def save_session_target_df(session_id, target_df):
    """Save the updated target dataframe for a session"""
```

#### 20.1.3 Code Execution Functions
```python
def create_code_file(code_content, query, is_double=False):
    """Create a permanent Python file with the provided code content"""
    
def execute_code(file_path, source_dfs, target_df, target_sap_fields):
    """Execute a Python file and return the result or detailed error traceback"""
    
def validation_handling(source_df, target_df, result, target_sap_fields):
    """Validate the transformation results for data integrity"""
    
def save_file(upload_file):
    """Save an uploaded file and return its dataframe, text, and metadata"""
    
def docx2tabular(docx_path):
    """Convert a DOCX table to a list of rows"""
```

#### 20.1.4 Transform Utility Functions
```python
def filter_dataframe(df, field, condition_type, value):
    """Filter a dataframe based on field and condition"""
    
def map_fields(source_df, target_df, field_mapping, key_field_pair=None):
    """Map fields from source to target dataframe"""
    
def conditional_mapping(source_df, condition_field, conditions, value_field=None, value_map=None, default=None):
    """Map values based on conditions"""
    
def join_tables(main_df, other_df, main_key, other_key, fields_to_add):
    """Join two tables and add specific fields"""
    
def aggregate_data(df, group_by, agg_functions):
    """Aggregate data by group with specified functions"""
    
def map_material_type(source_df, source_field='MTART', target_field='MATERIAL_TYPE_TEXT'):
    """Map SAP material type codes to readable text descriptions"""
    
def convert_sap_date(source_df, date_field, output_format='YYYY-MM-DD'):
    """Convert SAP date format (YYYYMMDD) to standard date format"""
    
def map_sap_language_code(source_df, lang_field='SPRAS'):
    """Map SAP language codes to full language names"""
    
def map_sap_unit_of_measure(source_df, uom_field='MEINS'):
    """Map SAP units of measure to full descriptions"""
    
def handle_sap_leading_zeros(source_df, field, length=10):
    """Ensure SAP numerical keys have proper leading zeros"""
```

#### 20.1.5 Token Tracking Functions
```python
@track_token_usage(log_to_file=True, log_path='token_usage.log')
def function_that_calls_gemini():
    """Decorator for tracking token usage"""
    
def estimate_tokens(text):
    """Estimate the number of tokens in a text string using tiktoken"""
    
def get_token_usage_stats():
    """Get the current token usage statistics"""
    
def reset_token_usage_stats():
    """Reset all token usage statistics to zero"""
```

### 20.2 Common Error Messages and Solutions

| Error Message | Possible Cause | Solution |
|---------------|----------------|----------|
| "GEMINI_API_KEY not found in environment variables" | Missing API key | Add API key to .env file |
| "Failed to call Gemini API" | Network or API issue | Check connectivity and API key validity |
| "Empty dataframe passed to _generate_simple_plan" | Missing or invalid input data | Verify input query and database content |
| "SQL identifier contains invalid characters" | Potential SQL injection attempt | Use only alphanumeric characters in table/field names |
| "Error executing code" | Issues in generated code | Check error traceback for specific issues |
| "No data found for object_id, segment_id, project_id" | Invalid mapping parameters | Verify object, segment, and project IDs |
| "Failed to resolve query" | Query couldn't be understood | Simplify or clarify the query |
| "Target field not found in target dataframe" | Missing field in target | Check target table schema |
