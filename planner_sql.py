import json
import uuid
import os
import pandas as pd
import re
import sqlite3
from io import StringIO
from datetime import datetime
import logging
from google import genai
from google.genai import types
from token_tracker import track_token_usage, get_token_usage_stats
from pathlib import Path
import spacy
import traceback
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import SQL related modules
from sql_executor import SQLExecutor

# Initialize SQL executor
sql_executor = SQLExecutor()

from spacy.matcher import Matcher
from spacy.tokens import Span

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    # Fallback to smaller model if medium not available
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # In case no model is installed
        print("Downloading spaCy model...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
def classify_query_with_spacy(query):
    """
    Use spaCy to classify the query type based on linguistic patterns
    
    Parameters:
    query (str): The natural language query
    
    Returns:
    str: Query classification (SIMPLE_TRANSFORMATION, JOIN_OPERATION, etc.)
    dict: Additional details about the classification
    """
    doc = nlp(query.lower())
    
    # Initialize matcher with vocabulary
    matcher = Matcher(nlp.vocab)
    
    # Define patterns for JOIN operations
    join_patterns = [
        [{"LOWER": "join"}, {"OP": "*"}, {"LOWER": {"IN": ["table", "tables"]}}],
        [{"LOWER": {"IN": ["merge", "combine", "link"]}}, {"OP": "*"}, {"LOWER": {"IN": ["data", "tables", "information"]}}],
        [{"LOWER": {"IN": ["from", "using"]}}, {"OP": "*"}, {"LOWER": {"IN": ["both", "all"]}}, {"OP": "*"}, {"LOWER": {"IN": ["tables", "segments"]}}],
        [{"LOWER": "where"}, {"OP": "*"}, {"LOWER": {"IN": ["equals", "matches", "="]}}, {"OP": "*"}, {"LOWER": {"IN": ["matnr", "material", "number"]}}]
    ]
    
    # Define patterns for CROSS_SEGMENT operations
    segment_patterns = [
        [{"LOWER": {"IN": ["segment", "basic", "marc", "makt", "mvke"]}}],
        [{"LOWER": {"IN": ["previous", "prior", "last", "earlier"]}}, {"OP": "*"}, {"LOWER": {"IN": ["segment", "transformation", "data"]}}],
        [{"LOWER": {"IN": ["use", "consider", "refer"]}}, {"OP": "*"}, {"LOWER": {"IN": ["segment", "basic", "marc", "makt"]}}]
    ]
    
    # Define patterns for VALIDATION operations
    validation_patterns = [
        [{"LOWER": {"IN": ["validate", "verify", "ensure"]}}],
        [{"LOWER": "if"}, {"OP": "*"}, {"LOWER": {"IN": ["exists", "valid", "present", "available"]}}],
        [{"LOWER": {"IN": ["missing", "invalid", "correct", "consistent"]}}],
        [{"LOWER": {"IN": ["every", "all", "each"]}}, {"OP": "*"}, {"LOWER": {"IN": ["must", "should", "has to"]}}]
    ]
    
    # Define patterns for AGGREGATION operations
    aggregation_patterns = [
        [{"LOWER": {"IN": ["count", "sum", "average", "mean", "calculate", "total"]}}],
        [{"LOWER": "group"}, {"LOWER": "by"}],
        [{"LOWER": {"IN": ["minimum", "maximum", "min", "max", "highest", "lowest"]}}],
        [{"LOWER": {"IN": ["statistics", "aggregation", "aggregate", "statistical"]}}]
    ]
    
    # Add patterns to matcher
    matcher.add("JOIN", join_patterns)
    matcher.add("SEGMENT", segment_patterns)
    matcher.add("VALIDATION", validation_patterns)
    matcher.add("AGGREGATION", aggregation_patterns)
    
    # Find matches
    matches = matcher(doc)
    
    # Count match types
    match_counts = {"JOIN": 0, "SEGMENT": 0, "VALIDATION": 0, "AGGREGATION": 0}
    match_details = {"JOIN": [], "SEGMENT": [], "VALIDATION": [], "AGGREGATION": []}
    
    for match_id, start, end in matches:
        match_type = nlp.vocab.strings[match_id]
        match_text = doc[start:end].text
        match_counts[match_type] += 1
        match_details[match_type].append(match_text)
    
    tables_mentioned = []
    common_sap_tables = ["MARA", "MARC", "MAKT", "MVKE", "MARM", "MLAN", "EKKO", "EKPO", "VBAK", "VBAP", "KNA1", "LFA1"]
    segment_keywords = ["BASIC", "PLANT", "SALES", "PURCHASING", "CLASSIFICATION", "MRP", "WAREHOUSE"]

    for token in doc:
        # Check for known SAP tables
        if token.text.upper() in common_sap_tables:
            tables_mentioned.append(token.text.upper())
        # Also detect uppercase tokens that might be table names
        elif token.text.isupper() and len(token.text) >= 3 and token.text.isalpha():
            tables_mentioned.append(token.text)
        # Check for segment mentions
        elif token.text.upper() in segment_keywords:
            # This could indicate a cross-segment operation
            if "SEGMENT" not in match_counts:
                match_counts["SEGMENT"] = 0
                match_details["SEGMENT"] = []
            match_counts["SEGMENT"] += 1
            match_details["SEGMENT"].append(f"{token.text} segment")
    
    # Determine primary classification based on match counts
    if match_counts["JOIN"] > 0 or len(set(tables_mentioned)) > 1:
        primary_class = "JOIN_OPERATION"
    elif match_counts["SEGMENT"] > 0:
        primary_class = "CROSS_SEGMENT"
    elif match_counts["VALIDATION"] > 0:
        primary_class = "VALIDATION_OPERATION"
    elif match_counts["AGGREGATION"] > 0:
        primary_class = "AGGREGATION_OPERATION"
    else:
        # Default to simple transformation
        primary_class = "SIMPLE_TRANSFORMATION"
    
    # Gather details about the classification
    details = {
        "match_counts": match_counts,
        "match_details": match_details,
        "tables_mentioned": tables_mentioned,
        "has_multiple_tables": len(set(tables_mentioned)) > 1,
        "tokens": [token.text for token in doc]
    }
    
    return primary_class, details

PROMPT_TEMPLATES = {
    # (Same templates as original planner.py)
    # Templates are not modified for SQL-based approach since planner's job
    # remains the same - extract intent, entities, etc. The actual SQL generation
    # happens in the SQLGenerator class
    
    "JOIN_OPERATION": """
    You are a data transformation assistant specializing in SAP data mappings and JOIN operations. 
    Your task is to analyze a natural language query about joining tables and map it to the appropriate source tables, fields, and join conditions.
    
    CONTEXT DATA SCHEMA: {table_desc}

    
    CURRENT TARGET TABLE STATE:
    {target_df_sample}
    
    USER QUERY: {question}

    Note:
    - Check segment names to identify correct tables if source tables are not mentioned, Use this Mapping to help with this {segment_mapping}
    

    Notes:
    - Check segment names to identify correct tables if source tables are not mentioned, Use this Mapping to help with this {segment_mapping}
    
    INSTRUCTIONS:
    1. Identify key entities in the join query:
       - All source tables needed for the join
       - Join fields for each pair of tables
       - Fields to select from each table
       - Filtering conditions
       - Target fields for insertion
    
    2. Specifically identify the join conditions:
       - Which table is joined to which
       - On which fields they are joined
       - The type of join (inner, left, right)
    
    3. Format your response as JSON with the following schema:
    ```json
    {{
        "query_type": "JOIN_OPERATION",
        "source_table_name": [List of all source tables, including previously visited segment tables],
        "source_field_names": [List of all fields to select],
        "filtering_fields": [List of filtering fields],
        "insertion_fields": [insertion_field],
        "target_sap_fields": [Target field(s)],
        "join_conditions": [
            {{
                "left_table": "table1",
                "right_table": "table2",
                "left_field": "join_field_left",
                "right_field": "join_field_right",
                "join_type": "inner"
            }}
        ],
        "Resolved_query": "Restructured query with resolved data"
    }}
    ```
    """,
    
    "CROSS_SEGMENT": """
    You are a data transformation assistant specializing in SAP data mappings across multiple segments. 
    Your task is to analyze a natural language query about data transformations involving previous segments.
    
    CONTEXT DATA SCHEMA: {table_desc}
    
    CURRENT TARGET TABLE STATE:
    {target_df_sample}
    
    USER QUERY: {question}

    Notes:
    - Check segment names to identify correct tables if source tables are not mentioned, Use this Mapping to help with this {segment_mapping}
    
    INSTRUCTIONS:
    1. Identify which previous segments are referenced in the query
    2. Determine how to link current data with segment data (join conditions)
    3. Identify which fields to extract from each segment
    4. Determine filtering conditions if any
    5. Identify the target fields for insertion
    
    Format your response as JSON with the following schema:
    ```json
    {{
        "query_type": "CROSS_SEGMENT",
        "source_table_name": [List of all source tables, including segment tables],
        "source_field_names": [List of all fields to select],
        "filtering_fields": [List of filtering fields],
        "insertion_fields": [List of fields to be inserted],
        "target_sap_fields": [Target field(s)],
        "segment_references": [
            {{
                "segment_id": "segment_id",
                "segment_name": "segment_name",
                "table_name": "table_name"
            }}
        ],
        "cross_segment_joins": [
            {{
                "left_table": "segment_table",
                "right_table": "current_table",
                "left_field": "join_field_left",
                "right_field": "join_field_right"
            }}
        ],
        "Resolved_query": "Restructured query with resolved data"
    }}
    ```
    """,
    
    "VALIDATION_OPERATION": """
    You are a data validation assistant specializing in SAP data. 
    Your task is to analyze a natural language query about data validation and map it to appropriate validation rules.
    
    CONTEXT DATA SCHEMA: {table_desc}
    
    CURRENT TARGET TABLE STATE:
    {target_df_sample}
    
    USER QUERY: {question}

    Notes:
    - Check segment names to identify correct tables if source tables are not mentioned, Use this Mapping to help with this {segment_mapping}
    
    
    INSTRUCTIONS:
    1. Identify the validation requirements in the query
    2. Determine which tables and fields need to be checked
    3. Formulate the validation rules in a structured way
    4. Specify what should happen for validation success/failure
    
    Format your response as JSON with the following schema:
    ```json
    {{
        "query_type": "VALIDATION_OPERATION",
        "source_table_name": [List of tables to validate],
        "source_field_names": [List of fields to validate],
        "validation_rules": [
            {{
                "field": "field_name",
                "rule_type": "not_null|unique|range|regex|exists_in",
                "parameters": {{
                    "min": minimum_value,
                    "max": maximum_value,
                    "pattern": "regex_pattern",
                    "reference_table": "table_name",
                    "reference_field": "field_name"
                }}
            }}
        ],
        "target_sap_fields": [Target field(s) to update with validation results],
        "Resolved_query": "Restructured query with resolved data"
    }}
    ```
    """,
    
    "AGGREGATION_OPERATION": """
    You are a data aggregation assistant specializing in SAP data. 
    Your task is to analyze a natural language query about data aggregation and map it to appropriate aggregation operations.
    
    CONTEXT DATA SCHEMA: {table_desc}
    
    
    CURRENT TARGET TABLE STATE:
    {target_df_sample}
    
    USER QUERY: {question}

    Notes:
    - Check segment names to identify correct tables if source tables are not mentioned, Use this Mapping to help with this {segment_mapping}
    
    
    INSTRUCTIONS:
    1. Identify the aggregation functions required (sum, count, average, etc.)
    2. Determine which tables and fields are involved
    3. Identify grouping fields if any
    4. Determine filtering conditions if any
    5. Identify where the results should be stored
    
    Format your response as JSON with the following schema:
    ```json
    {{
        "query_type": "AGGREGATION_OPERATION",
        "source_table_name": [Source tables],
        "source_field_names": [Fields to aggregate],
        "aggregation_functions": [
            {{
                "field": "field_name",
                "function": "sum|count|avg|min|max",
                "alias": "result_name"
            }}
        ],
        "group_by_fields": [Fields to group by],
        "filtering_fields": [Filtering fields],
        "filtering_conditions": {{
            "field_name": "condition_value"
        }},
        "target_sap_fields": [Target fields for results],
        "Resolved_query": "Restructured query with resolved data"
    }}
    ```
    """,
    
    "SIMPLE_TRANSFORMATION": """
    You are a data transformation assistant specializing in SAP data mappings. 
    Your task is to analyze a natural language query about data transformations and match it to the appropriate source and target tables and fields.
    
    CONTEXT DATA SCHEMA: {table_desc}
    
    CURRENT TARGET TABLE STATE:
    {target_df_sample}
    
    USER QUERY: {question}

    Note:
    - Check segment names to identify correct tables if source tables are not mentioned, Use this Mapping to help with this {segment_mapping}
    
    
    INSTRUCTIONS:
    1. Identify key entities in the query:
       - Source table(s)
       - Source field(s)
       - Filtering or transformation conditions
       - Logical flow (IF/THEN/ELSE statements)
       - Insertion fields
    
    2. Match these entities to the corresponding entries in the joined_data.csv schema
       - For each entity, find the closest match in the schema
       - Resolve ambiguities using the description field
       - Validate that the identified fields exist in the mentioned tables
    
    3. Generate a structured representation of the transformation logic:
       - JSON format showing the transformation flow
       - Include all source tables, fields, conditions, and targets
       - Map conditional logic to proper syntax
       - Handle fallback scenarios (ELSE conditions)
       - Use the provided key mappings to connect source and target fields correctly
       - Consider the current state of the target data shown above
    
    4. Create a resolved query that takes the actual field and table names, and does not change what is said in the query
    
    5. For the insertion fields, identify the fields that need to be inserted into the target table based on the User query.
    
    Respond with:
    ```json
    {{
        "query_type": "SIMPLE_TRANSFORMATION",
        "source_table_name": [List of all source_tables],
        "source_field_names": [List of all source_fields],
        "filtering_fields": [List of filtering fields],
        "insertion_fields": [field to be inserted],
        "target_sap_fields": [Target field(s)],
        "Resolved_query": [Rephrased query with resolved data]
    }}
    ```
    """
}

def process_query_by_type(object_id, segment_id, project_id, query, session_id=None, query_type=None, classification_details=None, target_sap_fields=None):
    """
    Process a query based on its classified type
    
    Parameters:
    object_id (int): Object ID
    segment_id (int): Segment ID
    project_id (int): Project ID
    query (str): The natural language query
    session_id (str): Optional session ID for context tracking
    query_type (str): Type of query (SIMPLE_TRANSFORMATION, JOIN_OPERATION, etc.)
    classification_details (dict): Details about the classification
    target_sap_fields (str/list): Optional target SAP fields to override
    
    Returns:
    dict: Processed information or None if errors
    """
    conn = None
    try:
        # Initialize context manager
        context_manager = ContextualSessionManager()
        
        # Get existing context and visited segments
        previous_context = context_manager.get_context(session_id) if session_id else None
        visited_segments = previous_context.get("segments_visited", {}) if previous_context else {}
        
        # Connect to database
        conn = sqlite3.connect("db.sqlite3")
        
        # Track current segment
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT segement_name FROM connection_segments WHERE segment_id = ?", (segment_id,))
            segment_result = cursor.fetchone()
            segment_name = segment_result[0] if segment_result else f"segment_{segment_id}"
            
            context_manager.track_segment(session_id, segment_id, segment_name, conn)
        except Exception as e:
            logger.warning(f"Error tracking segment: {e}")
        
        # Fetch mapping data
        joined_df = fetch_data_by_ids(object_id, segment_id, project_id, conn)
        
        # Handle missing values
        joined_df = missing_values_handling(joined_df)
        
        # Get target data sample - Use SQL directly instead of DataFrame
        target_df_sample = None
        try:
            # Get the target table name from joined_df
            target_table = joined_df["table_name"].unique().tolist()
            if target_table and len(target_table) > 0:
                # Get current target data using SQL
                target_df_sample = sql_executor.get_table_sample(target_table[0])
                
                # If SQL-based retrieval fails, try the original approach as fallback
                if isinstance(target_df_sample, dict) and "error_type" in target_df_sample:
                    logger.warning(f"SQL-based target data sample retrieval failed, using fallback")
                    # Get a connection to fetch current target data
                    target_df = get_or_create_session_target_df(
                        session_id, target_table[0], conn
                    )
                    target_df_sample = (
                        target_df.head(5).to_dict("records")
                        if not target_df.empty
                        else []
                    )
                else:
                    # Convert DataFrame to dict for consistency
                    target_df_sample = target_df_sample.head(5).to_dict("records") if not target_df_sample.empty else []
        except Exception as e:
            logger.warning(f"Error getting target data sample: {e}")
            target_df_sample = []
            
        # If query_type not provided, determine it now
        if not query_type:
            query_type, classification_details = classify_query_with_spacy(query)
        
        # Get the appropriate prompt template
        prompt_template = PROMPT_TEMPLATES.get(query_type, PROMPT_TEMPLATES["SIMPLE_TRANSFORMATION"])
        
        # Format target data sample for the prompt
        target_df_sample_str = "No current target data available"
        if target_df_sample:
            try:
                target_df_sample_str = json.dumps(target_df_sample, indent=2)
            except Exception as e:
                logger.warning(f"Error formatting target data sample: {e}")
                
        # Format visited segments for the prompt
        visited_segments_str = "No previously visited segments"
        if visited_segments:
            try:
                formatted_segments = []
                for seg_id, seg_info in visited_segments.items():
                    formatted_segments.append(
                        f"{seg_info.get('name')} (table: {seg_info.get('table_name')}, id: {seg_id})"
                    )
                visited_segments_str = "\n".join(formatted_segments)
            except Exception as e:
                logger.warning(f"Error formatting visited segments: {e}")
                
        # Format the prompt with all inputs
        table_desc = joined_df[joined_df.columns.tolist()[:-1]]
        
        # FIX: Change 'semgent_mapping' to 'segment_mapping'
        formatted_prompt = prompt_template.format(
            question=query,
            table_desc=list(table_desc.itertuples(index=False)),
            target_df_sample=target_df_sample_str,
            segment_mapping=context_manager.get_segments(session_id) if session_id else [],  # FIXED: was 'semgent_mapping'
        )
        
        # Call Gemini API with customized prompt
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise APIError("Gemini API key not configured")
            
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17", 
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                temperature=0.5, top_p=0.95, top_k=40
            ),
        )
        
        # Extract and parse JSON from response
        json_str = re.search(r"```json(.*?)```", response.text, re.DOTALL)
        if json_str:
            parsed_data = json.loads(json_str.group(1).strip())
        else:
            # Try to parse the whole response as JSON
            parsed_data = json.loads(response.text.strip())
            
        # Add query type to the parsed data
        parsed_data["query_type"] = query_type
        
        # Add other standard information
        parsed_data["target_table_name"] = joined_df["table_name"].unique().tolist()
        parsed_data["key_mapping"] = context_manager.get_key_mapping(session_id) if session_id else []
        parsed_data["visited_segments"] = visited_segments
        parsed_data["session_id"] = session_id
        
        # Add the classification details
        parsed_data["classification_details"] = classification_details
        if target_sap_fields is not None:
            if isinstance(target_sap_fields, list):
                parsed_data["target_sap_fields"] = target_sap_fields
            else:
                parsed_data["target_sap_fields"] = [target_sap_fields]
                
        # Fetch table schema information for SQL generation
        schema_info = {}
        for table_name in parsed_data.get("source_table_name", []):
            try:
                table_schema = sql_executor.get_table_schema(table_name)
                if isinstance(table_schema, list):
                    schema_info[table_name] = table_schema
            except Exception as e:
                logger.warning(f"Error fetching schema for {table_name}: {e}")
                
        parsed_data["table_schemas"] = schema_info
        
        # Check target table schema too
        target_table_names = parsed_data.get("target_table_name", [])
        if target_table_names:
            target_table = target_table_names[0] if isinstance(target_table_names, list) else target_table_names
            try:
                target_schema = sql_executor.get_table_schema(target_table)
                if isinstance(target_schema, list):
                    parsed_data["target_table_schema"] = target_schema
            except Exception as e:
                logger.warning(f"Error fetching schema for target table {target_table}: {e}")
                
        # Process the resolved data to get table information
        results = process_info(parsed_data, conn)
        
        # Handle key mapping differently based on query type
        if query_type == "SIMPLE_TRANSFORMATION":
            # For simple transformations, use original key mapping logic
            results = _handle_key_mapping_for_simple(results, joined_df, context_manager, session_id, conn)
        else:
            # For other operations, we don't enforce strict key mapping
            # Just pass through the existing key mappings
            results["key_mapping"] = parsed_data["key_mapping"]
        
        # Add session_id and other metadata
        results["session_id"] = session_id
        results["query_type"] = query_type
        results["visited_segments"] = visited_segments
        results["current_segment"] = {
            "id": segment_id,
            "name": segment_name if 'segment_name' in locals() else f"segment_{segment_id}"
        }
        
        # Add table schemas to results
        results["table_schemas"] = parsed_data.get("table_schemas", {})
        results["target_table_schema"] = parsed_data.get("target_table_schema", [])
        
        return results
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")

def _handle_key_mapping_for_simple(results, joined_df, context_manager, session_id, conn):
    """
    Handle key mapping specifically for simple transformations
    
    This uses the original key mapping logic for simple transformations
    """
    key_mapping = []
    key_mapping = context_manager.get_key_mapping(session_id)
    
    if not key_mapping:
        try:
            # Check if we have a target field and it's a key
            for target_field in results["target_sap_fields"]:
                target_field_filter = joined_df["target_sap_field"] == target_field
                if target_field_filter.any() and joined_df[target_field_filter]["isKey"].values[0] == "True":
                    # We're working with a primary key field
                    logger.info(f"Target field '{target_field}' is identified as a primary key")

                    # Check if we have insertion fields to map
                    if results["insertion_fields"] and len(results["insertion_fields"]) > 0:
                        # CRITICAL FIX: Don't use target field as source field
                        # Instead, use the actual insertion field from source table
                        source_field = None
                        
                        # First try to find a matching source field from the insertion fields
                        for field in results["insertion_fields"]:
                            if field in results["source_field_names"]:
                                source_field = field
                                break
                                
                        # If no direct match, take the first insertion field
                        if not source_field and results["insertion_fields"]:
                            source_field = results["insertion_fields"][0]
                            
                        # Get source table
                        source_table = (
                            results["source_table_name"][0]
                            if results["source_table_name"]
                            else None
                        )

                        # Verify the source data meets primary key requirements
                        if source_table and source_field:
                            error = None
                            try:
                                # Get the source data using SQL instead of DataFrame
                                has_nulls = False
                                has_duplicates = False
                                
                                try:
                                    # Validate table and field names to prevent SQL injection
                                    safe_table = validate_sql_identifier(source_table)
                                    safe_field = validate_sql_identifier(source_field)
                                    
                                    # Check for nulls
                                    null_query = f"SELECT COUNT(*) AS null_count FROM {safe_table} WHERE {safe_field} IS NULL"
                                    null_result = sql_executor.execute_query(null_query)
                                    
                                    if isinstance(null_result, list) and null_result:
                                        has_nulls = null_result[0].get("null_count", 0) > 0
                                    
                                    # Check for duplicates
                                    dup_query = f"""
                                    SELECT COUNT(*) AS dup_count
                                    FROM (
                                        SELECT {safe_field}, COUNT(*) as cnt
                                        FROM {safe_table}
                                        WHERE {safe_field} IS NOT NULL
                                        GROUP BY {safe_field}
                                        HAVING COUNT(*) > 1
                                    )
                                    """
                                    dup_result = sql_executor.execute_query(dup_query)
                                    
                                    if isinstance(dup_result, list) and dup_result:
                                        has_duplicates = dup_result[0].get("dup_count", 0) > 0
                                        
                                except Exception as e:
                                    logger.error(f"Failed to query source data for key validation: {e}")
                                    has_nulls = True  # Assume worst case
                                    has_duplicates = True  # Assume worst case
                                    
                                # Only proceed if the data satisfies primary key requirements
                                # or if the query explicitly indicates working with distinct values
                                if has_nulls or has_duplicates:
                                    # Check if the query is requesting distinct values
                                    restructured_query = results.get("restructured_query", "")
                                    is_distinct_query = (
                                        check_distinct_requirement(restructured_query) if restructured_query 
                                        else False
                                    )

                                    if not is_distinct_query:
                                        # The data doesn't meet primary key requirements and query doesn't indicate distinct values
                                        error_msg = f"Cannot use '{source_field}' as a primary key: "
                                        if has_nulls and has_duplicates:
                                            error_msg += "contains null values and duplicate entries"
                                        elif has_nulls:
                                            error_msg += "contains null values"
                                        else:
                                            error_msg += "contains duplicate entries"

                                        logger.error(error_msg)
                                        error = error_msg
                                    else:
                                        logger.info(
                                            f"Source data has integrity issues but query suggests distinct values will be used"
                                        )
                            except Exception as e:
                                logger.error(f"Error during primary key validation: {e}")
                                error = f"Error during key validation: {e}"
                                
                        if not error and source_field:
                            # If we've reached here, it's safe to add the key mapping
                            logger.info(f"Adding key mapping: {target_field} -> {source_field}")
                            key_mapping = context_manager.add_key_mapping(
                                session_id, target_field, source_field
                            )
                        else:
                            key_mapping = [error] if error else []
                    else:
                        logger.warning("No insertion fields found for key mapping")
                        key_mapping = context_manager.get_key_mapping(session_id)
                else:
                    # Not a key field, just get existing mappings
                    key_mapping = context_manager.get_key_mapping(session_id)
        except Exception as e:
            logger.error(f"Error processing key mapping: {e}")
            # Continue with empty key mapping
            key_mapping = []

    # Safely add key mapping to results
    results["key_mapping"] = key_mapping
    
    return results

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


def validate_sql_identifier(identifier):
    """
    Validate that an SQL identifier doesn't contain injection attempts
    Returns sanitized identifier or raises exception
    """
    if not identifier:
        raise SQLInjectionError("Empty SQL identifier provided")

    # Check for common SQL injection patterns
    dangerous_patterns = [
        ";",
        "--",
        "/*",
        "*/",
        "DROP",
        "DELETE",
        "INSERT",
        "UPDATE",
        "UNION",
        "EXEC",
        "EXECUTE",
    ]
    for pattern in dangerous_patterns:
        if pattern.lower() in identifier.lower():
            raise SQLInjectionError(
                f"Potentially dangerous SQL pattern found: {pattern}"
            )

    # Only allow alphanumeric characters, underscores, and some specific characters
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", identifier):
        raise SQLInjectionError("SQL identifier contains invalid characters")
    return identifier


def check_distinct_requirement(sentence):
    """
    Analyzes a sentence to determine if it contains words semantically similar to 'distinct' or 'unique',
    which would indicate a need for DISTINCT in SQL queries.

    Args:
        sentence (str): The input sentence/query to analyze

    Returns:
        bool: True if the sentence likely requires distinct values, False otherwise
    """
    # Load the spaCy model - using the medium English model for better word vectors
    nlp = spacy.load("en_core_web_md")

    # Process the input sentence
    doc = nlp(sentence.lower())

    # Target words we're looking for similarity to
    target_words = ["distinct", "unique", "different", "individual", "separate"]
    target_docs = [nlp(word) for word in target_words]

    similarity_threshold = 0.9

    direct_keywords = [
        "distinct",
        "unique",
        "duplicates",
        "duplicate",
        "duplicated",
        "deduplicate",
        "deduplication",
    ]
    for token in doc:
        if token.text in direct_keywords:
            return True

    for token in doc:
        if token.is_stop or token.is_punct:
            continue

        # Check similarity with each target word
        for target_doc in target_docs:
            similarity = token.similarity(target_doc[0])
            if similarity > similarity_threshold:
                return True

    return False


class ContextualSessionManager:
    """
    Manages context and state for sequential data transformations
    """

    def __init__(self, storage_path="sessions"):
        self.storage_path = storage_path
        try:
            os.makedirs(storage_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create session storage directory: {e}")
            raise SessionError(f"Failed to create session storage: {e}")

    def add_segment(self, session_id, segment_name,target_table_name):
        """Add a new segment to the session context"""
        try:
            if not session_id:
                logger.warning("No session ID provided for add_segment")
                return False
            session_path = f"{self.storage_path}/{session_id}"
            if not os.path.exists(session_path):
                logger.warning(f"Session directory not found: {session_path}")
                os.makedirs(session_path, exist_ok=True)
            context_path = f"{session_path}/segments.json"
            if not os.path.exists(context_path):
                segments = []
            else:
                try:
                    with open(context_path, "r") as f:
                        segments = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON in segments file, creating new segments"
                    )
                    segments = []
            
            # Add the new segment
            segments.append(
                {
                    "segment_name": segment_name,
                    "target_table_name": target_table_name,
                })
            
            # Save the updated segments
            with open(context_path, "w") as f:
                json.dump(segments, f, indent=2)
            return True
        except:
            pass

    def get_segments(self, session_id):
        """Get the segments for a session"""
        try:
            if not session_id:
                logger.warning("No session ID provided for get_segments")
                return []

            context_path = f"{self.storage_path}/{session_id}/segments.json"
            if not os.path.exists(context_path):
                logger.warning(f"Segments file not found for session {session_id}")
                return []

            with open(context_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in segments file for session {session_id}: {e}")
            return []

    def create_session(self):
        """Create a new session and return its ID"""
        try:
            session_id = str(uuid.uuid4())
            session_path = f"{self.storage_path}/{session_id}"
            os.makedirs(session_path, exist_ok=True)

            # Initialize empty context
            context = {
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

            # Save initial context
            with open(f"{session_path}/context.json", "w") as f:
                json.dump(context, f, indent=2)

            return session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise SessionError(f"Failed to create session: {e}")

    def get_context(self, session_id):
        """Get the current context for a session"""
        try:
            if not session_id:
                logger.warning("No session ID provided for get_context")
                return None

            context_path = f"{self.storage_path}/{session_id}/context.json"
            if not os.path.exists(context_path):
                logger.warning(f"Context file not found for session {session_id}")
                return None

            with open(context_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in context file for session {session_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get context for session {session_id}: {e}")
            return None

    def update_context(self, session_id, resolved_data):
        """Update the context with new resolved data"""
        try:
            if not session_id:
                logger.warning("No session ID provided for update_context")
                raise SessionError("No session ID provided")

            session_path = f"{self.storage_path}/{session_id}"
            if not os.path.exists(session_path):
                logger.warning(f"Session directory not found: {session_path}")
                os.makedirs(session_path, exist_ok=True)

            context_path = f"{session_path}/context.json"

            # Save the entire resolved data object (which includes the updated context)
            with open(context_path, "w") as f:
                json.dump(resolved_data, f, indent=2)

            # Also save as a versioned snapshot for history
            snapshot_path = (
                f"{session_path}/context_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            )
            with open(snapshot_path, "w") as f:
                json.dump(resolved_data, f, indent=2)

            return resolved_data
        except Exception as e:
            logger.error(f"Failed to update context for session {session_id}: {e}")
            raise SessionError(
                f"Failed to update context for session {session_id}: {e}"
            )

    def get_transformation_history(self, session_id):
        """Get the transformation history for a session"""
        try:
            context = self.get_context(session_id)
            if not context or "context" not in context:
                return []

            return context["context"]["transformation_history"]
        except Exception as e:
            logger.error(
                f"Failed to get transformation history for session {session_id}: {e}"
            )
            return []
    
    def track_segment(self, session_id, segment_id, segment_name, conn=None):
        """Track a visited segment in the session context"""
        try:
            if not session_id:
                logger.warning("No session ID provided for track_segment")
                return False
                
            # Get existing context or create new
            context_path = f"{self.storage_path}/{session_id}/context.json"
            context = None
            
            if os.path.exists(context_path):
                try:
                    with open(context_path, 'r') as f:
                        context = json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading context file: {e}")
                    context = {"session_id": session_id}
            else:
                # Create session directory if it doesn't exist
                os.makedirs(os.path.dirname(context_path), exist_ok=True)
                context = {"session_id": session_id}
                
            # Get segment name if not provided and conn exists
            if not segment_name and conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT segement_name FROM connection_segments WHERE segment_id = ?", (segment_id,))
                    result = cursor.fetchone()
                    if result:
                        segment_name = result[0]
                    else:
                        segment_name = f"segment_{segment_id}"
                except Exception as e:
                    logger.error(f"Error fetching segment name: {e}")
                    segment_name = f"segment_{segment_id}"
                    
            # Initialize segments_visited if needed
            if "segments_visited" not in context:
                context["segments_visited"] = {}
                
            # Add to visited segments
            context["segments_visited"][str(segment_id)] = {
                "name": segment_name,
                "visited_at": datetime.now().isoformat(),
                "table_name": ''.join(c if c.isalnum() else '_' for c in segment_name.lower())
            }
            
            # Save updated context
            with open(context_path, 'w') as f:
                json.dump(context, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error in track_segment: {e}")
            return False

    def add_key_mapping(self, session_id, target_col, source_col):
        """Add a key mapping for a session"""
        try:
            if not session_id:
                logger.warning("No session ID provided for add_key_mapping")
                raise SessionError("No session ID provided")

            # Validate parameters to prevent injection
            if not target_col or not isinstance(target_col, str):
                logger.warning(f"Invalid target column: {target_col}")
                return []

            if not source_col or not isinstance(source_col, str):
                logger.warning(f"Invalid source column: {source_col}")
                return []

            file_path = f"{self.storage_path}/{session_id}/key_mapping.json"

            # If directory doesn't exist, create it
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            if not os.path.exists(file_path):
                key_mappings = []
            else:
                try:
                    with open(file_path, "r") as f:
                        key_mappings = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON in key mapping file, creating new mapping"
                    )
                    key_mappings = []
                except Exception as e:
                    logger.error(f"Error reading key mapping file: {e}")
                    key_mappings = []

            # Add the new mapping
            if not any(
                mapping
                for mapping in key_mappings
                if mapping["target_col"] == target_col
                and mapping["source_col"] == source_col
            ):
                key_mappings.append(
                    {"target_col": target_col, "source_col": source_col}
                )

            # Save the updated mappings
            with open(file_path, "w") as f:
                json.dump(key_mappings, f, indent=2)

            return key_mappings
        except Exception as e:
            logger.error(f"Failed to add key mapping for session {session_id}: {e}")
            return []

    def get_key_mapping(self, session_id):
        """Get key mappings for a session"""
        try:
            if not session_id:
                logger.warning("No session ID provided for get_key_mapping")
                return []

            file_path = f"{self.storage_path}/{session_id}/key_mapping.json"
            if not os.path.exists(file_path):
                return []

            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in key mapping file for session {session_id}")
            return []
        except Exception as e:
            logger.error(f"Failed to get key mapping for session {session_id}: {e}")
            return []

    def get_segment_info(self, session_id, segment_id=None, segment_name=None):
        """
        Get information about a specific segment
        
        Parameters:
        session_id (str): Session ID
        segment_id (str, optional): Segment ID to look for
        segment_name (str, optional): Segment name to look for (partial match)
        
        Returns:
        dict: Segment information or None if not found
        """
        try:
            context = self.get_context(session_id)
            if not context:
                return None
                
            segments = context.get("segments_visited", {})
            
            # Direct lookup by segment_id
            if segment_id and segment_id in segments:
                return segments[segment_id]
                
            # Search by name (partial match)
            if segment_name:
                segment_name_lower = segment_name.lower()
                for seg_id, info in segments.items():
                    seg_name = info.get("name", "").lower()
                    if segment_name_lower in seg_name or seg_name in segment_name_lower:
                        return info
                        
            return None
        except Exception as e:
            logger.error(f"Error in get_segment_info: {e}")
            return None
            
    def is_cross_segment_query(self, session_id, query):
        """
        Determine if a query is likely a cross-segment operation
        
        Parameters:
        session_id (str): Session ID
        query (str): The query to analyze
        Parameters:
        session_id (str): Session ID
        query (str): The query to analyze
        
        Returns:
        bool: True if likely a cross-segment query
        """
        try:
            # No segments visited means it can't be cross-segment
            context = self.get_context(session_id)
            if not context:
                return False
                
            segments = context.get("segments_visited", {})
            if not segments:
                return False
                
            # Check if query mentions any visited segments
            query_lower = query.lower()
            for _, info in segments.items():
                segment_name = info.get("name", "").lower()
                table_name = info.get("table_name", "").lower()
                
                # Check for segment name mentions
                if segment_name and segment_name in query_lower:
                    return True
                    
                # Check for table name mentions
                if table_name and table_name in query_lower:
                    return True
                    
            # Check for general cross-segment terminology
            cross_segment_terms = [
                "previous segment", "last segment", "prior segment",
                "segment data", "from segment", "basic segment",
                "marc segment", "makt segment"
            ]
            
            for term in cross_segment_terms:
                if term in query_lower:
                    return True
                    
            return False
        except Exception as e:
            logger.error(f"Error in is_cross_segment_query: {e}")
            return False

def fetch_data_by_ids(object_id, segment_id, project_id, conn):
    """Fetch data mappings from the database"""
    try:
        # Validate parameters
        if not all(isinstance(x, int) for x in [object_id, segment_id, project_id]):
            logger.error("Invalid parameter types for fetch_data_by_ids")
            raise ValueError("Object ID, segment ID, and project ID must be integers")

        joined_query = """
        SELECT 
            f.description,
            f.isMandatory,
            f.isKey,
            r.source_field_name,
            r.target_sap_field,
            s.table_name
        FROM connection_fields f
        LEFT JOIN (
            SELECT r1.*
            FROM connection_rule r1
            INNER JOIN (
                SELECT field_id, MAX(version_id) as max_version
                FROM connection_rule
                WHERE object_id_id = ? 
                AND segment_id_id = ? 
                AND project_id_id = ? 
                GROUP BY field_id
            ) r2 ON r1.field_id = r2.field_id AND r1.version_id = r2.max_version
            WHERE r1.object_id_id = ? 
            AND r1.segment_id_id = ? 
            AND r1.project_id_id = ? 
        ) r ON f.field_id = r.field_id
        JOIN connection_segments s ON f.segement_id_id = s.segment_id
            AND f.obj_id_id = s.obj_id_id
            AND f.project_id_id = s.project_id_id
        WHERE f.obj_id_id = ? 
        AND f.segement_id_id = ? 
        AND f.project_id_id = ? 
        """

        params = [object_id, segment_id, project_id] * 3
        joined_df = pd.read_sql_query(joined_query, conn, params=params)

        if joined_df.empty:
            logger.warning(
                f"No data found for object_id={object_id}, segment_id={segment_id}, project_id={project_id}"
            )

        return joined_df
    except sqlite3.Error as e:
        logger.error(f"SQLite error in fetch_data_by_ids: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in fetch_data_by_ids: {e}")
        raise


def missing_values_handling(df):
    """Handle missing values in the dataframe"""
    try:
        # Check if dataframe is empty or None
        if df is None or df.empty:
            logger.warning("Empty dataframe passed to missing_values_handling")
            return df

        # Make a copy to avoid modifying the original
        df_processed = df.copy()

        # Handle source_table column
        if "source_table" in df_processed.columns:
            # Convert empty strings and whitespace-only to NaN first
            df_processed["source_table"] = df_processed["source_table"].replace(
                r"^\s*$", pd.NA, regex=True
            )

            # Fill NaN values if there are any non-NaN values
            if not df_processed["source_table"].dropna().empty:
                non_na_values = df_processed["source_table"].dropna()
                if len(non_na_values) > 0:
                    fill_value = non_na_values.iloc[0]
                    df_processed["source_table"] = df_processed["source_table"].fillna(
                        fill_value
                    )

        # Handle source_field_name based on target_sap_field
        if (
            "source_field_name" in df_processed.columns
            and "target_sap_field" in df_processed.columns
        ):
            # Convert empty strings to NaN first
            df_processed["source_field_name"] = df_processed[
                "source_field_name"
            ].replace(r"^\s*$", pd.NA, regex=True)

            # Count null values
            null_count = df_processed["source_field_name"].isna().sum()
            if null_count > 0:
                # Ensure we don't propagate empty strings from target_sap_field
                df_processed["target_sap_field"] = df_processed[
                    "target_sap_field"
                ].replace(r"^\s*$", pd.NA, regex=True)
                valid_targets = df_processed["target_sap_field"].notna()

                # Fill missing source fields with target fields where available
                missing_sources = df_processed["source_field_name"].isna()
                fill_indices = missing_sources & valid_targets

                if fill_indices.any():
                    df_processed.loc[fill_indices, "source_field_name"] = (
                        df_processed.loc[fill_indices, "target_sap_field"]
                    )

        return df_processed
    except Exception as e:
        logger.error(f"Error in missing_values_handling: {e}")
        # Return the original dataframe if there's an error
        return df


@track_token_usage(log_to_file=True, log_path="gemini_planner_usage.log")
def parse_data_with_context(
    joined_df, query, session_id=None, previous_context=None, target_table_desc=None
):
    """
    Parse data using Gemini API with token usage tracking and context awareness

    Parameters:
    joined_df (DataFrame): The joined dataframe with field mappings
    query (str): The natural language query
    session_id (str): The current session ID for retrieving key mappings
    previous_context (dict): Context from previous transformations

    Returns:
    dict: The parsed data with mapping information and updated context
    """
    try:
        # Validate inputs
        if joined_df is None or joined_df.empty:
            logger.error("Empty dataframe passed to parse_data_with_context")
            raise ValueError("Empty dataframe provided")

        if not query or not isinstance(query, str):
            logger.error(
                f"Invalid query passed to parse_data_with_context: {type(query)}"
            )
            raise ValueError("Query must be a non-empty string")

        # Get key mapping if session_id is provided
        key_mapping = []
        target_df_sample = None
        if session_id:
            try:
                # Initialize context manager to get key mappings
                context_manager = ContextualSessionManager()
                key_mapping = context_manager.get_key_mapping(session_id)
                session_names = context_manager.get_segments(session_id)

                # Get the current state of target data if available
                try:
                    # Get the target table name from joined_df
                    target_table = joined_df["table_name"].unique().tolist()
                    if target_table and len(target_table) > 0:
                        # Use SQL executor to get table sample instead of loading full DataFrame
                        target_df_sample = sql_executor.get_table_sample(target_table[0])
                        
                        if isinstance(target_df_sample, dict) and "error_type" in target_df_sample:
                            # If SQL approach fails, fall back to original method
                            logger.warning(f"SQL sample retrieval failed, using fallback: {target_df_sample.get('error_message')}")
                            # Get a connection to fetch current target data
                            conn = sqlite3.connect("db.sqlite3")
                            target_df = get_or_create_session_target_df(
                                session_id, target_table[0], conn
                            )
                            target_df_sample = (
                                target_df.head(5).to_dict("records")
                                if not target_df.empty
                                else []
                            )
                            conn.close()
                        else:
                            # Convert to dict records format for compatibility
                            target_df_sample = target_df_sample.head(5).to_dict("records") if not target_df_sample.empty else []
                except Exception as e:
                    logger.warning(f"Error getting target data sample: {e}")
                    target_df_sample = []
            except Exception as e:
                logger.warning(
                    f"Error retrieving key mapping for session {session_id}: {e}"
                )
                # Continue with empty key mapping if there's an error

        prompt = """
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
       - Target field
       - Filtering or transformation conditions
       - Logical flow (IF/THEN/ELSE statements)
       - insertion fields
     
    2. Match these entities to the corresponding entries in the joined_data.csv schema
       - For each entity, find the closest match in the schema
       - Resolve ambiguities using the description field
       - Validate that the identified fields exist in the mentioned tables.
       - Carefully extract the target_sap_field from the datascheme given.
     
    3. Generate a structured representation of the transformation logic:
       - JSON format showing the transformation flow
       - Include all source tables, fields, conditions, and targets
       - Map conditional logic to proper syntax
       - Handle fallback scenarios (ELSE conditions)
       - Use the provided key mappings to connect source and target fields correctly
       - Consider the current state of the target data shown above.
     
    4. Create a resolved query that takes the actual field and table names, and does not change what is said in the 

    5. For the insertion fields, identify the fields that need to be inserted into the target table based on the User query. Take note to not add the filtering fields to the insertion fields if not specifically requested.

    6. Restructure the user query with resolved data types and field names.

    Note:
    - In the Restrucutured query, only replace the textual descriptions with the field names.
    - Do not change the query itself, just replace the field names with the actual field names.
    - if target_sap_field is not found in the schema, return make source_field_name the target_sap_field.

    Respond with:
    ```json
    {{
    "source_table_name": [List of all source_tables],
    "source_field_names": [List of all source_fields],
    "filtering_fields": [List of filtering fields],
    "insertion_fields": [insertion_field],
    "target_sap_fields": target_sap_fields
    "Resolved_query": [Rephrased query with resolved data]
    }}
    ```
        """

        # Format the previous context for the prompt
        context_str = (
            "None"
            if previous_context is None
            else json.dumps(previous_context, indent=2)
        )

        # Format key mapping for the prompt
        key_mapping_str = "No key mappings available"
        if key_mapping:
            try:
                formatted_mappings = []
                for mapping in key_mapping:
                    if (
                        isinstance(mapping, dict)
                        and "target_col" in mapping
                        and "source_col" in mapping
                    ):
                        formatted_mappings.append(
                            f"{mapping['target_col']}:{mapping['source_col']}"
                        )
                key_mapping_str = (
                    ", ".join(formatted_mappings)
                    if formatted_mappings
                    else "No key mappings available"
                )
            except Exception as e:
                logger.warning(f"Error formatting key mappings: {e}")
                key_mapping_str = "Error processing key mappings"

        # Format target data sample for the prompt
        target_df_sample_str = "No current target data available"
        if target_df_sample:
            try:
                target_df_sample_str = json.dumps(target_df_sample, indent=2)
            except Exception as e:
                logger.warning(f"Error formatting target data sample: {e}")
                target_df_sample_str = "Error processing target data sample"

        # Format the prompt with all inputs
        table_desc = joined_df[joined_df.columns.tolist()[:-1]] 
        formatted_prompt = prompt.format(
            question=query,
            table_desc=list(table_desc.itertuples(index=False)),
            key_mapping=key_mapping_str,
            target_df_sample=target_df_sample_str,
        )

        # Get Gemini API key from environment
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise APIError("Gemini API key not configured")

        # Initialize Gemini client
        client = genai.Client(api_key=api_key)

        # Call Gemini API with token tracking and error handling
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=formatted_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.5, top_p=0.95, top_k=40
                ),
            )

            # Check if response is valid
            if not response or not hasattr(response, "text"):
                logger.error("Invalid response from Gemini API")
                raise APIError("Failed to get valid response from Gemini API")

            # Log token usage statistics

            # Parse the JSON response with error handling
            try:
                json_str = re.search(r"```json(.*?)```", response.text, re.DOTALL)
                if json_str:
                    parsed_data = json.loads(json_str.group(1).strip())
                else:
                    # Try to parse the whole response as JSON
                    parsed_data = json.loads(response.text.strip())

                # Validate the parsed data structure
                required_keys = [
                    "source_table_name",
                    "source_field_names",
                    "filtering_fields",
                    "insertion_fields",
                    "Resolved_query",
                    "target_sap_fields",
                ]

                for key in required_keys:
                    if key not in parsed_data:
                        logger.warning(f"Missing key in Gemini response: {key}")
                        parsed_data[key] = (
                            []
                            if key != "transformation_logic" and key != "Resolved_query"
                            else ""
                        )

                # Add target table information
                parsed_data["target_table_name"] = joined_df["table_name"].unique().tolist()

                # Add key mapping information to the result
                parsed_data["key_mapping"] = key_mapping

                return parsed_data
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {e}")
                logger.error(f"Raw response: {response.text}")
                raise DataProcessingError(
                    f"Failed to parse Gemini response as JSON: {e}"
                )
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise APIError(f"Failed to call Gemini API: {e}")
    except Exception as e:
        logger.error(f"Error in parse_data_with_context: {e}")
        return None


def process_query(object_id, segment_id, project_id, query, session_id=None, target_sap_fields=None):
    """
    Process a query with context awareness and automatic query type detection
    
    Parameters:
    object_id (int): Object ID
    segment_id (int): Segment ID
    project_id (int): Project ID
    query (str): The natural language query
    session_id (str): Optional session ID for context tracking
    target_sap_fields (str/list): Optional target SAP fields
    
    Returns:
    dict: Processed information including context or None if key validation fails
    """
    try:
        # Validate inputs
        if not query or not isinstance(query, str):
            logger.error(f"Invalid query: {query}")
            return None

        # Validate IDs
        if not all(isinstance(x, int) for x in [object_id, segment_id, project_id]):
            logger.error(
                f"Invalid ID types: object_id={type(object_id)}, segment_id={type(segment_id)}, project_id={type(project_id)}"
            )
            return None

        # Initialize context manager
        context_manager = ContextualSessionManager()

        # Create a session if none provided
        if not session_id:
            session_id = context_manager.create_session()
            logger.info(f"Created new session: {session_id}")
        
        # Classify the query type using spaCy
        query_type, classification_details = classify_query_with_spacy(query)
        
        # Process the query based on its type
        return process_query_by_type(
            object_id, 
            segment_id, 
            project_id, 
            query, 
            session_id, 
            query_type, 
            classification_details,
            target_sap_fields  # Pass target_sap_fields to process_query_by_type
        )
    except Exception as e:
        logger.error(f"Error in process_query: {e}")
        logger.error(traceback.format_exc())
        return None

def get_session_context(session_id):
    """
    Get the current context for a session

    Parameters:
    session_id (str): The session ID

    Returns:
    dict: The session context
    """
    try:
        if not session_id:
            logger.warning("No session ID provided for get_session_context")
            return None

        context_manager = ContextualSessionManager()
        return context_manager.get_context(session_id)
    except Exception as e:
        logger.error(f"Error in get_session_context: {e}")
        return None


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
    try:
        if not session_id:
            logger.warning("No session ID provided for get_or_create_session_target_df")
            return pd.DataFrame()

        if not target_table:
            logger.warning(
                "No target table provided for get_or_create_session_target_df"
            )
            return pd.DataFrame()

        if not conn:
            logger.warning(
                "No database connection provided for get_or_create_session_target_df"
            )
            return pd.DataFrame()

        session_path = f"sessions/{session_id}"
        target_path = f"{session_path}/target_latest.csv"

        if os.path.exists(target_path):
            # Load existing target data
            try:
                target_df = pd.read_csv(target_path)
                return target_df
            except Exception as e:
                logger.error(f"Error reading existing target CSV: {e}")
                # If there's an error reading the file, get fresh data from the database

        # Get fresh target data from the database - use SQL executor
        try:
            # Validate target table name
            safe_table = validate_sql_identifier(target_table)
            
            # Use SQL executor to get full table
            target_df = sql_executor.execute_and_fetch_df(f"SELECT * FROM {safe_table}")
            
            if isinstance(target_df, dict) and "error_type" in target_df:
                # If SQL executor failed, fall back to original approach
                logger.warning(f"SQL approach failed in get_or_create_session_target_df, using fallback: {target_df.get('error_message')}")
                
                # Use a parameterized query for safety
                query = f"SELECT * FROM {safe_table}"
                target_df = pd.read_sql_query(query, conn)
                
            return target_df
        except sqlite3.Error as e:
            logger.error(f"SQLite error in get_or_create_session_target_df: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in get_or_create_session_target_df: {e}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error in get_or_create_session_target_df: {e}")
        return pd.DataFrame()


def save_session_target_df(session_id, target_df):
    """
    Save the updated target dataframe for a session

    Parameters:
    session_id (str): Session ID
    target_df (DataFrame): The target dataframe to save

    Returns:
    bool: True if successful
    """
    try:
        if not session_id:
            logger.warning("No session ID provided for save_session_target_df")
            return False

        if target_df is None:
            logger.warning("None target_df provided for save_session_target_df")
            return False

        if not isinstance(target_df, pd.DataFrame):
            logger.warning(
                f"Invalid target_df type ({type(target_df)}) for save_session_target_df"
            )
            return False

        session_path = f"sessions/{session_id}"
        os.makedirs(session_path, exist_ok=True)
        target_path = f"{session_path}/target_latest.csv"

        # Also save a timestamped version for history
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        history_path = f"{session_path}/target_{timestamp}.csv"

        # Save the target dataframe
        try:
            target_df.to_csv(target_path, index=False)
            target_df.to_csv(history_path, index=False)
            logger.info(f"Saved target dataframe for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving target dataframe: {e}")
            return False
    except Exception as e:
        logger.error(f"Error in save_session_target_df: {e}")
        return False

def clean_table_name(table_name):
    """
    Clean table name by removing common suffixes like 'Table', 'table', etc.
    
    Parameters:
    table_name (str): The table name to clean
    
    Returns:
    str: Cleaned table name
    """
    if not table_name:
        return table_name
        
    # Remove common suffixes
    suffixes = [" Table", " table", " TABLE", "_Table", "_table", "_TABLE"]
    cleaned_name = table_name
    
    for suffix in suffixes:
        if cleaned_name.endswith(suffix):
            cleaned_name = cleaned_name[:-len(suffix)]
            break
            
    return cleaned_name


def process_info(resolved_data, conn):
    """
    Process the resolved data to extract table information based on the query type
    
    Parameters:
    resolved_data (dict): The resolved data from the language model
    conn (Connection): SQLite connection
    
    Returns:
    dict: Processed information including table samples
    """
    try:
        # Validate inputs
        if resolved_data is None:
            logger.error("None resolved_data passed to process_info")
            return None

        if conn is None:
            logger.error("None database connection passed to process_info")
            return None
            
        # Get query type - default to SIMPLE_TRANSFORMATION
        query_type = resolved_data.get("query_type", "SIMPLE_TRANSFORMATION")
        
        # Define required fields based on query type (same as original)
        required_fields = {
            "SIMPLE_TRANSFORMATION": [
                "source_table_name", "source_field_names", "target_table_name",
                "filtering_fields", "Resolved_query", "insertion_fields", 
                "target_sap_fields"
            ],
            "JOIN_OPERATION": [
                "source_table_name", "source_field_names", "target_table_name",
                "filtering_fields", "Resolved_query", "insertion_fields", 
                "target_sap_fields", "join_conditions"
            ],
            "CROSS_SEGMENT": [
                "source_table_name", "source_field_names", "target_table_name",
                "filtering_fields", "Resolved_query", "insertion_fields", 
                "target_sap_fields", "segment_references", "cross_segment_joins"
            ],
            "VALIDATION_OPERATION": [
                "source_table_name", "source_field_names", "target_table_name",
                "validation_rules", "target_sap_fields", "Resolved_query"
            ],
            "AGGREGATION_OPERATION": [
                "source_table_name", "source_field_names", "target_table_name",
                "aggregation_functions", "group_by_fields", "target_sap_fields", 
                "Resolved_query"
            ]
        }
        
        # Check if all required fields for this query type are present
        current_required_fields = required_fields.get(query_type, required_fields["SIMPLE_TRANSFORMATION"])
        
        for field in current_required_fields:
            if field not in resolved_data:
                logger.warning(f"Missing required field in resolved_data: {field}")
                # Initialize missing fields with sensible defaults
                if field in ["source_table_name", "source_field_names", "filtering_fields", 
                            "insertion_fields", "group_by_fields"]:
                    resolved_data[field] = []
                elif field in ["target_table_name", "target_sap_fields"]:
                    resolved_data[field] = []
                elif field == "Resolved_query":
                    resolved_data[field] = ""
                elif field == "join_conditions":
                    resolved_data[field] = []
                elif field == "validation_rules":
                    resolved_data[field] = []
                elif field == "aggregation_functions":
                    resolved_data[field] = []
                elif field == "segment_references":
                    resolved_data[field] = []
                elif field == "cross_segment_joins":
                    resolved_data[field] = []

        # Initialize result dictionary with fields based on query type
        result = {
            "query_type": query_type,
            "source_table_name": resolved_data["source_table_name"],
            "source_field_names": resolved_data["source_field_names"],
            "target_table_name": resolved_data["target_table_name"],
            "target_sap_fields": resolved_data["target_sap_fields"],
            "restructured_query": resolved_data["Resolved_query"],
        }
        
        # Add type-specific fields
        if query_type == "SIMPLE_TRANSFORMATION":
            result["filtering_fields"] = resolved_data["filtering_fields"]
            result["insertion_fields"] = resolved_data["insertion_fields"]
        elif query_type == "JOIN_OPERATION":
            result["filtering_fields"] = resolved_data["filtering_fields"]
            result["insertion_fields"] = resolved_data["insertion_fields"]
            result["join_conditions"] = resolved_data["join_conditions"]
        elif query_type == "CROSS_SEGMENT":
            result["filtering_fields"] = resolved_data["filtering_fields"]
            result["insertion_fields"] = resolved_data["insertion_fields"]
            result["segment_references"] = resolved_data["segment_references"]
            result["cross_segment_joins"] = resolved_data["cross_segment_joins"]
        elif query_type == "VALIDATION_OPERATION":
            result["validation_rules"] = resolved_data["validation_rules"]
        elif query_type == "AGGREGATION_OPERATION":
            result["aggregation_functions"] = resolved_data["aggregation_functions"]
            result["group_by_fields"] = resolved_data["group_by_fields"]
            
            # Add filtering fields if present
            if "filtering_fields" in resolved_data:
                result["filtering_fields"] = resolved_data["filtering_fields"]
            else:
                result["filtering_fields"] = []

        # Add data samples using SQL approach for better memory efficiency
        source_data = {}
        try:
            for table in resolved_data["source_table_name"]:
                # Clean the table name to remove suffixes
                cleaned_table = clean_table_name(table)
                
                try:
                    # Validate the table name
                    safe_table = validate_sql_identifier(cleaned_table)
                    
                    # Use SQL executor to get a sample of data
                    source_df = sql_executor.get_table_sample(safe_table, limit=5)
                    
                    if isinstance(source_df, dict) and "error_type" in source_df:
                        # If SQL executor failed, fall back to original approach
                        logger.warning(f"SQL source sample failed for {safe_table}, using fallback: {source_df.get('error_message')}")
                        
                        # Use pandas read_sql as fallback
                        query = f"SELECT * FROM {safe_table} LIMIT 5"
                        source_df = pd.read_sql_query(query, conn)
                        
                    # Store the sample data
                    source_data[table] = source_df
                except Exception as e:
                    logger.error(f"Error fetching source data for table {cleaned_table}: {e}")
                    source_data[table] = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting source data samples: {e}")
            source_data = {}
            
        result["source_data_samples"] = source_data
        
        # Add schema information
        result["table_schemas"] = resolved_data.get("table_schemas", {})
        result["target_table_schema"] = resolved_data.get("target_table_schema", [])
        
        return result
    except Exception as e:
        logger.error(f"Error in process_info: {e}")
        return None