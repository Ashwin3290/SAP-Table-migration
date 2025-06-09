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
from pathlib import Path
import spacy
import traceback
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import SQL related modules
from executor import SQLExecutor

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
        


from difflib import SequenceMatcher


def find_closest_match(query, word_list, threshold=0.6):
    """
    Find the closest matching word from a list using fuzzy string matching.
    
    Args:
        query (str): The search term (possibly with typos)
        word_list (list): List of valid words to match against
        threshold (float): Minimum similarity score (0.0 to 1.0)
        
    Returns:
        dict: Contains 'match' (best matching word), 'score' (similarity score), 
              and 'all_matches' (list of all matches above threshold)
    """
    if not query or not word_list:
        return {"match": None, "score": 0.0, "all_matches": []}
    
    # Clean the query (remove extra spaces, convert to lowercase)
    clean_query = re.sub(r'\s+', ' ', query.strip().lower())
    
    matches = []
    for word in word_list:
        clean_word = word.lower()
        
        # Calculate similarity using different methods
        # Method 1: Overall sequence similarity
        overall_sim = SequenceMatcher(None, clean_query, clean_word).ratio()
        
        # Method 2: Substring matching bonus
        substring_bonus = 0
        if clean_query in clean_word or clean_word in clean_query:
            substring_bonus = 0.2
        
        # Method 3: Word-level matching for multi-word strings
        query_words = clean_query.split()
        word_words = clean_word.split()
        word_level_sim = 0
        
        if len(query_words) > 1 or len(word_words) > 1:
            # For multi-word matching, check individual words
            word_matches = 0
            total_words = max(len(query_words), len(word_words))
            
            for q_word in query_words:
                best_word_match = 0
                for w_word in word_words:
                    word_sim = SequenceMatcher(None, q_word, w_word).ratio()
                    best_word_match = max(best_word_match, word_sim)
                word_matches += best_word_match
            
            word_level_sim = word_matches / total_words if total_words > 0 else 0
        
        # Combine scores with weights
        final_score = (overall_sim * 0.4) + (substring_bonus) + (word_level_sim * 0.5)
        final_score = min(final_score, 1.0)  # Cap at 1.0
        
        if final_score >= threshold:
            matches.append({
                'word': word,
                'score': final_score,
                'original_word': word
            })
    
    # Sort by score (highest first)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    # Return results
    result = {
        'match': matches[0]['word'] if matches else None,
        'score': matches[0]['score'] if matches else 0.0,
        'all_matches': [(m['word'], round(m['score'], 3)) for m in matches]
    }
    
    return result

class ClassificationEnhancer:
    """
    Enhances LLM classification details with fuzzy matching for tables, columns, and segments
    """
    
    def __init__(self, db_path=None, segments_csv_path="segments.csv"):
        """
        Initialize the ClassificationEnhancer
        
        Args:
            db_path (str): Path to the SQLite database
            segments_csv_path (str): Path to the segments CSV file
        """
        self.db_path = db_path or os.environ.get('DB_PATH')
        self.segments_csv_path = segments_csv_path
        self._available_tables = None
        self._table_columns = {}
        self._segments_df = None
        
    def _get_available_tables(self) -> List[str]:
        """
        Get list of available tables from the database
        
        Returns:
            List[str]: List of table names
        """
        if self._available_tables is not None:
            return self._available_tables
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            self._available_tables = tables
            return tables
            
        except Exception as e:
            logger.error(f"Error getting available tables: {e}")
            return []
    
    def _get_table_columns(self, table_name: str) -> List[str]:
        """
        Get columns for a specific table
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            List[str]: List of column names
        """
        if table_name in self._table_columns:
            return self._table_columns[table_name]
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]  # row[1] is column name
            
            conn.close()
            self._table_columns[table_name] = columns
            return columns
            
        except Exception as e:
            logger.error(f"Error getting columns for table {table_name}: {e}")
            return []
    
    def _get_all_table_columns(self, table_names: List[str]) -> Dict[str, List[str]]:
        """
        Get columns for multiple tables
        
        Args:
            table_names (List[str]): List of table names
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping table names to their columns
        """
        result = {}
        for table_name in table_names:
            result[table_name] = self._get_table_columns(table_name)
        return result
    
    def _load_segments_data(self) -> pd.DataFrame:
        """
        Load segments data from CSV
        
        Returns:
            pd.DataFrame: Segments dataframe
        """
        if self._segments_df is not None:
            return self._segments_df
            
        try:
            conn = sqlite3.connect(self.db_path)
            self._segments_df = pd.read_sql("SELECT * FROM connection_segments", conn)
            conn.close()
            return self._segments_df
        except Exception as e:
            logger.error(f"Error loading segments CSV: {e}")
            return pd.DataFrame()
    
    def _get_current_target_table_pattern(self, segment_id: int) -> Optional[str]:
        """
        Get the target table pattern (t_[number]) for the current segment
        
        Args:
            segment_id (int): Current segment ID
            
        Returns:
            Optional[str]: The t_[number] pattern or None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the target table for current segment
            cursor.execute("""
                SELECT table_name 
                FROM connection_segments 
                WHERE segment_id = ?
            """, (segment_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                table_name = result[0]
                # Extract t_[number] pattern
                match = re.match(r't_(\d+)', table_name.lower())
                if match:
                    return f"t_{match.group(1)}"
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current target table pattern: {e}")
            return None
        
    def _match_segments(self, segments_mentioned: List[str], current_segment_id: int) -> Dict[str, Any]:
            """
            Match mentioned segments with available segments and get target tables

            Args:
                segments_mentioned (List[str]): List of segment names mentioned
                current_segment_id (int): Current segment ID for pattern matching

            Returns:
                Dict[str, Any]: Enhanced segment information
            """
            segments_df = self._load_segments_data()
            if segments_df.empty or not segments_mentioned:
                return {"matched_segments": [], "segment_target_tables": {}}

            # Get available segment names
            available_segments = segments_df['segement_name'].unique().tolist()
            current_pattern = self._get_current_target_table_pattern(current_segment_id)

            matched_segments = []
            segment_target_tables = {}

            for mentioned_segment in segments_mentioned:
                # Find closest match
                match_result = find_closest_match(mentioned_segment, available_segments, threshold=0.3)

                if match_result['match']:
                    matched_segment = match_result['match']
                    matched_segments.append(matched_segment)

                    # Get target tables for this segment
                    segment_rows = segments_df[segments_df['segement_name'] == matched_segment]
                    target_tables = segment_rows['table_name'].tolist()

                    # If multiple target tables and we have a current pattern, prefer matching pattern
                    if len(target_tables) > 1 and current_pattern:
                        pattern_matches = [table for table in target_tables if table.lower().startswith(current_pattern)]
                        if pattern_matches:
                            target_tables = pattern_matches

                    segment_target_tables[matched_segment] = target_tables
                else:
                    # Keep original if no good match found
                    matched_segments.append(mentioned_segment)
                    segment_target_tables[mentioned_segment] = []

            return {
                "matched_segments": matched_segments,
                "segment_target_tables": segment_target_tables
            }

    def _match_tables(self, tables_mentioned: List[str]) -> Dict[str, Any]:
        """
        Match mentioned tables with available tables
        
        Args:
            tables_mentioned (List[str]): List of table names mentioned
            
        Returns:
            Dict[str, Any]: Enhanced table information
        """
        available_tables = self._get_available_tables()
        
        if not tables_mentioned or not available_tables:
            return {"matched_tables": [], "table_match_confidence": {}}
        
        matched_tables = []
        table_match_confidence = {}
        
        for mentioned_table in tables_mentioned:
            match_result = find_closest_match(mentioned_table, available_tables, threshold=0.6)
            
            if match_result['match']:
                matched_table = match_result['match']
                confidence = match_result['score']
                
                # Check if this is a high-confidence exact or near-exact match
                if confidence >= 0.9 or len(match_result['all_matches']) == 1:
                    # High confidence, use the match
                    matched_tables.append(matched_table)
                    table_match_confidence[matched_table] = confidence
                elif len(match_result['all_matches']) > 1:
                    # Multiple close matches, need to be careful
                    best_match = match_result['all_matches'][0]
                    second_best = match_result['all_matches'][1] if len(match_result['all_matches']) > 1 else None
                    
                    if second_best is None or (best_match[1] - second_best[1]) > 0.2:
                        # Clear winner
                        matched_tables.append(best_match[0])
                        table_match_confidence[best_match[0]] = best_match[1]
                    else:
                        # Ambiguous, keep original
                        matched_tables.append(mentioned_table)
                        table_match_confidence[mentioned_table] = 0.5
                else:
                    matched_tables.append(matched_table)
                    table_match_confidence[matched_table] = confidence
            else:
                # No good match, keep original
                matched_tables.append(mentioned_table)
                table_match_confidence[mentioned_table] = 0.0
        
        return {
            "matched_tables": matched_tables,
            "table_match_confidence": table_match_confidence
        }
    
    def _match_columns(self, columns_mentioned: List[str], matched_tables: List[str], 
                      segment_target_tables: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Match mentioned columns with available columns in tables
        
        Args:
            columns_mentioned (List[str]): List of column names mentioned
            matched_tables (List[str]): List of matched table names
            segment_target_tables (Dict[str, List[str]]): Segment target tables
            
        Returns:
            Dict[str, Any]: Enhanced column information
        """
        if not columns_mentioned:
            return {"matched_columns": [], "columns_in_mentioned_table": {}}
        
        # Get all potential tables to search (matched tables + segment target tables)
        all_potential_tables = set(matched_tables)
        for tables in segment_target_tables.values():
            all_potential_tables.update(tables)
        
        # Get columns for all potential tables
        table_columns = self._get_all_table_columns(list(all_potential_tables))
        
        matched_columns = []
        columns_in_mentioned_table = {}
        
        for mentioned_column in columns_mentioned:
            best_match = None
            best_score = 0.0
            column_found_in_tables = []
            
            # Search in all potential tables
            for table_name, available_columns in table_columns.items():
                if not available_columns:
                    continue
                    
                match_result = find_closest_match(mentioned_column, available_columns, threshold=0.6)
                
                if match_result['match']:
                    if match_result['score'] > best_score:
                        best_match = match_result['match']
                        best_score = match_result['score']
                    
                    # Track which tables contain this column (or close matches)
                    column_found_in_tables.append((table_name, match_result['match'], match_result['score']))
            
            # Use best match or keep original
            final_column = best_match if best_match and best_score >= 0.6 else mentioned_column
            matched_columns.append(final_column)
            
            # Build table->columns mapping
            for table_name, matched_col, score in column_found_in_tables:
                if score >= 0.6:  # Only include good matches
                    if table_name not in columns_in_mentioned_table:
                        columns_in_mentioned_table[table_name] = []
                    if matched_col not in columns_in_mentioned_table[table_name]:
                        columns_in_mentioned_table[table_name].append(matched_col)
        
        return {
            "matched_columns": matched_columns,
            "columns_in_mentioned_table": columns_in_mentioned_table
        }
    
    def enhance_classification_details(self, classification_details: Dict[str, Any], 
                                     current_segment_id: int) -> Dict[str, Any]:
        """
        Main function to enhance classification details with fuzzy matching
        
        Args:
            classification_details (Dict[str, Any]): Original classification details
            current_segment_id (int): Current segment ID for context
            
        Returns:
            Dict[str, Any]: Enhanced classification details
        """
        try:
            # Make a copy to avoid modifying original
            enhanced_details = classification_details.copy()
            
            # Extract mentioned items
            tables_mentioned = classification_details.get("detected_elements", {}).get("sap_tables_mentioned", [])
            columns_mentioned = classification_details.get("detected_elements", {}).get("columns_Mentioned", [])
            segments_mentioned = classification_details.get("detected_elements", {}).get("segments_mentioned", [])
            
            segment_match_result = self._match_segments(segments_mentioned, current_segment_id)
            
            # 2. Match tables
            table_match_result = self._match_tables(tables_mentioned)
            
            # 3. Match columns (using both matched tables and segment target tables)
            column_match_result = self._match_columns(
                columns_mentioned,
                table_match_result["matched_tables"],
                segment_match_result["segment_target_tables"]
            )
            
            # 4. Create the enhanced structure
            enhanced_matching_info = {
                "tables_mentioned": table_match_result["matched_tables"],
                "columns_mentioned": column_match_result["matched_columns"],
                "columns_in_mentioned_table": column_match_result["columns_in_mentioned_table"],
                "segments_mentioned": segment_match_result["matched_segments"],
                "segment_target_tables": segment_match_result["segment_target_tables"],
                "table_match_confidence": table_match_result["table_match_confidence"]
            }

            enhanced_matching_info["segment glossary"] = segment_match_result["segment_target_tables"]
            # 5. Update the enhanced details
            enhanced_details["enhanced_matching"] = enhanced_matching_info
            
            # 6. Update the detected_elements with matched values
            if "detected_elements" not in enhanced_details:
                enhanced_details["detected_elements"] = {}
            
            enhanced_details["detected_elements"]["sap_tables_mentioned"] = table_match_result["matched_tables"]
            enhanced_details["detected_elements"]["columns_Mentioned"] = column_match_result["matched_columns"]
            enhanced_details["detected_elements"]["segments_mentioned"] = segment_match_result["matched_segments"]
            
            logger.info(f"Enhanced classification details: {enhanced_matching_info}")
            
            return enhanced_details
            
        except Exception as e:
            logger.error(f"Error enhancing classification details: {e}")
            return classification_details

def enhance_classification_before_processing(classification_details: Dict[str, Any], 
                                           current_segment_id: int,
                                           db_path: str = None,
                                           segments_csv_path: str = "segments.csv") -> Dict[str, Any]:
    """
    Convenience function to enhance classification details
    
    Args:
        classification_details (Dict[str, Any]): Original classification details
        current_segment_id (int): Current segment ID
        db_path (str, optional): Database path
        segments_csv_path (str, optional): Segments CSV path
        
    Returns:
        Dict[str, Any]: Enhanced classification details
    """
    enhancer = ClassificationEnhancer(db_path, segments_csv_path)
    return enhancer.enhance_classification_details(classification_details, current_segment_id)

def classify_query_with_llm(query):
    """
    Use LLM to classify the query type based on linguistic patterns and semantic understanding
    
    Parameters:
    query (str): The natural language query
    
    Returns:
    str: Query classification (SIMPLE_TRANSFORMATION, JOIN_OPERATION, etc.)
    dict: Additional details about the classification
    """
    try:
        # Create comprehensive prompt for query classification
        prompt = f"""
You are an expert data transformation analyst. Analyze the following natural language query and classify it into one of these categories:

1. **SIMPLE_TRANSFORMATION**: Basic data operations like filtering, single table operations, field transformations
2. **JOIN_OPERATION**: Operations involving multiple tables that need to be joined together
3. **CROSS_SEGMENT**: Operations that reference previous segments or transformations in a workflow
4. **VALIDATION_OPERATION**: Data validation, checking data quality, ensuring data integrity
5. **AGGREGATION_OPERATION**: Statistical operations like sum, count, average, grouping operations

USER QUERY: "{query}"

CLASSIFICATION CRITERIA:

**JOIN_OPERATION indicators:**
- Words like: "join", "merge", "combine", "link", "from both", "using data from"
- References to connecting data between tables

**CROSS_SEGMENT indicators:**
- References to previous segments: "basic segment", "marc segment", "makt segment"
- Temporal references: "previous", "prior", "last", "earlier" segment/transformation
- Segment keywords: "BASIC", "PLANT", "SALES", "PURCHASING", "CLASSIFICATION", "MRP", "WAREHOUSE"
- Phrases like: "from segment", "use segment", "based on segment", "transformation X"

**VALIDATION_OPERATION indicators:**
- Words like: "validate", "verify", "ensure", "check", "confirm"
- Data quality terms: "missing", "invalid", "correct", "consistent", "duplicate"
- Conditional validation: "if exists", "must be", "should have"

**AGGREGATION_OPERATION indicators:**
- Statistical functions: "count", "sum", "average", "mean", "total", "calculate"
- Grouping operations: "group by", "per", "for each"
- Comparative terms: "minimum", "maximum", "highest", "lowest"

**SIMPLE_TRANSFORMATION indicators:**
- Single table operations
- Basic filtering: "where", "with condition", "having"
- Field transformations: "bring", "get", "extract", "convert"
- Simple data operations without joins or complex logic

ADDITIONAL DETECTION:
- **SAP Tables**: Look for mentions of SAP Tables
- **Segment Names**: Look for: Basic, Plant, Sales, Purchasing, Classification, MRP, Warehouse segments
- **Table Count**: If multiple SAP tables mentioned, likely JOIN_OPERATION
- **Segment References**: Any reference to numbered transformations (Transformation 1, Step 2, etc.)


Respond with a JSON object:
```json
{{
    "primary_classification": "CATEGORY_NAME",
    "confidence": 0.95,
    "reasoning": "Brief explanation of why this classification was chosen",
    "detected_elements": {{
        "sap_tables_mentioned": ["Table1", "Table2"],
        "segments_mentioned": ["Segment_name1", "Segment_name2"],
        "join_indicators": ["merge", "combine"],
        "validation_indicators": [],
        "aggregation_indicators": [],
        "transformation_references": ["previous segment"],
        "has_multiple_tables": true
        "columns_Mentioned": ["Column1", "Column2"]
    }},
    "secondary_possibilities": ["OTHER_CATEGORY"]
}}
```
"""

        # Call Gemini API for classification
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            # Fallback to simple classification
            return _fallback_classification(query)
            
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17", 
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1)  # Low temperature for consistent classification
        )
        
        if not response or not hasattr(response, "text"):
            logger.warning("Invalid response from Gemini API for query classification")
            return _fallback_classification(query)
            
        # Parse JSON response
        try:
            json_str = re.search(r"```json(.*?)```", response.text, re.DOTALL)
            if json_str:
                result = json.loads(json_str.group(1).strip())
            else:
                # Try to parse the whole response as JSON
                result = json.loads(response.text.strip())
            # Extract classification and details
            primary_class = result.get("primary_classification", "SIMPLE_TRANSFORMATION")
            
            # Create detailed response in the expected format
            details = {
                "confidence": result.get("confidence", 0.8),
                "reasoning": result.get("reasoning", "LLM-based classification"),
                "detected_elements": result.get("detected_elements", {}),
                "secondary_possibilities": result.get("secondary_possibilities", []),
                "sap_tables_mentioned": result.get("detected_elements", {}).get("sap_tables_mentioned", []),
                "segments_mentioned": result.get("detected_elements", {}).get("segments_mentioned", []),
                "has_multiple_tables": result.get("detected_elements", {}).get("has_multiple_tables", False),
                "join_indicators": result.get("detected_elements", {}).get("join_indicators", []),
                "validation_indicators": result.get("detected_elements", {}).get("validation_indicators", []),
                "aggregation_indicators": result.get("detected_elements", {}).get("aggregation_indicators", []),
                "transformation_references": result.get("detected_elements", {}).get("transformation_references", []),
                "columns_Mentioned": result.get("detected_elements", {}).get("columns_Mentioned", [])
            }
            with open("classification_response.json", "w") as f:
                json.dump(result, f, indent=4)
            
            return primary_class, details
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing LLM classification response: {e}")
            logger.error(f"Raw response: {response.text}")
            return _fallback_classification(query)
            
    except Exception as e:
        logger.error(f"Error in classify_query_with_llm: {e}")
        return _fallback_classification(query)


def _fallback_classification(query):
    """
    Fallback classification using simple keyword matching when LLM fails
    """
    query_lower = query.lower()
    
    # Simple keyword-based classification
    join_keywords = ["join", "merge", "combine", "link", "both tables", "multiple tables"]
    segment_keywords = ["segment", "basic", "marc", "makt", "previous", "prior", "transformation"]
    validation_keywords = ["validate", "verify", "check", "ensure", "missing", "invalid"]
    aggregation_keywords = ["count", "sum", "average", "total", "group by", "calculate"]
    
    # Check for multiple SAP tables
    sap_tables = ["mara", "marc", "makt", "mvke", "marm", "mlan", "ekko", "ekpo", "vbak", "vbap", "kna1", "lfa1"]
    tables_found = [table for table in sap_tables if table in query_lower]
    
    # Classification logic
    if len(tables_found) > 1 or any(keyword in query_lower for keyword in join_keywords):
        primary_class = "JOIN_OPERATION"
    elif any(keyword in query_lower for keyword in segment_keywords):
        primary_class = "CROSS_SEGMENT"
    elif any(keyword in query_lower for keyword in validation_keywords):
        primary_class = "VALIDATION_OPERATION"
    elif any(keyword in query_lower for keyword in aggregation_keywords):
        primary_class = "AGGREGATION_OPERATION"
    else:
        primary_class = "SIMPLE_TRANSFORMATION"
    
    # Create basic details
    details = {
        "confidence": 0.6,  # Lower confidence for fallback
        "reasoning": "Fallback keyword-based classification",
        "detected_elements": {
            "sap_tables_mentioned": [table.upper() for table in tables_found],
            "segments_mentioned": [],
            "has_multiple_tables": len(tables_found) > 1,
            "join_indicators": [kw for kw in join_keywords if kw in query_lower],
            "validation_indicators": [kw for kw in validation_keywords if kw in query_lower],
            "aggregation_indicators": [kw for kw in aggregation_keywords if kw in query_lower],
            "transformation_references": [kw for kw in segment_keywords if kw in query_lower]
        },
        "secondary_possibilities": []
    }
    
    return primary_class, details



PROMPT_TEMPLATES = {
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
    
    These are the Extracted and processed information from the query, you have to strictly adhere to the this and use reasoning to generate the response
    Do not mention any other table if it's name hasnt been mentioned in the query, and for segments striclty use the segment glossary in the given important context
    Important Query Context:{additional_context}

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

    These are the Extracted and processed information from the query, you have to strictly adhere to the this and use reasoning to generate the response
    Do not mention any other table if it's name hasnt been mentioned in the query, and for segments striclty use the segment glossary in the given important context
    Important Query Context:{additional_context}

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

    These are the Extracted and processed information from the query, you have to strictly adhere to the this and use reasoning to generate the response
    Do not mention any other table if it's name hasnt been mentioned in the query, and for segments striclty use the segment glossary in the given important context
    Important Query Context:{additional_context}

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

    These are the Extracted and processed information from the query, you have to strictly adhere to the this and use reasoning to generate the response
    Do not mention any other table if it's name hasnt been mentioned in the query, and for segments striclty use the segment glossary in the given important context
    Important Query Context:{additional_context}    
    
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

    These are the Extracted and processed information from the query, you have to strictly adhere to the this and use reasoning to generate the response
    Do not mention any other table if it's name hasnt been mentioned in the query, and for segments striclty use the segment glossary in the given important context
    Important Query Context:{additional_context}
    
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
        conn = sqlite3.connect(os.environ.get('DB_PATH'))
        
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
            query_type, classification_details = classify_query_with_llm(query)
            enhanced_classification = enhance_classification_before_processing(
                classification_details, segment_id, db_path=os.environ.get('DB_PATH')
            )
            classification_details = enhanced_classification
        with open("classification_details.json", "w") as f:
            json.dump(classification_details, f, indent=4)
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
        
        formatted_prompt = prompt_template.format(
            question=query,
            table_desc=list(table_desc.itertuples(index=False)),
            target_df_sample=target_df_sample_str,
            segment_mapping=context_manager.get_segments(session_id) if session_id else [],
            additional_context=classification_details
        )
        with open("formatted_prompt.txt", "w") as f:
            f.write(formatted_prompt)
        
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
        with open("parsed_data.json", "w") as f:
            json.dump(parsed_data, f, indent=4)
        
        logger.info(f"Parsed data: {parsed_data}")
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
            segment_info ={
                    "segment_name": segment_name,
                    "target_table_name": target_table_name,
                }
            if segment_info not in segments:
                segments.append()
            
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
                df_processed["target_sap_field"] = df_processed[
                    "target_sap_field"
                ].replace(r"^\s*$", pd.NA, regex=True)
                valid_targets = df_processed["target_sap_field"].notna()
                missing_sources = df_processed["source_field_name"].isna()
                fill_indices = missing_sources & valid_targets

                if fill_indices.any():
                    df_processed.loc[fill_indices, "source_field_name"] = (
                        df_processed.loc[fill_indices, "target_sap_field"]
                    )

        return df_processed
    except Exception as e:
        logger.error(f"Error in missing_values_handling: {e}")
        return df

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
        query_type, classification_details = classify_query_with_llm(query)
        enhanced_classification = enhance_classification_before_processing(
            classification_details, segment_id, db_path=os.environ.get('DB_PATH')
        )
        classification_details = enhanced_classification
        logger.info(f"Query type: {query_type}")
        logger.info(f"Classification details: {classification_details}")

        
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