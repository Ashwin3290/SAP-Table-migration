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
from difflib import SequenceMatcher
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

load_dotenv()

from DMtool.executor import SQLExecutor

sql_executor = SQLExecutor()

try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:

        print("Downloading spaCy model...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

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
    
    clean_query = re.sub(r'\s+', ' ', query.strip().lower())
    
    matches = []
    for word in word_list:
        clean_word = word.lower()
        
        if clean_query == clean_word:
            final_score = 1.0
        else:
            
            char_similarity = SequenceMatcher(None, clean_query, clean_word).ratio()
            
            substring_score = 0
            if clean_query in clean_word:

                substring_score = len(clean_query) / len(clean_word)
            elif clean_word in clean_query:
                substring_score = len(clean_word) / len(clean_query)
            
            query_words = clean_query.split()
            word_words = clean_word.split()
            
            word_level_score = 0
            if len(query_words) == 1 and len(word_words) == 1:
                word_level_score = char_similarity
            else:
                exact_word_matches = 0
                for q_word in query_words:
                    if q_word in word_words:
                        exact_word_matches += 1
                
                fuzzy_matches = 0
                total_comparisons = max(len(query_words), len(word_words))
                
                for q_word in query_words:
                    best_match_score = 0
                    for w_word in word_words:
                        if q_word == w_word:
                            best_match_score = 1.0
                            break
                        else:
                            word_sim = SequenceMatcher(None, q_word, w_word).ratio()
                            best_match_score = max(best_match_score, word_sim)
                    fuzzy_matches += best_match_score
                
                word_level_score = fuzzy_matches / total_comparisons if total_comparisons > 0 else 0
            
            prefix_score = 0
            min_len = min(len(clean_query), len(clean_word))
            if min_len >= 3:
                common_prefix_len = 0
                for i in range(min_len):
                    if clean_query[i] == clean_word[i]:
                        common_prefix_len += 1
                    else:
                        break
                if common_prefix_len >= 3:
                    prefix_score = common_prefix_len / max(len(clean_query), len(clean_word))
            

            case_insensitive_exact = query.strip() == word.strip()
            case_bonus = 0.1 if case_insensitive_exact and not (clean_query == clean_word) else 0
            
            # Calculate final score with weighted components
            if len(query_words) == 1 and len(word_words) == 1:
                final_score = (
                    char_similarity * 0.7 +      # Heavy weight on character similarity
                    substring_score * 0.2 +       # Some weight on substring matches
                    prefix_score * 0.1 +          # Small weight on prefix matches
                    case_bonus                    # Bonus for case-only differences
                )
            else:
                final_score = (
                    word_level_score * 0.5 +      # Word-level matching
                    char_similarity * 0.3 +       # Character-level similarity
                    substring_score * 0.15 +      # Substring matches
                    prefix_score * 0.05 +         # Prefix matches
                    case_bonus                    # Case bonus
                )
            
            final_score = min(final_score, 1.0)
        
        if final_score >= threshold:
            matches.append({
                'word': word,
                'score': final_score,
                'original_word': word
            })
    
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    result = {
        'match': matches[0]['word'] if matches else None,
        'score': matches[0]['score'] if matches else 0.0,
        'all_matches': [(m['word'], round(m['score'], 3)) for m in matches]
    }
    
    return result

def check_column_exists_in_table(column_name: str, table_name: str, db_path: str = None) -> bool:
    """
    Check if a specific column exists in a specific table
    """
    try:
        if not db_path:
            db_path = os.environ.get('DB_PATH')
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()
        
        return column_name in columns
        
    except Exception as e:
        logger.error(f"Error checking column {column_name} in table {table_name}: {e}")
        return False

def generate_table_specific_column_mapping(segment_glossary: Dict[str, List[str]], 
                                         key_mapping: List[Dict[str, str]], 
                                         source_tables_from_classification: List[str],
                                         db_path: str = None) -> Dict[str, Dict[str, str]]:
    """
    Generate table-specific column mapping for both target tables used as source and regular source tables
    """
    table_column_instructions = {}
    
    # Get all target tables from segment glossary
    target_tables_used_as_source = []
    for segment_name, tables in segment_glossary.items():
        target_tables_used_as_source.extend(tables)
    
    # For target tables used as source - check which target_col from key_mapping exists
    for table in target_tables_used_as_source:
        table_column_instructions[table] = {
            "table_type": "target_table_used_as_source",
            "segment_name": None,
            "column_mappings": {}
        }
        
        # Find which segment this table belongs to
        for segment_name, tables in segment_glossary.items():
            if table in tables:
                table_column_instructions[table]["segment_name"] = segment_name
                break
        
        # Check key mappings for this table
        for mapping in key_mapping:
            if isinstance(mapping, dict):
                source_col = mapping.get('source_col')
                target_col = mapping.get('target_col')
                
                if source_col and target_col:
                    # Check if target_col exists in this target table
                    if check_column_exists_in_table(target_col, table, db_path):
                        table_column_instructions[table]["column_mappings"][source_col] = target_col
                        logger.info(f"Table {table}: Use '{target_col}' instead of '{source_col}'")
    
    # For regular source tables - check which source columns actually exist
    for table in source_tables_from_classification:
        if table not in target_tables_used_as_source:  # Don't duplicate target tables
            table_column_instructions[table] = {
                "table_type": "regular_source_table",
                "column_mappings": {}
            }
            
            # For regular source tables, verify the original column names exist
            for mapping in key_mapping:
                if isinstance(mapping, dict):
                    source_col = mapping.get('source_col')
                    
                    if source_col:
                        # Check if source_col exists in this regular source table
                        if check_column_exists_in_table(source_col, table, db_path):
                            table_column_instructions[table]["column_mappings"][source_col] = source_col
                            logger.info(f"Table {table}: Use original column '{source_col}'")
    
    return table_column_instructions

def create_enhanced_segment_column_prompt(segment_glossary: Dict[str, List[str]], 
                                        key_mapping: List[Dict[str, str]], 
                                        source_tables_from_classification: List[str],
                                        query_type: str) -> str:
    """
    Create enhanced prompt for segment column mapping
    """
    if query_type not in ["CROSS_SEGMENT", "JOIN_OPERATION"]:
        return ""
    
    if not segment_glossary and not key_mapping:
        return ""
    
    # Generate table-specific column mappings
    table_instructions = generate_table_specific_column_mapping(
        segment_glossary, key_mapping, source_tables_from_classification
    )
    
    if not table_instructions:
        return ""
    
    prompt = """
CRITICAL COLUMN NAMING INSTRUCTIONS:

"""
    
    # Instructions for target tables used as source
    target_table_instructions = []
    for table, info in table_instructions.items():
        if info["table_type"] == "target_table_used_as_source":
            if info["column_mappings"]:
                segment_info = f" (from {info['segment_name']} segment)" if info["segment_name"] else ""
                target_table_instructions.append(f"""
For table '{table}'{segment_info}:
  - This is a PREVIOUSLY POPULATED TARGET TABLE now being used as SOURCE
  - Column name changes:""")
                
                for original_name, current_name in info["column_mappings"].items():
                    target_table_instructions.append(f"    * DO NOT use '{original_name}' → USE '{current_name}' instead")
    
    if target_table_instructions:
        prompt += "TARGET TABLES USED AS SOURCE:\n"
        prompt += "\n".join(target_table_instructions)
        prompt += "\n"
    
    # Instructions for regular source tables
    source_table_instructions = []
    for table, info in table_instructions.items():
        if info["table_type"] == "regular_source_table":
            if info["column_mappings"]:
                source_table_instructions.append(f"""
For table '{table}':
  - This is a REGULAR SOURCE TABLE
  - Use these verified column names:""")
                
                for col_name, verified_name in info["column_mappings"].items():
                    source_table_instructions.append(f"    * Use '{verified_name}'")
    
    if source_table_instructions:
        prompt += "\nREGULAR SOURCE TABLES:\n"
        prompt += "\n".join(source_table_instructions)
        prompt += "\n"
    
    prompt += """
IMPORTANT REMINDERS:
1. Target tables (t_[number]) have DIFFERENT column names than original source tables
2. When joining target tables with regular source tables, use the CORRECT column names for each table
3. Example: If joining t_24_Product_Basic_Data_mandatory_Ext with MARC, use:
   - t_24_Product_Basic_Data_mandatory_Ext.PRODUCT (not MATNR)
   - MARC.MATNR (not PRODUCT)
4. Always verify column names match the actual table schema

"""
    
    return prompt

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
            

            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            
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
            

            cursor.execute("""
                SELECT table_name 
                FROM connection_segments 
                WHERE segment_id = ?
            """, (segment_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                table_name = result[0]

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


            available_segments = segments_df['segement_name'].unique().tolist()
            current_pattern = self._get_current_target_table_pattern(current_segment_id)

            matched_segments = []
            segment_target_tables = {}

            for mentioned_segment in segments_mentioned:

                match_result = find_closest_match(mentioned_segment, available_segments, threshold=0.3)

                if match_result['match'] and match_result['score'] >= 0.3:
                    matched_segment = match_result['match']
                    matched_segments.append(matched_segment)


                    segment_rows = segments_df[segments_df['segement_name'] == matched_segment]
                    target_tables = segment_rows['table_name'].tolist()


                    if len(target_tables) > 1 and current_pattern:
                        pattern_matches = [table for table in target_tables if table.lower().startswith(current_pattern)]
                        if pattern_matches:
                            target_tables = pattern_matches

                    segment_target_tables[matched_segment] = target_tables
                else:

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
            match_result = find_closest_match(mentioned_table, available_tables, threshold=0.4)
            
            if match_result['match']:
                matched_table = match_result['match']
                confidence = match_result['score']
                

                if confidence >= 0.9 or len(match_result['all_matches']) == 1:

                    matched_tables.append(matched_table)
                    table_match_confidence[matched_table] = confidence
                elif len(match_result['all_matches']) > 1:

                    best_match = match_result['all_matches'][0]
                    second_best = match_result['all_matches'][1] if len(match_result['all_matches']) > 1 else None
                    
                    if second_best is None or (best_match[1] - second_best[1]) > 0.2:

                        matched_tables.append(best_match[0])
                        table_match_confidence[best_match[0]] = best_match[1]
                    else:

                        matched_tables.append(mentioned_table)
                        table_match_confidence[mentioned_table] = 0.5
                else:
                    matched_tables.append(matched_table)
                    table_match_confidence[matched_table] = confidence
            else:

                matched_tables.append(mentioned_table)
                table_match_confidence[mentioned_table] = 0.0
        
        return {
            "matched_tables": matched_tables,
            "table_match_confidence": table_match_confidence
        }
    
    def _match_columns(self, columns_mentioned: List[str], matched_tables: List[str], 
                    segment_target_tables: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Enhanced column matching with better fuzzy matching and glossary integration
        
        Args:
            columns_mentioned: List of column names mentioned in query
            matched_tables: List of matched table names
            segment_target_tables: Segment target tables dictionary
            
        Returns:
            Dictionary with detailed column matching results
        """
        if not columns_mentioned:
            return {
                "matched_columns": [],
                "columns_in_mentioned_table": {},
                "column_validation": {
                    "columns_found_in_tables": {},
                    "missing_columns": [],
                    "table_columns": {}
                }
            }
        
        # Get all potential tables (SAP tables + segment tables)
        all_potential_tables = set(matched_tables)
        for tables in segment_target_tables.values():
            all_potential_tables.update(tables)
        
        # Get all columns for these tables
        table_columns = self._get_all_table_columns(list(all_potential_tables))
        
        # Prepare results structure
        matched_columns = []
        columns_in_mentioned_table = {}
        column_validation = {
            "columns_found_in_tables": {},
            "missing_columns": [],
            "table_columns": table_columns
        }
        
        # First pass - exact matching
        for mentioned_column in columns_mentioned:
            found_in_tables = []
            
            for table_name, available_columns in table_columns.items():
                if mentioned_column in available_columns:
                    found_in_tables.append({
                        "table": table_name,
                        "actual_column_name": mentioned_column,
                        "match_type": "exact",
                        "confidence": 1.0
                    })
                    
            if found_in_tables:
                column_validation["columns_found_in_tables"][mentioned_column] = found_in_tables
                matched_columns.append(mentioned_column)
                
                # Track which tables contain this column
                for table_info in found_in_tables:
                    if table_info["table"] not in columns_in_mentioned_table:
                        columns_in_mentioned_table[table_info["table"]] = []
                    if mentioned_column not in columns_in_mentioned_table[table_info["table"]]:
                        columns_in_mentioned_table[table_info["table"]].append(mentioned_column)
        
        # Second pass - fuzzy matching for unmatched columns
        for mentioned_column in columns_mentioned:
            if mentioned_column in column_validation["columns_found_in_tables"]:
                continue
                
            best_match = None
            best_score = 0.0
            column_found_in_tables = []
            
            for table_name, available_columns in table_columns.items():
                if not available_columns:
                    continue
                    
                # Enhanced fuzzy matching with better thresholding
                match_result = find_closest_match(mentioned_column, available_columns, threshold=0.5)
                
                if match_result['match']:
                    # Only consider if score is significantly better than current best
                    if match_result['score'] > best_score + 0.1:
                        best_match = match_result['match']
                        best_score = match_result['score']
                    
                    # Track all potential matches for analysis
                    column_found_in_tables.append({
                        "table": table_name,
                        "original_column": mentioned_column,
                        "matched_column": match_result['match'],
                        "score": match_result['score'],
                        "match_type": "fuzzy"
                    })
            
            # Only accept fuzzy match if confidence is high enough
            if best_match and best_score >= 0.7:
                matched_columns.append(best_match)
                column_validation["columns_found_in_tables"][mentioned_column] = [{
                    "table": table_name,
                    "actual_column_name": best_match,
                    "match_type": "fuzzy",
                    "confidence": best_score
                } for table_name, match_info in 
                [(t["table"], t) for t in column_found_in_tables 
                    if t["matched_column"] == best_match]]
                
                # Track which tables contain this matched column
                for match_info in column_found_in_tables:
                    if match_info["matched_column"] == best_match:
                        if match_info["table"] not in columns_in_mentioned_table:
                            columns_in_mentioned_table[match_info["table"]] = []
                        if best_match not in columns_in_mentioned_table[match_info["table"]]:
                            columns_in_mentioned_table[match_info["table"]].append(best_match)
            else:
                column_validation["missing_columns"].append(mentioned_column)
        
        return {
            "matched_columns": matched_columns,
            "columns_in_mentioned_table": columns_in_mentioned_table,
            "column_validation": column_validation
        }
        
    def _match_columns_with_glossary(self, columns_mentioned: List[str], 
                                    joined_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced column matching that includes glossary term matching from joined_df
        
        Args:
            columns_mentioned (List[str]): List of column names mentioned in query
            joined_df (pd.DataFrame): DataFrame containing description and target_sap_field mappings
            
        Returns:
            Dict[str, Any]: Enhanced column matching results with glossary mappings
        """
        try:
            if joined_df.empty or not columns_mentioned:
                return {"glossary_matches": {}, "column_hints": {}}
            
            # Get available actual column names from database tables
            available_tables = self._get_available_tables()
            all_db_columns = {}
            for table in available_tables:
                table_columns = self._get_table_columns(table)
                all_db_columns[table] = table_columns
            
            # Extract glossary mappings from joined_df
            glossary_mappings = {}
            if 'description' in joined_df.columns and 'target_sap_field' in joined_df.columns:
                for _, row in joined_df.iterrows():
                    description = str(row.get('description', '')).strip()
                    target_field = str(row.get('target_sap_field', '')).strip()
                    if description and target_field and description != 'nan' and target_field != 'nan':
                        glossary_mappings[description.lower()] = target_field
            
            glossary_matches = {}
            column_hints = {}
            
            for mentioned_column in columns_mentioned:
                mentioned_lower = mentioned_column.lower().strip()
                
                # First, check if this column exists directly in any database table
                found_in_db = False
                for table, columns in all_db_columns.items():
                    if mentioned_column in columns:
                        found_in_db = True
                        break
                
                # If not found in DB, try glossary matching
                if not found_in_db:
                    # Direct glossary lookup first
                    if mentioned_lower in glossary_mappings:
                        actual_column = glossary_mappings[mentioned_lower]
                        glossary_matches[mentioned_column] = {
                            'actual_column': actual_column,
                            'match_type': 'direct_glossary',
                            'confidence': 1.0
                        }
                    else:
                        # Fuzzy matching against glossary terms
                        glossary_terms = list(glossary_mappings.keys())
                        fuzzy_result = find_closest_match(mentioned_lower, glossary_terms, threshold=0.4)
                        
                        if fuzzy_result['match']:
                            matched_glossary = fuzzy_result['match']
                            actual_column = glossary_mappings[matched_glossary]
                            confidence = fuzzy_result['score']
                            
                            glossary_matches[mentioned_column] = {
                                'actual_column': actual_column,
                                'matched_glossary_term': matched_glossary,
                                'match_type': 'fuzzy_glossary',
                                'confidence': confidence
                            }
                            
                            # Add as hint for the LLM
                            column_hints[mentioned_column] = f"User likely meant '{actual_column}' (matched via glossary term '{matched_glossary}' with {confidence:.2%} confidence)"
            
            return {
                "glossary_matches": glossary_matches,
                "column_hints": column_hints,
                "total_glossary_mappings": len(glossary_mappings)
            }
            
        except Exception as e:
            logger.error(f"Error in _match_columns_with_glossary: {e}")
            return {"glossary_matches": {}, "column_hints": {}}

    def _enhance_column_matching_with_glossary(self, original_column_result: Dict[str, Any],
                                            columns_mentioned: List[str],
                                            joined_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhance existing column matching results with glossary term matching
        
        Args:
            original_column_result (Dict): Original column matching results
            columns_mentioned (List[str]): Columns mentioned in query
            joined_df (pd.DataFrame): DataFrame with glossary mappings
            
        Returns:
            Dict[str, Any]: Enhanced column matching results
        """
        try:
            # Get glossary matching results
            glossary_results = self._match_columns_with_glossary(columns_mentioned, joined_df)
            
            # Enhance the original results
            enhanced_result = original_column_result.copy()
            enhanced_result["glossary_column_matching"] = glossary_results
            
            # Add glossary hints to the existing matched columns
            if "matched_columns" in enhanced_result:
                enhanced_matched_columns = enhanced_result["matched_columns"].copy()
                
                for mentioned_col, glossary_info in glossary_results["glossary_matches"].items():
                    if mentioned_col not in enhanced_matched_columns:
                        # Add the actual column name from glossary
                        enhanced_matched_columns.append(glossary_info["actual_column"])
                
                enhanced_result["matched_columns"] = enhanced_matched_columns
            
            # Enhance columns_in_mentioned_table with glossary mappings
            if "columns_in_mentioned_table" not in enhanced_result:
                enhanced_result["columns_in_mentioned_table"] = {}
                
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in _enhance_column_matching_with_glossary: {e}")
            return original_column_result

    def map_columns_to_tables(self, tables: List[str], columns: List[str], 
                            key_mapping: List[Dict[str, str]] = None,
                            db_path: str = None) -> Dict[str, Dict[str, List[str]]]:
        """
        Enhanced column-to-table mapping with key mapping integration
        
        Args:
            tables: List of table names to check
            columns: List of column names to map
            key_mapping: Optional list of key mappings (source_col -> target_col)
            db_path: Optional database path
            
        Returns:
            Dictionary with two keys:
            - 'table_to_columns': Dict mapping each table to all its columns
            - 'column_to_tables': Dict mapping each column to tables where it exists
            - 'key_mapping_applied': Dict showing how key mappings were applied
        """
        if not db_path:
            db_path = self.db_path or os.environ.get('DB_PATH')
        
        result = {
            'table_to_columns': {},
            'column_to_tables': {},
            'key_mapping_applied': {}
        }
        
        # Initialize column_to_tables with empty lists
        for column in columns:
            result['column_to_tables'][column] = []
        
        # Add key mapping columns to our search
        extended_columns = set(columns)
        if key_mapping:
            for mapping in key_mapping:
                if isinstance(mapping, dict):
                    if 'source_col' in mapping:
                        extended_columns.add(mapping['source_col'])
                    if 'target_col' in mapping:
                        extended_columns.add(mapping['target_col'])
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # For each table, get all its columns
            for table in tables:
                try:
                    # Get table schema
                    cursor.execute(f"PRAGMA table_info({table})")
                    table_columns = [row[1] for row in cursor.fetchall()]
                    
                    # Store all columns for this table
                    result['table_to_columns'][table] = table_columns
                    
                    # Check which of our target columns exist in this table
                    for column in extended_columns:
                        # Check direct match
                        if column in table_columns:
                            result['column_to_tables'][column].append(table)
                            continue
                        
                        # Check key mapping variants
                        if key_mapping:
                            for mapping in key_mapping:
                                if isinstance(mapping, dict):
                                    # If this column matches a source_col, check if target_col exists
                                    if column == mapping.get('source_col') and mapping.get('target_col') in table_columns:
                                        result['column_to_tables'][column].append(table)
                                        result['key_mapping_applied'][f"{column}->{mapping['target_col']}"] = table
                                        break
                                    # If this column matches a target_col, check if source_col exists
                                    elif column == mapping.get('target_col') and mapping.get('source_col') in table_columns:
                                        result['column_to_tables'][column].append(table)
                                        result['key_mapping_applied'][f"{mapping['source_col']}->{column}"] = table
                                        break
                                    
                    # Also check for case-insensitive matches
                    lower_columns = [c.lower() for c in table_columns]
                    for column in extended_columns:
                        if column.lower() in lower_columns:
                            actual_col = table_columns[lower_columns.index(column.lower())]
                            if table not in result['column_to_tables'][column]:
                                result['column_to_tables'][column].append(table)
                                result['key_mapping_applied'][f"{column}(case_insensitive)->{actual_col}"] = table
                                
                except Exception as e:
                    logger.warning(f"Error getting columns for table {table}: {e}")
                    result['table_to_columns'][table] = []
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error in map_columns_to_tables: {e}")
        
        return result
    
    def enhance_classification_details(self, classification_details: Dict[str, Any], 
                                    current_segment_id: int, joined_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Main function to enhance classification details with fuzzy matching
        """
        try:
            enhanced_details = classification_details.copy()
            
            # Extract elements from classification
            tables_mentioned = classification_details.get("detected_elements", {}).get("sap_tables_mentioned", [])
            columns_mentioned = classification_details.get("detected_elements", {}).get("columns_Mentioned", [])
            segments_mentioned = classification_details.get("detected_elements", {}).get("segments_mentioned", [])
            
            # Get key mappings from context if available
            context_manager = ContextualSessionManager()
            session_id = classification_details.get("session_id")
            key_mapping = context_manager.get_key_mapping(session_id) if session_id else []
            
            # Match segments and get their target tables
            segment_match_result = self._match_segments(segments_mentioned, current_segment_id)
            
            # Match tables
            table_match_result = self._match_tables(tables_mentioned)
            
            # Combine all tables (SAP tables + segment tables)
            all_tables = set(table_match_result["matched_tables"])
            for tables in segment_match_result["segment_target_tables"].values():
                all_tables.update(tables)
            
            # Enhanced column matching with all tables
            column_match_result = self._match_columns(
                columns_mentioned,
                list(all_tables),
                segment_match_result["segment_target_tables"]
            )
            
            # Create comprehensive column-to-table mapping
            column_table_mapping = self.map_columns_to_tables(
                list(all_tables),
                columns_mentioned,
                key_mapping
            )
            
            # Enhance with glossary if available
            if joined_df is not None and not joined_df.empty:
                column_match_result = self._enhance_column_matching_with_glossary(
                    column_match_result,
                    columns_mentioned,
                    joined_df
                )
            
            # Build enhanced matching info
            enhanced_matching_info = {
                "tables_mentioned": table_match_result["matched_tables"],
                "columns_mentioned": column_match_result["matched_columns"],
                "columns_in_mentioned_table": column_match_result["columns_in_mentioned_table"],
                "segments_mentioned": segment_match_result["matched_segments"],
                "segment_target_tables": segment_match_result["segment_target_tables"],
                "table_match_confidence": table_match_result["table_match_confidence"],
                "column_glossary_matching": column_match_result.get("glossary_column_matching", {}),
                "column_hints": column_match_result.get("glossary_column_matching", {}).get("column_hints", {}),
                "column_table_mapping": column_table_mapping,
                "column_validation": column_match_result.get("column_validation", {})
            }
            
            enhanced_details["enhanced_matching"] = enhanced_matching_info
            
            # Update detected elements with validated information
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
                                        segments_csv_path: str = "segments.csv",
                                        joined_df : pd.DataFrame = None) -> Dict[str, Any]:
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
    return enhancer.enhance_classification_details(classification_details, current_segment_id,joined_df)

def classify_query_with_llm(query, target_table, table_desc):
    """
    Use LLM to classify the query type based on linguistic patterns and semantic understanding
    
    Parameters:
    query (str): The natural language query
    
    Returns:
    str: Query classification (SIMPLE_TRANSFORMATION, JOIN_OPERATION, etc.)
    dict: Additional details about the classification
    """
    try:

        prompt = f"""
You are an expert data transformation analyst. Analyze the following natural language query and classify it into one of these categories:

1. **SIMPLE_TRANSFORMATION**: Basic data operations like filtering, single table operations, field transformations
2. **JOIN_OPERATION**: Operations involving multiple tables that need to be joined together
3. **CROSS_SEGMENT**: Operations that reference previous segments or transformations in a workflow
4. **VALIDATION_OPERATION**: Data validation, checking data quality, ensuring data integrity
5. **AGGREGATION_OPERATION**: Statistical operations like sum, count, average, grouping operations

USER QUERY: "{query}"

Note:
if the query says Target table then it indicates {target_table}

CLASSIFICATION CRITERIA:

**JOIN_OPERATION indicators:**
- Words like: "join", "merge", "combine", "link", "from both", "using data from"
- References to connecting data between tables

**CROSS_SEGMENT indicators:**
- References to previous segments: "basic segment", "marc segment", "makt segment"
- Segment keywords: "BASIC", "PLANT", "SALES", "PURCHASING", "CLASSIFICATION", "WAREHOUSE"
- Phrases like: "from segment", "use segment", "based on segment"

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

Following is a Target Table glossary: {table_desc}

Note:
- if a query mentions a previous transformation that means that we need find that transformation in the context history and use that target table
it does not influence the classification but it is important to understand the context

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

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")

            return _fallback_classification(query)
            
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
            config=types.GenerateContentConfig(
            temperature=0.05,
            top_p=0.8,
            top_k=20
        ))
        
        if not response or not hasattr(response, "text"):
            logger.warning("Invalid response from Gemini API for query classification")
            return _fallback_classification(query)
            

        try:
            json_str = re.search(r"```json(.*?)```", response.text, re.DOTALL)
            if json_str:
                result = json.loads(json_str.group(1).strip())
            else:

                result = json.loads(response.text.strip())

            primary_class = result.get("primary_classification", "SIMPLE_TRANSFORMATION")
            

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
    

    join_keywords = ["join", "merge", "combine", "link", "both tables", "multiple tables"]
    segment_keywords = ["segment", "basic", "marc", "makt", "previous", "prior", "transformation"]
    validation_keywords = ["validate", "verify", "check", "ensure", "missing", "invalid"]
    aggregation_keywords = ["count", "sum", "average", "total", "group by", "calculate"]
    

    sap_tables = ["mara", "marc", "makt", "mvke", "marm", "mlan", "ekko", "ekpo", "vbak", "vbap", "kna1", "lfa1"]
    tables_found = [table for table in sap_tables if table in query_lower]
    

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
    

    details = {
        "confidence": 0.6,
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
    
    Target Table glossary: {table_desc}

    
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
    
    Target Table glossary: {table_desc}
    
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
    
    Target Table glossary: {table_desc}
    
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
    
    Target Table glossary: {table_desc}
    
    
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
    
    Target Table glossary: {table_desc}
    
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

def process_query_by_type(object_id, segment_id, project_id, query, session_id=None, target_sap_fields=None):
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

        context_manager = ContextualSessionManager()
        

        previous_context = context_manager.get_context(session_id) if session_id else None
        visited_segments = previous_context.get("segments_visited", {}) if previous_context else {}
        

        conn = sqlite3.connect(os.environ.get('DB_PATH'))
        

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT segement_name FROM connection_segments WHERE segment_id = ?", (segment_id,))
            segment_result = cursor.fetchone()
            segment_name = segment_result[0] if segment_result else f"segment_{segment_id}"
            
            context_manager.track_segment(session_id, segment_id, segment_name, conn)
        except Exception as e:
            logger.warning(f"Error tracking segment: {e}")
        

        joined_df = fetch_data_by_ids(object_id, segment_id, project_id, conn)
        

        joined_df = missing_values_handling(joined_df)
        

        target_df_sample = None
        try:

            target_table = joined_df["table_name"].unique().tolist()
            if target_table and len(target_table) > 0:

                target_df_sample = sql_executor.get_table_sample(target_table[0])
                

                if isinstance(target_df_sample, dict) and "error_type" in target_df_sample:
                    logger.warning(f"SQL-based target data sample retrieval failed, using fallback")

                    target_df = get_or_create_session_target_df(
                        session_id, target_table[0], conn
                    )
                    target_df_sample = (
                        target_df.head(5).to_dict("records")
                        if not target_df.empty
                        else []
                    )
                else:

                    target_df_sample = target_df_sample.head(5).to_dict("records") if not target_df_sample.empty else []
        except Exception as e:
            logger.warning(f"Error getting target data sample: {e}")
            target_df_sample = []
            

        query_type, classification_details = classify_query_with_llm(query,target_table,list(joined_df.itertuples(index=False)))
        enhanced_classification = enhance_classification_before_processing(
            classification_details, segment_id, db_path=os.environ.get('DB_PATH'),joined_df=joined_df
        )
        classification_details = enhanced_classification
        logger.info(f"Query type: {query_type}")
        logger.info(f"Classification details: {classification_details}")


        prompt_template = PROMPT_TEMPLATES.get(query_type, PROMPT_TEMPLATES["SIMPLE_TRANSFORMATION"])
        

        target_df_sample_str = "No current target data available"
        if target_df_sample:
            try:
                target_df_sample_str = json.dumps(target_df_sample, indent=2)
            except Exception as e:
                logger.warning(f"Error formatting target data sample: {e}")
        
        segment_prompt = ""
        key_mapping = context_manager.get_key_mapping(session_id) if session_id else []
        if query_type in ["CROSS_SEGMENT", "JOIN_OPERATION"]:
            segment_glossary = enhanced_classification.get('enhanced_matching', {}).get('segment_target_tables', {})
            source_tables_from_classification = enhanced_classification.get('enhanced_matching', {}).get('tables_mentioned', [])
            
            segment_prompt = create_enhanced_segment_column_prompt(
                segment_glossary, 
                key_mapping, 
                source_tables_from_classification,
                query_type
            )
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
                

        table_desc = joined_df
        additional_context_prompt = f"""
COMPREHENSIVE QUERY ANALYSIS:

CLASSIFICATION RESULTS:
- Query Type: {classification_details.get('primary_classification', 'Unknown')}
- Confidence: {classification_details.get('confidence', 0)}
- Reasoning: {classification_details.get('reasoning', 'Not provided')}

DETECTED ELEMENTS:
- SAP Tables Mentioned: {', '.join(classification_details.get('detected_elements', {}).get('sap_tables_mentioned', [])) or 'None'}
- Columns Mentioned: {', '.join(classification_details.get('detected_elements', {}).get('columns_Mentioned', [])) or 'None'}
- Segments Referenced: {', '.join(classification_details.get('detected_elements', {}).get('segments_mentioned', [])) or 'None'}
- Join Indicators: {', '.join(classification_details.get('detected_elements', {}).get('join_indicators', [])) or 'None'}
- Validation Indicators: {', '.join(classification_details.get('detected_elements', {}).get('validation_indicators', [])) or 'None'}
- Aggregation Indicators: {', '.join(classification_details.get('detected_elements', {}).get('aggregation_indicators', [])) or 'None'}
- Transformation References: {', '.join(classification_details.get('detected_elements', {}).get('transformation_references', [])) or 'None'}
- Has Multiple Tables: {classification_details.get('detected_elements', {}).get('has_multiple_tables', False)}

TABLE VALIDATION:
- Valid Tables: {', '.join(classification_details.get('enhanced_matching', {}).get('table_validation', {}).get('valid_tables', {}).keys()) or 'None found'}
- Invalid Tables: {', '.join([t.get('original_name', '') for t in classification_details.get('enhanced_matching', {}).get('table_validation', {}).get('invalid_tables', [])]) or 'None'}
- Schema Errors: {len(classification_details.get('enhanced_matching', {}).get('table_validation', {}).get('schema_errors', []))} errors

COLUMN VALIDATION:
- Columns Found in Database: {', '.join(classification_details.get('enhanced_matching', {}).get('column_validation', {}).get('columns_found_in_tables', {}).keys()) or 'None'}
- Missing Columns: {', '.join(classification_details.get('enhanced_matching', {}).get('column_validation', {}).get('missing_columns', [])) or 'None'}
- Total Tables Scanned: {len(classification_details.get('enhanced_matching', {}).get('column_validation', {}).get('table_columns', {}))}

DETAILED COLUMN-TABLE MAPPING:
{chr(10).join([f"- Column '{col}': found in {len(tables)} table(s) → {', '.join([t.get('table', '') + '(' + t.get('actual_column_name', '') + ')' for t in tables])}" for col, tables in classification_details.get('enhanced_matching', {}).get('column_validation', {}).get('columns_found_in_tables', {}).items()]) or '- No valid column mappings found'}

TABLE COLUMNS AVAILABLE:
{chr(10).join([f"- {table}: {', '.join(cols[:5])}{' ...' if len(cols) > 5 else ''} ({len(cols)} total)" for table, cols in classification_details.get('enhanced_matching', {}).get('column_validation', {}).get('table_columns', {}).items()]) or '- No table schemas available'}

SEGMENT MAPPINGS:
{chr(10).join([f"- Segment '{seg}' → Tables: {', '.join(tables)}" for seg, tables in classification_details.get('enhanced_matching', {}).get('segment_target_tables', {}).items()]) or '- No segment mappings available'}

FUZZY MATCHING RESULTS:
- Tables with Confidence: {', '.join([f"{table}({conf:.0%})" for table, conf in classification_details.get('enhanced_matching', {}).get('table_match_confidence', {}).items()]) or 'None'}
- Columns in Mentioned Tables: {', '.join([f"{table}:[{', '.join(cols)}]" for table, cols in classification_details.get('enhanced_matching', {}).get('columns_in_mentioned_table', {}).items()]) or 'None'}

Column Glossary Matching:
- Columns Matched: {classification_details.get('enhanced_matching', {}).get('column_glossary_matching', {}).get('matched_columns', []) or 'None'}
- Column Hints: {classification_details.get('enhanced_matching', {}).get('column_glossary_matching', {}).get('column_hints', {}) or 'None'}

Key Mapping:
- Key Mappings: {key_mapping if classification_details.get('key_mapping') else 'No key mappings available'}

{"""SEGMENT AND COLUMN MAPPING DETAILS:
{segment_prompt}
VERIFICATION RESULTS:
- Verified column mappings have been checked against actual database schemas
- Use the EXACT column names specified above for each table
- Target tables used as source have different column names than original source tables""" if segment_prompt else ""}

INSTRUCTIONS: Use ONLY the validated table and column names from the mappings above. If a column is missing or a table is invalid, do not include it in your SQL generation.
Do not invent new tables or columns that are not present in the provided mappings.
Use the provided segment mappings to identify the correct tables for each segment mentioned in the query.
"""
        
        formatted_prompt = prompt_template.format(
            question=query,
            table_desc=list(table_desc.itertuples(index=False)),
            target_df_sample=target_df_sample_str,
            segment_mapping=context_manager.get_segments(session_id) if session_id else [],
            additional_context=additional_context_prompt
        )
        
        logger.info(f"Formatted prompt for Gemini API: {formatted_prompt}")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise APIError("Gemini API key not configured")
            
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                temperature=0.5, top_p=0.95, top_k=30
            ),
        )
        

        json_str = re.search(r"```json(.*?)```", response.text, re.DOTALL)
        if json_str:
            parsed_data = json.loads(json_str.group(1).strip())
        else:

            parsed_data = json.loads(response.text.strip())
        
        logger.info(f"Parsed data: {parsed_data}")
        parsed_data["query_type"] = query_type
        

        parsed_data["target_table_name"] = joined_df["table_name"].unique().tolist()
        parsed_data["key_mapping"] = context_manager.get_key_mapping(session_id) if session_id else []
        parsed_data["visited_segments"] = visited_segments
        parsed_data["session_id"] = session_id
        

        parsed_data["classification_details"] = classification_details
        if target_sap_fields is not None:
            if isinstance(target_sap_fields, list):
                parsed_data["target_sap_fields"] = target_sap_fields
            else:
                parsed_data["target_sap_fields"] = [target_sap_fields]
                

        schema_info = {}
        for table_name in parsed_data.get("source_table_name", []):
            try:
                table_schema = sql_executor.get_table_schema(table_name)
                if isinstance(table_schema, list):
                    schema_info[table_name] = table_schema
            except Exception as e:
                logger.warning(f"Error fetching schema for {table_name}: {e}")
                
        parsed_data["table_schemas"] = schema_info
        

        target_table_names = parsed_data.get("target_table_name", [])
        if target_table_names:
            target_table = target_table_names[0] if isinstance(target_table_names, list) else target_table_names
            try:
                target_schema = sql_executor.get_table_schema(target_table)
                if isinstance(target_schema, list):
                    parsed_data["target_table_schema"] = target_schema
            except Exception as e:
                logger.warning(f"Error fetching schema for target table {target_table}: {e}")
                

        results = process_info(parsed_data, conn)
        

        if query_type == "SIMPLE_TRANSFORMATION":

            results = _handle_key_mapping_for_simple(results, joined_df, context_manager, session_id, conn)
        else:


            results["key_mapping"] = parsed_data["key_mapping"]
        

        results["session_id"] = session_id
        results["query_type"] = query_type
        results["visited_segments"] = visited_segments
        results["current_segment"] = {
            "id": segment_id,
            "name": segment_name if 'segment_name' in locals() else f"segment_{segment_id}"
        }
        

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

            for target_field in results["target_sap_fields"]:
                target_field_filter = joined_df["target_sap_field"] == target_field
                if target_field_filter.any() and joined_df[target_field_filter]["isKey"].values[0] == "True":

                    logger.info(f"Target field '{target_field}' is identified as a primary key")


                    if results["insertion_fields"] and len(results["insertion_fields"]) > 0:


                        source_field = None
                        

                        for field in results["insertion_fields"]:
                            if field in results["source_field_names"]:
                                source_field = field
                                break
                                

                        if not source_field and results["insertion_fields"]:
                            source_field = results["insertion_fields"][0]
                            

                        source_table = (
                            results["source_table_name"][0]
                            if results["source_table_name"]
                            else None
                        )


                        if source_table and source_field:
                            error = None
                            try:

                                has_nulls = False
                                has_duplicates = False
                                
                                try:

                                    safe_table = validate_sql_identifier(source_table)
                                    safe_field = validate_sql_identifier(source_field)
                                    

                                    null_query = f"SELECT COUNT(*) AS null_count FROM {safe_table} WHERE {safe_field} IS NULL"
                                    null_result = sql_executor.execute_query(null_query)
                                    
                                    if isinstance(null_result, list) and null_result:
                                        has_nulls = null_result[0].get("null_count", 0) > 0
                                    

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
                                    has_nulls = True
                                    has_duplicates = True
                                    


                                if has_nulls or has_duplicates:

                                    restructured_query = results.get("restructured_query", "")
                                    is_distinct_query = (
                                        check_distinct_requirement(restructured_query) if restructured_query 
                                        else False
                                    )

                                    if not is_distinct_query:

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

                    key_mapping = context_manager.get_key_mapping(session_id)
        except Exception as e:
            logger.error(f"Error processing key mapping: {e}")

            key_mapping = []


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

    nlp = spacy.load("en_core_web_md")


    doc = nlp(sentence.lower())


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
            

            segment_info ={
                    "segment_name": segment_name,
                    "target_table_name": target_table_name,
                }
            if segment_info not in segments:
                segments.append()
            

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


            with open(f"{session_path}/context.json", "w") as f:
                json.dump(context, f, indent=2)

            return session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise SessionError(f"Failed to create session: {e}")
        
    def record_executed_query(self, session_id, query):
        """Record an executed query"""
        try:
            if not session_id:
                logger.warning("No session ID provided for record_executed_query")
                return False

            context_path = f"{self.storage_path}/{session_id}/queries.json"
            if not os.path.exists(context_path):
                queries = []
            else:
                try:
                    with open(context_path, "r") as f:
                        queries = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON in queries file, creating new queries"
                    )
                    queries = []

            queries.append(query)

            with open(context_path, "w") as f:
                json.dump(queries, f, indent=2)

            return True
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in context file for session {session_id}: {e}")
            return False

    def get_executed_queries(self,session_id):
        """Get the executed queries for a session"""
        try:
            if not session_id:
                logger.warning("No session ID provided for record_executed_query")
                return False

            context_path = f"{self.storage_path}/{session_id}/queries.json"
            if not os.path.exists(context_path):
                queries = []
            else:
                try:
                    with open(context_path, "r") as f:
                        queries = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON in queries file, creating new queries"
                    )
                    queries = []
            return queries
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in context file for session {session_id}: {e}")
            return []

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

                os.makedirs(os.path.dirname(context_path), exist_ok=True)
                context = {"session_id": session_id}
                

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
                    

            if "segments_visited" not in context:
                context["segments_visited"] = {}
                

            context["segments_visited"][str(segment_id)] = {
                "name": segment_name,
                "visited_at": datetime.now().isoformat(),
                "table_name": ''.join(c if c.isalnum() else '_' for c in segment_name.lower())
            }
            

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


            if not target_col or not isinstance(target_col, str):
                logger.warning(f"Invalid target column: {target_col}")
                return []

            if not source_col or not isinstance(source_col, str):
                logger.warning(f"Invalid source column: {source_col}")
                return []

            file_path = f"{self.storage_path}/{session_id}/key_mapping.json"


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


            if not any(
                mapping
                for mapping in key_mappings
                if mapping["target_col"] == target_col
                and mapping["source_col"] == source_col
            ):
                key_mappings.append(
                    {"target_col": target_col, "source_col": source_col}
                )


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

    def add_transformation_record(self, session_id, transformation_data):
        """Add a transformation record to the session context"""
        try:
            if not session_id:
                logger.warning("No session ID provided for add_transformation_record")
                return False
                
            context_path = f"{self.storage_path}/{session_id}/context.json"
            

            context = self.get_context(session_id)
            if not context:
                context = {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "transformation_history": [],
                    "target_table_state": {
                        "populated_fields": [],
                        "remaining_mandatory_fields": [],
                        "total_rows": 0,
                        "rows_with_data": 0,
                    }
                }
            

            if "transformation_history" not in context:
                context["transformation_history"] = []
            

            transformation_record = {
                "transformation_id": f"t_{len(context['transformation_history']) + 1}",
                "timestamp": datetime.now().isoformat(),
                "original_query": transformation_data.get("original_query", ""),
                "generated_sql": transformation_data.get("generated_sql", ""),
                "query_type": transformation_data.get("query_type", ""),
                "source_tables": transformation_data.get("source_tables", []),
                "target_table": transformation_data.get("target_table", ""),
                "fields_affected": transformation_data.get("fields_affected", []),
                "execution_result": transformation_data.get("execution_result", {}),
                "is_multi_step": transformation_data.get("is_multi_step", False),
                "steps_completed": transformation_data.get("steps_completed", 1)
            }
            
            context["transformation_history"].append(transformation_record)
            

            os.makedirs(os.path.dirname(context_path), exist_ok=True)
            with open(context_path, "w") as f:
                json.dump(context, f, indent=2)
                
            logger.info(f"Added transformation record {transformation_record['transformation_id']} to session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding transformation record: {e}")
            return False

    def get_transformation_history(self, session_id):
        """Get all transformation records for a session"""
        try:
            context = self.get_context(session_id)
            if context and "transformation_history" in context:
                return context["transformation_history"]
            return []
        except Exception as e:
            logger.error(f"Error getting transformation history: {e}")
            return []

    def find_transformation_by_reference(self, session_id, reference_text):
        """Find transformation records that match a reference (e.g., 'previous transformation', 'transformation 2')"""
        try:
            history = self.get_transformation_history(session_id)
            if not history:
                return []
            
            reference_lower = reference_text.lower()
            matches = []
            

            import re
            number_match = re.search(r'(?:transformation|step|query)\s*(\d+)', reference_lower)
            if number_match:
                transform_num = int(number_match.group(1))
                if 1 <= transform_num <= len(history):
                    matches.append(history[transform_num - 1])
            

            elif any(word in reference_lower for word in ['previous', 'last', 'prior', 'earlier']):
                if history:
                    matches.append(history[-1])
            

            else:
                for record in history:
                    if (reference_lower in record.get("original_query", "").lower() or
                        any(table.lower() in reference_lower for table in record.get("source_tables", [])) or
                        record.get("target_table", "").lower() in reference_lower):
                        matches.append(record)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error finding transformation by reference: {e}")
            return []

    def save_multi_query_state(self, session_id, multi_query_state):
        """Save multi-query execution state"""
        try:
            if not session_id:
                return False
                
            session_dir = f"{self.storage_path}/{session_id}"
            os.makedirs(session_dir, exist_ok=True)
            
            state_file = f"{session_dir}/multi_query_state.json"
            with open(state_file, 'w') as f:
                json.dump(multi_query_state, f, indent=2, default=str)
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving multi-query state: {e}")
            return False

    def load_multi_query_state(self, session_id):
        """Load multi-query execution state"""
        try:
            if not session_id:
                return None
                
            state_file = f"{self.storage_path}/{session_id}/multi_query_state.json"
            
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    return json.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading multi-query state: {e}")
        
        return None

    def cleanup_multi_query_state(self, session_id):
        """Clean up multi-query execution state after successful completion"""
        try:
            if not session_id:
                return
                
            state_file = f"{self.storage_path}/{session_id}/multi_query_state.json"
            if os.path.exists(state_file):
                os.remove(state_file)
                
        except Exception as e:
            logger.error(f"Error cleaning up multi-query state: {e}")

    def get_transformation_context_for_query(self, session_id, current_query):
        """Get relevant transformation context that might be referenced in current query"""
        try:
            history = self.get_transformation_history(session_id)
            if not history:
                return None
            

            references = self.find_transformation_by_reference(session_id, current_query)
            
            if references:
                return {
                    "referenced_transformations": references,
                    "transformation_summary": {
                        "total_transformations": len(history),
                        "last_transformation": history[-1] if history else None,
                        "available_tables": list(set([t.get("target_table") for t in history if t.get("target_table")]))
                    }
                }
            
            return {
                "transformation_summary": {
                    "total_transformations": len(history),
                    "last_transformation": history[-1] if history else None,
                    "available_tables": list(set([t.get("target_table") for t in history if t.get("target_table")]))
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting transformation context: {e}")
            return None

def fetch_data_by_ids(object_id, segment_id, project_id, conn):
    """Fetch data mappings from the database"""
    try:

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

        if df is None or df.empty:
            logger.warning("Empty dataframe passed to missing_values_handling")
            return df


        df_processed = df.copy()


        if "source_table" in df_processed.columns:

            df_processed["source_table"] = df_processed["source_table"].replace(
                r"^\s*$", pd.NA, regex=True
            )


            if not df_processed["source_table"].dropna().empty:
                non_na_values = df_processed["source_table"].dropna()
                if len(non_na_values) > 0:
                    fill_value = non_na_values.iloc[0]
                    df_processed["source_table"] = df_processed["source_table"].fillna(
                        fill_value
                    )


        if (
            "source_field_name" in df_processed.columns
            and "target_sap_field" in df_processed.columns
        ):

            df_processed["source_field_name"] = df_processed[
                "source_field_name"
            ].replace(r"^\s*$", pd.NA, regex=True)


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
        if not query or not isinstance(query, str):
            logger.error(f"Invalid query: {query}")
            return None


        if not all(isinstance(x, int) for x in [object_id, segment_id, project_id]):
            logger.error(
                f"Invalid ID types: object_id={type(object_id)}, segment_id={type(segment_id)}, project_id={type(project_id)}"
            )
            return None

        context_manager = ContextualSessionManager()

        if not session_id:
            session_id = context_manager.create_session()
            logger.info(f"Created new session: {session_id}")
    
        return process_query_by_type(
            object_id, 
            segment_id, 
            project_id, 
            query, 
            session_id, 
            target_sap_fields
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
            try:
                target_df = pd.read_csv(target_path)
                return target_df
            except Exception as e:
                logger.error(f"Error reading existing target CSV: {e}")
        try:
            safe_table = validate_sql_identifier(target_table)
            
            target_df = sql_executor.execute_and_fetch_df(f"SELECT * FROM {safe_table}")
            
            if isinstance(target_df, dict) and "error_type" in target_df:
                logger.warning(f"SQL approach failed in get_or_create_session_target_df, using fallback: {target_df.get('error_message')}")
                

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

        if resolved_data is None:
            logger.error("None resolved_data passed to process_info")
            return None

        if conn is None:
            logger.error("None database connection passed to process_info")
            return None
            

        query_type = resolved_data.get("query_type", "SIMPLE_TRANSFORMATION")
        

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
        

        current_required_fields = required_fields.get(query_type, required_fields["SIMPLE_TRANSFORMATION"])
        
        for field in current_required_fields:
            if field not in resolved_data:
                logger.warning(f"Missing required field in resolved_data: {field}")

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


        result = {
            "query_type": query_type,
            "source_table_name": resolved_data["source_table_name"],
            "source_field_names": resolved_data["source_field_names"],
            "target_table_name": resolved_data["target_table_name"],
            "target_sap_fields": resolved_data["target_sap_fields"],
            "restructured_query": resolved_data["Resolved_query"],
        }
        

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
            

            if "filtering_fields" in resolved_data:
                result["filtering_fields"] = resolved_data["filtering_fields"]
            else:
                result["filtering_fields"] = []


        source_data = {}
        try:
            for table in resolved_data["source_table_name"]:

                cleaned_table = clean_table_name(table)
                
                try:

                    safe_table = validate_sql_identifier(cleaned_table)
                    

                    source_df = sql_executor.get_table_sample(safe_table, limit=5)
                    
                    if isinstance(source_df, dict) and "error_type" in source_df:

                        logger.warning(f"SQL source sample failed for {safe_table}, using fallback: {source_df.get('error_message')}")
                        

                        query = f"SELECT * FROM {safe_table} LIMIT 5"
                        source_df = pd.read_sql_query(query, conn)
                        

                    source_data[table] = source_df
                except Exception as e:
                    logger.error(f"Error fetching source data for table {cleaned_table}: {e}")
                    source_data[table] = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting source data samples: {e}")
            source_data = {}
            
        result["source_data_samples"] = source_data
        

        result["table_schemas"] = resolved_data.get("table_schemas", {})
        result["target_table_schema"] = resolved_data.get("target_table_schema", [])
        
        return result
    except Exception as e:
        logger.error(f"Error in process_info: {e}")
        return None