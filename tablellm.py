import os
import logging
import json
import pandas as pd
import numpy as np
import sqlite3
from dotenv import load_dotenv
from google import genai
from google.genai import types
from token_tracker import track_token_usage, get_token_usage_stats

# Import planner functions (keeping these since they work well)
from planner import process_query as planner_process_query
from planner import get_session_context, get_or_create_session_target_df, save_session_target_df
from code_exec import create_code_file, execute_code

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TableLLM:
    """Improved TableLLM with optimized code generation and better information flow"""
    
    def __init__(self):
        """Initialize the TableLLM instance"""
        # Configure Gemini
        api_key = os.environ.get('GEMINI_API_KEY')
        self.client = genai.Client(api_key=api_key)
        
        # Load code templates
        self.code_templates = self._initialize_templates()
        
        # Current session context
        self.current_context = None

    def _extract_planner_info(self, resolved_data):
        """
        Extract and organize all relevant information from planner's resolved data
        to make it easily accessible in all prompting phases
        """
        # Create a comprehensive context object
        planner_info = {
            # Table information
            "source_table": resolved_data.get('source_table_name'),
            "target_table": resolved_data.get('target_table_name'),
            "additional_sources": resolved_data.get('additional_source_table', []),
            
            # Field information
            "source_fields": resolved_data.get('source_field_names', []),
            "target_fields": resolved_data.get('target_sap_fields', []),
            "filtering_fields": resolved_data.get('filtering_fields', []),
            "insertion_fields": resolved_data.get('insertion_fields', []),
            
            # Data samples
            "source_data": {
                "sample": resolved_data.get('source_info', pd.DataFrame()).head(3).to_dict('records'),
                "describe": resolved_data.get('source_describe', {})
            },
            "target_data": {
                "sample": resolved_data.get('target_info', pd.DataFrame()).head(3).to_dict('records'),
                "describe": resolved_data.get('target_describe', {})
            },
            
            # Query understanding
            "original_query": resolved_data.get('original_query', ''),
            "restructured_query": resolved_data.get('restructured_question', ''),
            
            # Transformation history and context
            "transformation_history": resolved_data.get('context', {}).get('transformation_history', []),
            "target_table_state": resolved_data.get('context', {}).get('target_table_state', {}),
            
            # Session information
            "session_id": resolved_data.get('session_id')
        }
        
        # Extract specific filtering conditions from the restructured query
        query_text = planner_info["restructured_query"]
        conditions = {}
        
        # Look for common filter patterns in the restructured query
        if "=" in query_text:
            for field in planner_info["filtering_fields"]:
                pattern = f"{field}\\s*=\\s*['\"](.*?)['\"]"
                import re
                matches = re.findall(pattern, query_text)
                if matches:
                    conditions[field] = matches[0]
        
        # Look for IN conditions
        if " in " in query_text.lower():
            for field in planner_info["filtering_fields"]:
                pattern = f"{field}\\s+in\\s+\\(([^)]+)\\)"
                import re
                matches = re.findall(pattern, query_text, re.IGNORECASE)
                if matches:
                    # Parse the values in the parentheses
                    values_str = matches[0]
                    values = [v.strip().strip("'\"") for v in values_str.split(',')]
                    conditions[field] = values
        
        # Add specific conditions
        planner_info["extracted_conditions"] = conditions
        
        # Store context for use in all prompting phases
        self.current_context = planner_info
        return planner_info

    def find_value_intersections(self, source_df, target_df, target_fields):
        """
        Find source columns that have values intersecting with a target column
        using direct value comparison approach.
        
        Args:
            source_df: Source dataframe
            target_df: Target dataframe
            target_fields: List of target fields we're trying to populate

        Returns:
            Dict mapping source columns to their potential target matches with scores
        """
        logger.info(f"Finding value intersections for target fields: {target_fields}")
        
        # Handle empty dataframes
        if source_df.empty or target_df.empty:
            logger.warning("Empty dataframe detected in intersection analysis")
            return []
        
        # Pre-filter columns from both dataframes - eliminate poor candidates
        # 1. Remove columns with too many nulls
        source_null_ratios = source_df.isnull().mean()
        valid_source_cols = [col for col in source_df.columns 
                           if source_null_ratios[col] < 0.9]  # Less than 90% nulls
        
        # 2. Remove columns with too low uniqueness (e.g., all identical values)
        source_uniqueness = {col: source_df[col].nunique() / len(source_df) 
                           for col in valid_source_cols}
        valid_source_cols = [col for col in valid_source_cols 
                           if source_uniqueness.get(col, 0) > 0.01]  # At least 1% unique
        
        # 3. Skip columns that are likely not useful for joining (like large text fields)
        excluded_patterns = ['descr', 'text', 'note', 'comment', 'detail', 'paragraph']
        valid_source_cols = [col for col in valid_source_cols if not any(
            pattern in str(col).lower() for pattern in excluded_patterns)]
        
        # Now collect all target values for each target field
        target_values = {}
        for field in target_fields:
            if field in target_df.columns:
                # Handle different data types by converting to string
                target_values[field] = set(target_df[field].dropna().astype(str).tolist())
            else:
                target_values[field] = set()  # Empty set if field doesn't exist
        
        # Check each source column for intersection with each target field
        intersection_results = []
        
        for source_col in valid_source_cols:
            # Convert source values to strings for comparison
            try:
                source_values = set(source_df[source_col].dropna().astype(str).tolist())
                
                # Skip if source has no values
                if not source_values:
                    continue
                
                # Check intersection with each target field
                for target_field, target_field_values in target_values.items():
                    # Skip if target has no values
                    if not target_field_values:
                        continue
                    
                    # Find intersection
                    intersection = source_values.intersection(target_field_values)
                    
                    # Calculate metrics
                    if intersection:
                        # Percentage of target values found in source
                        coverage_ratio = len(intersection) / max(len(target_field_values), 1) 
                        
                        # Uniqueness of source column
                        uniqueness_ratio = source_uniqueness.get(source_col, 0)
                        
                        # Name similarity score (0-1)
                        name_similarity = 0
                        s_col_lower = source_col.lower()
                        t_field_lower = target_field.lower()
                        
                        # Exact match
                        if s_col_lower == t_field_lower:
                            name_similarity = 1.0
                        # One contains the other
                        elif s_col_lower in t_field_lower or t_field_lower in s_col_lower:
                            name_similarity = 0.7
                        # Share words
                        elif any(word in t_field_lower.split('_') 
                                for word in s_col_lower.split('_') if word):
                            name_similarity = 0.5
                        
                        # Combined score (weighted)
                        score = (coverage_ratio * 0.6 + 
                                uniqueness_ratio * 0.2 + 
                                name_similarity * 0.2)
                        
                        # Must have at least some coverage
                        if coverage_ratio > 0:
                            intersection_results.append({
                                'source_col': source_col,
                                'target_col': target_field,
                                'intersection_size': len(intersection),
                                'coverage_ratio': coverage_ratio,
                                'uniqueness_ratio': uniqueness_ratio,
                                'name_similarity': name_similarity,
                                'score': score
                            })
            except Exception as e:
                logger.warning(f"Error processing column {source_col}: {str(e)}")
                continue
        
        # Sort by score (descending)
        intersection_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Log results
        if intersection_results:
            logger.info(f"Found {len(intersection_results)} potential value intersections.")
            logger.info(f"Top match: {intersection_results[0]}")
        else:
            logger.warning("No value intersections found between source and target columns.")
        
        return intersection_results

    def identify_join_keys(self, source_df, target_df):
        """Identify potential join keys between source and target dataframes"""
        logger.info("Starting join key identification...")
        
        # Check if dataframes are empty
        if source_df.empty:
            logger.warning("Source dataframe is empty")
            return []
        if target_df.empty:
            logger.info("Target dataframe is empty - no join keys needed for initial population")
            # Return a special indicator for empty target
            return [{"source_col": "_empty_target_", 
                     "target_col": "_empty_target_", 
                     "overlap_score": 1.0,
                     "source_unique_ratio": 1.0,
                     "target_unique_ratio": 1.0,
                     "match_type": "empty_target"}]
        
        # Pre-filter columns - remove problematic ones
        # 1. Remove completely null columns
        valid_source_cols = [col for col in source_df.columns if not source_df[col].isna().all()]
        valid_target_cols = [col for col in target_df.columns if not target_df[col].isna().all()]
        
        # 2. Remove columns with too low uniqueness (e.g., all identical values)
        valid_source_cols = [col for col in valid_source_cols 
                           if source_df[col].nunique() > 1]
        valid_target_cols = [col for col in valid_target_cols 
                           if target_df[col].nunique() > 1]
        
        logger.info(f"Valid source columns after filtering: {len(valid_source_cols)}")
        logger.info(f"Valid target columns after filtering: {len(valid_target_cols)}")
        
        potential_keys = []
        
        # PHASE 1: Look for exact column name matches
        exact_matches = set(valid_source_cols).intersection(set(valid_target_cols))
        logger.info(f"Exact column name matches: {exact_matches}")
        
        for col in exact_matches:
            try:
                # Convert to strings for comparison
                source_values = set(source_df[col].dropna().astype(str).unique())
                target_values = set(target_df[col].dropna().astype(str).unique())
                
                if source_values and target_values:
                    # Calculate intersection
                    intersection = source_values.intersection(target_values)
                    
                    # Calculate metrics
                    if intersection:
                        overlap = len(intersection) / len(target_values) if target_values else 0
                        source_unique_ratio = len(source_values) / len(source_df) if len(source_df) > 0 else 0
                        target_unique_ratio = len(target_values) / len(target_df) if len(target_df) > 0 else 0
                        
                        # Much more permissive criteria - accept even small overlaps
                        potential_keys.append({
                            'source_col': col,
                            'target_col': col,
                            'overlap_score': overlap,
                            'source_unique_ratio': source_unique_ratio,
                            'target_unique_ratio': target_unique_ratio,
                            'match_type': 'exact_name',
                            'intersection_size': len(intersection)
                        })
            except Exception as e:
                logger.warning(f"Error comparing column {col}: {str(e)}")
        
        # PHASE 2: Use the direct value intersection method to check target fields against source columns
        # Get list of target fields - we'll be filling these
        target_fields = list(set(valid_target_cols))
        
        # Apply the direct intersection method
        intersection_results = self.find_value_intersections(source_df, target_df, target_fields)
        
        # Convert intersection results to potential keys format
        for result in intersection_results:
            # Avoid duplicating exact name matches
            if not any(k['source_col'] == result['source_col'] and 
                    k['target_col'] == result['target_col'] for k in potential_keys):
                potential_keys.append({
                    'source_col': result['source_col'],
                    'target_col': result['target_col'],
                    'overlap_score': result['coverage_ratio'],
                    'source_unique_ratio': result['uniqueness_ratio'],
                    'target_unique_ratio': 0.0,  # Not calculated in intersection method
                    'match_type': 'value_intersection',
                    'intersection_size': result['intersection_size']
                })
        
        # PHASE 3: Look for columns with similar names as a fallback
        if len(potential_keys) < 2:
            logger.info("Looking for similar named columns...")
            
            for s_col in valid_source_cols:
                s_col_lower = s_col.lower()
                for t_col in valid_target_cols:
                    # Skip if it's an exact match we already processed
                    if s_col == t_col:
                        continue
                    
                    t_col_lower = t_col.lower()
                    # Check for name similarity
                    if (s_col_lower in t_col_lower or 
                        t_col_lower in s_col_lower or
                        any(word in t_col_lower.split('_') 
                            for word in s_col_lower.split('_') if word)):
                        
                        try:
                            # Check for value overlap
                            source_values = set(source_df[s_col].dropna().astype(str).unique())
                            target_values = set(target_df[t_col].dropna().astype(str).unique())
                            
                            if source_values and target_values:
                                intersection = source_values.intersection(target_values)
                                if intersection:
                                    overlap = len(intersection) / len(target_values) if target_values else 0
                                    source_unique_ratio = len(source_values) / len(source_df) if len(source_df) > 0 else 0
                                    target_unique_ratio = len(target_values) / len(target_df) if len(target_df) > 0 else 0
                                    
                                    potential_keys.append({
                                        'source_col': s_col,
                                        'target_col': t_col,
                                        'overlap_score': overlap,
                                        'source_unique_ratio': source_unique_ratio,
                                        'target_unique_ratio': target_unique_ratio,
                                        'match_type': 'similar_name',
                                        'intersection_size': len(intersection)
                                    })
                        except Exception as e:
                            logger.warning(f"Error comparing {s_col} with {t_col}: {str(e)}")
        
        # If no keys found, use metadata to suggest potential keys
        if not potential_keys:
            logger.warning("No overlap found, suggesting potential keys based on column properties")
            
            # Find columns that look like ID fields
            id_patterns = ['id', 'key', 'code', 'num', 'no', 'nr', 'number', 'name', 'mat']
            
            source_id_cols = [col for col in valid_source_cols 
                             if any(pattern in col.lower() for pattern in id_patterns)]
            target_id_cols = [col for col in valid_target_cols 
                             if any(pattern in col.lower() for pattern in id_patterns)]
            
            # If ID columns found, suggest the first pair
            if source_id_cols and target_id_cols:
                potential_keys.append({
                    'source_col': source_id_cols[0],
                    'target_col': target_id_cols[0],
                    'overlap_score': 0.0,
                    'source_unique_ratio': 0.0,
                    'target_unique_ratio': 0.0,
                    'match_type': 'suggested_id',
                    'intersection_size': 0
                })
            # Otherwise, use columns with highest uniqueness
            else:
                source_uniqueness = [(col, source_df[col].nunique() / len(source_df)) 
                                   for col in valid_source_cols]
                target_uniqueness = [(col, target_df[col].nunique() / len(target_df)) 
                                   for col in valid_target_cols]
                
                # Sort by uniqueness
                source_uniqueness.sort(key=lambda x: x[1], reverse=True)
                target_uniqueness.sort(key=lambda x: x[1], reverse=True)
                
                if source_uniqueness and target_uniqueness:
                    potential_keys.append({
                        'source_col': source_uniqueness[0][0],
                        'target_col': target_uniqueness[0][0],
                        'overlap_score': 0.0,
                        'source_unique_ratio': source_uniqueness[0][1],
                        'target_unique_ratio': target_uniqueness[0][1],
                        'match_type': 'high_uniqueness',
                        'intersection_size': 0
                    })
        
        # Sort keys by a composite score
        def get_sort_score(key):
            match_type_score = {
                'exact_name': 5.0,
                'value_intersection': 4.0,
                'similar_name': 3.0,
                'empty_target': 10.0,
                'suggested_id': 2.0,
                'high_uniqueness': 1.0
            }.get(key['match_type'], 0.0)
            
            return (
                match_type_score * 1000 +  # Match type is most important
                key['overlap_score'] * 100 +  # Then overlap score
                key['intersection_size'] * 0.1 +  # Then actual intersection size
                (key['source_unique_ratio'] + key['target_unique_ratio']) * 10  # Then uniqueness
            )
        
        potential_keys.sort(key=get_sort_score, reverse=True)
        
        # Log results
        if potential_keys:
            logger.info(f"Top potential join key: {potential_keys[0]}")
            logger.info(f"Total potential join keys identified: {len(potential_keys)}")
        else:
            logger.warning("No potential join keys identified.")
        
        return potential_keys

    @track_token_usage()
    def _select_best_join_keys(self, query, planner_info, source_df, target_df):
        """Use LLM to select the best join keys from potential candidates"""
        # Identify potential keys first using data analysis
        potential_keys = self.identify_join_keys(source_df, target_df)
        
        # If no potential keys found, return an empty result
        if not potential_keys:
            logger.warning("No potential join keys found for selection")
            return {
                "selected_keys": [],
                "join_type": "none",
                "reasoning": "No potential join keys identified between source and target tables."
            }
        
        # Prepare data samples
        source_sample = source_df.head(3).to_string()
        target_sample = target_df.head(3).to_string()
        
        # Extract the key fields
        source_fields = planner_info['source_fields']
        source_field = source_fields[0] if isinstance(source_fields, list) and len(source_fields) > 0 else "SOURCE_FIELD"
        
        target_fields = planner_info['target_fields']
        target_field = target_fields[0] if isinstance(target_fields, list) and len(target_fields) > 0 else "TARGET_FIELD"
        
        prompt = f"""
You are a data mapping expert. Select the BEST join key(s) to map data from source to target tables.

QUERY: {query}
RESTRUCTURED QUERY: {planner_info['restructured_query']}

SOURCE TABLE: {planner_info['source_table']}
TARGET TABLE: {planner_info['target_table']}

SOURCE FIELDS TO EXTRACT: {source_field}
TARGET FIELDS TO UPDATE: {target_field}
FILTERING FIELDS: {planner_info['filtering_fields']}
FILTERING CONDITIONS: {json.dumps(planner_info['extracted_conditions'], indent=2)}

SOURCE TABLE SAMPLE:
{source_sample}

TARGET TABLE SAMPLE:
{target_sample}

POTENTIAL JOIN KEYS (ranked by data overlap):
{json.dumps(potential_keys[:5], indent=2)}

Select the best key(s) for mapping data by analyzing:
1. Which key has good overlap of values between source and target
2. Which key has good uniqueness (not all values are the same)
3. Which key most logically connects the tables based on the query
4. If multiple keys should be used together as a composite key

Return a JSON object with:
1. "selected_keys": A list of column pairs to use as join keys (each with source_col and target_col)
2. "join_type": The recommended join type (inner, left, right)
3. "reasoning": Brief explanation of your choice

Example response:
```json
{
    "selected_keys": [
        {"source_col": "material_id", "target_col": "material_id"},
        {"source_col": "plant", "target_col": "plant_id"}
    ],
    "join_type": "left",
    "reasoning": "Selected material_id and plant as composite key since they uniquely identify rows and have good overlap between source and target tables."
}
```
"""
        
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Parse the response
        try:
            import re
            json_str = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
            if json_str:
                selected_keys = json.loads(json_str.group(1).strip())
            else:
                selected_keys = json.loads(response.text.strip())
            
            logger.info(f"Selected join keys: {selected_keys}")
            return selected_keys
        except Exception as e:
            logger.error(f"Error parsing join key selection response: {e}")
            # Fallback if parsing fails
            logger.info("Using fallback join key selection")
            return {
                "selected_keys": [{"source_col": potential_keys[0]['source_col'], 
                                 "target_col": potential_keys[0]['target_col']}],
                "join_type": "left",
                "reasoning": "Automatically selected based on highest overlap score"
            }

    @track_token_usage()
    def _generate_simple_plan_with_keys(self, planner_info, selected_keys):
        """
        Generate a simple, step-by-step plan in natural language using selected join keys
        """
        # Extract key fields
        source_fields = planner_info["source_fields"]
        source_field = source_fields[0] if isinstance(source_fields, list) and len(source_fields) > 0 else "SOURCE_FIELD"
        
        target_fields = planner_info["target_fields"]
        target_field = target_fields[0] if isinstance(target_fields, list) and len(target_fields) > 0 else "TARGET_FIELD"
        
        # Extract filtering conditions
        conditions_str = "No specific conditions found"
        if planner_info["extracted_conditions"]:
            conditions_str = json.dumps(planner_info["extracted_conditions"], indent=2)
        
        # Format the join keys for the prompt
        join_keys_str = json.dumps(selected_keys["selected_keys"], indent=2)
        join_type = selected_keys["join_type"]
        join_reasoning = selected_keys.get("reasoning", "No reasoning provided")
        
        # Generate prompt for plan creation
        base_prompt = f"""
Create a simplified step-by-step plan for code that will extract data from source and add it to the target table.

QUERY DETAILS:
User's intent: {planner_info['restructured_query']}
Source table: {planner_info['source_table']}
Target table: {planner_info['target_table']}
Source field to extract: {source_field}
Target field to update: {target_field}
Filtering conditions: {conditions_str}

JOIN KEYS IDENTIFIED:
{join_keys_str}
Join type recommended: {join_type}

CRITICAL REQUIREMENTS:
1. The source field '{source_field}' must be copied to the target field '{target_field}'
2. You MUST handle both cases: empty target table and non-empty target table
3. For empty target table, create new rows with the filtered source data
4. For non-empty target table, update existing rows using join keys

SAMPLE DATA:
Source data sample: {json.dumps(planner_info['source_data']['sample'][:2], indent=2)}
Target data sample: {json.dumps(planner_info['target_data']['sample'][:2], indent=2)}

Write a precise, step-by-step plan that handles ALL scenarios:
1. Create a copy of the source dataframe
2. Filter source rows based on filtering conditions
3. Check if filtered dataframe is empty, if empty return original target df
4. Check if target dataframe is empty
5. If target is empty, create new dataframe with filtered source data 
6. If target is not empty, use join keys to merge source and target
7. Update target field with source field values
8. Return the updated target dataframe

Your steps (numbered, 6-10 steps maximum):
"""
        
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=base_prompt
        )
        
        return response.text.strip()

    @track_token_usage()
    def _generate_code_from_plan_with_keys(self, simple_plan, planner_info, selected_keys):
        """
        Generate code based on a step-by-step plan with join keys
        """
        # Extract key information
        source_fields = planner_info["source_fields"]
        source_field = source_fields[0] if isinstance(source_fields, list) and len(source_fields) > 0 else "SOURCE_FIELD"
        
        target_fields = planner_info["target_fields"]
        target_field = target_fields[0] if isinstance(target_fields, list) and len(target_fields) > 0 else "TARGET_FIELD"
        
        # Extract filtering conditions
        filter_conditions = []
        if planner_info["extracted_conditions"]:
            # Convert the extracted conditions to pandas filter syntax
            for field, value in planner_info["extracted_conditions"].items():
                if isinstance(value, list):
                    conditions = [f"df1['{field}'] == '{v}'" for v in value]
                    filter_conditions.append(f"({' | '.join(conditions)})")
                else:
                    filter_conditions.append(f"df1['{field}'] == '{value}'")
        
        # Use a default condition if none found
        if not filter_conditions:
            filter_conditions = ["df1[df1.columns[0]] != ''"]  # Default condition that selects all rows
        
        filter_condition_str = " & ".join(filter_conditions)
        
        # Format join keys information
        join_keys = selected_keys["selected_keys"]
        join_type = selected_keys["join_type"]
        
        # Example code for handling empty target table
        empty_target_handling = f"""
# Handle empty target table
if len(df2) == 0:
    # Create a new dataframe from filtered source data
    # Make sure to include needed columns
    result = pd.DataFrame()
    result['{target_field}'] = filtered_df['{source_field}']
    # Add any other necessary columns
    for key in join_keys:
        if key in filtered_df.columns:
            result[key] = filtered_df[key]
    return result
"""
        
        # Example code for merge operation with join keys
        merge_example = ""
        if join_keys:
            join_cols_source = [k["source_col"] for k in join_keys]
            join_cols_target = [k["target_col"] for k in join_keys]
            
            source_cols_str = ", ".join([f"'{col}'" for col in join_cols_source])
            source_cols_str = f"[{source_cols_str}, '{source_field}']"
            
            # Create rename dictionary if needed
            rename_dict = {}
            for source_key, target_key in zip(join_cols_source, join_cols_target):
                if source_key != target_key:
                    rename_dict[source_key] = target_key
            
            merge_example = f"""
# Example merge operation:
# 1. Prepare source data with only necessary columns
source_for_merge = filtered_df[{source_cols_str}].copy()

# 2. Rename columns if needed
{'rename_dict = ' + str(rename_dict) if rename_dict else '# No column renaming needed'}
{'source_for_merge = source_for_merge.rename(columns=rename_dict)' if rename_dict else ''}

# 3. Perform the merge operation
if len(df2) > 0:  # Only merge if target table is not empty
    join_cols = {join_cols_target if rename_dict else join_cols_source}
    join_on = join_cols[0] if len(join_cols) == 1 else join_cols
    
    result = pd.merge(
        df2,
        source_for_merge,
        on=join_on,
        how='{join_type}'
    )
    
    # 4. Update target field with source field values where matches exist
    if '{source_field}' in source_for_merge.columns and '{target_field}' in result.columns:
        mask = result['{source_field}'].notna()
        result.loc[mask, '{target_field}'] = result.loc[mask, '{source_field}']
    
    # 5. Drop the temporary source field column if it was added during merge
    if '{source_field}' != '{target_field}' and '{source_field}' in result.columns:
        result = result.drop(columns=['{source_field}'])
"""
        
        # Create template with the simple plan as a guide and join key context
        prompt = f"""
Write Python code that follows these EXACT steps:

{simple_plan}

CRITICAL REQUIREMENTS:
1. You MUST handle BOTH empty and non-empty target dataframes properly
2. For empty target dataframe (len(df2) == 0), create a new dataframe from filtered source data
3. For non-empty target dataframe, use join keys to update existing records
4. Make sure to actually copy the '{source_field}' value to the '{target_field}' column
5. If the target field doesn't exist in the target table, create it
6. Return the final dataframe with the updated/added data

KEY INFORMATION:
- Source field to extract: '{source_field}'
- Target field to update: '{target_field}'
- Filter condition: {filter_condition_str}
- Join keys: {json.dumps(join_keys, indent=2)}

EXAMPLE CODE FOR EMPTY TARGET HANDLING:
{empty_target_handling}

EXAMPLE CODE FOR MERGE OPERATION:
{merge_example}

SAMPLE DATA:
Source data sample: {json.dumps(planner_info['source_data']['sample'][:2], indent=2)}
Target data sample: {json.dumps(planner_info['target_data']['sample'][:2], indent=2)}

Complete this function, implementing ALL the requirements and following the plan:
```python
def analyze_data(df1, df2, additional_tables=None):
    # 1. Make a copy of the source dataframe
    source_df = df1.copy()
    
    # 2. Filter based on conditions
    filtered_df = source_df[{filter_condition_str}].copy()
    
    # Your complete implementation here
    # Handle both empty and non-empty target cases
    # Make sure to copy source_field to target_field
    
    # Return the final result
    return result
```

Return ONLY the complete Python function:
"""
        
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Extract the code
        import re
        code_match = re.search(r'```python\s*(.*?)\s*```', response.text, re.DOTALL)
        if code_match:
            return code_match.group(1)
        else:
            # If no code block, try to extract the function
            function_match = re.search(r'def analyze_data.*?return', response.text, re.DOTALL)
            if function_match:
                return function_match.group(0) + " df2"  # Add return value if missing
            return response.text

    @track_token_usage()
    def _classify_query(self, query, planner_info):
        """
        Classify the query type to determine code generation approach
        Uses extracted planner information for better context
        """
        # Create a comprehensive prompt with detailed context
        prompt = f"""
Classify this data transformation query into ONE of these categories:
- FILTER_AND_EXTRACT: Filtering records from source and extracting specific fields
- UPDATE_EXISTING: Updating values in existing target records only
- CONDITIONAL_MAPPING: Applying if/else logic to determine values
- EXTRACTION: Extracting data from source to target without complex filtering
- TIERED_LOOKUP: Looking up data in multiple tables in a specific order
- AGGREGATION: Performing calculations or aggregations on source data

QUERY INFORMATION:
Original query: {query}
Restructured query: {planner_info['restructured_query']}
Source fields: {planner_info['source_fields']}
Target fields: {planner_info['target_fields']}
Filter fields: {planner_info['filtering_fields']}
Insertion fields: {planner_info['insertion_fields']}


TRANSFORMATION CONTEXT:
Previously populated fields: {planner_info['target_table_state'].get('populated_fields', [])}
Previously completed transformations: {len(planner_info['transformation_history'])}

EXTRACTED CONDITIONS:
{json.dumps(planner_info['extracted_conditions'], indent=2)}

Return ONLY the classification name with no explanation.
"""
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Return the classification
        return response.text.strip()
    
    @track_token_usage()
    def _generate_simple_plan(self, query_type, planner_info):
        """
        Generate a simple, step-by-step plan in natural language
        Uses comprehensive planner information
        """
        # Extract key fields
        source_fields = planner_info["source_fields"]
        source_field = source_fields[0] if isinstance(source_fields, list) and len(source_fields) > 0 else "SOURCE_FIELD"
        
        target_fields = planner_info["target_fields"]
        target_field = target_fields[0] if isinstance(target_fields, list) and len(target_fields) > 0 else "TARGET_FIELD"
        
        # Extract filtering conditions
        conditions_str = "No specific conditions found"
        if planner_info["extracted_conditions"]:
            conditions_str = json.dumps(planner_info["extracted_conditions"], indent=2)
        
        # Add context from transformation history
        history_context = ""
        if planner_info["transformation_history"]:
            last_transform = planner_info["transformation_history"][-1]
            history_context = f"""
Last transformation: {last_transform.get('description', 'Unknown')}
Fields modified: {last_transform.get('fields_modified', [])}
Filter conditions used: {json.dumps(last_transform.get('filter_conditions', {}))}
"""
        print(planner_info)
        # Generate prompt with different templates based on query type
        base_prompt = f"""
Create a simplified step-by-step plan for code that will perform a {query_type} operation.

QUERY DETAILS:
User's intent: {planner_info['restructured_query']}
Source table: {planner_info['source_table']}
Target table: {planner_info['target_table']}
Source field(s): {source_fields}
Target field(s): {target_fields}
Filtering field(s): {planner_info['filtering_fields']}
Filtering conditions: {conditions_str}
Insertion field(s): {planner_info['insertion_fields']}

Current state of target table:
- Populated fields: {planner_info['target_table_state'].get('populated_fields', [])}
- Remaining fields: {planner_info['target_table_state'].get('remaining_mandatory_fields', [])}

Note:
1. Only update a the Insertion field in the target table do not add anything else
2. Use the source table for the initial data 
3. Do not add any additional fields to the target table

SAMPLE DATA:
Source data sample: {json.dumps(planner_info['source_data']['sample'][:2], indent=2)}
Target data sample: {json.dumps(planner_info['target_data']['sample'][:2], indent=2)}
"""

        # Add query-type specific prompting
        if query_type == "CONDITIONAL_MAPPING":
            base_prompt += """
For CONDITIONAL_MAPPING, include steps that:
1. Identify all the conditions mentioned in the query
2. Define a clear mapping from conditions to output values
3. Explain how to apply the conditions in the right order
4. Handle the default case
"""
        elif query_type == "TIERED_LOOKUP":
            base_prompt += """
For TIERED_LOOKUP, include steps that:
1. Define the lookup order across multiple tables
2. Specify what to do when a match is found in each table
3. Handle the case when no matches are found in any table
"""
        
        base_prompt += """
Write ONLY simple, clear steps that a code generator must follow exactly, like this example:
1. Make copy of the source dataframe 
2. Filter rows where MTART value is ROH
3. Take only MATNR column from filtered source data
4. Check if target dataframe is empty
5. If empty, create new dataframe with MATNR as target field
6. If not empty, update the target field with source field values
7. Return the updated target dataframe

Your steps (numbered, 5-10 steps maximum):
"""
        
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-thinking-exp-01-21",
            contents=base_prompt
        )
        
        return response.text.strip()

    @track_token_usage()
    def _generate_code_from_simple_plan(self, simple_plan, planner_info):
        """
        Generate code based on a simple, step-by-step plan
        With improved context from planner
        """
        # Extract key information
        source_fields = planner_info["source_fields"]
        source_field = source_fields[0] if isinstance(source_fields, list) and len(source_fields) > 0 else "SOURCE_FIELD"
        
        target_fields = planner_info["target_fields"]
        target_field = target_fields[0] if isinstance(target_fields, list) and len(target_fields) > 0 else "TARGET_FIELD"
        
        # Extract filtering conditions
        filter_conditions = []
        if planner_info["extracted_conditions"]:
            # Convert the extracted conditions to pandas filter syntax
            for field, value in planner_info["extracted_conditions"].items():
                if isinstance(value, list):
                    conditions = [f"df1['{field}'] == '{v}'" for v in value]
                    filter_conditions.append(f"({' | '.join(conditions)})")
                else:
                    filter_conditions.append(f"df1['{field}'] == '{value}'")
        
        # Use a default condition if none found
        if not filter_conditions:
            filter_conditions = ["df1[df1.columns[0]] != ''"]  # Default condition that selects all rows
        
        filter_condition_str = " & ".join(filter_conditions)
        
        # Determine if there are multiple source tables to handle
        additional_tables = planner_info.get("additional_sources", [])
        additional_tables_handling = ""
        if additional_tables and isinstance(additional_tables, list) and len(additional_tables) > 0:
            additional_tables_handling = f"""
# The additional_tables parameter contains these tables: {additional_tables}
# Access them like this: additional_tables['{additional_tables[0]}']
# Make sure to check if they exist before using them:
if additional_tables and '{additional_tables[0]}' in additional_tables:
    secondary_table = additional_tables['{additional_tables[0]}']
    # Use the secondary table for lookups
"""

        # Create template with the simple plan as a guide and extensive context
        prompt = f"""
Write Python code that follows these EXACT steps:

{simple_plan}

DETAILED INFORMATION:
- Source table: {planner_info['source_table']}
- Target table: {planner_info['target_table']}
- Source field(s): {source_fields}
- Target field(s): {target_fields}
- Filter condition to use: {filter_condition_str}
{additional_tables_handling}

SAMPLE DATA:
Source data sample: {json.dumps(planner_info['source_data']['sample'][:2], indent=2)}
Target data sample: {json.dumps(planner_info['target_data']['sample'][:2], indent=2)}

REQUIREMENTS:
1. Follow the steps PRECISELY in order
2. Handle both empty and non-empty target dataframes
3. Handle additional tables correctly if they're provided
4. Return the modified target dataframe (df2)
5. Use numpy for conditional logic if needed (already imported as np)
6. If you dont find target_field in target_table but it is present in the sample data then add the column to the target table

Complete this function:
```python
def analyze_data(df1, df2, additional_tables=None):
    # Your implementation of the steps above
    
    # Make sure to return the modified df2
    return df2
```

Return ONLY the complete Python function:
"""
        
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Extract the code
        import re
        code_match = re.search(r'```python\s*(.*?)\s*```', response.text, re.DOTALL)
        if code_match:
            return code_match.group(1)
        else:
            # If no code block, try to extract the function
            function_match = re.search(r'def analyze_data.*?return', response.text, re.DOTALL)
            if function_match:
                return function_match.group(0) + " df2"  # Add return value if missing
            return response.text

    def _initialize_templates(self):
        """Initialize code templates for common operations"""
        return {
            "filter": """
# Filter source data based on condition
mask = {filter_condition}
filtered_df = df1[mask].copy()
""",
            "update": """
# Create a copy of df2 to avoid modifying the original
result = df2.copy()

# Check if target table is empty
if len(result) == 0:
    # For empty target, create a new dataframe with necessary columns
    # and only the data we need from source
    key_data = df1[['{key_field}']].copy()
    key_data['{target_field}'] = df1['{source_field}']
    return key_data
else:
    # For non-empty target, update only the target field
    # Find matching rows using the key field
    for idx, row in df1.iterrows():
        # Get the key value
        key_value = row['{key_field}']
        
        # Find matching rows in the target
        target_indices = result[result['{key_field}'] == key_value].index
        
        # Update the target field for matching rows
        if len(target_indices) > 0:
            result.loc[target_indices, '{target_field}'] = row['{source_field}']
        
    return result
""",
            "conditional_mapping": """
# Create a copy of df2 to avoid modifying the original
result = df2.copy()

# Check if target table is empty
if len(result) == 0:
    # For empty target, we need to create initial structure with key fields
    # and apply our conditional logic directly to the source data
    key_data = df1[['{key_field}']].copy()
    
    # Define conditions and choices
    conditions = [
        {conditions}
    ]
    choices = [
        {choices}
    ]
    default = '{default_value}'
    
    # Apply conditional mapping to create the target field
    key_data['{target_field}'] = np.select(conditions, choices, default=default)
    return key_data
else:
    # Define conditions and choices
    conditions = [
        {conditions}
    ]
    choices = [
        {choices}
    ]
    default = '{default_value}'
    
    # Create temporary mapping from source data
    source_mapping = pd.Series(index=df1['{key_field}'], data=np.select(conditions, choices, default=default))
    
    # Apply mapping to target based on key field
    for idx, row in result.iterrows():
        key_value = row['{key_field}']
        if key_value in source_mapping.index:
            result.loc[idx, '{target_field}'] = source_mapping[key_value]
    
    return result
"""
        }
    
    def post_proccess_result(self, result):
        """
        Post-process the result DataFrame to remove any columns added due to reindexing
        
        Parameters:
        result (DataFrame): The result DataFrame to clean
        
        Returns:
        DataFrame: The cleaned DataFrame
        """
        if not isinstance(result, pd.DataFrame):
            return result
        
        # Create a copy to avoid modifying the original
        cleaned_df = result.copy()
        
        # Find columns that match the pattern "Unnamed: X" where X is a number
        unnamed_cols = [col for col in cleaned_df.columns if 'unnamed' in str(col).lower() and ':' in str(col)]
        
        # Drop these columns
        if unnamed_cols:
            cleaned_df = cleaned_df.drop(columns=unnamed_cols)
            logger.info(f"Removed {len(unnamed_cols)} unnamed columns: {unnamed_cols}")
        
        return cleaned_df

    def process_sequential_query(self, query, object_id=29, segment_id=336, project_id=24, session_id=None):
        """
        Process a query as part of a sequential transformation
        With improved information flow from planner and data-driven join key selection
        
        Parameters:
        query (str): The user's query
        object_id (int): Object ID for mapping
        segment_id (int): Segment ID for mapping
        project_id (int): Project ID for mapping
        session_id (str): Optional session ID, creates new session if None
        
        Returns:
        tuple: (code, result, session_id)
        """
        # 1. Process query with the planner
        resolved_data = planner_process_query(object_id, segment_id, project_id, query, session_id)
        if not resolved_data:
            return None, "Failed to resolve query", session_id
        
        # 2. Extract and organize all relevant information from the planner
        resolved_data['original_query'] = query  # Add original query for context
        planner_info = self._extract_planner_info(resolved_data)
        # Get session ID from the results
        session_id = planner_info["session_id"]
        
        # 3. Connect to database
        conn = sqlite3.connect('db.sqlite3')
        
        # 4. Extract table names
        source_table = planner_info['source_table']
        target_table = planner_info['target_table']
        additional_tables = planner_info.get('additional_sources', [])
        
        # 5. Get source and target dataframes
        source_df = pd.read_sql_query(f"SELECT * FROM {source_table}", conn)
        target_df = get_or_create_session_target_df(session_id, target_table, conn)
        
        # 6. Prepare additional tables if present
        additional_source_tables = None
        if additional_tables:
            additional_source_tables = {}
            for table in additional_tables:
                if table and table.lower() != "none":
                    additional_source_tables[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        
        # Try the data-driven approach for key selection
        try:
            # Select best join keys using data-driven approach
            logger.info("Using data-driven join key selection approach")
            selected_keys = self._select_best_join_keys(query, planner_info, source_df, target_df)
            
            # Generate plan using join keys
            logger.info("Generating plan with join keys")
            simple_plan = self._generate_simple_plan_with_keys(planner_info, selected_keys)
            
            # Generate code with join keys
            logger.info("Generating code with join keys")
            code_content = self._generate_code_from_plan_with_keys(simple_plan, planner_info, selected_keys)
            
        except Exception as e:
            # Fallback to the original approach if data-driven approach fails
            logger.warning(f"Data-driven approach failed: {str(e)}, falling back to original method")
            
            # 7. Generate query classification with context
            query_type = self._classify_query(query, planner_info)
            logger.info(f"Query classified as: {query_type}")
            
            # 8. Generate a simple, step-by-step plan in natural language
            simple_plan = self._generate_simple_plan(query_type, planner_info)
            logger.info(f"Simple plan generated: {simple_plan}")
            
            # 9. Generate code from the simple plan with full context
            code_content = self._generate_code_from_simple_plan(simple_plan, planner_info)
        
        logger.info(f"Code generated: {len(code_content)} chars")
        
        # 10. Execute the generated code
        code_file = create_code_file(code_content, query, is_double=True)
        result = execute_code(code_file, (source_df, target_df), additional_tables=additional_source_tables, is_double=True)
        
        # 11. Save the updated target dataframe if it's a DataFrame
        if isinstance(result, pd.DataFrame):
            result = self.post_proccess_result(result)
            save_session_target_df(session_id, result)
        
        conn.close()
        return code_content, result, session_id
    
    def get_session_info(self, session_id):
        """Get information about a session"""
        context = get_session_context(session_id)
        return {
            "session_id": session_id,
            "context": context,
            "transformation_history": context.get("context", {}).get("transformation_history", []) if context else [],
            "target_table_state": context.get("context", {}).get("target_table_state", {}) if context else {}
        }