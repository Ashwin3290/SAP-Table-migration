"""
Transformation Pattern Library for TableLLM
This module provides a library of common transformation patterns
"""
import re
import json
from utils.logging_utils import main_logger as logger
from utils.token_utils import track_token_usage
import config

class PatternLibrary:
    """
    Library of common SAP transformation patterns
    """
    
    def __init__(self, client=None):
        """
        Initialize the pattern library
        
        Parameters:
        client (genai.Client, optional): LLM client for pattern matching
        """
        # Store client for pattern matching
        self.client = client
        
        # Initialize patterns
        self.patterns = self._initialize_patterns()
        
        logger.info(f"Initialized PatternLibrary with {len(self.patterns)} patterns")
    
    def _initialize_patterns(self):
        """
        Initialize common transformation patterns
        
        Returns:
        dict: Dictionary of transformation patterns
        """
        return {
            "field_mapping": {
                "description": "Simple field mapping from source to target",
                "example": """
# Map fields from source to target dataframe
def analyze_data(source_dfs, target_df):
    # Import required utilities
    import pandas as pd
    
    # Get source dataframe
    source_df = source_dfs['SOURCE_TABLE']
    
    # Check if target dataframe is empty
    if len(target_df) == 0:
        # Create new dataframe with mapped fields
        result_df = pd.DataFrame()
        result_df['TARGET_FIELD'] = source_df['SOURCE_FIELD']
        return result_df
    else:
        # Update existing records
        target_df['TARGET_FIELD'] = source_df['SOURCE_FIELD']
        return target_df
"""
            },
            "filter_and_extract": {
                "description": "Filter records based on condition and extract fields",
                "example": """
# Filter source data and extract fields
def analyze_data(source_dfs, target_df):
    # Import required utilities
    import pandas as pd
    
    # Get source dataframe
    source_df = source_dfs['SOURCE_TABLE']
    
    # Filter records based on condition
    filtered_df = source_df[source_df['FILTER_FIELD'] == 'FILTER_VALUE'].copy()
    
    # Check if target dataframe is empty
    if len(target_df) == 0:
        # Create new dataframe with extracted fields
        result_df = pd.DataFrame()
        result_df['TARGET_FIELD1'] = filtered_df['SOURCE_FIELD1']
        result_df['TARGET_FIELD2'] = filtered_df['SOURCE_FIELD2']
        return result_df
    else:
        # Update existing records based on key
        for _, row in filtered_df.iterrows():
            # Find matching rows in target
            mask = target_df['KEY_FIELD'] == row['KEY_FIELD']
            if mask.any():
                # Update existing rows
                target_df.loc[mask, 'TARGET_FIELD1'] = row['SOURCE_FIELD1']
                target_df.loc[mask, 'TARGET_FIELD2'] = row['SOURCE_FIELD2']
            else:
                # Add new row
                new_row = pd.DataFrame({
                    'KEY_FIELD': [row['KEY_FIELD']],
                    'TARGET_FIELD1': [row['SOURCE_FIELD1']],
                    'TARGET_FIELD2': [row['SOURCE_FIELD2']]
                })
                target_df = pd.concat([target_df, new_row], ignore_index=True)
        
        return target_df
"""
            },
            "conditional_mapping": {
                "description": "Apply if/else logic to determine values",
                "example": """
# Apply conditional mapping to determine values
def analyze_data(source_dfs, target_df):
    # Import required utilities
    import pandas as pd
    import numpy as np
    
    # Get source dataframe
    source_df = source_dfs['SOURCE_TABLE']
    
    # Define conditions and corresponding values
    conditions = [
        source_df['CONDITION_FIELD'].isin(['VALUE1', 'VALUE2']),
        source_df['CONDITION_FIELD'].isin(['VALUE3']),
        source_df['CONDITION_FIELD'].isin(['VALUE4', 'VALUE5'])
    ]
    
    values = ['RESULT1', 'RESULT2', 'RESULT3']
    
    # Apply conditional logic using np.select
    result_values = np.select(conditions, values, default='DEFAULT_VALUE')
    
    # Check if target dataframe is empty
    if len(target_df) == 0:
        # Create new dataframe with results
        result_df = pd.DataFrame()
        result_df['KEY_FIELD'] = source_df['KEY_FIELD']
        result_df['TARGET_FIELD'] = result_values
        return result_df
    else:
        # Update existing records based on key
        for i, row in source_df.iterrows():
            # Find matching rows in target
            mask = target_df['KEY_FIELD'] == row['KEY_FIELD']
            if mask.any():
                # Update existing rows
                target_df.loc[mask, 'TARGET_FIELD'] = result_values[i]
        
        return target_df
"""
            },
            "tiered_lookup": {
                "description": "Look up values in multiple tables in sequence",
                "example": """
# Perform tiered lookup across multiple tables
def analyze_data(source_dfs, target_df):
    # Import required utilities
    import pandas as pd
    
    # Function to get value from tiered tables
    def get_from_tiered_lookup(key_value, field_name):
        # First check in PRIMARY_TABLE
        if 'PRIMARY_TABLE' in source_dfs:
            primary_df = source_dfs['PRIMARY_TABLE']
            if field_name in primary_df.columns:
                matches = primary_df[primary_df['KEY_FIELD'] == key_value]
                if not matches.empty:
                    return matches[field_name].iloc[0]
        
        # Then check in SECONDARY_TABLE
        if 'SECONDARY_TABLE' in source_dfs:
            secondary_df = source_dfs['SECONDARY_TABLE']
            if field_name in secondary_df.columns:
                matches = secondary_df[secondary_df['KEY_FIELD'] == key_value]
                if not matches.empty:
                    return matches[field_name].iloc[0]
        
        # Finally check in FALLBACK_TABLE
        if 'FALLBACK_TABLE' in source_dfs:
            fallback_df = source_dfs['FALLBACK_TABLE']
            if field_name in fallback_df.columns:
                matches = fallback_df[fallback_df['KEY_FIELD'] == key_value]
                if not matches.empty:
                    return matches[field_name].iloc[0]
        
        # Return default value if not found
        return None
    
    # Get key values to process
    if 'MAIN_TABLE' in source_dfs:
        key_values = source_dfs['MAIN_TABLE']['KEY_FIELD'].unique()
    else:
        # Use first available table
        first_table = list(source_dfs.keys())[0]
        key_values = source_dfs[first_table]['KEY_FIELD'].unique()
    
    # Process each key value
    result_data = {'KEY_FIELD': [], 'TARGET_FIELD': []}
    for key in key_values:
        result_data['KEY_FIELD'].append(key)
        result_data['TARGET_FIELD'].append(get_from_tiered_lookup(key, 'SOURCE_FIELD'))
    
    # Create result dataframe
    result_df = pd.DataFrame(result_data)
    
    # Merge with target if not empty
    if len(target_df) == 0:
        return result_df
    else:
        # Update based on key field
        merged_df = pd.merge(
            target_df, 
            result_df, 
            on='KEY_FIELD', 
            how='left',
            suffixes=('', '_new')
        )
        
        # Update target field with new values where available
        mask = merged_df['TARGET_FIELD_new'].notna()
        merged_df.loc[mask, 'TARGET_FIELD'] = merged_df.loc[mask, 'TARGET_FIELD_new']
        
        # Drop the temporary column
        merged_df = merged_df.drop('TARGET_FIELD_new', axis=1)
        
        return merged_df
"""
            },
            "string_cleansing": {
                "description": "Clean text data by removing special characters",
                "example": """
# Clean text data by removing special characters
def analyze_data(source_dfs, target_df):
    # Import required utilities
    import pandas as pd
    import re
    
    # Get source dataframe
    source_df = source_dfs['SOURCE_TABLE']
    
    # Function to remove special characters
    def clean_text(text):
        if pd.isna(text):
            return text
        # Remove special characters
        return re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    
    # Apply cleaning to source field
    cleaned_values = source_df['SOURCE_FIELD'].apply(clean_text)
    
    # Check if target dataframe is empty
    if len(target_df) == 0:
        # Create new dataframe with cleaned values
        result_df = pd.DataFrame()
        result_df['KEY_FIELD'] = source_df['KEY_FIELD']
        result_df['TARGET_FIELD'] = cleaned_values
        return result_df
    else:
        # Update existing records based on key
        for i, row in source_df.iterrows():
            # Find matching rows in target
            mask = target_df['KEY_FIELD'] == row['KEY_FIELD']
            if mask.any():
                # Update existing rows
                target_df.loc[mask, 'TARGET_FIELD'] = cleaned_values.iloc[i]
        
        return target_df
"""
            },
            "join_tables": {
                "description": "Join data from multiple tables",
                "example": """
# Join data from multiple tables
def analyze_data(source_dfs, target_df):
    # Import required utilities
    import pandas as pd
    
    # Get source dataframes
    primary_df = source_dfs['PRIMARY_TABLE']
    secondary_df = source_dfs['SECONDARY_TABLE']
    
    # Perform join operation
    joined_df = pd.merge(
        primary_df,
        secondary_df,
        left_on='PRIMARY_KEY',
        right_on='SECONDARY_KEY',
        how='inner'
    )
    
    # Extract relevant fields
    result_data = {
        'KEY_FIELD': joined_df['PRIMARY_KEY'],
        'FIELD1': joined_df['PRIMARY_FIELD'],
        'FIELD2': joined_df['SECONDARY_FIELD']
    }
    
    # Create result dataframe
    result_df = pd.DataFrame(result_data)
    
    # Merge with target if not empty
    if len(target_df) == 0:
        return result_df
    else:
        # Update based on key field
        for _, row in result_df.iterrows():
            # Find matching rows in target
            mask = target_df['KEY_FIELD'] == row['KEY_FIELD']
            if mask.any():
                # Update existing rows
                target_df.loc[mask, 'FIELD1'] = row['FIELD1']
                target_df.loc[mask, 'FIELD2'] = row['FIELD2']
            else:
                # Add new row
                new_row = pd.DataFrame({
                    'KEY_FIELD': [row['KEY_FIELD']],
                    'FIELD1': [row['FIELD1']],
                    'FIELD2': [row['FIELD2']]
                })
                target_df = pd.concat([target_df, new_row], ignore_index=True)
        
        return target_df
"""
            },
            "length_calculation": {
                "description": "Calculate length of text fields and conditionally set values",
                "example": """
# Calculate length of text fields and set conditional values
def analyze_data(source_dfs, target_df):
    # Import required utilities
    import pandas as pd
    
    # Get source dataframe
    source_df = source_dfs['SOURCE_TABLE']
    
    # Calculate lengths of text field
    text_lengths = source_df['TEXT_FIELD'].astype(str).apply(len)
    
    # Apply conditional logic based on length
    result_values = ['G' if length > 30 else 'L' for length in text_lengths]
    
    # Check if target dataframe is empty
    if len(target_df) == 0:
        # Create new dataframe with results
        result_df = pd.DataFrame()
        result_df['KEY_FIELD'] = source_df['KEY_FIELD'] 
        result_df['LENGTH_FIELD'] = text_lengths
        result_df['INDICATOR_FIELD'] = result_values
        return result_df
    else:
        # Update existing records
        for i, row in source_df.iterrows():
            # Find matching rows in target
            mask = target_df['KEY_FIELD'] == row['KEY_FIELD']
            if mask.any():
                # Update existing rows
                target_df.loc[mask, 'LENGTH_FIELD'] = text_lengths.iloc[i]
                target_df.loc[mask, 'INDICATOR_FIELD'] = result_values[i]
        
        return target_df
"""
            },
            "validation_check": {
                "description": "Validate data against reference tables",
                "example": """
# Validate data against reference tables
def analyze_data(source_dfs, target_df):
    # Import required utilities
    import pandas as pd
    
    # Get source and reference dataframes
    source_df = source_dfs['SOURCE_TABLE']
    reference_df = source_dfs['REFERENCE_TABLE']
    
    # Create validation function
    def validate_value(value, validation_field):
        # Check if value exists in reference table
        if value in reference_df[validation_field].values:
            return 'Valid'
        else:
            return 'Invalid'
    
    # Apply validation to each value
    validation_results = source_df['SOURCE_FIELD'].apply(
        lambda x: validate_value(x, 'REFERENCE_FIELD')
    )
    
    # Check if target dataframe is empty
    if len(target_df) == 0:
        # Create new dataframe with validation results
        result_df = pd.DataFrame()
        result_df['KEY_FIELD'] = source_df['KEY_FIELD']
        result_df['VALIDATION_FIELD'] = validation_results
        return result_df
    else:
        # Update existing records
        for i, row in source_df.iterrows():
            # Find matching rows in target
            mask = target_df['KEY_FIELD'] == row['KEY_FIELD']
            if mask.any():
                # Update existing rows
                target_df.loc[mask, 'VALIDATION_FIELD'] = validation_results.iloc[i]
        
        return target_df
"""
            },
            "language_specific_extraction": {
                "description": "Extract data based on language key",
                "example": """
# Extract data based on language key
def analyze_data(source_dfs, target_df):
    # Import required utilities
    import pandas as pd
    
    # Get source dataframe
    source_df = source_dfs['SOURCE_TABLE']
    
    # Filter for specific language
    language_filter = source_df['LANGUAGE_FIELD'] == 'LANGUAGE_VALUE'
    filtered_df = source_df[language_filter].copy()
    
    # Check if target dataframe is empty
    if len(target_df) == 0:
        # Create new dataframe with extracted fields
        result_df = pd.DataFrame()
        result_df['KEY_FIELD'] = filtered_df['KEY_FIELD']
        result_df['DESCRIPTION_FIELD'] = filtered_df['DESCRIPTION_FIELD']
        return result_df
    else:
        # Update existing records based on key
        for _, row in filtered_df.iterrows():
            # Find matching rows in target
            mask = target_df['KEY_FIELD'] == row['KEY_FIELD']
            if mask.any():
                # Update existing rows
                target_df.loc[mask, 'DESCRIPTION_FIELD'] = row['DESCRIPTION_FIELD']
        
        return target_df
"""
            },
            "multi_step_transformation": {
                "description": "Perform multiple transformation steps in sequence",
                "example": """
# Perform multiple transformation steps in sequence
def analyze_data(source_dfs, target_df):
    # Import required utilities
    import pandas as pd
    import numpy as np
    import re
    
    # Get source dataframe
    source_df = source_dfs['SOURCE_TABLE']
    
    # Step 1: Filter records
    filtered_df = source_df[source_df['FILTER_FIELD'] == 'FILTER_VALUE'].copy()
    
    # Step 2: Clean text fields
    filtered_df['CLEAN_TEXT'] = filtered_df['TEXT_FIELD'].apply(
        lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)) if pd.notna(x) else x
    )
    
    # Step 3: Apply conditional logic
    conditions = [
        filtered_df['CONDITION_FIELD'].isin(['VALUE1', 'VALUE2']),
        filtered_df['CONDITION_FIELD'].isin(['VALUE3'])
    ]
    choices = ['RESULT1', 'RESULT2']
    filtered_df['RESULT_FIELD'] = np.select(conditions, choices, default='DEFAULT')
    
    # Check if target dataframe is empty
    if len(target_df) == 0:
        # Create new dataframe with results
        result_df = pd.DataFrame()
        result_df['KEY_FIELD'] = filtered_df['KEY_FIELD']
        result_df['CLEAN_TEXT_FIELD'] = filtered_df['CLEAN_TEXT']
        result_df['RESULT_FIELD'] = filtered_df['RESULT_FIELD']
        return result_df
    else:
        # Update existing records based on key
        for _, row in filtered_df.iterrows():
            # Find matching rows in target
            mask = target_df['KEY_FIELD'] == row['KEY_FIELD']
            if mask.any():
                # Update existing rows
                target_df.loc[mask, 'CLEAN_TEXT_FIELD'] = row['CLEAN_TEXT']
                target_df.loc[mask, 'RESULT_FIELD'] = row['RESULT_FIELD']
        
        return target_df
"""
            }
        }
    
    def get_pattern(self, pattern_name):
        """
        Get a pattern by name
        
        Parameters:
        pattern_name (str): Pattern name
        
        Returns:
        dict: Pattern details or None if not found
        """
        return self.patterns.get(pattern_name)
    
    def get_all_patterns(self):
        """
        Get all available patterns
        
        Returns:
        dict: Dictionary of all patterns
        """
        return self.patterns
    
    @track_token_usage()
    def get_matching_patterns(self, query, transformation_type=None):
        """
        Find patterns that match a query and transformation type
        Uses LLM for intelligent pattern matching
        
        Parameters:
        query (str): Natural language query
        transformation_type (str, optional): Transformation type from intent agent
        
        Returns:
        list: List of matching pattern dictionaries
        """
        # If no client is available, use simple matching
        if not self.client:
            return self._simple_pattern_matching(query, transformation_type)
        
        try:
            # Create prompt for pattern matching
            prompt = f"""
You are a data transformation pattern expert. Based on the query below, identify the most appropriate transformation patterns.

QUERY: {query}

TRANSFORMATION TYPE: {transformation_type if transformation_type else "Not specified"}

AVAILABLE PATTERNS:
{json.dumps({name: details["description"] for name, details in self.patterns.items()}, indent=2)}

Select up to 3 patterns that would be most useful for implementing this transformation. Consider:
1. The specific operations mentioned in the query
2. The data manipulation requirements
3. The complexity of the transformation

Return ONLY a JSON array with the names of the selected patterns, in order of relevance:
["pattern1", "pattern2", "pattern3"]
"""
            
            # Call LLM
            response = self.client.models.generate_content(
                model=config.PLANNING_MODEL,
                contents=prompt,
                config={"temperature": 0.2}
            )
            
            # Parse response
            if hasattr(response, 'text'):
                # Extract JSON array
                pattern_match = re.search(r'\[.*\]', response.text, re.DOTALL)
                if pattern_match:
                    try:
                        pattern_names = json.loads(pattern_match.group(0))
                        
                        # Validate and get patterns
                        result = []
                        for name in pattern_names:
                            if name in self.patterns:
                                result.append({
                                    "name": name,
                                    "description": self.patterns[name]["description"],
                                    "example": self.patterns[name]["example"]
                                })
                        
                        return result
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse pattern matching response: {response.text}")
            
            # Fallback to simple matching if LLM fails
            return self._simple_pattern_matching(query, transformation_type)
        except Exception as e:
            logger.error(f"Error in get_matching_patterns: {e}")
            return self._simple_pattern_matching(query, transformation_type)
    
    def _simple_pattern_matching(self, query, transformation_type=None):
        """
        Simple pattern matching based on keywords
        
        Parameters:
        query (str): Natural language query
        transformation_type (str, optional): Transformation type from intent agent
        
        Returns:
        list: List of matching pattern dictionaries
        """
        # Define pattern keywords
        pattern_keywords = {
            "field_mapping": ["map", "copy", "extract", "field", "column"],
            "filter_and_extract": ["filter", "where", "condition", "extract", "specific"],
            "conditional_mapping": ["if", "else", "condition", "case", "when", "depending"],
            "tiered_lookup": ["check", "lookup", "sequence", "first", "then", "else"],
            "string_cleansing": ["clean", "remove", "special", "character", "text"],
            "join_tables": ["join", "combine", "multiple", "tables"],
            "length_calculation": ["length", "calculate", "text", "size"],
            "validation_check": ["validate", "check", "reference", "valid"],
            "language_specific_extraction": ["language", "specific", "extraction", "spras"],
            "multi_step_transformation": ["multi", "step", "complex", "sequence"]
        }
        
        # Type-based pattern recommendations
        type_patterns = {
            "FILTER_AND_EXTRACT": ["filter_and_extract", "field_mapping"],
            "UPDATE_EXISTING": ["field_mapping", "conditional_mapping"],
            "CONDITIONAL_MAPPING": ["conditional_mapping", "tiered_lookup"],
            "EXTRACTION": ["field_mapping", "join_tables"],
            "TIERED_LOOKUP": ["tiered_lookup", "conditional_mapping"],
            "AGGREGATION": ["join_tables", "multi_step_transformation"],
            "JOIN": ["join_tables", "field_mapping"],
            "VALIDATION": ["validation_check", "field_mapping"],
            "CLEANSING": ["string_cleansing", "length_calculation"]
        }
        
        # Count keyword matches for each pattern
        scores = {pattern: 0 for pattern in self.patterns.keys()}
        query_lower = query.lower()
        
        for pattern, keywords in pattern_keywords.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    scores[pattern] += 1
        
        # Add type-based recommendations
        if transformation_type and transformation_type in type_patterns:
            for pattern in type_patterns[transformation_type]:
                scores[pattern] += 2  # Give higher weight to type-based patterns
        
        # Sort patterns by score
        sorted_patterns = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 3 patterns with score > 0
        result = []
        for name, score in sorted_patterns:
            if score > 0 and len(result) < 3:
                result.append({
                    "name": name,
                    "description": self.patterns[name]["description"],
                    "example": self.patterns[name]["example"]
                })
        
        # If no matches found, return top 2 patterns
        if not result:
            for name, _ in sorted_patterns[:2]:
                result.append({
                    "name": name,
                    "description": self.patterns[name]["description"],
                    "example": self.patterns[name]["example"]
                })
        
        return result
