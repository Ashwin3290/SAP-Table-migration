"""
Enhancement module for TableLLM.
This file contains new methods and functionality to enhance the TableLLM class.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import sqlite3
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def validate_key_column_operation(planner_info, source_dfs, target_df):
    """
    Validate operations on key columns to prevent incorrect data insertion
    
    Parameters:
    planner_info (dict): Information from the planner
    source_dfs (dict): Dictionary of source dataframes
    target_df (DataFrame): Target dataframe
    
    Returns:
    tuple: (is_valid, message)
    """
    key_columns = planner_info.get('key_columns', [])
    
    if not key_columns:
        return True, "No key columns defined"
    
    # Check if we're trying to insert into key columns
    for key in key_columns:
        # Check if the source data has problems (nulls, duplicates) for this key
        for source_table, source_df in source_dfs.items():
            if key in source_df.columns:
                has_nulls = source_df[key].isna().any()
                has_duplicates = source_df[key].duplicated().any()
                
                if has_nulls or has_duplicates:
                    message = f"Cannot insert into key column '{key}': data contains "
                    if has_nulls:
                        message += "null values"
                    if has_nulls and has_duplicates:
                        message += " and "
                    if has_duplicates:
                        message += "duplicate values"
                    
                    return False, message
    
    return True, "Operation validated"

def format_conversational_response(result, validation_message=None):
    """
    Format a conversational response for the user
    
    Parameters:
    result: The operation result
    validation_message (str): Optional validation message
    
    Returns:
    str: Formatted response
    """
    try:
        if isinstance(result, pd.DataFrame):
            # For dataframe results
            message = f"I've processed your request and updated the data. "
            
            if not result.empty:
                message += f"The result contains {len(result)} rows and {len(result.columns)} columns. "
                
                # Add sample data
                if len(result) > 0:
                    message += "Here's a sample of the result:\n\n"
                    message += result.head(3).to_string()
            else:
                message += "The result is empty. No data was found matching your criteria."
                
        elif isinstance(result, dict) and "error_type" in result:
            # For error results
            message = f"I encountered an error while processing your request: {result['error_message']}"
            
            # Add more details for technical users
            if "traceback" in result:
                message += "\n\nTechnical details:\n" + result["traceback"]
                
        elif isinstance(result, str):
            # For string results
            message = result
            
        else:
            # For other result types
            message = f"Operation completed successfully with result type: {type(result).__name__}"
            
        # Add validation message if provided
        if validation_message:
            message = f"{validation_message}\n\n{message}"
            
        return message
    except Exception as e:
        logger.error(f"Error formatting conversational response: {e}")
        return str(result)

def generate_join_operation_code(parent_table, child_table, key_columns, result_fields, join_type="inner"):
    """
    Generate code for a join operation between parent and child tables
    
    Parameters:
    parent_table (str): Name of parent table
    child_table (str): Name of child table
    key_columns (list): List of key columns for the join
    result_fields (list): List of fields to include in the result
    join_type (str): Type of join (inner, left, right, outer)
    
    Returns:
    str: Generated code
    """
    try:
        # Validate inputs
        if not parent_table or not child_table:
            logger.error("Missing table names for join operation")
            raise ValueError("Parent and child table names are required")
            
        if not key_columns:
            logger.error("Missing key columns for join operation")
            raise ValueError("Key columns are required for join operation")
            
        if not result_fields:
            logger.error("Missing result fields for join operation")
            raise ValueError("Result fields are required")
            
        # Format lists for code
        key_columns_str = json.dumps(key_columns)
        result_fields_str = json.dumps(result_fields)
        
        # Generate the code using the template
        join_code = f"""
def analyze_data(source_dfs, target_df):
    # Import required utilities
    import pandas as pd
    import numpy as np
    
    # Get source table references
    parent_table = '{parent_table}'
    child_table = '{child_table}'
    
    # Get parent and child dataframes
    parent_df = source_dfs[parent_table] if parent_table in source_dfs else pd.DataFrame()
    child_df = source_dfs[child_table] if child_table in source_dfs else pd.DataFrame()
    
    # Check if dataframes are empty
    if parent_df.empty:
        raise ValueError(f"Parent table '{{parent_table}}' is empty or not found")
        
    if child_df.empty:
        raise ValueError(f"Child table '{{child_table}}' is empty or not found")
    
    # Prepare key columns for join
    key_columns = {key_columns_str}
    result_fields = {result_fields_str}
    
    # Perform the join operation
    try:
        joined_df = pd.merge(
            parent_df,
            child_df,
            on=key_columns,
            how='{join_type}'
        )
        
        # Select only the requested fields
        if result_fields:
            # Check if all result fields exist in the joined dataframe
            missing_fields = [field for field in result_fields if field not in joined_df.columns]
            if missing_fields:
                raise ValueError(f"Missing fields in joined result: {{missing_fields}}")
                
            result = joined_df[result_fields].copy()
        else:
            # If no specific fields requested, return all columns
            result = joined_df.copy()
            
        # Update target dataframe with join result
        if target_df.empty:
            # For empty target, use the join result directly
            target_df = result.copy()
        else:
            # For existing target, update or append based on key columns
            # This is a simplified approach - modify as needed for specific requirements
            for key in key_columns:
                if key not in target_df.columns:
                    target_df[key] = None
                    
            # Create a mask for matching rows in target
            if len(key_columns) > 0:
                # For each source row, find matching target row
                for idx, row in result.iterrows():
                    # Create key matching condition
                    mask = pd.Series(True, index=target_df.index)
                    for key in key_columns:
                        mask &= target_df[key] == row[key]
                        
                    # Update existing or append new
                    if mask.any():
                        # Update matching rows
                        for col in result.columns:
                            if col in target_df.columns:
                                target_df.loc[mask, col] = row[col]
                    else:
                        # Append new row
                        new_row = pd.DataFrame([row])
                        target_df = pd.concat([target_df, new_row], ignore_index=True)
            else:
                # Without key columns, just append all results
                target_df = pd.concat([target_df, result], ignore_index=True)
        
        return target_df
        
    except Exception as e:
        raise RuntimeError(f"Error during join operation: {{str(e)}}")
"""
        return join_code
    except Exception as e:
        logger.error(f"Error generating join operation code: {e}")
        return None

def initialize_join_templates():
    """Initialize join templates for TableLLM"""
    return {
        "join": """
# Join operation between parent and child tables
parent_table = '{parent_table}'
child_table = '{child_table}'
key_column = '{key_column}'

# Get parent and child dataframes
parent_df = source_dfs[parent_table] if parent_table in source_dfs else target_df
child_df = source_dfs[child_table] if child_table in source_dfs else target_df

# Perform the join
joined_df = pd.merge(
    parent_df,
    child_df,
    left_on=key_column,
    right_on=key_column,
    how='{join_type}'
)

# Select only the columns we need
result_fields = {result_fields}
result = joined_df[result_fields].copy()

return result
"""
    }

# Additional methods for enhanced TableLLM capabilities can be added here
