"""
Enhancement module for code_exec.py.
This file contains new functions to enhance the code execution module.
"""

import os
import sys
import importlib.util
import pandas as pd
import numpy as np
import inspect
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import classes from code_exec to avoid circular imports
from code_exec import ValidationError

def execute_code_multi_target(file_path, source_dfs, target_dfs, target_sap_fields=None):
    """
    Execute a Python file with support for multiple target tables
    
    Parameters:
    file_path (str): Path to the Python file to execute
    source_dfs (dict): Dictionary of source dataframes
    target_dfs (dict): Dictionary of target dataframes
    target_sap_fields (dict): Dictionary of target SAP fields by table
    
    Returns:
    dict/object: Result of the code execution or error information
    """
    try:
        # Add the current directory to sys.path to ensure utilities can be imported
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import the module
        module_name = os.path.basename(file_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check for different function signatures
        sig = inspect.signature(module.analyze_data)
        param_count = len(sig.parameters)
        
        # Execute with appropriate arguments based on function signature
        if param_count == 3:  # source_dfs, target_dfs, target_sap_fields
            result = module.analyze_data(source_dfs, target_dfs, target_sap_fields)
        elif param_count == 2:  # source_dfs, target_df
            # Handle legacy signature
            if isinstance(target_dfs, dict):
                # Use the first target dataframe
                target_df = next(iter(target_dfs.values())) if target_dfs else pd.DataFrame()
            else:
                target_df = target_dfs
                
            result = module.analyze_data(source_dfs, target_df)
        else:
            return {
                "error_type": "FunctionSignatureError",
                "error_message": f"Unexpected function signature: {param_count} parameters",
                "traceback": "The analyze_data function should have 2 or 3 parameters"
            }
        
        # Validate the result
        if isinstance(result, dict) and all(isinstance(df, pd.DataFrame) for df in result.values()):
            # Multi-table result
            for table_name, df in result.items():
                # Run validation for each table
                if isinstance(target_sap_fields, dict) and table_name in target_sap_fields:
                    enhanced_validation_handling(source_dfs, target_dfs.get(table_name, pd.DataFrame()), 
                                       df, target_sap_fields[table_name])
        elif isinstance(result, pd.DataFrame):
            # Single table result
            if isinstance(target_dfs, dict):
                # Use the first target dataframe for validation
                target_df = next(iter(target_dfs.values())) if target_dfs else pd.DataFrame()
            else:
                target_df = target_dfs
                
            # Use the first target SAP field for validation
            target_field = target_sap_fields
            if isinstance(target_sap_fields, dict):
                target_field = next(iter(target_sap_fields.values())) if target_sap_fields else None
                
            enhanced_validation_handling(source_dfs, target_df, result, target_field)
            
        return result
    except Exception as e:
        # Capture the full traceback with detailed information
        error_traceback = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "code_file": file_path
        }
        
        # Return the detailed error information
        return error_traceback
    finally:
        # Restore sys.path
        if os.path.dirname(os.path.abspath(__file__)) in sys.path:
            sys.path.remove(os.path.dirname(os.path.abspath(__file__)))


def enhanced_validation_handling(source_dfs, target_df, result, target_sap_fields):
    """
    Enhanced validation handling with better error messages
    
    Parameters:
    source_dfs (dict): Dictionary of source dataframes
    target_df (DataFrame): Target dataframe
    result (DataFrame): Result dataframe
    target_sap_fields (str/list): Target SAP fields
    
    Raises:
    ValidationError: If validation fails
    """
    error_msg = ""
    
    # Check 1: Length validation - target_df should never be smaller than result
    if len(target_df) < len(result) and len(target_df) != 0:
        error_msg += f"Error: Target dataframe has fewer rows than result. Target: {len(target_df)}, Result: {len(result)}\n"
    
    # Check 2: Column existence
    if isinstance(target_sap_fields, list):
        missing_fields = [field for field in target_sap_fields if field not in result.columns]
        if missing_fields:
            error_msg += f"Error: Required fields are missing in the result: {missing_fields}\n"
    elif target_sap_fields and target_sap_fields not in result.columns:
        error_msg += f"Error: Required field '{target_sap_fields}' is missing in the result\n"
    
    # Get non-null columns in both dataframes
    target_not_null_columns = set(target_df.columns[target_df.notna().any()].tolist())
    result_not_null_columns = set(result.columns[result.notna().any()].tolist())
    
    # For debugging purposes
    print("Target non-null columns:", target_not_null_columns)
    print("#########################################")
    print("Result non-null columns:", result_not_null_columns)
    print("#########################################")
    
    # Check if the target_sap_field should be included
    expected_not_null_columns = target_not_null_columns.copy()
    
    # Handle multiple target fields
    target_fields = []
    if isinstance(target_sap_fields, list):
        target_fields = target_sap_fields
    elif target_sap_fields:
        target_fields = [target_sap_fields]
        
    for field in target_fields:
        if field in target_df.columns:
            # If the field has non-null values or isn't already in the not-null columns,
            # we add it to our expected non-null columns
            if field not in target_not_null_columns or target_df[field].notna().any():
                expected_not_null_columns.add(field)
    
    # Now validate that result has the expected non-null columns
    if result_not_null_columns != expected_not_null_columns:
        # Find specific differences
        missing_in_result = expected_not_null_columns - result_not_null_columns
        if missing_in_result:
            error_msg += f"Error: Expected non-null columns missing in result: {missing_in_result}\n"
        
        unexpected_in_result = result_not_null_columns - expected_not_null_columns
        if unexpected_in_result:
            error_msg += f"Error: Unexpected non-null columns in result: {unexpected_in_result}\n"
        
        # Even if the counts match but the columns are different, it's still an error
        if len(result_not_null_columns) == len(expected_not_null_columns):
            error_msg += "Error: Result has the same number of non-null columns as expected, but they are different columns\n"
    else:
        print("Validation passed: Result contains the correct set of non-null columns")
    
    # Check 3: Data type validation
    for field in target_fields:
        if field in target_df.columns and field in result.columns:
            target_dtype = target_df[field].dtype
            result_dtype = result[field].dtype
            
            # Check if the data types are compatible
            try:
                # Try to cast one to the other
                sample = result[field].iloc[0] if not result.empty else None
                if sample is not None:
                    target_df[field].iloc[0] if not target_df.empty else pd.Series([sample], dtype=target_dtype)
            except Exception:
                error_msg += f"Error: Data type mismatch for field '{field}'. Target: {target_dtype}, Result: {result_dtype}\n"
    
    if error_msg:
        raise ValidationError(f"Validation errors:\n{error_msg}")

# Wrapper for the existing execute_code function to use enhanced validation
def enhanced_execute_code(file_path, source_dfs, target_df, target_sap_fields):
    """
    Enhanced version of execute_code with better validation and multi-table support
    
    Parameters:
    file_path (str): Path to the Python file to execute
    source_dfs (dict): Dictionary of source dataframes
    target_df (DataFrame/dict): Target dataframe or dictionary of target dataframes
    target_sap_fields (str/list/dict): Target SAP fields
    
    Returns:
    object: Result of the code execution or error information
    """
    # Check if we're dealing with multiple targets
    if isinstance(target_df, dict):
        return execute_code_multi_target(file_path, source_dfs, target_df, target_sap_fields)
        
    try:
        # Add the current directory to sys.path to ensure utilities can be imported
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import the module
        module_name = os.path.basename(file_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        result = module.analyze_data(source_dfs, target_df)
        enhanced_validation_handling(source_dfs, target_df, result, target_sap_fields)
        
        # Update the target dataframe with the results
        if isinstance(target_sap_fields, list):
            for field in target_sap_fields:
                if field in result.columns:
                    target_df[field] = result[field]
        elif target_sap_fields and target_sap_fields in result.columns:
            target_df[target_sap_fields] = result[target_sap_fields]
            
        return target_df
    except Exception as e:
        # Capture the full traceback with detailed information
        error_traceback = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "code_file": file_path
        }
        
        # Return the detailed error information
        return error_traceback
    finally:
        # Restore sys.path
        if os.path.dirname(os.path.abspath(__file__)) in sys.path:
            sys.path.remove(os.path.dirname(os.path.abspath(__file__)))
