import os
import sys
import uuid
import tempfile
import importlib.util
import pandas as pd
import matplotlib.pyplot as plt
from docx import Document
from io import StringIO
import sqlite3
import json

def validation_handling(target_df, result, target_sap_fields):
    error_msg = ""
    
    # Check 1: Length validation - target_df should never be smaller than result
    if len(target_df) < len(result) or len(target_df) == 0:
        error_msg += f"Error: Target dataframe has fewer rows than result. Target: {len(target_df)}, Result: {len(result)}\n"
    
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
    if target_sap_fields in target_df.columns:
        # If the target_sap_field has non-null values or isn't already in the not-null columns,
        # we add it to our expected non-null columns
        if target_sap_fields not in target_not_null_columns or target_df[target_sap_fields].notna().any():
            expected_not_null_columns.add(target_sap_fields)
    
    # Now validate that result has exactly the expected non-null columns
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
    
    return error_msg if error_msg else "All validations passed successfully"

if __name__=="__main__":
    target_df = pd.read_csv("D:/TableLLM-test/sessions/dd9f4899-ed00-470e-8d4e-e2f27048100d/target_20250415171722.csv")
    result = pd.read_csv("D:/TableLLM-test/sessions/dd9f4899-ed00-470e-8d4e-e2f27048100d/target_20250415172517.csv")
    target_sap_fields = "MEINS"
    print(validation_handling(target_df,result,target_sap_fields)) # Example SAP fields