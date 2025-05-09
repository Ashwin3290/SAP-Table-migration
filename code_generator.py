import os
import pandas as pd


def generate_exploration_code(df, filename, is_double=False):
    """Generate boilerplate code for exploring a dataframe"""
    
    code_dir = 'generated_code'
    os.makedirs(code_dir, exist_ok=True)
    
    # Create a sanitized filename
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    sanitized_name = "".join(c if c.isalnum() else "_" for c in base_filename)
    
    # Determine numeric, categorical, and date columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Generate code for the appropriate type (single or double table)
    if is_double:
        code = generate_double_table_code(sanitized_name, numeric_cols, categorical_cols, date_cols)
    else:
        code = generate_single_table_code(sanitized_name, numeric_cols, categorical_cols, date_cols)
    
    # Save the code to a file
    output_file = f"{code_dir}/exploration_{sanitized_name}.py"
    with open(output_file, 'w') as f:
        f.write(code)
    
    return output_file


def generate_single_table_code(file_name, numeric_cols, categorical_cols, date_cols):
    """Generate code for single table analysis"""
    
    # Create the imports and function definition
    code = f"""# Auto-generated exploration code for {file_name}
# Generated by TableLLM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(df):
    \"\"\"
    Perform exploratory data analysis on the dataframe
    
    Parameters:
    df (pandas.DataFrame): The dataframe to analyze
    
    Returns:
    dict: Dictionary containing analysis results
    \"\"\"
    # Basic dataframe information
    result = {{}}
    result['shape'] = df.shape
    result['columns'] = df.columns.tolist()
    result['dtypes'] = df.dtypes.to_dict()
    result['missing_values'] = df.isnull().sum().to_dict()
    result['duplicates'] = df.duplicated().sum()
    
    # Summary statistics for numeric columns
    if df.select_dtypes(include=['number']).shape[1] > 0:
        result['numeric_stats'] = df.describe().to_dict()
"""
    
    # Add code for numeric columns
    if numeric_cols:
        code += """
    # Create visualizations for numeric columns
    # Correlation heatmap
    if len(df.select_dtypes(include=['number']).columns) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        result['correlation_heatmap'] = plt.gcf()
        plt.close()
"""
    
    # Add code for categorical columns
    if categorical_cols:
        code += """
    # Categorical column analysis
    categorical_stats = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        categorical_stats[col] = df[col].value_counts().to_dict()
    result['categorical_stats'] = categorical_stats
    
    # Create visualizations for categorical columns
    for col in list(df.select_dtypes(include=['object', 'category']).columns)[:5]:  # Limit to first 5
        plt.figure(figsize=(10, 6))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Value Counts for {col}')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        result[f'{col}_plot'] = plt.gcf()
        plt.close()
"""
    
    # Add code for datetime columns
    if date_cols:
        code += """
    # Time series analysis for date columns
    for col in df.select_dtypes(include=['datetime']).columns:
        if df[col].notna().any():
            plt.figure(figsize=(12, 6))
            df.set_index(col).count(axis=1).plot()
            plt.title(f'Records Over Time ({col})')
            plt.ylabel('Count')
            plt.tight_layout()
            result[f'{col}_timeseries'] = plt.gcf()
            plt.close()
"""
    
    # Add return statement
    code += """
    return result
"""
    
    return code


def generate_double_table_code(file_name, numeric_cols, categorical_cols, date_cols):
    """Generate code for double table analysis"""
    
    code = f"""# Auto-generated exploration code for dual tables based on {file_name}
# Generated by TableLLM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(df1, df2):
    \"\"\"
    Perform comparative analysis between two dataframes
    
    Parameters:
    df1 (pandas.DataFrame): First dataframe
    df2 (pandas.DataFrame): Second dataframe
    
    Returns:
    dict: Dictionary containing analysis results
    \"\"\"
    # Basic information about both dataframes
    result = {{}}
    
    # Compare dataframe shapes
    result['shapes'] = {{
        'df1': df1.shape,
        'df2': df2.shape
    }}
    
    # Compare columns
    df1_cols = set(df1.columns)
    df2_cols = set(df2.columns)
    result['common_columns'] = list(df1_cols.intersection(df2_cols))
    result['unique_to_df1'] = list(df1_cols - df2_cols)
    result['unique_to_df2'] = list(df2_cols - df1_cols)
    
    # Compare data types for common columns
    common_cols = result['common_columns']
    dtypes_comparison = {{}}
    for col in common_cols:
        dtypes_comparison[col] = {{
            'df1': str(df1[col].dtype),
            'df2': str(df2[col].dtype)
        }}
    result['dtypes_comparison'] = dtypes_comparison
    
    # Check for potential join/merge keys
    potential_keys = []
    for col in common_cols:
        if df1[col].nunique() > 0 and df2[col].nunique() > 0:
            unique_ratio1 = df1[col].nunique() / len(df1)
            unique_ratio2 = df2[col].nunique() / len(df2)
            if unique_ratio1 > 0.1 or unique_ratio2 > 0.1:  # Reasonable unique ratio
                potential_keys.append({{
                    'column': col,
                    'unique_values_df1': df1[col].nunique(),
                    'unique_values_df2': df2[col].nunique(),
                    'unique_ratio_df1': unique_ratio1,
                    'unique_ratio_df2': unique_ratio2
                }})
    result['potential_keys'] = potential_keys
    
    # Attempt a basic inner join if there are potential keys
    if potential_keys and len(potential_keys) > 0:
        best_key = potential_keys[0]['column']
        try:
            merged_df = pd.merge(df1, df2, on=best_key, how='inner', suffixes=('_df1', '_df2'))
            result['sample_merge'] = {{
                'key': best_key,
                'merged_shape': merged_df.shape,
                'sample': merged_df.head(5).to_dict()
            }}
        except Exception as e:
            result['sample_merge'] = {{'error': str(e)}}
"""
    
    # Add numeric comparison if there are common numeric columns
    code += """
    # Compare statistics for common numeric columns
    common_numeric_cols = [col for col in common_cols if 
                          col in df1.select_dtypes(include=['number']).columns and 
                          col in df2.select_dtypes(include=['number']).columns]
    
    if common_numeric_cols:
        numeric_comparison = {}
        for col in common_numeric_cols:
            numeric_comparison[col] = {
                'df1_stats': df1[col].describe().to_dict(),
                'df2_stats': df2[col].describe().to_dict()
            }
            
            # Create comparative histogram
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.histplot(df1[col], kde=True)
            plt.title(f'Distribution in DF1: {col}')
            
            plt.subplot(1, 2, 2)
            sns.histplot(df2[col], kde=True)
            plt.title(f'Distribution in DF2: {col}')
            
            plt.tight_layout()
            numeric_comparison[f'{col}_histogram'] = plt.gcf()
            plt.close()
            
        result['numeric_comparison'] = numeric_comparison
"""
    
    # Add categorical comparison if there are common categorical columns
    code += """
    # Compare categorical distributions for common categorical columns
    common_cat_cols = [col for col in common_cols if 
                      col in df1.select_dtypes(include=['object', 'category']).columns and 
                      col in df2.select_dtypes(include=['object', 'category']).columns]
    
    if common_cat_cols:
        cat_comparison = {}
        for col in common_cat_cols:
            df1_counts = df1[col].value_counts().to_dict()
            df2_counts = df2[col].value_counts().to_dict()
            
            cat_comparison[col] = {
                'df1_counts': df1_counts,
                'df2_counts': df2_counts,
                'common_values': set(df1_counts.keys()) & set(df2_counts.keys()),
                'only_in_df1': set(df1_counts.keys()) - set(df2_counts.keys()),
                'only_in_df2': set(df2_counts.keys()) - set(df1_counts.keys())
            }
        
        result['categorical_comparison'] = cat_comparison
"""
    
    # Add return statement
    code += """
    return result
"""
    
    return code