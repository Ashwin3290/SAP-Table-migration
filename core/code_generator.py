"""
Code Generator for TableLLM
This module generates Python code for data transformations
"""
import re
import json
import traceback
from utils.logging_utils import main_logger as logger
from utils.token_utils import track_token_usage
import config

class CodeGenerator:
    """
    Generates code for data transformations with pattern matching
    """
    
    def __init__(self, client=None, pattern_library=None):
        """
        Initialize the code generator
        
        Parameters:
        client (genai.Client, optional): LLM client for code generation
        pattern_library (PatternLibrary, optional): Library of transformation patterns
        """
        # Store client
        self.client = client
        
        # Store pattern library
        self.pattern_library = pattern_library
        
        logger.info("Initialized CodeGenerator")
    
    @track_token_usage()
    def generate_code(self, query, extracted_info, patterns=None):
        """
        Generate code for transformation
        
        Parameters:
        query (str): Original query
        extracted_info (dict): Information from all agents
        patterns (list, optional): Specific patterns to use
        
        Returns:
        str: Generated Python code
        """
        try:
            # Get matching patterns if not provided
            if not patterns and self.pattern_library:
                intent_info = extracted_info.get("intent_info", {})
                intent_type = intent_info.get("intent_type") if intent_info else None
                patterns = self.pattern_library.get_matching_patterns(query, intent_type)
            
            # Create pattern examples string if patterns available
            pattern_examples = ""
            if patterns:
                pattern_examples = "\n\n".join([
                    f"PATTERN: {p['name']} - {p['description']}\nEXAMPLE:\n{p['example']}"
                    for p in patterns
                ])
            
            # Prepare data for prompt
            intent_info = extracted_info.get("intent_info", {})
            table_info = extracted_info.get("table_info", {})
            segment_info = extracted_info.get("segment_info", {})
            column_info = extracted_info.get("column_info", {})
            
            # Format source tables information
            source_tables = table_info.get("source_tables", [])
            source_tables_str = json.dumps(source_tables) if source_tables else "[]"
            
            # Format target table information
            target_table = table_info.get("target_table")
            target_table_str = f'"{target_table}"' if target_table else "None"
            
            # Format column information for each table
            column_info_formatted = {}
            for table, cols in column_info.items():
                if isinstance(cols, dict) and "required" in cols and "optional" in cols:
                    column_info_formatted[table] = {
                        "required": cols["required"],
                        "optional": cols["optional"]
                    }
            
            # Create comprehensive prompt
            prompt = f"""
You are an expert Python developer specializing in data transformation scripts for SAP data. Your task is to write a Python function that implements the following transformation.

ORIGINAL QUERY: {query}

TRANSFORMATION TYPE: {intent_info.get('intent_type', 'Not specified')}

SOURCE TABLES: {source_tables_str}
TARGET TABLE: {target_table_str}

SEGMENT INFORMATION:
{json.dumps(segment_info, indent=2) if segment_info else "Not available"}

REQUIRED COLUMNS FOR EACH TABLE:
{json.dumps(column_info_formatted, indent=2) if column_info_formatted else "Not available"}

{pattern_examples if pattern_examples else ""}

REQUIREMENTS:
1. Your function must be named 'analyze_data' and take exactly two parameters:
   - source_dfs: Dictionary where keys are table names and values are pandas DataFrames
   - target_df: The target DataFrame to be updated

2. Basic requirements:
   - Import only necessary packages (pandas, numpy, re)
   - Return the target_df after all transformations are applied
   - The function should correctly handle both empty and non-empty target DataFrames
   - Use descriptive variable names and include basic comments
   - Do not print intermediate results - return only the final DataFrame

3. Look for specific implementation requirements in the query, including:
   - Filtering conditions
   - Field mappings
   - Conditional logic
   - Multiple table lookups
   - Text transformations
   - Tiered lookup logic

4. Exception handling:
   - Validate that required fields exist before using them
   - Handle missing or empty tables gracefully
   - Check dataframe is not empty before performing operations

Please generate only a Python function. Do not include any explanation text, just the 'analyze_data' function itself.

def analyze_data(source_dfs, target_df):
    # Import required packages
    import pandas as pd
    import numpy as np
    
"""
            
            # Call LLM for code generation
            if not self.client:
                logger.warning("No LLM client available for code generation")
                return self._generate_fallback_code(extracted_info)
            
            response = self.client.models.generate_content(
                model=config.GENERATION_MODEL,
                contents=prompt
            )
            
            # Parse response
            if hasattr(response, 'text'):
                # Extract function definition
                code = self._extract_code(response.text)
                if code:
                    return code
            
            # Fallback to template if LLM fails
            logger.warning("LLM code generation failed, using fallback template")
            return self._generate_fallback_code(extracted_info)
        except Exception as e:
            logger.error(f"Error in generate_code: {e}")
            logger.error(traceback.format_exc())
            return self._generate_fallback_code(extracted_info)
    
    def _extract_code(self, response_text):
        """
        Extract Python code from LLM response
        
        Parameters:
        response_text (str): LLM response text
        
        Returns:
        str: Extracted code or None if extraction fails
        """
        try:
            # First try to extract code between triple backticks
            code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", response_text, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            
            # If that fails, try to extract the function definition directly
            function_match = re.search(r"def\s+analyze_data\s*\(.*?\).*?(?:return\s+.*?)?$", response_text, re.DOTALL)
            if function_match:
                return function_match.group(0).strip()
            
            # If no function definition found, return the whole response as a fallback
            return response_text.strip()
        except Exception as e:
            logger.error(f"Error extracting code: {e}")
            return None
    
    def _generate_fallback_code(self, extracted_info):
        """
        Generate fallback code when LLM generation fails
        
        Parameters:
        extracted_info (dict): Information from all agents
        
        Returns:
        str: Fallback Python code
        """
        try:
            # Get table and column information
            table_info = extracted_info.get("table_info", {})
            column_info = extracted_info.get("column_info", {})
            
            # Get source and target tables
            source_tables = table_info.get("source_tables", [])
            target_table = table_info.get("target_table")
            
            # Choose first source table as primary if available
            primary_table = source_tables[0] if source_tables else "unknown_table"
            
            # Get required columns for primary table if available
            source_columns = []
            if primary_table in column_info and isinstance(column_info[primary_table], dict):
                source_columns = column_info[primary_table].get("required", [])
            
            # Get required columns for target table if available
            target_columns = []
            if target_table in column_info and isinstance(column_info[target_table], dict):
                target_columns = column_info[target_table].get("required", [])
            
            # Build the fallback code
            fallback_code = f"""def analyze_data(source_dfs, target_df):
    # Import required packages
    import pandas as pd
    import numpy as np
    
    # Safety check for source dataframes
    if not source_dfs or len(source_dfs) == 0:
        return target_df
    
    # Get source table(s)
    available_tables = list(source_dfs.keys())
    
    # Use primary table if available
    primary_table = "{primary_table}" if "{primary_table}" in available_tables else available_tables[0]
    source_df = source_dfs[primary_table]
    
    # Validate source dataframe
    if source_df.empty:
        return target_df
    
    # Check if required columns exist in source
    required_source_columns = {json.dumps(source_columns)}
    for col in required_source_columns:
        if col not in source_df.columns:
            # Skip if required column doesn't exist
            return target_df
    
    # Create copy of source dataframe with required columns
    result_df = source_df[required_source_columns].copy() if required_source_columns else source_df.copy()
    
    # If target is empty, return source as new target
    if len(target_df) == 0:
        return result_df
    else:
        # Basic update: If columns match, update target with source values
        for col in result_df.columns:
            if col in target_df.columns:
                target_df[col] = result_df[col]
        
        return target_df
"""
            
            return fallback_code
        except Exception as e:
            logger.error(f"Error generating fallback code: {e}")
            
            # Return ultra-simple fallback if even the regular fallback fails
            return """def analyze_data(source_dfs, target_df):
    # Import required packages
    import pandas as pd
    
    # Return unmodified target (safety fallback)
    return target_df
"""
    
    @track_token_usage()
    def fix_code(self, code_content, error_info, extracted_info=None, attempt=1, max_attempts=3):
        """
        Attempt to fix code based on error information
        
        Parameters:
        code_content (str): Original code content
        error_info (dict): Error information (type, message, traceback)
        extracted_info (dict, optional): Information from agents
        attempt (int): Current attempt number
        max_attempts (int): Maximum number of attempts
        
        Returns:
        str: Fixed code or None if fixing failed
        """
        if attempt > max_attempts:
            logger.warning(f"Max fix attempts ({max_attempts}) reached")
            return None
        
        try:
            # Extract error details
            error_type = error_info.get("error_type", "Unknown error")
            error_message = error_info.get("error_message", "No error message")
            traceback_text = error_info.get("traceback", "No traceback available")
            
            # Prepare context information
            context_info = "No additional context available"
            if extracted_info:
                # Format simplified context for the prompt
                table_info = extracted_info.get("table_info", {})
                column_info = extracted_info.get("column_info", {})
                
                context_parts = []
                
                # Add source tables
                source_tables = table_info.get("source_tables", [])
                if source_tables:
                    context_parts.append(f"Source Tables: {json.dumps(source_tables)}")
                
                # Add target table
                target_table = table_info.get("target_table")
                if target_table:
                    context_parts.append(f"Target Table: {target_table}")
                
                # Add abbreviated column information
                if column_info:
                    table_columns = {}
                    for table, cols in column_info.items():
                        if isinstance(cols, dict) and "required" in cols:
                            table_columns[table] = cols["required"]
                    
                    if table_columns:
                        context_parts.append(f"Required Columns: {json.dumps(table_columns)}")
                
                # Combine context parts
                if context_parts:
                    context_info = "\n".join(context_parts)
            
            # Create prompt for code fixing
            prompt = f"""
You are an expert Python developer specializing in fixing data transformation code. An error occurred while executing the following code. Your task is to fix the code to resolve the error.

THE CODE THAT FAILED:
```python
{code_content}
```

ERROR INFORMATION:
Error Type: {error_type}
Error Message: {error_message}

TRACEBACK:
{traceback_text}

CONTEXT INFORMATION:
{context_info}

COMMON ERRORS TO CHECK FOR:
- Field name typos or case sensitivity issues
- Missing field validation before access
- Incorrect handling of empty dataframes
- Type conversion errors (e.g., treating None or NaN values)
- Index errors when accessing specific rows
- Using .loc vs direct column access incorrectly
- Missing import statements

FIXING GUIDELINES:
1. Focus on fixing the specific error, not rewriting the entire function
2. Maintain the function signature and return type
3. Add proper validation for fields and dataframes
4. Handle edge cases more carefully
5. Add defensive checks where appropriate

Return ONLY the fixed code with no explanations or comments about your changes.
"""
            
            # Call LLM for code fixing
            if not self.client:
                logger.warning("No LLM client available for code fixing")
                return None
            
            response = self.client.models.generate_content(
                model=config.GENERATION_MODEL,
                contents=prompt
            )
            
            # Parse response
            if hasattr(response, 'text'):
                # Extract fixed code
                fixed_code = self._extract_code(response.text)
                if fixed_code:
                    return fixed_code
            
            # If extraction failed, return None
            logger.warning("Failed to extract fixed code from LLM response")
            return None
        except Exception as e:
            logger.error(f"Error in fix_code: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_plan(self, query, extracted_info):
        """
        Generate a step-by-step transformation plan
        
        Parameters:
        query (str): Original query
        extracted_info (dict): Information from all agents
        
        Returns:
        str: Transformation plan
        """
        try:
            # Prepare data for prompt
            intent_info = extracted_info.get("intent_info", {})
            table_info = extracted_info.get("table_info", {})
            segment_info = extracted_info.get("segment_info", {})
            column_info = extracted_info.get("column_info", {})
            
            # Format source tables information
            source_tables = table_info.get("source_tables", [])
            
            # Format target table information
            target_table = table_info.get("target_table")
            
            # Format column information briefly
            column_summary = {}
            for table, cols in column_info.items():
                if isinstance(cols, dict) and "required" in cols:
                    column_summary[table] = cols["required"]
            
            # Create prompt for plan generation
            prompt = f"""
You are an expert data transformation architect. Create a detailed, step-by-step plan for the following data transformation task.

ORIGINAL QUERY: {query}

TRANSFORMATION TYPE: {intent_info.get('intent_type', 'Not specified')}

SOURCE TABLES: {json.dumps(source_tables)}
TARGET TABLE: {target_table}

REQUIRED COLUMNS: {json.dumps(column_summary)}

SEGMENT INFORMATION:
{json.dumps(segment_info, indent=2) if segment_info else "Not available"}

Create a numbered plan with 5-10 clear, detailed steps that will achieve this transformation. Each step should be specific and actionable. The plan should cover:

1. Data validation and preparation steps
2. How to handle the specific transformation requirements
3. How to process both empty and populated target dataframes
4. Any conditional logic or special handling needed
5. Data validation before returning results

Return ONLY the numbered plan with no additional explanation.
"""
            
            # Call LLM for plan generation
            if not self.client:
                logger.warning("No LLM client available for plan generation")
                return self._generate_fallback_plan(extracted_info)
            
            response = self.client.models.generate_content(
                model=config.PLANNING_MODEL,
                contents=prompt
            )
            
            # Parse response
            if hasattr(response, 'text'):
                return response.text.strip()
            
            # Fallback to template if LLM fails
            logger.warning("LLM plan generation failed, using fallback template")
            return self._generate_fallback_plan(extracted_info)
        except Exception as e:
            logger.error(f"Error in generate_plan: {e}")
            return self._generate_fallback_plan(extracted_info)
    
    def _generate_fallback_plan(self, extracted_info):
        """
        Generate fallback plan when LLM generation fails
        
        Parameters:
        extracted_info (dict): Information from all agents
        
        Returns:
        str: Fallback transformation plan
        """
        # Get table information
        table_info = extracted_info.get("table_info", {})
        
        # Get source and target tables
        source_tables = table_info.get("source_tables", [])
        target_table = table_info.get("target_table", "target table")
        
        # Choose first source table as primary if available
        primary_table = source_tables[0] if source_tables else "source table"
        
        return f"""
1. Validate that required source tables ({', '.join(source_tables) if source_tables else 'source tables'}) exist in the provided input
2. Check if required fields exist in the source tables
3. Perform any necessary data filtering and cleaning
4. Check if the target dataframe is empty
5. If target is empty, create a new dataframe with the required fields
6. If target is not empty, update existing records with matching keys
7. Apply any specified transformations to the data
8. Validate the transformed data for integrity
9. Return the updated target dataframe
"""
