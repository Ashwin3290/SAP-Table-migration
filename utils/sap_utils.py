"""
SAP-specific utilities for TableLLM
"""
import pandas as pd
import numpy as np
import re
from utils.logging_utils import main_logger as logger
import config

def handle_sap_leading_zeros(source_df, field, length=None):
    """
    Ensure SAP numerical keys have proper leading zeros
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    field (str): Field to process
    length (int, optional): Target field length, uses config values if None
    
    Returns:
    pd.Series: Series with properly formatted values
    """
    try:
        if field not in source_df.columns:
            logger.warning(f"Field {field} not found in dataframe")
            return pd.Series(index=source_df.index)
        
        if length is None:
            # Use configured length for common fields
            length = config.SAP_FIELD_LENGTH.get(field, 10)
        
        # Convert to string and pad with zeros
        return source_df[field].astype(str).str.zfill(length)
    except Exception as e:
        logger.error(f"Error in handle_sap_leading_zeros: {e}")
        return source_df[field].copy() if field in source_df.columns else pd.Series(index=source_df.index)

def convert_sap_date(source_df, date_field, output_format='YYYY-MM-DD'):
    """
    Convert SAP date format (YYYYMMDD) to standard date format
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    date_field (str): Field containing SAP dates
    output_format (str): Output format specification
    
    Returns:
    pd.Series: Series with formatted dates
    """
    try:
        if date_field not in source_df.columns:
            logger.warning(f"Date field {date_field} not found in dataframe")
            return pd.Series(index=source_df.index)
        
        # Convert to string
        date_series = source_df[date_field].astype(str)
        
        # Replace zeros and empty strings with NaN
        date_series = date_series.replace(['00000000', '0', ''], np.nan)
        
        # Format based on output_format
        if output_format == 'YYYY-MM-DD':
            # Extract components and format
            return date_series.apply(
                lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if pd.notna(x) and len(x) == 8 else np.nan
            )
        elif output_format == 'MM/DD/YYYY':
            return date_series.apply(
                lambda x: f"{x[4:6]}/{x[6:8]}/{x[:4]}" if pd.notna(x) and len(x) == 8 else np.nan
            )
        else:
            logger.warning(f"Unsupported date format: {output_format}")
            return date_series
    except Exception as e:
        logger.error(f"Error in convert_sap_date: {e}")
        return source_df[date_field].copy() if date_field in source_df.columns else pd.Series(index=source_df.index)

def map_material_type(source_df, source_field='MTART'):
    """
    Map SAP material type codes to readable text descriptions
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    source_field (str): Field containing material type codes
    
    Returns:
    pd.Series: Series with mapped text descriptions
    """
    try:
        if source_field not in source_df.columns:
            logger.warning(f"Field {source_field} not found in dataframe")
            return pd.Series(index=source_df.index)
        
        # SAP material type mapping
        material_type_map = {
            'ROH': 'Raw Material',
            'HALB': 'Semi-Finished Product',
            'FERT': 'Finished Product',
            'HIBE': 'Trading Goods',
            'VERP': 'Packaging Material',
            'NLAG': 'Non-Stock Material',
            'DIEN': 'Service',
            'ERSA': 'Spare Part',
            # Add more mappings as needed
        }
        
        # Apply mapping with fallback to original value
        return source_df[source_field].apply(
            lambda x: material_type_map.get(x, x) if pd.notna(x) else np.nan
        )
    except Exception as e:
        logger.error(f"Error in map_material_type: {e}")
        return source_df[source_field].copy() if source_field in source_df.columns else pd.Series(index=source_df.index)

def map_sap_language_code(source_df, lang_field='SPRAS'):
    """
    Map SAP language codes to full language names
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    lang_field (str): Field containing language codes
    
    Returns:
    pd.Series: Series with full language names
    """
    try:
        if lang_field not in source_df.columns:
            logger.warning(f"Field {lang_field} not found in dataframe")
            return pd.Series(index=source_df.index)
        
        # SAP language code mapping
        language_map = {
            'EN': 'English',
            'DE': 'German',
            'FR': 'French',
            'ES': 'Spanish',
            'IT': 'Italian',
            'PT': 'Portuguese',
            'RU': 'Russian',
            'ZH': 'Chinese',
            'JA': 'Japanese',
            'KO': 'Korean',
            # Add more mappings as needed
        }
        
        # Apply mapping with fallback to original value
        return source_df[lang_field].apply(
            lambda x: language_map.get(x, x) if pd.notna(x) else np.nan
        )
    except Exception as e:
        logger.error(f"Error in map_sap_language_code: {e}")
        return source_df[lang_field].copy() if lang_field in source_df.columns else pd.Series(index=source_df.index)

def map_sap_unit_of_measure(source_df, uom_field='MEINS'):
    """
    Map SAP units of measure to full descriptions
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    uom_field (str): Field containing units of measure
    
    Returns:
    pd.Series: Series with full unit descriptions
    """
    try:
        if uom_field not in source_df.columns:
            logger.warning(f"Field {uom_field} not found in dataframe")
            return pd.Series(index=source_df.index)
        
        # SAP unit of measure mapping
        uom_map = {
            'EA': 'Each',
            'PC': 'Piece',
            'KG': 'Kilogram',
            'G': 'Gram',
            'MG': 'Milligram',
            'L': 'Liter',
            'ML': 'Milliliter',
            'M': 'Meter',
            'CM': 'Centimeter',
            'MM': 'Millimeter',
            'KM': 'Kilometer',
            'SQM': 'Square Meter',
            'CBM': 'Cubic Meter',
            # Add more mappings as needed
        }
        
        # Apply mapping with fallback to original value
        return source_df[uom_field].apply(
            lambda x: uom_map.get(x, x) if pd.notna(x) else np.nan
        )
    except Exception as e:
        logger.error(f"Error in map_sap_unit_of_measure: {e}")
        return source_df[uom_field].copy() if uom_field in source_df.columns else pd.Series(index=source_df.index)

def filter_dataframe(df, field, condition_type, value):
    """
    Filter a dataframe based on field and condition
    
    Parameters:
    df (pd.DataFrame): Source dataframe
    field (str): Field name to filter on
    condition_type (str): One of: 'equals', 'not_equals', 'contains', 'in', 'not_in', 'greater_than', 'less_than', 'is_null', 'not_null'
    value: Value to compare against
    
    Returns:
    pd.DataFrame: Filtered dataframe
    """
    try:
        if field not in df.columns:
            logger.warning(f"Field {field} not found in dataframe for filtering")
            return df
        
        if condition_type == 'equals':
            return df[df[field] == value]
        elif condition_type == 'not_equals':
            return df[df[field] != value]
        elif condition_type == 'contains':
            return df[df[field].astype(str).str.contains(str(value), na=False)]
        elif condition_type == 'in':
            return df[df[field].isin(value)]
        elif condition_type == 'not_in':
            return df[~df[field].isin(value)]
        elif condition_type == 'greater_than':
            return df[df[field] > value]
        elif condition_type == 'less_than':
            return df[df[field] < value]
        elif condition_type == 'is_null':
            return df[df[field].isna()]
        elif condition_type == 'not_null':
            return df[df[field].notna()]
        else:
            logger.warning(f"Unknown condition type: {condition_type}")
            return df
    except Exception as e:
        logger.error(f"Error in filter_dataframe: {e}")
        return df

def map_fields(source_df, target_df, field_mapping, key_field_pair=None):
    """
    Map fields from source to target dataframe
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    target_df (pd.DataFrame): Target dataframe
    field_mapping (Dict[str, str]): Dictionary mapping target fields to source fields
    key_field_pair (Optional[Tuple[str, str]]): Optional tuple containing (target_key, source_key)
    
    Returns:
    pd.DataFrame: Updated target dataframe with mapped fields
    """
    try:
        # Create a copy of the target dataframe to avoid modifying the original
        result = target_df.copy()
        
        # Check if target dataframe is empty
        if len(result) == 0:
            # For empty target, create a new dataframe with key field and mapped fields
            if key_field_pair:
                target_key, source_key = key_field_pair
                # Get key values from source
                key_values = source_df[source_key].drop_duplicates().reset_index(drop=True)
                # Create dataframe with key values
                result = pd.DataFrame({target_key: key_values})
                # Add mapped fields
                for target_field, source_field in field_mapping.items():
                    # Use dictionary for mapping
                    mapping_dict = dict(zip(source_df[source_key], source_df[source_field]))
                    result[target_field] = result[target_key].map(mapping_dict)
            else:
                # No key field, just create a mapping from the first valid source row
                result = pd.DataFrame()
                if len(source_df) > 0:
                    for target_field, source_field in field_mapping.items():
                        # Check if source field exists
                        if source_field in source_df.columns:
                            result[target_field] = source_df[source_field]
                        else:
                            logger.warning(f"Source field {source_field} not found in source dataframe")
                            result[target_field] = np.nan
        else:
            # For non-empty target, update fields based on key mapping
            if key_field_pair:
                target_key, source_key = key_field_pair
                # Update fields based on key mapping
                for target_field, source_field in field_mapping.items():
                    # Check if source field exists
                    if source_field in source_df.columns:
                        # Create mapping dictionary
                        mapping_dict = dict(zip(source_df[source_key], source_df[source_field]))
                        # Apply mapping
                        result[target_field] = result[target_key].map(mapping_dict).fillna(result[target_field])
                    else:
                        logger.warning(f"Source field {source_field} not found in source dataframe")
            else:
                # No key field, just map directly (assumes same number of rows or broadcasting)
                for target_field, source_field in field_mapping.items():
                    # Check if source field exists
                    if source_field in source_df.columns:
                        result[target_field] = source_df[source_field].values
                    else:
                        logger.warning(f"Source field {source_field} not found in source dataframe")
        
        return result
    except Exception as e:
        logger.error(f"Error in map_fields: {e}")
        return target_df

def conditional_mapping(source_df, condition_field, conditions, value_field=None, value_map=None, default=None):
    """
    Map values based on conditions
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    condition_field (str): Field to apply conditions to
    conditions (List[str]): List of condition expressions as strings
    value_field (Optional[str]): Field to get values from if value_map not provided
    value_map (Optional[List[Any]]): List of values corresponding to conditions
    default (Optional[Any]): Value to use when no conditions match
    
    Returns:
    pd.Series: Series with mapped values
    """
    try:
        if condition_field not in source_df.columns:
            logger.warning(f"Condition field {condition_field} not found in dataframe")
            return pd.Series(index=source_df.index)
        
        # Convert conditions to executable expressions
        condition_results = []
        for condition in conditions:
            # Handle common condition formats
            if '==' in condition:
                field, value = condition.split('==')
                field = field.strip()
                value = value.strip().strip("'\"")
                if field == condition_field:
                    condition_results.append(source_df[field] == value)
            elif 'in' in condition.lower():
                # Extract values inside parentheses
                match = re.search(r'\((.*?)\)', condition)
                if match:
                    values_str = match.group(1)
                    values = [v.strip().strip("'\"") for v in values_str.split(',')]
                    condition_results.append(source_df[condition_field].isin(values))
            else:
                # For more complex conditions, try to evaluate as is
                try:
                    condition_results.append(eval(f"source_df['{condition_field}'] {condition}"))
                except Exception as e:
                    logger.error(f"Error evaluating condition '{condition}': {e}")
                    condition_results.append(pd.Series(False, index=source_df.index))
        
        # Create result series
        if value_field is not None and value_field in source_df.columns:
            # Use np.select with source values
            values_to_use = []
            for i, _ in enumerate(conditions):
                mask = condition_results[i]
                values_to_use.append(source_df.loc[mask, value_field].iloc[0] if any(mask) else np.nan)
            
            return pd.Series(
                np.select(condition_results, values_to_use, default=default if default is not None else np.nan),
                index=source_df.index
            )
        elif value_map is not None:
            # Use np.select with provided value map
            return pd.Series(
                np.select(condition_results, value_map, default=default if default is not None else np.nan),
                index=source_df.index
            )
        else:
            # No values provided, return boolean results
            logger.warning("No value_field or value_map provided, returning boolean results")
            return pd.Series(
                np.select(condition_results, [True] * len(condition_results), default=False),
                index=source_df.index
            )
    except Exception as e:
        logger.error(f"Error in conditional_mapping: {e}")
        return pd.Series(index=source_df.index)

def join_tables(main_df, other_df, main_key, other_key, fields_to_add):
    """
    Join two tables and add specific fields
    
    Parameters:
    main_df (pd.DataFrame): Main dataframe
    other_df (pd.DataFrame): Other dataframe to join with
    main_key (str): Key field in main dataframe
    other_key (str): Key field in other dataframe
    fields_to_add (List[str]): Fields to add from other dataframe
    
    Returns:
    pd.DataFrame: Main dataframe with added fields
    """
    try:
        # Validate inputs
        if main_key not in main_df.columns:
            logger.warning(f"Main key field {main_key} not found in main dataframe")
            return main_df
        
        if other_key not in other_df.columns:
            logger.warning(f"Other key field {other_key} not found in other dataframe")
            return main_df
        
        # Create a copy of the main dataframe
        result_df = main_df.copy()
        
        # Validate fields to add
        valid_fields = [field for field in fields_to_add if field in other_df.columns]
        if len(valid_fields) < len(fields_to_add):
            missing_fields = set(fields_to_add) - set(valid_fields)
            logger.warning(f"Fields not found in other dataframe: {missing_fields}")
        
        if not valid_fields:
            logger.warning("No valid fields to add")
            return result_df
        
        # Create a mapping dictionary for each field to add
        for field in valid_fields:
            mapping_dict = dict(zip(other_df[other_key], other_df[field]))
            result_df[field] = result_df[main_key].map(mapping_dict)
        
        return result_df
    except Exception as e:
        logger.error(f"Error in join_tables: {e}")
        return main_df

def aggregate_data(df, group_by, agg_functions):
    """
    Aggregate data by group with specified functions
    
    Parameters:
    df (pd.DataFrame): Source dataframe
    group_by (str or List[str]): Fields to group by
    agg_functions (Dict[str, str]): Dictionary mapping fields to aggregation functions
    
    Returns:
    pd.DataFrame: Aggregated dataframe
    """
    try:
        # Validate inputs
        if isinstance(group_by, str):
            group_by = [group_by]
        
        # Check if all group_by fields exist
        for field in group_by:
            if field not in df.columns:
                logger.warning(f"Group by field {field} not found in dataframe")
                return df
        
        # Validate aggregation functions
        valid_agg = {}
        for field, func in agg_functions.items():
            if field in df.columns:
                valid_agg[field] = func
            else:
                logger.warning(f"Aggregation field {field} not found in dataframe")
        
        if not valid_agg:
            logger.warning("No valid aggregation functions")
            return df
        
        # Perform aggregation
        return df.groupby(group_by).agg(valid_agg).reset_index()
    except Exception as e:
        logger.error(f"Error in aggregate_data: {e}")
        return df

def clean_special_characters(source_df, field):
    """
    Remove special characters from a text field
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    field (str): Field to clean
    
    Returns:
    pd.Series: Cleaned series
    """
    try:
        if field not in source_df.columns:
            logger.warning(f"Field {field} not found in dataframe")
            return pd.Series(index=source_df.index)
        
        # Convert to string and remove special characters
        return source_df[field].astype(str).apply(
            lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x) if pd.notna(x) else np.nan
        )
    except Exception as e:
        logger.error(f"Error in clean_special_characters: {e}")
        return source_df[field].copy() if field in source_df.columns else pd.Series(index=source_df.index)

def tiered_lookup(source_dfs, lookup_tables, key_field, value_field, default=None):
    """
    Look up values in multiple tables in sequence
    
    Parameters:
    source_dfs (Dict[str, pd.DataFrame]): Dictionary of source dataframes
    lookup_tables (List[str]): Ordered list of table names to check
    key_field (str): Key field to use for lookup
    value_field (str): Value field to retrieve
    default (Optional[Any]): Default value if not found in any table
    
    Returns:
    Dict[Any, Any]: Dictionary mapping keys to values
    """
    try:
        result = {}
        keys = set()
        
        # Collect all keys from source dataframes
        for table_name, df in source_dfs.items():
            if key_field in df.columns:
                keys.update(df[key_field].dropna().unique())
        
        # Look up each key in the tables in sequence
        for key in keys:
            value = default
            for table_name in lookup_tables:
                if table_name in source_dfs and key_field in source_dfs[table_name].columns and value_field in source_dfs[table_name].columns:
                    matches = source_dfs[table_name][source_dfs[table_name][key_field] == key]
                    if not matches.empty:
                        value = matches[value_field].iloc[0]
                        break
            result[key] = value
        
        return result
    except Exception as e:
        logger.error(f"Error in tiered_lookup: {e}")
        return {}
