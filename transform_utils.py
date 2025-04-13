# transform_utils.py
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Any, Optional, Callable

def filter_dataframe(df: pd.DataFrame, 
                    field: str, 
                    condition_type: str, 
                    value: Any) -> pd.DataFrame:
    """
    Filter a dataframe based on field and condition
    
    Parameters:
    df (pd.DataFrame): Source dataframe
    field (str): Field name to filter on
    condition_type (str): One of:
        - 'equals': Field equals value (==)
        - 'not_equals': Field not equal to value (!=)
        - 'contains': Field contains value (str.contains)
        - 'greater_than': Field greater than value (>)
        - 'less_than': Field less than value (<)
        - 'in_list': Field value is in provided list (isin)
        - 'not_in_list': Field value is not in provided list (~isin)
        - 'is_null': Field is null (isna)
        - 'is_not_null': Field is not null (~isna)
    value (Any): Value to compare against (not used for is_null/is_not_null)
    
    Returns:
    pd.DataFrame: Filtered dataframe
    """
    if condition_type == 'equals':
        return df[df[field] == value].copy()
    elif condition_type == 'not_equals':
        return df[df[field] != value].copy()
    elif condition_type == 'contains':
        if df[field].dtype == 'object':  # String column
            return df[df[field].str.contains(str(value), na=False)].copy()
        else:
            raise ValueError(f"Cannot use 'contains' on non-string column {field}")
    elif condition_type == 'greater_than':
        return df[df[field] > value].copy()
    elif condition_type == 'less_than':
        return df[df[field] < value].copy()
    elif condition_type == 'in_list':
        return df[df[field].isin(value)].copy()
    elif condition_type == 'not_in_list':
        return df[~df[field].isin(value)].copy()
    elif condition_type == 'is_null':
        return df[df[field].isna()].copy()
    elif condition_type == 'is_not_null':
        return df[~df[field].isna()].copy()
    else:
        raise ValueError(f"Unsupported condition type: {condition_type}. " 
                         f"Must be one of: equals, not_equals, contains, greater_than, less_than, "
                         f"in_list, not_in_list, is_null, is_not_null")

def map_fields(source_df: pd.DataFrame, 
              target_df: pd.DataFrame, 
              field_mapping: Dict[str, str], 
              key_field_pair: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
    """
    Map fields from source to target dataframe
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    target_df (pd.DataFrame): Target dataframe 
    field_mapping (Dict[str, str]): Dictionary mapping target fields to source fields, e.g.:
                                   {'TARGET_FIELD1': 'SOURCE_FIELD1', 'TARGET_FIELD2': 'SOURCE_FIELD2'}
    key_field_pair (Optional[Tuple[str, str]]): Optional tuple containing (target_key, source_key)
                                               for record matching. If None, direct mapping is used.
    
    Returns:
    pd.DataFrame: Updated target dataframe with mapped fields
    """
    # Create a copy of target dataframe
    result_df = target_df.copy()
    
    # If target is empty, create a new dataframe with mapped fields
    if result_df.empty:
        # Create new dataframe with keys first
        if key_field_pair and key_field_pair[1] in source_df.columns:
            target_key, source_key = key_field_pair
            result_df = pd.DataFrame({target_key: source_df[source_key].values})
        else:
            result_df = pd.DataFrame(index=source_df.index)
        
        # Add mapped fields
        for target_field, source_field in field_mapping.items():
            if source_field in source_df.columns:
                result_df[target_field] = source_df[source_field].values
        
        return result_df
    
    # If we have key fields, use them to match records
    if key_field_pair:
        target_key, source_key = key_field_pair
        
        if target_key in result_df.columns and source_key in source_df.columns:
            # Create mapping from source keys to indices
            source_map = dict(zip(source_df[source_key], source_df.index))
            
            # Update target fields where keys match
            for idx, row in result_df.iterrows():
                if pd.notna(row[target_key]) and row[target_key] in source_map:
                    source_idx = source_map[row[target_key]]
                    
                    # Map each field
                    for target_field, source_field in field_mapping.items():
                        if source_field in source_df.columns:
                            result_df.loc[idx, target_field] = source_df.loc[source_idx, source_field]
    else:
        # No key fields, so map fields directly
        for target_field, source_field in field_mapping.items():
            if source_field in source_df.columns:
                result_df[target_field] = source_df[source_field].values
    
    return result_df

def match_by_keys(source_df: pd.DataFrame, 
                 target_df: pd.DataFrame, 
                 key_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Match records between source and target dataframes based on key mapping
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    target_df (pd.DataFrame): Target dataframe
    key_mapping (Dict[str, str]): Dictionary mapping target keys to source keys, e.g.:
                                 {'TARGET_KEY': 'SOURCE_KEY'}
    
    Returns:
    pd.DataFrame: Updated target dataframe with matched records
    """
    # Create a copy of target dataframe
    result_df = target_df.copy()
    
    # If target is empty, initialize with keys from source
    if result_df.empty and not source_df.empty:
        # Create new dataframe with just the key columns
        new_df = pd.DataFrame()
        for target_key, source_key in key_mapping.items():
            if source_key in source_df.columns:
                new_df[target_key] = source_df[source_key]
        return new_df
    
    # For each mapping, match records
    for target_key, source_key in key_mapping.items():
        # Ensure columns exist in both dataframes
        if target_key not in result_df.columns:
            result_df[target_key] = None
            
        if source_key in source_df.columns and target_key in result_df.columns:
            # Create a map from source key values to rows
            key_to_row = {}
            for idx, row in source_df.iterrows():
                key_value = row[source_key]
                if pd.notna(key_value):
                    key_to_row[key_value] = row.to_dict()
            
            # Update target where keys match
            for idx, row in result_df.iterrows():
                target_key_value = row[target_key]
                if pd.notna(target_key_value) and target_key_value in key_to_row:
                    # Just update the key field for now
                    result_df.loc[idx, target_key] = key_to_row[target_key_value][source_key]
    
    return result_df

def conditional_mapping(source_df: pd.DataFrame, 
                       condition_field: str, 
                       conditions: List[str], 
                       value_field: Optional[str] = None, 
                       value_map: Optional[List[Any]] = None, 
                       default: Optional[Any] = None) -> pd.Series:
    """
    Map values based on conditions
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    condition_field (str): Field to apply conditions to
    conditions (List[str]): List of condition expressions as strings. Format options:
                           - "== 'value'"    (equals string value)
                           - "== 50"         (equals numeric value)
                           - "!= 'value'"    (not equals string value)
                           - "> 100"         (greater than numeric value)
                           - "< 200"         (less than numeric value)
                           - "in ['A', 'B']" (value in list)
    value_field (Optional[str]): Field to get values from if value_map not provided
    value_map (Optional[List[Any]]): List of values corresponding to conditions
    default (Optional[Any]): Value to use when no conditions match
    
    Returns:
    pd.Series: Series with mapped values
    """
    result = pd.Series(index=source_df.index, dtype=object)
    
    # Apply each condition in sequence
    for i, condition in enumerate(conditions):
        # Parse the condition
        if '==' in condition:
            field, val = condition.split('==')
            field = field.strip()
            val = val.strip()
            # Handle string literals in quotes
            if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1]  # Remove quotes
                mask = source_df[field] == val
            else:
                # Try to convert to numeric
                try:
                    val = float(val)
                    mask = source_df[field] == val
                except ValueError:
                    mask = source_df[field] == val
        elif '!=' in condition:
            field, val = condition.split('!=')
            field = field.strip()
            val = val.strip()
            if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1]  # Remove quotes
                mask = source_df[field] != val
            else:
                try:
                    val = float(val)
                    mask = source_df[field] != val
                except ValueError:
                    mask = source_df[field] != val
        elif '>' in condition:
            field, val = condition.split('>')
            field = field.strip()
            val = float(val.strip())
            mask = source_df[field] > val
        elif '<' in condition:
            field, val = condition.split('<')
            field = field.strip()
            val = float(val.strip())
            mask = source_df[field] < val
        elif 'in' in condition.lower():
            # Handle "in list" condition, expected format: "FIELD in ['A', 'B', 'C']"
            parts = condition.split('in')
            field = parts[0].strip()
            values_str = parts[1].strip()
            # Extract the list values
            try:
                values = eval(values_str)  # This evaluates the list expression
                mask = source_df[field].isin(values)
            except:
                raise ValueError(f"Invalid 'in' condition format: {condition}")
        else:
            # Custom condition - would need to use eval carefully
            try:
                mask = eval(f"source_df['{condition_field}']" + condition)
            except:
                raise ValueError(f"Invalid condition format: {condition}")
        
        # Apply the value where condition is met
        if value_map and i < len(value_map):
            result[mask] = value_map[i]
        elif value_field:
            result[mask] = source_df.loc[mask, value_field]
    
    # Apply default value where no condition was met
    if default is not None:
        result[result.isna()] = default
        
    return result

def initialize_target(target_df: Optional[pd.DataFrame], 
                     required_columns: List[str]) -> pd.DataFrame:
    """
    Initialize target dataframe with required columns
    
    Parameters:
    target_df (Optional[pd.DataFrame]): Target dataframe, can be None
    required_columns (List[str]): List of column names that must exist in result
    
    Returns:
    pd.DataFrame: Initialized dataframe with all required columns
    """
    if target_df is None or target_df.empty:
        return pd.DataFrame(columns=required_columns)
    
    result = target_df.copy()
    
    # Ensure all required columns exist
    for col in required_columns:
        if col not in result.columns:
            result[col] = None
            
    return result

def join_tables(main_df: pd.DataFrame, 
               other_df: pd.DataFrame, 
               main_key: str, 
               other_key: str, 
               fields_to_add: List[str]) -> pd.DataFrame:
    """
    Join two tables and add specific fields
    
    Parameters:
    main_df (pd.DataFrame): Main dataframe to update
    other_df (pd.DataFrame): Other dataframe to get data from
    main_key (str): Key field in main_df
    other_key (str): Key field in other_df
    fields_to_add (List[str]): List of field names from other_df to add to main_df
    
    Returns:
    pd.DataFrame: Updated main_df with fields from other_df
    """
    # Create a copy to avoid modifying the original
    result = main_df.copy()
    
    # Create a mapping from other_key to fields_to_add
    mapping = {}
    for idx, row in other_df.iterrows():
        key = row[other_key]
        if pd.notna(key):
            mapping[key] = {field: row[field] for field in fields_to_add if field in other_df.columns}
    
    # Apply the mapping to main_df
    for idx, row in result.iterrows():
        key = row[main_key]
        if pd.notna(key) and key in mapping:
            for field in fields_to_add:
                if field not in result.columns:
                    result[field] = None
                result.loc[idx, field] = mapping[key].get(field)
    
    return result

def aggregate_data(df: pd.DataFrame, 
                  group_by: Union[str, List[str]], 
                  agg_functions: Dict[str, Union[str, List[str]]]) -> pd.DataFrame:
    """
    Aggregate data by group with specified functions
    
    Parameters:
    df (pd.DataFrame): Dataframe to aggregate
    group_by (Union[str, List[str]]): Column(s) to group by
    agg_functions (Dict[str, Union[str, List[str]]]): Dictionary mapping columns to aggregation functions
                                                     Supported functions: 'sum', 'mean', 'min', 'max', 'count'
                                                     Example: {'SALES': 'sum', 'PRICE': 'mean'}
    
    Returns:
    pd.DataFrame: Aggregated dataframe
    """
    return df.groupby(group_by).agg(agg_functions).reset_index()

def safe_combine_tables(df1: Optional[pd.DataFrame], 
                       df2: Optional[pd.DataFrame], 
                       key: Optional[str] = None) -> pd.DataFrame:
    """
    Safely combine two tables, either by concatenation or by key matching
    
    Parameters:
    df1 (Optional[pd.DataFrame]): First dataframe, can be None
    df2 (Optional[pd.DataFrame]): Second dataframe, can be None
    key (Optional[str]): Optional key field for matching records
                        If provided, will update existing records in df1 with matching records from df2
                        If not provided, will simply concatenate the dataframes
    
    Returns:
    pd.DataFrame: Combined dataframe
    """
    # Create copies to avoid modifying originals
    result = df1.copy() if df1 is not None and not df1.empty else None
    df2_copy = df2.copy() if df2 is not None and not df2.empty else None
    
    # Handle empty dataframes
    if result is None or result.empty:
        return df2_copy if df2_copy is not None else pd.DataFrame()
    if df2_copy is None or df2_copy.empty:
        return result
    
    # Align columns if needed
    for col in df2_copy.columns:
        if col not in result.columns:
            result[col] = None
    
    # If key is provided, update existing rows and append new ones
    if key:
        # Only process if key exists in both dataframes
        if key in result.columns and key in df2_copy.columns:
            # Get existing keys
            existing_keys = set(result[key].dropna().unique())
            
            # Update existing rows
            for idx, row in df2_copy.iterrows():
                key_value = row[key]
                if pd.notna(key_value) and key_value in existing_keys:
                    # Find matching rows in result
                    mask = result[key] == key_value
                    # Update all columns from df2
                    for col in df2_copy.columns:
                        if col in result.columns:
                            result.loc[mask, col] = row[col]
            
            # Append new rows
            new_rows = df2_copy[~df2_copy[key].isin(existing_keys)]
            if not new_rows.empty:
                result = pd.concat([result, new_rows], ignore_index=True)
        else:
            # Key not found in both, just concatenate
            result = pd.concat([result, df2_copy], ignore_index=True)
    else:
        # No key, just concatenate
        result = pd.concat([result, df2_copy], ignore_index=True)
    
    return result

def map_material_type(source_df: pd.DataFrame, 
                     source_field: str = 'MTART', 
                     target_field: str = 'MATERIAL_TYPE_TEXT') -> pd.Series:
    """
    Map SAP material type codes to readable text descriptions
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe with material types
    source_field (str): Field containing material type codes (default: 'MTART')
    target_field (str): Field name for the text descriptions (default: 'MATERIAL_TYPE_TEXT')
    
    Returns:
    pd.Series: Series with mapped text descriptions
    """
    # SAP material type mapping
    mtart_mapping = {
        'ROH': 'Raw Material',
        'HALB': 'Semi-Finished Product',
        'FERT': 'Finished Product',
        'HAWA': 'Trading Goods',
        'NLAG': 'Non-Stock Material',
        'DIEN': 'Service',
        'ERSA': 'Spare Part',
        'UNBW': 'Non-Valuated Material',
        'VERP': 'Packaging Material',
        'PROD': 'Production Resources/Tools',
        'WETT': 'Competitor Product',
        'LEIH': 'Rental Material',
        'FRHT': 'Freight',
        'IBAU': 'In-House Production',
        'KMAT': 'Configurable Material',
        'HERS': 'Manufacturer Part',
        'PIPE': 'Pipeline Material',
        'DIPL': 'Digital Product',
        'KUND': 'Customer Material',
        'LEER': 'Empties'
    }
    
    # Create a mapping series
    result = pd.Series(index=source_df.index, dtype='object')
    
    # Apply mapping
    if source_field in source_df.columns:
        for idx, value in source_df[source_field].items():
            if pd.notna(value) and value in mtart_mapping:
                result.iloc[idx] = mtart_mapping[value]
            else:
                result.iloc[idx] = f"Unknown Type: {value}" if pd.notna(value) else None
    
    return result

def convert_sap_date(source_df: pd.DataFrame, 
                    date_field: str,
                    output_format: str = 'YYYY-MM-DD') -> pd.Series:
    """
    Convert SAP date format (YYYYMMDD) to standard date format
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe with SAP dates
    date_field (str): Field containing SAP dates
    output_format (str): Output format specification, one of:
                        'YYYY-MM-DD', 'MM/DD/YYYY', 'DD.MM.YYYY'
    
    Returns:
    pd.Series: Series with formatted dates
    """
    if date_field not in source_df.columns:
        return pd.Series(index=source_df.index, dtype='object')
    
    result = pd.Series(index=source_df.index, dtype='object')
    
    for idx, value in source_df[date_field].items():
        if pd.isna(value) or not str(value).strip():
            result.iloc[idx] = None
            continue
            
        # Convert to string if not already
        date_str = str(value).strip()
        
        # Check if it's a valid SAP date format (8 digits)
        if len(date_str) != 8 or not date_str.isdigit():
            result.iloc[idx] = value  # Keep original if not valid
            continue
            
        try:
            year = date_str[0:4]
            month = date_str[4:6]
            day = date_str[6:8]
            
            # Format according to output_format
            if output_format == 'YYYY-MM-DD':
                result.iloc[idx] = f"{year}-{month}-{day}"
            elif output_format == 'MM/DD/YYYY':
                result.iloc[idx] = f"{month}/{day}/{year}"
            elif output_format == 'DD.MM.YYYY':
                result.iloc[idx] = f"{day}.{month}.{year}"
            else:
                result.iloc[idx] = f"{year}-{month}-{day}"  # Default
        except:
            result.iloc[idx] = value  # Keep original if conversion fails
    
    return result

def map_sap_language_code(source_df: pd.DataFrame, 
                         lang_field: str = 'SPRAS') -> pd.Series:
    """
    Map SAP language codes to full language names
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe with language codes
    lang_field (str): Field containing language codes (default: 'SPRAS')
    
    Returns:
    pd.Series: Series with full language names
    """
    # SAP language code mapping
    language_mapping = {
        'EN': 'English',
        'DE': 'German',
        'FR': 'French',
        'ES': 'Spanish',
        'IT': 'Italian',
        'JA': 'Japanese',
        'DA': 'Danish',
        'FI': 'Finnish',
        'NL': 'Dutch',
        'NO': 'Norwegian',
        'PT': 'Portuguese',
        'SV': 'Swedish',
        'ZH': 'Chinese',
        'RU': 'Russian',
        'PL': 'Polish',
        'CS': 'Czech',
        'SK': 'Slovak',
        'HU': 'Hungarian',
        'TR': 'Turkish',
        'BG': 'Bulgarian',
        'RO': 'Romanian',
        'SR': 'Serbian',
        'HR': 'Croatian',
        'SL': 'Slovenian',
        'KO': 'Korean',
        'AR': 'Arabic',
        'HE': 'Hebrew',
        'TH': 'Thai',
        'UK': 'Ukrainian',
        'ET': 'Estonian',
        'LV': 'Latvian',
        'LT': 'Lithuanian',
        'EL': 'Greek'
    }
    
    # Create a mapping series
    result = pd.Series(index=source_df.index, dtype='object')
    
    # Apply mapping
    if lang_field in source_df.columns:
        for idx, value in source_df[lang_field].items():
            if pd.notna(value):
                # Normalize to uppercase
                code = str(value).strip().upper()
                if code in language_mapping:
                    result.iloc[idx] = language_mapping[code]
                else:
                    result.iloc[idx] = f"Unknown Language: {code}"
            else:
                result.iloc[idx] = None
    
    return result

def map_sap_unit_of_measure(source_df: pd.DataFrame, 
                           uom_field: str = 'MEINS') -> pd.Series:
    """
    Map SAP units of measure to full descriptions
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe with UoM codes
    uom_field (str): Field containing UoM codes (default: 'MEINS')
    
    Returns:
    pd.Series: Series with full UoM descriptions
    """
    # SAP UoM mapping
    uom_mapping = {
        'ST': 'Piece',
        'PC': 'Piece',
        'EA': 'Each',
        'KG': 'Kilogram',
        'G': 'Gram',
        'MG': 'Milligram',
        'L': 'Liter',
        'ML': 'Milliliter',
        'M': 'Meter',
        'MM': 'Millimeter',
        'CM': 'Centimeter',
        'KM': 'Kilometer',
        'M2': 'Square Meter',
        'M3': 'Cubic Meter',
        'H': 'Hour',
        'MIN': 'Minute',
        'SEC': 'Second',
        'D': 'Day',
        'WK': 'Week',
        'MO': 'Month',
        'YR': 'Year',
        'PAL': 'Pallet',
        'BOX': 'Box',
        'CAN': 'Can',
        'BTL': 'Bottle',
        'CS': 'Case',
        'DR': 'Drum',
        'CTN': 'Carton',
        'PAA': 'Pair',
        'PK': 'Pack',
        'RL': 'Roll',
        'SHT': 'Sheet',
        'TB': 'Tube',
        'BAG': 'Bag',
        'BKT': 'Bucket',
        'TRY': 'Tray'
    }
    
    # Create a mapping series
    result = pd.Series(index=source_df.index, dtype='object')
    
    # Apply mapping
    if uom_field in source_df.columns:
        for idx, value in source_df[uom_field].items():
            if pd.notna(value):
                # Normalize to uppercase
                code = str(value).strip().upper()
                if code in uom_mapping:
                    result.iloc[idx] = uom_mapping[code]
                else:
                    result.iloc[idx] = code  # Keep original if not in mapping
            else:
                result.iloc[idx] = None
    
    return result

def handle_sap_leading_zeros(source_df: pd.DataFrame, 
                            field: str, 
                            length: int = 10) -> pd.Series:
    """
    Ensure SAP numerical keys have proper leading zeros
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe
    field (str): Field to process
    length (int): Target field length (default: 10 for material numbers)
    
    Returns:
    pd.Series: Series with properly formatted values
    """
    if field not in source_df.columns:
        return pd.Series(index=source_df.index, dtype='object')
    
    result = pd.Series(index=source_df.index, dtype='object')
    
    for idx, value in source_df[field].items():
        if pd.isna(value):
            result.iloc[idx] = None
            continue
            
        # Convert to string if not already
        val_str = str(value).strip()
        
        # Skip if not numeric
        if not val_str.isdigit():
            result.iloc[idx] = val_str
            continue
            
        # Add leading zeros if needed
        result.iloc[idx] = val_str.zfill(length)
    
    return result

def standardize_product_hierarchy(source_df: pd.DataFrame, 
                                 hierarchy_field: str = 'PRDHA',
                                 delimiter: str = '-',
                                 level_names: Optional[List[str]] = None) -> Dict[str, pd.Series]:
    """
    Standardize SAP product hierarchy into separate level fields
    
    Parameters:
    source_df (pd.DataFrame): Source dataframe with product hierarchy
    hierarchy_field (str): Field containing hierarchy codes (default: 'PRDHA')
    delimiter (str): Delimiter to use when splitting levels
    level_names (Optional[List[str]]): Names for hierarchy levels, defaults to Level1, Level2, etc.
    
    Returns:
    Dict[str, pd.Series]: Dictionary of Series for each hierarchy level
    """
    if hierarchy_field not in source_df.columns:
        return {}
    
    # Extract unique hierarchy values
    unique_hierarchies = source_df[hierarchy_field].dropna().unique()
    
    # Determine max hierarchy depth
    max_depth = 0
    for h in unique_hierarchies:
        if pd.notna(h):
            depth = len(str(h))
            if depth > max_depth:
                max_depth = depth
    
    # Default level step is 2 characters for SAP hierarchies
    level_step = 2
    max_levels = (max_depth + level_step - 1) // level_step
    
    # Create level names if not provided
    if not level_names:
        level_names = [f"Level{i+1}" for i in range(max_levels)]
    else:
        # Ensure we have enough level names
        while len(level_names) < max_levels:
            level_names.append(f"Level{len(level_names)+1}")
    
    # Create a dictionary to store each level
    result = {name: pd.Series(index=source_df.index, dtype='object') for name in level_names}
    
    # Process each row
    for idx, value in source_df[hierarchy_field].items():
        if pd.isna(value):
            # Set all levels to None for missing values
            for name in level_names:
                result[name].iloc[idx] = None
            continue
            
        # Convert to string
        hierarchy = str(value).strip()
        
        # Process each level
        for i in range(max_levels):
            level_name = level_names[i]
            start = 0
            end = (i + 1) * level_step
            
            if end <= len(hierarchy):
                # Extract the level value
                level_value = hierarchy[start:end]
                result[level_name].iloc[idx] = level_value
            else:
                # Beyond the hierarchy depth for this value
                result[level_name].iloc[idx] = None
    
    return result