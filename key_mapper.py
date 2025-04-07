import json
import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Any, Tuple

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

def analyze_tables_for_key_fields(config_json_path: str, db_path: str = "db.sqlite3", output_json_path: str = "analysis_results.json"):
    """
    Analyzes tables from a configuration JSON file to identify potential key fields.
    
    Args:
        config_json_path: Path to the configuration JSON file
        db_path: Path to the SQLite database
        output_json_path: Path to save the analysis results
        
    Returns:
        Dict containing the analysis results
    """
    # Load configuration
    with open(config_json_path, 'r') as f:
        config = json.load(f)
    
    # Initialize result dictionary
    results = {
        "original_config": config,
        "key_analysis": {},
        "field_metrics": {}
    }
    
    # Function to read table data from SQLite database
    def read_table_data(table_name: str) -> pd.DataFrame:
        """
        Reads data from a table in the SQLite database.
        
        Args:
            table_name: Name of the table to read
            
        Returns:
            DataFrame containing the table data
        """
        try:
            # Connect to SQLite database
            conn = sqlite3.connect(db_path)
            
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone():
                print(f"Warning: Table '{table_name}' does not exist in the database")
                return pd.DataFrame()
            
            # Read table data into DataFrame
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            
            # Close connection
            conn.close()
            
            return df
            
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error reading table data: {e}")
            return pd.DataFrame()
    
    # Analyze all tables mentioned in the config
    all_tables = set(config.get("Source Tables", []) + config.get("Target Tables", []))
    
    for table_name in all_tables:
        print(f"Analyzing table: {table_name}")
        
        # Read table data
        df = read_table_data(table_name)
        
        if df.empty:
            results["key_analysis"][table_name] = {"error": f"Unable to read table data for {table_name}"}
            continue
        
        # Column metrics
        column_metrics = {}
        potential_keys = []
        
        # Analyze each column
        for column in df.columns:
            total_count = len(df)
            unique_count = df[column].nunique()
            null_count = df[column].isna().sum()
            uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Convert sample values to Python native types to avoid serialization issues
            sample_values = []
            for val in df[column].head(3):
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    sample_values.append(int(val))
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    sample_values.append(float(val))
                else:
                    sample_values.append(val)
            
            # Store column metrics
            column_metrics[column] = {
                "total_rows": int(total_count),
                "unique_values": int(unique_count),
                "null_values": int(null_count),
                "uniqueness_ratio": float(round(uniqueness_ratio, 4)),
                "data_type": str(df[column].dtype),
                "avg_length": float(round(df[column].astype(str).str.len().mean(), 2)) if df[column].dtype == object else None,
                "max_length": int(df[column].astype(str).str.len().max()) if df[column].dtype == object else None,
                "sample_values": sample_values
            }
            
            # Determine if column could be a key
            # Criteria: high uniqueness ratio (>0.9) and no null values
            is_potential_key = uniqueness_ratio > 0.9 and null_count == 0
            column_metrics[column]["is_potential_key"] = bool(is_potential_key)
            
            if is_potential_key:
                potential_keys.append(column)
        
        # Store results for this table
        results["field_metrics"][table_name] = column_metrics
        results["key_analysis"][table_name] = {
            "potential_key_fields": potential_keys,
            "recommended_primary_key": potential_keys[0] if potential_keys else None,
            "composite_key_recommendations": []
        }
        
        # Check for composite keys if no single field is a perfect key
        if not any(column_metrics[col]["uniqueness_ratio"] == 1.0 for col in df.columns if column_metrics[col]["is_potential_key"]):
            # Try pairs of columns
            for i, col1 in enumerate(df.columns):
                for col2 in df.columns[i+1:]:
                    # Skip if either column has null values
                    if df[col1].isna().any() or df[col2].isna().any():
                        continue
                        
                    # Check uniqueness of combination
                    combined = df[[col1, col2]].drop_duplicates()
                    combined_uniqueness = len(combined) / len(df)
                    
                    if combined_uniqueness > 0.95:  # 95% threshold for composite key
                        results["key_analysis"][table_name]["composite_key_recommendations"].append({
                            "fields": [col1, col2],
                            "uniqueness_ratio": float(round(combined_uniqueness, 4))
                        })
    
    # Add analysis of the mapping relationship
    results["mapping_analysis"] = analyze_mapping_relationship(config, results["field_metrics"])
    
    # Handle missing tables gracefully - create empty entries for tables that weren't found
    for table in all_tables:
        if table not in results["key_analysis"]:
            results["key_analysis"][table] = {"error": f"Table {table} not found or empty"}
        if table not in results["field_metrics"]:
            results["field_metrics"][table] = {}
    
    # Save results to output file
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"Analysis completed and saved to {output_json_path}")
    return results

def analyze_mapping_relationship(config: Dict, field_metrics: Dict) -> Dict:
    """
    Analyzes the mapping relationship between source and target fields.
    
    Args:
        config: The configuration dictionary
        field_metrics: The field metrics dictionary
        
    Returns:
        Dict containing mapping analysis
    """
    source_tables = config.get("Source Tables", [])
    target_tables = config.get("Target Tables", [])
    source_fields = config.get("Source Fields", [])
    target_fields = config.get("Target Fields", [])
    
    mapping_analysis = {
        "source_to_target_mapping": {}
    }
    
    # Check if source fields can be properly mapped to target fields
    for i, source_field in enumerate(source_fields):
        if i < len(target_fields):
            target_field = target_fields[i]
            
            # Find which source table contains this field
            source_table_with_field = None
            for table in source_tables:
                if table in field_metrics and source_field in field_metrics.get(table, {}):
                    source_table_with_field = table
                    break
            
            # Find which target table contains this field
            target_table_with_field = None
            for table in target_tables:
                if table in field_metrics and target_field in field_metrics.get(table, {}):
                    target_table_with_field = table
                    break
            
            mapping_info = {
                "source_table": source_table_with_field,
                "target_table": target_table_with_field,
            }
            
            # Check if both fields exist in their respective tables
            if source_table_with_field and source_field in field_metrics.get(source_table_with_field, {}) and \
               target_table_with_field and target_field in field_metrics.get(target_table_with_field, {}):
                source_metrics = field_metrics[source_table_with_field][source_field]
                target_metrics = field_metrics[target_table_with_field][target_field]
                
                mapping_info.update({
                    "data_type_match": source_metrics["data_type"] == target_metrics["data_type"],
                    "length_compatibility": check_length_compatibility(source_metrics, target_metrics),
                    "is_source_key": source_metrics.get("is_potential_key", False),
                    "is_target_key": target_metrics.get("is_potential_key", False),
                    "recommendation": get_mapping_recommendation(source_metrics, target_metrics)
                })
            else:
                # Handle case where one or both fields aren't found
                if not source_table_with_field or source_field not in field_metrics.get(source_table_with_field, {}):
                    mapping_info["source_field_error"] = f"Field {source_field} not found in any source table"
                
                if not target_table_with_field or target_field not in field_metrics.get(target_table_with_field, {}):
                    mapping_info["target_field_error"] = f"Field {target_field} not found in any target table"
                
                mapping_info["recommendation"] = "ERROR: Cannot analyze mapping due to missing fields"
            
            mapping_analysis["source_to_target_mapping"][f"{source_field} -> {target_field}"] = mapping_info
    
    return mapping_analysis

def check_length_compatibility(source_metrics: Dict, target_metrics: Dict) -> bool:
    """
    Checks if the field lengths are compatible between source and target.
    
    Args:
        source_metrics: Metrics for the source field
        target_metrics: Metrics for the target field
        
    Returns:
        Boolean indicating if lengths are compatible
    """
    # Only applicable for string fields
    if source_metrics.get("max_length") is None or target_metrics.get("max_length") is None:
        return True
    
    # Check if source data fits in target field
    return source_metrics["max_length"] <= target_metrics["max_length"]

def get_mapping_recommendation(source_metrics: Dict, target_metrics: Dict) -> str:
    """
    Provides a recommendation for the field mapping.
    
    Args:
        source_metrics: Metrics for the source field
        target_metrics: Metrics for the target field
        
    Returns:
        String with a recommendation
    """
    if not source_metrics.get("data_type") == target_metrics.get("data_type"):
        return "WARNING: Data type mismatch may require conversion"
    
    if source_metrics.get("max_length") and target_metrics.get("max_length"):
        if source_metrics["max_length"] > target_metrics["max_length"]:
            return "WARNING: Source data may be truncated in target field"
    
    if source_metrics.get("is_potential_key", False) and not target_metrics.get("is_potential_key", False):
        return "NOTE: Mapping a key field to a non-key field"
    
    if not source_metrics.get("is_potential_key", False) and target_metrics.get("is_potential_key", False):
        return "WARNING: Mapping a non-key field to a key field may cause issues"
    
    return "OK: Compatible mapping"

if __name__ == "__main__":
    # Example usage
    # First, save the configuration to a file
    config = {
        "Source Tables": ["MARA", "MAKT"],
        "Target Tables": ["S_MARA"],
        "Source Fields": ["MATNR", "MAKTX"],
        "Target Fields": ["MAKTX"],
        "Insertion Fields": ["MAKTX"],
        "Filtering Conditions": ["Transformation 1 contains materials"],
        "Transformation Logic": [
            "For each material in Transformation 1, retrieve corresponding material descriptions from MAKT table and load into S_MARA table."
        ],
        "Additional Information": [
            "The query requests to retrieve material descriptions (MAKTX) from the MAKT table for materials present in Transformation 1 and map them to the S_MARA table. The source table is MAKT and the target table is S_MARA. The join condition is based on the material number (MATNR). The source field is MAKTX and the target field is also MAKTX."
        ]
    }
    
    with open("transformation_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run the analysis
    results = analyze_tables_for_key_fields("transformation_config.json", "db.sqlite3", "analysis_results.json")
    
    # Print a summary
    print("\nKey Field Analysis Summary:")
    for table, analysis in results["key_analysis"].items():
        print(f"\nTable: {table}")
        if "error" in analysis:
            print(f"  Error: {analysis['error']}")
            continue
            
        print(f"  Potential key fields: {', '.join(analysis['potential_key_fields'])}")
        print(f"  Recommended primary key: {analysis['recommended_primary_key']}")
        
        if analysis["composite_key_recommendations"]:
            print("  Composite key recommendations:")
            for rec in analysis["composite_key_recommendations"]:
                print(f"    {' + '.join(rec['fields'])} (uniqueness: {rec['uniqueness_ratio']})")