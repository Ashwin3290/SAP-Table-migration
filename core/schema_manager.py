"""
Schema Manager for TableLLM
This module manages SAP schema information
"""
import sqlite3
import pandas as pd
from utils.logging_utils import main_logger as logger
from utils.sql_utils import get_table_schema, validate_sql_identifier, table_exists
import config

class SchemaManager:
    """
    Manages SAP schema information with field mappings and relationships
    """
    
    def __init__(self, db_connection=None):
        """
        Initialize the schema manager
        
        Parameters:
        db_connection (sqlite3.Connection, optional): Database connection
        """
        # Initialize database connection
        self.conn = db_connection
        if not self.conn:
            try:
                self.conn = sqlite3.connect(config.DATABASE_PATH)
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                self.conn = None
        
        # Initialize schema cache
        self.schema_cache = {}
        self.relationship_cache = {}
        
        logger.info("Initialized SchemaManager")
    
    def get_schema(self, table_name, prune_columns=None):
        """
        Get schema information for an SAP table
        
        Parameters:
        table_name (str): SAP table name
        prune_columns (list, optional): Optional list of columns to keep, prunes others
        
        Returns:
        dict: Schema information including columns, keys, relationships
        """
        try:
            # Validate table name
            safe_table = validate_sql_identifier(table_name)
            
            # Check cache first
            if safe_table in self.schema_cache:
                schema = self.schema_cache[safe_table].copy()
                
                # Prune columns if requested
                if prune_columns:
                    schema["columns"] = {k: v for k, v in schema["columns"].items() 
                                        if k in prune_columns}
                
                return schema
            
            # Not in cache, get schema from database
            if not self.conn:
                logger.warning(f"No database connection to get schema for {safe_table}")
                return {"columns": {}, "keys": [], "relationships": []}
            
            # Check if table exists
            if not table_exists(self.conn, safe_table):
                logger.warning(f"Table {safe_table} does not exist")
                return {"columns": {}, "keys": [], "relationships": []}
            
            # Get schema
            schema_df = get_table_schema(self.conn, safe_table)
            
            # Process schema information
            columns = {}
            keys = []
            
            for _, row in schema_df.iterrows():
                column_name = row["name"]
                
                # Skip if pruning and column not in prune list
                if prune_columns and column_name not in prune_columns:
                    continue
                
                # Column information
                column_info = {
                    "type": row["type"],
                    "nullable": row["notnull"] == 0
                }
                
                # Check if primary key
                if "pk" in schema_df.columns and row["pk"] > 0:
                    column_info["is_key"] = True
                    keys.append(column_name)
                else:
                    column_info["is_key"] = False
                
                columns[column_name] = column_info
            
            # Get relationships
            relationships = self._get_table_relationships(safe_table)
            
            # Create schema object
            schema = {
                "columns": columns,
                "keys": keys,
                "relationships": relationships
            }
            
            # Cache schema (without pruning)
            if not prune_columns:
                self.schema_cache[safe_table] = schema.copy()
            
            return schema
        except Exception as e:
            logger.error(f"Error getting schema for {table_name}: {e}")
            return {"columns": {}, "keys": [], "relationships": []}
    
    def _get_table_relationships(self, table_name):
        """
        Get relationships for a table from foreign key constraints
        
        Parameters:
        table_name (str): Table name
        
        Returns:
        list: List of relationship dictionaries
        """
        # Check relationship cache first
        if table_name in self.relationship_cache:
            return self.relationship_cache[table_name].copy()
        
        relationships = []
        
        try:
            if not self.conn:
                return relationships
            
            # Get foreign key information
            fk_query = f"""
            PRAGMA foreign_key_list({table_name})
            """
            
            fk_df = pd.read_sql_query(fk_query, self.conn)
            
            # Process foreign keys
            for _, row in fk_df.iterrows():
                relationship = {
                    "type": "foreign_key",
                    "from_table": table_name,
                    "from_column": row["from"],
                    "to_table": row["table"],
                    "to_column": row["to"]
                }
                relationships.append(relationship)
            
            # Also include common SAP relationships
            sap_relationships = self._get_sap_table_relationships(table_name)
            if sap_relationships:
                relationships.extend(sap_relationships)
            
            # Cache relationships
            self.relationship_cache[table_name] = relationships.copy()
            
            return relationships
        except Exception as e:
            logger.error(f"Error getting relationships for {table_name}: {e}")
            return relationships
    
    def _get_sap_table_relationships(self, table_name):
        """
        Get common SAP table relationships that may not be defined as foreign keys
        
        Parameters:
        table_name (str): Table name
        
        Returns:
        list: List of relationship dictionaries
        """
        relationships = []
        
        # Common SAP relationships
        sap_relationships = {
            "MARA": [
                {"type": "common_key", "from_table": "MARA", "from_column": "MATNR", 
                 "to_table": "MARC", "to_column": "MATNR", "description": "Material to Plant Data"},
                {"type": "common_key", "from_table": "MARA", "from_column": "MATNR", 
                 "to_table": "MAKT", "to_column": "MATNR", "description": "Material to Description"},
                {"type": "common_key", "from_table": "MARA", "from_column": "MATNR", 
                 "to_table": "MARD", "to_column": "MATNR", "description": "Material to Storage Location"}
            ],
            "MAKT": [
                {"type": "common_key", "from_table": "MAKT", "from_column": "MATNR", 
                 "to_table": "MARA", "to_column": "MATNR", "description": "Description to Material"}
            ],
            "MARC": [
                {"type": "common_key", "from_table": "MARC", "from_column": "MATNR", 
                 "to_table": "MARA", "to_column": "MATNR", "description": "Plant Data to Material"},
                {"type": "common_key", "from_table": "MARC", "from_column": "WERKS", 
                 "to_table": "T001W", "to_column": "WERKS", "description": "Plant to Plant Master"}
            ],
            "KNA1": [
                {"type": "common_key", "from_table": "KNA1", "from_column": "KUNNR", 
                 "to_table": "KNB1", "to_column": "KUNNR", "description": "Customer to Company Code Data"},
                {"type": "common_key", "from_table": "KNA1", "from_column": "KUNNR", 
                 "to_table": "KNVV", "to_column": "KUNNR", "description": "Customer to Sales Data"}
            ],
            "KNB1": [
                {"type": "common_key", "from_table": "KNB1", "from_column": "KUNNR", 
                 "to_table": "KNA1", "to_column": "KUNNR", "description": "Company Code Data to Customer"}
            ],
            "KNVV": [
                {"type": "common_key", "from_table": "KNVV", "from_column": "KUNNR", 
                 "to_table": "KNA1", "to_column": "KUNNR", "description": "Sales Data to Customer"}
            ],
            "MARA_500": [
                {"type": "common_key", "from_table": "MARA_500", "from_column": "MATNR", 
                 "to_table": "MARA", "to_column": "MATNR", "description": "Material 500 to Material"}
            ],
            "MARA_800": [
                {"type": "common_key", "from_table": "MARA_800", "from_column": "MATNR", 
                 "to_table": "MARA", "to_column": "MATNR", "description": "Material 800 to Material"}
            ]
        }
        
        # Return relationships for this table if defined
        if table_name in sap_relationships:
            return sap_relationships[table_name]
        
        return []
    
    def get_field_mappings(self, source_table, target_table):
        """
        Get common field mappings between source and target tables
        
        Parameters:
        source_table (str): Source table name
        target_table (str): Target table name
        
        Returns:
        dict: Dictionary mapping target fields to source fields
        """
        mappings = {}
        
        try:
            # Get schemas for both tables
            source_schema = self.get_schema(source_table)
            target_schema = self.get_schema(target_table)
            
            # Find common field names or common relationships
            source_columns = set(source_schema["columns"].keys())
            target_columns = set(target_schema["columns"].keys())
            
            # First, look for exact field name matches
            common_fields = source_columns.intersection(target_columns)
            for field in common_fields:
                mappings[field] = field
            
            # Next, look for relationships in the relationships list
            for rel in source_schema["relationships"]:
                if rel["to_table"] == target_table:
                    # This is a relationship from source to target
                    mappings[rel["to_column"]] = rel["from_column"]
            
            for rel in target_schema["relationships"]:
                if rel["to_table"] == source_table:
                    # This is a relationship from target to source
                    mappings[rel["from_column"]] = rel["to_column"]
            
            # SAP-specific field mappings
            sap_mappings = self._get_sap_field_mappings(source_table, target_table)
            mappings.update(sap_mappings)
            
            return mappings
        except Exception as e:
            logger.error(f"Error getting field mappings between {source_table} and {target_table}: {e}")
            return mappings
    
    def _get_sap_field_mappings(self, source_table, target_table):
        """
        Get SAP-specific field mappings between tables
        
        Parameters:
        source_table (str): Source table name
        target_table (str): Target table name
        
        Returns:
        dict: Dictionary mapping target fields to source fields
        """
        # Common SAP field mappings between tables
        table_mappings = {
            ("MARA", "Material_Basic_Segment"): {
                "MATNR": "MATNR",   # Material Number
                "MTART": "MTART",   # Material Type
                "MBRSH": "MBRSH",   # Industry Sector
                "MEINS": "MEINS"    # Base Unit of Measure
            },
            ("MAKT", "Material_Description_Segment"): {
                "MATNR": "MATNR",   # Material Number
                "MAKTX": "MAKTX",   # Material Description
                "SPRAS": "SPRAS"    # Language Key
            },
            ("MARC", "Material_Plant_Segment"): {
                "MATNR": "MATNR",   # Material Number
                "WERKS": "WERKS",   # Plant
                "LGORT": "LGORT"    # Storage Location
            },
            ("KNA1", "Customer"): {
                "KUNNR": "KUNNR",   # Customer Number
                "NAME1": "NAME1",   # Name
                "ORT01": "ORT01",   # City
                "LAND1": "LAND1"    # Country Key
            }
        }
        
        # Check for exact match
        key = (source_table, target_table)
        if key in table_mappings:
            return table_mappings[key]
        
        # Check for partial table name match
        for (src, tgt), mappings in table_mappings.items():
            if source_table.startswith(src) and target_table.startswith(tgt):
                return mappings
        
        # No specific mappings found
        return {}
    
    def get_all_schemas(self):
        """
        Get schemas for all tables in the database
        
        Returns:
        dict: Dictionary mapping table names to schemas
        """
        all_schemas = {}
        
        try:
            if not self.conn:
                logger.warning("No database connection to get all schemas")
                return all_schemas
            
            # Get all tables
            tables_query = """
            SELECT name FROM sqlite_master WHERE type='table'
            """
            tables_df = pd.read_sql_query(tables_query, self.conn)
            
            # Get schema for each table
            for _, row in tables_df.iterrows():
                table_name = row["name"]
                all_schemas[table_name] = self.get_schema(table_name)
            
            return all_schemas
        except Exception as e:
            logger.error(f"Error getting all schemas: {e}")
            return all_schemas
    
    def close(self):
        """
        Close the database connection
        """
        if self.conn:
            self.conn.close()
            self.conn = None
