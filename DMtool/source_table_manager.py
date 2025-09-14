import os
import logging
import sqlite3
import re
import csv
import json
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

# Import LLM manager for transformation detection
from DMtool.llm_config import LLMManager

load_dotenv()
logger = logging.getLogger(__name__)

class EnhancedSourceTableManager:
    """
    Enhanced manager that handles:
    1. Source table (_src) operations for data preservation
    2. Column-level lineage tracking for transformations
    3. Pre-load/post-load CSV report generation
    """
    
    # Patterns that indicate data transformation/modification - DO NOT SYNC these to _src
    TRANSFORMATION_PATTERNS = [
        # String manipulations that change data
        r'\b(substr|substring|trim|ltrim|rtrim|upper|lower|replace|reverse)\s*\(',
        r'regexp_replace\s*\(',
        r'split_string\s*\(',
        r'proper_case\s*\(',
        r'left_pad\s*\(',
        r'right_pad\s*\(',
        
        # Mathematical operations (except simple column references)
        r'[\+\-\*\/\%]\s*(?:\d+|[\'"][^\'"]+[\'"])',  # Math with literals
        r'\b(round|abs|ceil|floor|sqrt|power|log)\s*\(',
        r'safe_divide\s*\(',
        r'percentage\s*\(',
        
        # Date/time manipulations that transform data
        r'date\s*\(\s*[\'"]now[\'\"]\s*\)',  # Current date
        r'datetime\s*\(\s*[\'"]now[\'\"]\s*\)',
        r'strftime\s*\(',  # Date formatting
        r'date_add_days\s*\(',
        r'date_diff_days\s*\(',
        r'format_date\s*\(',
        r'to_date\s*\(',
        
        # CASE statements with hardcoded values (business logic)
        r'CASE\s+WHEN\s+.*?\s+THEN\s+[\'"][^\'"]+[\'"]',  # CASE with literal strings
        r'CASE\s+WHEN\s+.*?\s+THEN\s+\d+',  # CASE with literal numbers
        
        # Hardcoded value assignments (not from another table)
        r'SET\s+\w+\s*=\s*[\'"][^\'"]+[\'"](?!\s*FROM)',  # String literal not from subquery
        r'SET\s+\w+\s*=\s*\d+(?:\.\d+)?(?!\s*FROM)',  # Number literal not from subquery
        r'SET\s+\w+\s*=\s*NULL',  # Setting to NULL (data deletion)
        
        # Calculated/derived values
        r'length\s*\(',  # String length calculation
        r'count\s*\(',  # Aggregations
        r'sum\s*\(',
        r'avg\s*\(',
        r'min\s*\(',
        r'max\s*\(',
        
        # Conditional transformations
        r'COALESCE\s*\([^,]+,[^)]*[\'"][^\'"]+[\'"][^)]*\)',  # COALESCE with literals
        r'IFNULL\s*\([^,]+,\s*[\'"][^\'"]+[\'\"]\s*\)',  # IFNULL with literals
        r'IIF\s*\(',  # IIF function
        
        # JSON operations
        r'json_extract_value\s*\(',
        
        # Validation functions (these compute new values)
        r'is_numeric\s*\(',
        r'is_email\s*\(',
        r'is_phone\s*\(',
        r'is_valid_json\s*\(',
    ]
    
    # Operations that should NEVER sync (even if they don't transform data)
    NEVER_SYNC_PATTERNS = [
        r'DROP\s+TABLE',
        r'TRUNCATE\s+TABLE',
        r'CREATE\s+(?:TEMP|TEMPORARY)\s+',  # Temporary objects
        r'CREATE\s+VIEW',  # Views
        r'CREATE\s+INDEX',  # Indexes
    ]
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize Enhanced Source Table Manager"""
        self.db_path = db_path or os.environ.get('DB_PATH')
        self.sync_enabled = os.environ.get('ENABLE_SOURCE_TABLE_BACKUP', 'true').lower() == 'true'
        self.lineage_enabled = os.environ.get('ENABLE_LINEAGE_TRACKING', 'true').lower() == 'true'
        
        logger.info(f"EnhancedSourceTableManager initialized - Sync: {self.sync_enabled}, Lineage: {self.lineage_enabled}")
        
        self.stats = {
            'sync_attempts': 0,
            'sync_successes': 0,
            'sync_failures': 0,
            'tables_created': 0,
            'lineage_captures': 0,
            'reports_generated': 0,
            'llm_transformation_calls': 0,
            'last_sync': None,
            'last_lineage_capture': None
        }

    def handle_query_execution(self, query: str, target_table: str, planner_info: Dict[str, Any],
                              params: Optional[Dict] = None, 
                              main_execution_successful: bool = True,
                              session_id: Optional[str] = None,
                              is_selection_criteria = False) -> Dict[str, Any]:
        """
        Main function to handle both source table sync AND lineage tracking
        
        Parameters:
        query (str): The executed SQL query
        target_table (str): Target table name
        planner_info (Dict): Information from the planner
        params (Optional[Dict]): Query parameters
        main_execution_successful (bool): Whether main execution succeeded
        session_id (Optional[str]): Session ID for tracking
        
        Returns:
        Dict: Combined results from both sync and lineage operations
        """
        start_time = datetime.now()
        
        result = {
            "sync_attempted": False,
            "sync_successful": False,
            "lineage_captured": False,
            "should_sync": False,
            "has_transformation": False,
            "reason": "",
            "target_table": target_table,
            "execution_time_ms": 0,
            "lineage_metadata": {}
        }
        
        if not main_execution_successful:
            result["reason"] = "Main execution failed"
            return result
        
        # 1. Determine if query has transformations (using LLM)
        has_transformation = self._has_transformation(query)
        result["has_transformation"] = has_transformation

        # 3. Handle lineage tracking (especially for transformations)
        if self.lineage_enabled and planner_info:
            lineage_result = self._capture_column_lineage(query, target_table, planner_info, session_id)
            result["lineage_captured"] = bool(lineage_result)
            result["lineage_metadata"] = lineage_result
            
            if lineage_result:
                self.stats['lineage_captures'] += 1
                self.stats['last_lineage_capture'] = datetime.now().isoformat()
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        result["execution_time_ms"] = round(execution_time, 2)
        
        return result
    
    def _has_transformation(self, query: str) -> bool:
        """Check if query contains data transformations using LLM"""
        try:
            # Use LLM for a general transformation check
            prompt = f"""
Does this SQL query transform/alter data or just move existing data as-is?

QUERY: {query}

Answer with only: YES (if it transforms data) or NO (if it just moves data)
"""
            
            llm = LLMManager(
                provider="google",
                model="gemini/gemini-2.5-flash",
                api_key=os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
            )
            
            response = llm.generate(prompt, temperature=0.1, max_tokens=10)
            self.stats['llm_transformation_calls'] += 1
            
            if response and response.strip().upper().startswith("YES"):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in LLM transformation check: {e}")
            # Fallback to regex patterns
            return self._has_complex_transformation_regex(query)
    
    def _has_complex_transformation_regex(self, query: str) -> bool:
        """Fallback regex-based transformation detection"""
        # Check transformation patterns
        for pattern in self.TRANSFORMATION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        # Check for complex transformations
        return self._has_complex_transformation(query)
    
    def _handle_source_table_sync(self, query: str, target_table: str, 
                                 params: Optional[Dict] = None, 
                                 main_execution_successful: bool = True) -> Dict[str, Any]:
        """Handle source table synchronization (existing logic)"""
        result = {
            "sync_attempted": False,
            "sync_successful": False,
            "should_sync": False,
            "reason": "",
            "src_table_exists": False,
        }
        
        self.stats['sync_attempts'] += 1
        
        # Check if we should sync
        should_sync, reason = self.should_sync_to_source(query, target_table)
        result["should_sync"] = should_sync
        result["reason"] = reason
        
        if not should_sync:
            logger.info(f"Source sync not needed for {target_table}: {reason}")
            return result
        
        logger.info(f"Starting source sync for {target_table}")
        
        # Ensure source table exists
        src_exists = self.ensure_src_table_exists(target_table)
        result["src_table_exists"] = src_exists
        
        if not src_exists:
            result["reason"] = f"Could not create/access _src table for {target_table}"
            self.stats['sync_failures'] += 1
            return result
        
        # Attempt the sync
        result["sync_attempted"] = True
        sync_success = self.execute_on_src_table(query, target_table, params)
        result["sync_successful"] = sync_success
        
        if sync_success:
            result["reason"] = f"Successfully synced to {target_table}_src"
            self.stats['sync_successes'] += 1
            self.stats['last_sync'] = datetime.now().isoformat()
        else:
            result["reason"] = f"Failed to sync to {target_table}_src"
            self.stats['sync_failures'] += 1
        
        return result
    
    def _capture_column_lineage(self, query: str, target_table: str, planner_info: Dict[str, Any], 
                               session_id: Optional[str] = None) -> Dict[str, Any]:
        """Capture column-level lineage information"""
        try:
            lineage_metadata = {
                "session_id": session_id,
                "target_table": target_table,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "columns": {}
            }
            
            # Extract field mappings from planner
            qualified_source_fields = planner_info.get("qualified_source_fields", [])
            qualified_target_fields = planner_info.get("qualified_target_fields", [])
            insertion_fields = planner_info.get("insertion_fields", [])
            target_sap_fields = planner_info.get("target_sap_fields", [])
            
            # Map each target column to its source
            for i, target_field in enumerate(target_sap_fields):
                # Clean the target field name (remove table prefix if present)
                clean_target_field = self._clean_column_name(target_field)
                
                column_lineage = {
                    "target_column": clean_target_field,
                    "source_info": None,
                    "transformation_applied": False,
                    "transformation_type": "none",
                    "query_executed": query
                }
                
                # Find corresponding source field
                if i < len(insertion_fields):
                    source_field = insertion_fields[i]
                    # Find qualified source field
                    for qualified_field in qualified_source_fields:
                        if qualified_field.endswith(f".{source_field}"):
                            table_name = qualified_field.split('.')[0]
                            column_lineage["source_info"] = {
                                "table": table_name,
                                "column": source_field,
                                "qualified_name": qualified_field
                            }
                            break
                
                # Use LLM to detect transformation for this specific column
                is_transformation, transformation_type = self._detect_transformation_with_llm(query, clean_target_field)
                column_lineage["transformation_applied"] = is_transformation
                column_lineage["transformation_type"] = transformation_type
                
                lineage_metadata["columns"][clean_target_field] = column_lineage
            
            # Store lineage metadata in session if session_id provided
            if session_id:
                self._store_lineage_metadata(session_id, lineage_metadata)
            
            return lineage_metadata
            
        except Exception as e:
            logger.error(f"Error capturing column lineage: {e}")
            return {}
    
    def _detect_transformation_with_llm(self, query: str, target_column: str) -> Tuple[bool, str]:
        """
        Use LLM to determine if a query transforms data or just moves it as-is
        
        Returns:
        Tuple[bool, str]: (is_transformation, transformation_type)
        """
        try:
            prompt = f"""
Analyze this SQL query and determine if it transforms/alters data or just moves data as-is.

SQL QUERY:
{query}

TARGET COLUMN: {target_column}

QUESTION: Does this query transform/alter the data for column '{target_column}' such that the values in the target would be DIFFERENT from the source values?

EXAMPLES OF TRANSFORMATION (answer YES):
- CASE WHEN price > 100 THEN 'EXPENSIVE' ELSE 'CHEAP' END (creates new values)
- SUM(amount) (aggregates multiple values)
- UPPER(name) (changes text case)
- price * 1.1 (mathematical calculation)
- 'CONSTANT_VALUE' (hardcoded values)

EXAMPLES OF DATA MOVEMENT (answer NO):
- CASE WHEN table1.id IS NOT NULL THEN table1.field ELSE table2.field END (picks existing values)
- COALESCE(field1, field2) (picks first non-null existing value)
- Simple SELECT field FROM table (direct copy)
- JOIN operations that combine existing values without changing them

RESPOND WITH ONLY:
- "YES" if data is transformed/altered
- "NO" if data is just moved/copied as-is

If YES, also provide the transformation type from: aggregation, string_manipulation, date_transformation, conditional_logic, constant_assignment, mathematical, value_mapping

FORMAT: YES|transformation_type OR NO
"""

            llm = LLMManager(
                provider="google",
                model="gemini/gemini-2.5-flash",
                api_key=os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
            )
            
            response = llm.generate(prompt, temperature=0.1, max_tokens=50)
            self.stats['llm_transformation_calls'] += 1
            
            if response:
                response = response.strip().upper()
                
                if response.startswith("YES"):
                    # Extract transformation type if provided
                    if "|" in response:
                        parts = response.split("|")
                        transformation_type = parts[1].strip().lower() if len(parts) > 1 else "conditional_logic"
                    else:
                        transformation_type = "conditional_logic"  # default for YES responses
                    return True, transformation_type
                elif response.startswith("NO"):
                    return False, "none"
                else:
                    # Fallback: if response is unclear, assume no transformation
                    logger.warning(f"Unclear LLM response for transformation detection: {response}")
                    return False, "none"
            else:
                logger.warning("No response from LLM for transformation detection")
                return False, "none"
                
        except Exception as e:
            logger.error(f"Error in LLM transformation detection: {e}")
            # Fallback to regex-based detection for critical errors
            return self._fallback_regex_detection(query, target_column)

    def _fallback_regex_detection(self, query: str, target_column: str) -> Tuple[bool, str]:
        """Fallback regex-based detection for when LLM fails"""
        query_upper = query.upper()
        
        # Simple fallback patterns for obvious transformations
        obvious_transformations = [
            (r'SUM\(', 'aggregation'),
            (r'COUNT\(', 'aggregation'),
            (r'AVG\(', 'aggregation'),
            (r'UPPER\(', 'string_manipulation'),
            (r'LOWER\(', 'string_manipulation'),
            (r'SUBSTR\(', 'string_manipulation'),
            (r'[\+\-\*\/]\s*\d+', 'mathematical'),
            (r"THEN\s+['\"][^'\"]*['\"]", 'constant_assignment'),
            (r'DATE\(', 'date_transformation')
        ]
        
        for pattern, trans_type in obvious_transformations:
            if re.search(pattern, query, re.IGNORECASE):
                return True, trans_type
        
        return False, "none"
    
    def _clean_column_name(self, column_name: str) -> str:
        """
        Clean column name by removing table prefixes
        Examples:
        - 't_24_Product_Basic_Data_mandatory_Ext.MATKL' → 'MATKL'
        - 'MARA.MATNR' → 'MATNR'
        - 'PRODUCT' → 'PRODUCT' (unchanged)
        """
        if not column_name:
            return column_name
        
        # If it contains a dot, take everything after the last dot
        if '.' in column_name:
            return column_name.split('.')[-1]
        
        return column_name
    
    def generate_preload_postload_csv(self, target_table: str, session_id: Optional[str] = None) -> str:
        """
        Generate CSV report showing pre-load vs post-load data comparison
        
        Parameters:
        target_table (str): Target table name
        session_id (str): Session ID to get lineage metadata
        
        Returns:
        str: Path to generated CSV file
        """
        if not self.lineage_enabled:
            logger.warning("Lineage tracking disabled, cannot generate report")
            return ""
        
        try:
            # Get lineage metadata
            lineage_metadata = []
            if session_id:
                lineage_metadata = self._get_lineage_metadata(session_id)
            
            if not lineage_metadata:
                logger.warning(f"No lineage metadata found for session {session_id}")
                return ""
            
            # Build comparison query
            comparison_query = self._build_comparison_query(target_table, lineage_metadata)
            if not comparison_query:
                logger.error("Failed to build comparison query")
                return ""
            
            # Execute query and generate CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{target_table}_preload_postload_{timestamp}.csv"
            csv_path = os.path.join("reports", csv_filename)
            
            # Ensure reports directory exists
            os.makedirs("reports", exist_ok=True)
            
            # Execute query and write to CSV
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(comparison_query)
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            # Write CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_names)
                writer.writerows(cursor.fetchall())
            
            conn.close()
            
            self.stats['reports_generated'] += 1
            logger.info(f"Generated pre-load/post-load report: {csv_path}")
            return csv_path
            
        except Exception as e:
            logger.error(f"Error generating CSV report: {e}")
            return ""
    
    def _build_comparison_query(self, target_table: str, lineage_metadata: List[Dict[str, Any]]) -> str:
        """Build a query that joins source tables and shows side-by-side comparison"""
        try:
            if not lineage_metadata:
                return ""
            
            # Get primary key
            primary_key = self._get_primary_key(target_table)
            if not primary_key:
                primary_key = "ROWID"
            
            select_parts = [f"t.{primary_key} as primary_key"]
            joins = []
            join_tables = set()
            
            # Process each transformation's lineage
            for transformation in lineage_metadata:
                columns = transformation.get("columns", {})
                
                for column_name, column_info in columns.items():
                    source_info = column_info.get("source_info")
                    transformation_applied = column_info.get("transformation_applied", False)
                    
                    if source_info and source_info.get("table"):
                        source_table = source_info["table"]
                        source_column = source_info["column"]
                        
                        # Add source column
                        if transformation_applied:
                            alias = f"s_{source_table.lower()}"
                            select_parts.append(f"{alias}.{source_column} as {column_name}_source")
                            
                            if source_table not in join_tables:
                                joins.append(f"LEFT JOIN {source_table} {alias} ON t.{primary_key} = {alias}.{primary_key}")
                                join_tables.add(source_table)
                        else:
                            # For non-transformed data, source and target should be same
                            select_parts.append(f"t.{column_name} as {column_name}_source")
                    else:
                        # Handle constants or missing source info
                        select_parts.append(f"NULL as {column_name}_source")
                    
                    # Always add target column
                    select_parts.append(f"t.{column_name} as {column_name}_target")
            
            # Build final query
            query = f"""
            SELECT {', '.join(select_parts)}
            FROM {target_table} t
            {' '.join(joins)}
            ORDER BY t.{primary_key}
            """
            
            return query.strip()
            
        except Exception as e:
            logger.error(f"Error building comparison query: {e}")
            return ""
    
    def should_sync_to_source(self, query: str, target_table: str) -> Tuple[bool, str]:
        """Determine if a query should be synchronized to the _src table."""
        if not self.sync_enabled:
            return False, "Source table sync disabled"
        
        if not query or not target_table:
            return False, "Missing query or target table"
        
        query_upper = query.upper().strip()
        
        # Check if this operates on the target table
        if not self._is_target_table_operation(query_upper, target_table):
            return False, "Not a target table operation"
        
        # NEVER sync these operations
        for pattern in self.NEVER_SYNC_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.info(f"Never-sync pattern detected: {pattern[:30]}...")
                return False, f"Operation type not suitable for sync"
        
        # Check for transformation patterns - if found, DON'T sync
        for pattern in self.TRANSFORMATION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.info(f"Transformation pattern detected: {pattern[:50]}...")
                return False, f"Data transformation detected - preserving original data"
        
        # Special checks for complex cases
        if self._has_complex_transformation(query):
            return False, "Complex transformation detected"
        
        # If we get here, it's likely a simple data movement operation - SYNC IT
        query_type = self._identify_query_type(query_upper)
        logger.info(f"Query type '{query_type}' approved for sync to {target_table}_src")
        return True, f"Data preservation operation - {query_type}"
    
    def _has_complex_transformation(self, query: str) -> bool:
        """Additional checks for complex transformations"""
        query_upper = query.upper()
        
        # Check for UPDATE with complex SET clause
        if 'UPDATE' in query_upper and 'SET' in query_upper:
            set_match = re.search(r'SET\s+(.*?)(?:WHERE|FROM|$)', query_upper, re.DOTALL)
            if set_match:
                set_clause = set_match.group(1)
                
                # If SET clause has any function calls (except simple column references)
                if re.search(r'\w+\s*\([^)]*\)', set_clause):
                    # Check if it's just a simple subquery
                    if not re.search(r'^\s*\(\s*SELECT\s+\w+\s+FROM\s+\w+', set_clause):
                        return True
                
                # Multiple CASE statements usually indicate complex business logic
                if set_clause.count('CASE') > 1:
                    return True
        
        # Check for INSERT with complex SELECT
        if 'INSERT' in query_upper and 'SELECT' in query_upper:
            select_match = re.search(r'SELECT\s+(.*?)(?:FROM|$)', query_upper, re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                
                # If SELECT has calculations or functions (not just column names)
                if re.search(r'[\+\-\*\/]', select_clause):
                    return True
                if re.search(r'\w+\s*\([^)]*\)', select_clause):
                    # Allow simple column selections from subqueries
                    if not re.search(r'^\s*\w+\s*,?\s*$', select_clause):
                        return True
        
        return False
    
    def _identify_query_type(self, query_upper: str) -> str:
        """Identify the type of query for logging purposes"""
        if 'INSERT' in query_upper and 'SELECT' in query_upper:
            return "INSERT-SELECT"
        elif 'INSERT' in query_upper:
            return "INSERT"
        elif 'UPDATE' in query_upper and 'FROM' in query_upper:
            return "UPDATE-FROM"
        elif 'UPDATE' in query_upper:
            return "UPDATE"
        elif 'ALTER TABLE' in query_upper and 'ADD COLUMN' in query_upper:
            return "ALTER-ADD"
        elif 'ALTER TABLE' in query_upper and 'DROP COLUMN' in query_upper:
            return "ALTER-DROP"
        elif 'DELETE' in query_upper:
            return "DELETE"
        else:
            return "OTHER"
    
    def _is_target_table_operation(self, query_upper: str, target_table: str) -> bool:
        """Check if query operates on the target table"""
        target_upper = target_table.upper()
        
        patterns = [
            rf'\bINSERT\s+(?:OR\s+\w+\s+)?INTO\s+\[?{re.escape(target_upper)}\]?\b',
            rf'\bUPDATE\s+\[?{re.escape(target_upper)}\]?\b',
            rf'\bALTER\s+TABLE\s+\[?{re.escape(target_upper)}\]?\b',
            rf'\bDELETE\s+FROM\s+\[?{re.escape(target_upper)}\]?\b',
        ]
        
        return any(re.search(pattern, query_upper) for pattern in patterns)
    
    def get_table_schema(self, table_name: str, conn: sqlite3.Connection) -> str:
        """Get the complete CREATE TABLE statement for a table"""
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT sql FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            
            result = cursor.fetchone()
            if result:
                return result[0]
            
            logger.warning(f"No schema found for table {table_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {e}")
            return None
    
    def replicate_table_schema(self, source_table: str, target_table: str, conn: sqlite3.Connection) -> bool:
        """Replicate the exact schema from source table to target table"""
        try:
            logger.info(f"Replicating schema from {source_table} to {target_table}")
            
            create_sql = self.get_table_schema(source_table, conn)
            if not create_sql:
                logger.error(f"Could not get schema for {source_table}")
                return False
            
            # Replace table name in CREATE statement
            create_sql = re.sub(
                r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["\']?\w+["\']?',
                f'CREATE TABLE IF NOT EXISTS {target_table}',
                create_sql,
                count=1,
                flags=re.IGNORECASE
            )
            
            cursor = conn.cursor()
            cursor.execute(create_sql)
            
            # Copy indexes
            cursor.execute("""
                SELECT sql FROM sqlite_master 
                WHERE type='index' AND tbl_name=? AND sql IS NOT NULL
            """, (source_table,))
            
            indexes = cursor.fetchall()
            for index_sql in indexes:
                if index_sql[0]:
                    new_index_sql = index_sql[0].replace(source_table, target_table)
                    new_index_sql = re.sub(
                        r'CREATE\s+(UNIQUE\s+)?INDEX\s+(\w+)',
                        lambda m: f"CREATE {m.group(1) or ''}INDEX {m.group(2)}_src",
                        new_index_sql
                    )
                    try:
                        cursor.execute(new_index_sql)
                    except sqlite3.Error as e:
                        logger.warning(f"Could not create index for {target_table}: {e}")
            
            conn.commit()
            logger.info(f"Successfully replicated schema for {target_table}")
            return True
            
        except Exception as e:
            logger.error(f"Error replicating schema: {e}")
            return False
    
    def ensure_src_table_exists(self, target_table: str) -> bool:
        """Ensure the _src table exists with the same schema as the target table"""
        if not self.sync_enabled or not self.db_path:
            return False
        
        src_table = f"{target_table}_src"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if source table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (src_table,))
            
            if cursor.fetchone():
                logger.debug(f"Source table {src_table} already exists")
                conn.close()
                return True
            
            logger.info(f"Creating source table {src_table}")
            
            if self.replicate_table_schema(target_table, src_table, conn):
                # Copy existing data
                cursor.execute(f"INSERT INTO {src_table} SELECT * FROM {target_table}")
                conn.commit()
                
                cursor.execute(f"SELECT COUNT(*) FROM {src_table}")
                row_count = cursor.fetchone()[0]
                
                logger.info(f"Created source table {src_table} with {row_count} rows")
                self.stats['tables_created'] += 1
                
                conn.close()
                return True
            else:
                conn.close()
                return False
                
        except Exception as e:
            logger.error(f"Error creating source table {src_table}: {e}")
            return False
    
    def execute_on_src_table(self, query: str, target_table: str, params: Optional[Dict] = None) -> bool:
        """Execute the query on the _src table"""
        if not self.sync_enabled or not self.db_path:
            return False
        
        src_table = f"{target_table}_src"
        
        # Replace table name in query
        src_query = re.sub(
            r'\b' + re.escape(target_table) + r'\b',
            src_table,
            query
        )
        
        logger.debug(f"Executing on source table {src_table}: {src_query[:100]}...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if params:
                cursor.execute(src_query, params)
            else:
                cursor.execute(src_query)
            
            rows_affected = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully executed on source table {src_table}, {rows_affected} rows affected")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute on source table {src_table}: {e}")
            return False
    
    def _store_lineage_metadata(self, session_id: str, lineage_metadata: Dict[str, Any]):
        """Store lineage metadata in session storage"""
        try:
            session_dir = f"sessions/{session_id}"
            os.makedirs(session_dir, exist_ok=True)
            
            lineage_file = f"{session_dir}/lineage_metadata.json"
            
            # Load existing lineage data
            existing_lineage = []
            if os.path.exists(lineage_file):
                try:
                    with open(lineage_file, 'r') as f:
                        existing_lineage = json.load(f)
                except json.JSONDecodeError:
                    existing_lineage = []
            
            # Append new lineage metadata
            existing_lineage.append(lineage_metadata)
            
            # Save updated lineage data
            with open(lineage_file, 'w') as f:
                json.dump(existing_lineage, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error storing lineage metadata: {e}")
    
    def _get_lineage_metadata(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve lineage metadata for a session"""
        try:
            lineage_file = f"sessions/{session_id}/lineage_metadata.json"
            if os.path.exists(lineage_file):
                with open(lineage_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error retrieving lineage metadata: {e}")
            return []
    
    def _get_primary_key(self, table_name: str) -> Optional[str]:
        """Get primary key column for a table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table info to find primary key
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            for column in columns:
                if column[5]:  # pk column in PRAGMA table_info
                    conn.close()
                    return column[1]  # column name
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error getting primary key for {table_name}: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get enhanced source table manager statistics"""
        return self.stats.copy()
    
    def verify_sync_integrity(self, target_table: str) -> Dict[str, Any]:
        """Verify that source and target tables are in sync"""
        if not self.sync_enabled or not self.db_path:
            return {"error": "Source sync disabled or no DB path"}
        
        src_table = f"{target_table}_src"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (src_table,))
            
            if not cursor.fetchone():
                return {"error": f"Source table {src_table} does not exist"}
            
            cursor.execute(f"SELECT COUNT(*) FROM {target_table}")
            target_count = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM {src_table}")
            src_count = cursor.fetchone()[0]
            
            target_schema = self.get_table_schema(target_table, conn)
            src_schema = self.get_table_schema(src_table, conn)
            
            target_schema_normalized = re.sub(r'\b' + re.escape(target_table) + r'\b', 'TABLE', target_schema or '')
            src_schema_normalized = re.sub(r'\b' + re.escape(src_table) + r'\b', 'TABLE', src_schema or '')
            
            schemas_match = target_schema_normalized == src_schema_normalized
            
            conn.close()
            
            result = {
                "target_table": target_table,
                "source_table": src_table,
                "target_row_count": target_count,
                "source_row_count": src_count,
                "row_counts_match": target_count == src_count,
                "schemas_match": schemas_match,
                "in_sync": (target_count == src_count) and schemas_match
            }
            
            if not result["in_sync"]:
                logger.warning(f"Tables not in sync: {target_table} vs {src_table}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error verifying sync integrity: {e}")
            return {"error": str(e)}

# Global instance for easy access
_enhanced_source_manager = None

def get_enhanced_source_manager() -> EnhancedSourceTableManager:
    """Get global enhanced source table manager instance"""
    global _enhanced_source_manager
    if _enhanced_source_manager is None:
        _enhanced_source_manager = EnhancedSourceTableManager()
    return _enhanced_source_manager

def handle_is_selection_criteria(query: str, target_table: str, 
                      params: Optional[Dict] = None,
                      main_execution_successful: bool = True) -> Dict[str, Any]:
    """Convenience function for handling source table sync (backward compatibility)"""
    manager = get_enhanced_source_manager()
    return manager._handle_source_table_sync(query, target_table, params, main_execution_successful)

def handle_query_execution(query: str, target_table: str, planner_info: Dict[str, Any],
                          params: Optional[Dict] = None, 
                          main_execution_successful: bool = True,
                          session_id: Optional[str] = None,
                          is_selection_criteria = False) -> Dict[str, Any]:
    """Enhanced function for handling both sync and lineage tracking"""
    manager = get_enhanced_source_manager()
    return manager.handle_query_execution(query, target_table, planner_info, params, main_execution_successful, session_id,is_selection_criteria=is_selection_criteria)

def generate_lineage_report(target_table: str, session_id: Optional[str] = None) -> str:
    """Convenience function for generating lineage report"""
    manager = get_enhanced_source_manager()
    return manager.generate_preload_postload_csv(target_table, session_id)