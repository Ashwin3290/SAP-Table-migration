import os
import sqlite3
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure pandas display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.precision', 3)

# Custom JSON encoder to handle NumPy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

class DatabaseAnalyzer:
    """Class to analyze the SQLite database structure and content"""
    
    def __init__(self, db_path='db.sqlite3'):
        """Initialize with database path"""
        self.db_path = db_path
        self.conn = None
        self.tables = []
        self.analysis_results = {}
        self.output_dir = "database_analysis_results"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def connect(self):
        """Connect to the database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"Successfully connected to {self.db_path}")
            return True
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            return False
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            print("Connection closed")
    
    def get_table_list(self):
        """Get list of all tables in the database"""
        if not self.conn:
            self.connect()
        
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(query, self.conn)
        self.tables = tables['name'].tolist()
        print(f"Found {len(self.tables)} tables in the database")
        return self.tables
    
    def get_table_schema(self, table_name):
        """Get schema for a specific table"""
        if not self.conn:
            self.connect()
        
        query = f"PRAGMA table_info({table_name});"
        try:
            schema = pd.read_sql_query(query, self.conn)
            return schema
        except Exception as e:
            print(f"Error getting schema for table {table_name}: {str(e)}")
            return pd.DataFrame()
    
    def get_table_data(self, table_name, limit=100):
        """Get data from a specific table with optional limit"""
        if not self.conn:
            self.connect()
        
        query = f"SELECT * FROM {table_name} LIMIT {limit};"
        try:
            data = pd.read_sql_query(query, self.conn)
            return data
        except Exception as e:
            print(f"Error querying table {table_name}: {str(e)}")
            return pd.DataFrame()
    
    def get_row_count(self, table_name):
        """Get the number of rows in a table"""
        if not self.conn:
            self.connect()
        
        query = f"SELECT COUNT(*) as count FROM {table_name};"
        try:
            result = pd.read_sql_query(query, self.conn)
            return result['count'].iloc[0]
        except Exception as e:
            print(f"Error counting rows in {table_name}: {str(e)}")
            return 0
    
    def analyze_table(self, table_name):
        """Analyze a specific table - schema, data samples, statistics"""
        try:
            schema = self.get_table_schema(table_name)
            row_count = self.get_row_count(table_name)
            sample_data = self.get_table_data(table_name, limit=5)
            
            # Get column statistics if there's data
            if not sample_data.empty:
                # Handle different data types appropriately
                numeric_columns = sample_data.select_dtypes(include=['number']).columns.tolist()
                categorical_columns = sample_data.select_dtypes(include=['object']).columns.tolist()
                
                # Basic statistics for numeric columns
                numeric_stats = {}
                if len(numeric_columns) > 0:
                    try:
                        # Convert to Python native types to avoid JSON serialization issues
                        stats_df = sample_data[numeric_columns].describe()
                        
                        for col in numeric_columns:
                            if col in stats_df:
                                col_stats = {}
                                for stat, value in stats_df[col].items():
                                    # Convert numpy types to Python native types
                                    col_stats[stat] = float(value) if not pd.isna(value) else None
                                numeric_stats[col] = col_stats
                    except Exception as e:
                        print(f"Error calculating numeric stats for {table_name}: {str(e)}")
                
                # Value counts for categorical columns (limited to top 10)
                categorical_stats = {}
                for col in categorical_columns:
                    try:
                        value_counts = sample_data[col].value_counts().head(10)
                        # Convert to dict with Python native types
                        value_counts_dict = {str(k): int(v) for k, v in value_counts.items()}
                        
                        categorical_stats[col] = {
                            'value_counts': value_counts_dict,
                            'null_count': int(sample_data[col].isna().sum()),
                            'unique_count': int(sample_data[col].nunique())
                        }
                    except Exception as e:
                        print(f"Error analyzing column {col}: {str(e)}")
            
                # Make sure schema is serializable
                schema_list = []
                for _, row in schema.iterrows():
                    row_dict = {}
                    for col_name, value in row.items():
                        # Convert any non-serializable values to strings
                        try:
                            json.dumps(value)
                            row_dict[col_name] = value
                        except (TypeError, OverflowError):
                            row_dict[col_name] = str(value)
                    schema_list.append(row_dict)
                
                # Convert sample data to serializable format
                sample_list = []
                for _, row in sample_data.iterrows():
                    row_dict = {}
                    for col_name, value in row.items():
                        # Convert any non-serializable values to strings
                        try:
                            json.dumps(value)
                            row_dict[col_name] = value
                        except (TypeError, OverflowError):
                            row_dict[col_name] = str(value)
                    sample_list.append(row_dict)
                
                # Generate a JSON-serializable result
                table_analysis = {
                    'table_name': table_name,
                    'row_count': int(row_count),
                    'schema': schema_list,
                    'sample_data': sample_list,
                    'numeric_statistics': numeric_stats,
                    'categorical_statistics': categorical_stats
                }
                
                return table_analysis
            else:
                return {
                    'table_name': table_name,
                    'row_count': int(row_count),
                    'schema': schema.to_dict('records'),
                    'sample_data': [],
                    'error': 'No data available or error querying table'
                }
        except Exception as e:
            print(f"Error analyzing table {table_name}: {str(e)}")
            return {
                'table_name': table_name,
                'error': f"Analysis failed: {str(e)}"
            }
    
    def analyze_database(self):
        """Analyze the entire database structure and content"""
        if not self.tables:
            self.get_table_list()
        
        # Analyze each table
        for table in self.tables:
            print(f"Analyzing table: {table}")
            try:
                self.analysis_results[table] = self.analyze_table(table)
            except Exception as e:
                print(f"Error during analysis of {table}: {str(e)}")
                self.analysis_results[table] = {
                    'table_name': table,
                    'error': f"Analysis failed: {str(e)}"
                }
        
        return self.analysis_results
    
    def find_relationships(self):
        """Attempt to identify relationships between tables based on column names"""
        if not self.tables:
            self.get_table_list()
        
        relationships = []
        
        # Get all column names for each table
        table_columns = {}
        for table in self.tables:
            try:
                schema = self.get_table_schema(table)
                if not schema.empty:
                    table_columns[table] = schema['name'].tolist()
                else:
                    table_columns[table] = []
            except Exception as e:
                print(f"Error getting columns for {table}: {str(e)}")
                table_columns[table] = []
        
        # Look for potential foreign keys
        for table1 in self.tables[:50]:  # Limit the search to first 50 tables to avoid timeout
            for table2 in self.tables[:50]:
                if table1 != table2:
                    for col1 in table_columns.get(table1, []):
                        # Check for exact column name matches
                        if col1 in table_columns.get(table2, []):
                            # Check if values overlap
                            try:
                                query = f"""
                                SELECT COUNT(*) as count FROM 
                                (SELECT DISTINCT "{col1}" FROM "{table1}"
                                INTERSECT
                                SELECT DISTINCT "{col1}" FROM "{table2}")
                                """
                                result = pd.read_sql_query(query, self.conn)
                                overlap_count = int(result['count'].iloc[0])
                                
                                if overlap_count > 0:
                                    relationships.append({
                                        'table1': table1,
                                        'table2': table2,
                                        'column': col1,
                                        'overlap_count': overlap_count
                                    })
                            except Exception as e:
                                # Skip problematic comparisons
                                print(f"Error checking relationship between {table1}.{col1} and {table2}.{col1}: {str(e)}")
        
        return relationships
    
    def export_analysis_to_json(self):
        """Export the analysis results to a JSON file"""
        if not self.analysis_results:
            self.analyze_database()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/database_analysis_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, cls=NumpyEncoder)
            
            print(f"Analysis exported to {filename}")
            return filename
        except Exception as e:
            print(f"Error exporting analysis to JSON: {str(e)}")
            
            # Try to save partial results
            partial_filename = f"{self.output_dir}/partial_analysis_{timestamp}.json"
            try:
                # Save tables individually
                successful_tables = {}
                for table_name, analysis in self.analysis_results.items():
                    try:
                        # Test if the table analysis can be serialized
                        json.dumps(analysis, cls=NumpyEncoder)
                        successful_tables[table_name] = analysis
                    except:
                        successful_tables[table_name] = {
                            'table_name': table_name,
                            'error': 'Could not serialize table analysis'
                        }
                
                with open(partial_filename, 'w') as f:
                    json.dump(successful_tables, f, indent=2, cls=NumpyEncoder)
                
                print(f"Partial analysis exported to {partial_filename}")
                return partial_filename
            except Exception as e2:
                print(f"Error saving partial results: {str(e2)}")
                return None
    
    def generate_summary_report(self):
        """Generate a summary report of the database"""
        if not self.analysis_results:
            self.analyze_database()
        
        # Create summary dataframe
        summary_data = []
        for table_name, analysis in self.analysis_results.items():
            summary_data.append({
                'table_name': table_name,
                'row_count': analysis.get('row_count', 0),
                'column_count': len(analysis.get('schema', [])),
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Export summary to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"{self.output_dir}/summary_report_{timestamp}.csv"
        summary_df.to_csv(summary_filename, index=False)
        
        print(f"Summary report exported to {summary_filename}")
        
        # Generate visualization only if there's meaningful data
        if not summary_df.empty and summary_df['row_count'].sum() > 0:
            # Show only the top 30 tables by row count for better visualization
            plot_df = summary_df.sort_values('row_count', ascending=False).head(30)
            
            plt.figure(figsize=(15, 8))
            sns.barplot(x='table_name', y='row_count', data=plot_df)
            plt.title('Number of Rows in Top 30 Tables')
            plt.xticks(rotation=90)
            plt.tight_layout()
            chart_filename = f"{self.output_dir}/row_count_chart_{timestamp}.png"
            plt.savefig(chart_filename)
            plt.close()
        else:
            chart_filename = None
        
        # Analyze relationships
        try:
            relationships = self.find_relationships()
            relationships_df = pd.DataFrame(relationships)
            if not relationships_df.empty:
                rel_filename = f"{self.output_dir}/relationships_{timestamp}.csv"
                relationships_df.to_csv(rel_filename, index=False)
                print(f"Relationship analysis exported to {rel_filename}")
            else:
                rel_filename = None
                print("No relationships found or relationship analysis failed")
        except Exception as e:
            print(f"Error analyzing relationships: {str(e)}")
            rel_filename = None
        
        return {
            'summary': summary_filename,
            'visualization': chart_filename,
            'relationships': rel_filename
        }
    
    def analyze_key_tables(self):
        """Analyze key tables (MARA, MAKT, etc.) in more detail"""
        key_tables = ['MARA', 'MAKT', 'MARC', 'VBAK', 'VBAP', 'KNA1']
        key_tables_analysis = {}
        
        for table in key_tables:
            if table in self.tables:
                try:
                    print(f"Performing detailed analysis of key table: {table}")
                    
                    # Get all rows for detailed analysis
                    query = f"SELECT * FROM {table};"
                    df = pd.read_sql_query(query, self.conn)
                    
                    if not df.empty:
                        # Detailed statistics
                        stats = {
                            'row_count': len(df),
                            'column_count': len(df.columns),
                            'memory_usage': str(df.memory_usage(deep=True).sum() / (1024 * 1024)) + " MB",
                            'columns': df.columns.tolist(),
                            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                            'null_counts': {col: int(df[col].isna().sum()) for col in df.columns},
                            'unique_counts': {col: int(df[col].nunique()) for col in df.columns}
                        }
                        
                        # Get sample rows (maximum 10)
                        sample_rows = []
                        for _, row in df.head(10).iterrows():
                            sample_row = {}
                            for col, value in row.items():
                                # Make sure the value is JSON serializable
                                if isinstance(value, (np.integer, np.floating)):
                                    sample_row[col] = value.item()
                                elif pd.isna(value):
                                    sample_row[col] = None
                                else:
                                    sample_row[col] = str(value)
                            sample_rows.append(sample_row)
                        
                        key_tables_analysis[table] = {
                            'statistics': stats,
                            'sample_rows': sample_rows
                        }
                    else:
                        key_tables_analysis[table] = {'error': 'Table is empty'}
                
                except Exception as e:
                    print(f"Error analyzing key table {table}: {str(e)}")
                    key_tables_analysis[table] = {'error': str(e)}
            else:
                key_tables_analysis[table] = {'error': 'Table not found in database'}
        
        # Export the detailed analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/key_tables_analysis_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(key_tables_analysis, f, indent=2, cls=NumpyEncoder)
            print(f"Key tables analysis exported to {filename}")
            return filename
        except Exception as e:
            print(f"Error exporting key tables analysis: {str(e)}")
            return None
    
    def analyze_session_data(self):
        """Analyze the session data in the sessions folder"""
        sessions_dir = "sessions"
        if not os.path.exists(sessions_dir):
            print(f"Sessions directory not found: {sessions_dir}")
            return None
        
        session_folders = [f for f in os.listdir(sessions_dir) if os.path.isdir(os.path.join(sessions_dir, f))]
        session_data = []
        
        for session_id in session_folders:
            session_path = os.path.join(sessions_dir, session_id)
            context_file = os.path.join(session_path, "context.json")
            target_file = os.path.join(session_path, "target_latest.csv")
            
            session_info = {"session_id": session_id}
            
            # Extract context information if available
            if os.path.exists(context_file):
                try:
                    with open(context_file, 'r') as f:
                        context = json.load(f)
                    session_info['source_table'] = context.get('source_table_name')
                    session_info['target_table'] = context.get('target_table_name')
                    session_info['query'] = context.get('restructured_question')
                    
                    # Extract source and target fields
                    session_info['source_fields'] = context.get('source_field_names', [])
                    session_info['target_fields'] = context.get('target_sap_fields', [])
                    session_info['filtering_fields'] = context.get('filtering_fields', [])
                    
                    # Extract insertion mappings
                    if 'insertion_fields' in context:
                        session_info['insertion_mappings'] = context['insertion_fields']
                    
                    # Extract transformation history
                    if 'context' in context and 'transformation_history' in context['context']:
                        session_info['transformations'] = len(context['context']['transformation_history'])
                        session_info['fields_modified'] = []
                        session_info['transformation_history'] = []
                        
                        for transform in context['context']['transformation_history']:
                            if 'fields_modified' in transform:
                                session_info['fields_modified'].extend(transform['fields_modified'])
                            
                            # Add detailed transformation info
                            session_info['transformation_history'].append({
                                'description': transform.get('description', ''),
                                'fields_modified': transform.get('fields_modified', []),
                                'filter_conditions': transform.get('filter_conditions', {})
                            })
                            
                        session_info['fields_modified'] = list(set(session_info['fields_modified']))
                        
                    # Extract target table state
                    if 'context' in context and 'target_table_state' in context['context']:
                        session_info['target_table_state'] = context['context']['target_table_state']
                except Exception as e:
                    print(f"Error reading context for session {session_id}: {str(e)}")
            
            # Extract target data information if available
            if os.path.exists(target_file):
                try:
                    target_df = pd.read_csv(target_file)
                    session_info['target_rows'] = len(target_df)
                    session_info['target_columns'] = list(target_df.columns)
                    session_info['non_empty_columns'] = [col for col in target_df.columns 
                                                        if not target_df[col].isna().all()]
                    
                    # Include sample data (first 3 rows)
                    sample_rows = []
                    for _, row in target_df.head(3).iterrows():
                        sample_row = {}
                        for col, value in row.items():
                            if pd.isna(value):
                                sample_row[col] = None
                            elif isinstance(value, (np.integer, np.floating)):
                                sample_row[col] = value.item()
                            else:
                                sample_row[col] = str(value)
                        sample_rows.append(sample_row)
                    
                    session_info['target_sample'] = sample_rows
                except Exception as e:
                    print(f"Error reading target data for session {session_id}: {str(e)}")
            
            session_data.append(session_info)
        
        # Save the session analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_filename = f"{self.output_dir}/session_analysis_{timestamp}.json"
        try:
            with open(session_filename, 'w') as f:
                json.dump(session_data, f, indent=2, cls=NumpyEncoder)
            
            print(f"Session analysis exported to {session_filename}")
        except Exception as e:
            print(f"Error exporting session analysis: {str(e)}")
            
            # Try to save a simplified version
            try:
                simplified_data = []
                for session in session_data:
                    simplified = {
                        'session_id': session['session_id'],
                        'source_table': session.get('source_table'),
                        'target_table': session.get('target_table'),
                        'query': session.get('query'),
                        'transformations': session.get('transformations')
                    }
                    simplified_data.append(simplified)
                
                simple_filename = f"{self.output_dir}/simple_session_analysis_{timestamp}.json"
                with open(simple_filename, 'w') as f:
                    json.dump(simplified_data, f, indent=2)
                
                print(f"Simplified session analysis exported to {simple_filename}")
                session_filename = simple_filename
            except:
                print("Failed to save even simplified session data")
        
        # Create a summary dataframe
        try:
            if session_data:
                # Create a simplified DataFrame with key information
                summary_data = []
                for session in session_data:
                    summary_data.append({
                        'session_id': session['session_id'],
                        'source_table': session.get('source_table', ''),
                        'target_table': session.get('target_table', ''),
                        'transformations': session.get('transformations', 0),
                        'fields_modified_count': len(session.get('fields_modified', [])),
                        'target_rows': session.get('target_rows', 0)
                    })
                
                session_df = pd.DataFrame(summary_data)
                session_summary = f"{self.output_dir}/session_summary_{timestamp}.csv"
                session_df.to_csv(session_summary, index=False)
                print(f"Session summary exported to {session_summary}")
        except Exception as e:
            print(f"Error creating session summary: {str(e)}")
        
        return session_data

def main():
    print("Starting TableLLM Database Analysis...")
    analyzer = DatabaseAnalyzer()
    
    if analyzer.connect():
        tables = analyzer.get_table_list()
        print(f"Tables found: {', '.join(tables[:20])}{'...' if len(tables) > 20 else ''}")
        
        # Analyze the entire database
        print("\nAnalyzing database structure and content...")
        analyzer.analyze_database()
        
        # Export analysis to JSON
        json_file = analyzer.export_analysis_to_json()
        print(f"Complete analysis exported to: {json_file}")
        
        # Generate summary report
        print("\nGenerating summary report...")
        summary_files = analyzer.generate_summary_report()
        print(f"Summary report files: {summary_files}")
        
        # Analyze key tables in more detail
        print("\nAnalyzing key SAP tables...")
        key_tables_file = analyzer.analyze_key_tables()
        
        # Analyze session data
        print("\nAnalyzing session data...")
        session_data = analyzer.analyze_session_data()
        
        print("\nAnalysis completed successfully!")
    else:
        print("Failed to connect to the database.")
    
    analyzer.close()

if __name__ == "__main__":
    main()
