import os
import pandas as pd
import streamlit as st
from pymongo import MongoClient
import sqlite3
import json
from datetime import datetime

from code_exec import save_file
from tablellm import TableLLM

# Page configuration
st.set_page_config(
    page_title="TableLLM", 
    layout="wide",
    page_icon="📊"
)

# Initialize MongoDB connection if available
db_client = None
try:
    db_client = MongoClient('localhost', 27017, serverSelectionTimeoutMS=1000).table_llm
    db_client.admin.command('ping')  # Check connection
except Exception as e:
    st.sidebar.warning(f"Database connection failed: {e}")

# Initialize SQLite connection for data access
sqlite_conn = None
try:
    sqlite_conn = sqlite3.connect('db.sqlite3')
    st.sidebar.success("Connected to SQLite database")
except Exception as e:
    st.sidebar.warning(f"SQLite connection failed: {e}")

# Initialize TableLLM
tablellm = TableLLM()

# Initialize session state variables
if 'show_code' not in st.session_state:
    st.session_state['show_code'] = True

if 'transformation_session_id' not in st.session_state:
    st.session_state['transformation_session_id'] = None

if 'tables' not in st.session_state:
    st.session_state['tables'] = {
        'single': {
            'file': None,
            'df': None,
            'text': None,
            'details': None
        },
        'first': {
            'file': None,
            'df': None,
            'text': None,
            'details': None
        },
        'second': {
            'file': None,
            'df': None,
            'text': None,
            'details': None
        }
    }

if 'history' not in st.session_state:
    st.session_state['history'] = {
        'single': {
            'question': None,
            'code': None,
            'result': None,
            'session_id': 0
        },
        'transformation': {
            'questions': [],
            'codes': [],
            'results': [],
            'session_id': None
        }
    }

if 'transformation_history' not in st.session_state:
    st.session_state['transformation_history'] = []

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .code-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .example-query {
        padding: 0.5rem;
        background-color: #f1f3f5;
        border-radius: 4px;
        cursor: pointer;
        margin-bottom: 0.5rem;
    }
    .example-query:hover {
        background-color: #e9ecef;
    }
    .history-item {
        padding: 0.5rem;
        background-color: #e9ecef;
        border-left: 4px solid #0066cc;
        margin-bottom: 0.5rem;
    }
    .context-info {
        padding: 0.8rem;
        background-color: #f0f7ff;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main app header
st.markdown('<p class="main-header">TableLLM: Context-Aware Data Transformations</p>', unsafe_allow_html=True)
st.markdown("Process sequential data transformations with context awareness, preserving state between queries.")

# Create tabs for different operations
tab1, tab2 = st.tabs(["Single Table Analysis", "Sequential Data Transformations"])

# Single Table Analysis Tab
with tab1:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<p class="sub-header">Upload Your Data</p>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV, Excel, or DOCX file", 
            type=["csv", "xlsx", "xls", "docx"], 
            key="single_file_uploader"
        )
        
        # Set mode to Code
        mode = "Code"
        
        # Process uploaded file
        if uploaded_file:
            if uploaded_file != st.session_state['tables']['single']['file']:
                try:
                    with st.spinner("Processing file..."):
                        df, text, details = save_file(uploaded_file)
                        st.session_state['tables']['single'] = {
                            'file': uploaded_file,
                            'df': df,
                            'text': text,
                            'details': details
                        }
                except Exception as e:
                    st.error(f"Error processing file: {e}")
            
            # Display the DataFrame
            st.dataframe(
                st.session_state['tables']['single']['df'],
                height=400,
                use_container_width=True
            )
            
            # Show file info
            if st.session_state['tables']['single']['details']:
                with st.expander("File Details"):
                    st.write(f"Filename: {st.session_state['tables']['single']['details']['name']}")
                    st.write(f"Rows: {len(st.session_state['tables']['single']['df'])}")
                    st.write(f"Columns: {len(st.session_state['tables']['single']['df'].columns)}")
        else:
            st.info("Please upload a file to begin analysis")
    
    with col2:
        st.markdown('<p class="sub-header">Ask a Question</p>', unsafe_allow_html=True)
        
        # Query input
        query = st.text_area(
            "What would you like to know about your data?",
            height=100,
            key="single_query"
        )
        
        # Submit button and progress indicator
        col_button1, col_button2, col_button3 = st.columns([1, 1, 1])
        with col_button3:
            submit = st.button("📊 Analyze", key="single_analyze_button", use_container_width=True)
        
        # Process the query
        if submit and query and st.session_state['tables']['single']['df'] is not None:
            with st.spinner("Analyzing data..."):
                # Get response from TableLLM
                code, result = tablellm.process_query(
                    query,
                    st.session_state['tables']['single']['text'],
                    st.session_state['tables']['single']['df'],
                    mode=mode
                )
                
                st.session_state['history']['single'] = {
                    'question': query,
                    'code': code,
                    'result': result,
                    'session_id': 0
                }
            
            # Display results
            st.success("Analysis complete!")
            st.markdown("### Result")
            
            # Display code if in Code mode
            if mode == "Code":
                if st.session_state['show_code']:
                    with st.expander("Generated Python Code", expanded=True):
                        st.code(code, language="python")
                else:
                    if st.button("Show Code", key="show_code_button_single"):
                        st.session_state['show_code'] = True
                        st.rerun()
                
                # Display different result types appropriately
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result, use_container_width=True)
                elif str(type(result)).find('matplotlib') != -1:
                    st.pyplot(result)
                elif isinstance(result, (list, dict)):
                    st.json(result)
                else:
                    st.write(result)
            else:
                # QA mode - just display the text response
                st.markdown(code)
        
        # Example queries
        with st.expander("Example Queries"):
            example_queries = [
                "What's the average value in each column?",
                "Create a bar chart of the top 5 values",
                "Find correlations between numeric columns",
                "Show summary statistics for all columns",
                "Identify outliers in the data"
            ]
            
            for i, query in enumerate(example_queries):
                if st.button(query, key=f"example_{i}"):
                    st.session_state["single_query"] = query
                    st.rerun()

# Sequential Data Transformations Tab
with tab2:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<p class="sub-header">Configuration</p>', unsafe_allow_html=True)
        
        # Object and segment selection widgets
        object_id = st.number_input("Object ID", min_value=1, value=1, step=1, key="object_id")
        segment_id = st.number_input("Segment ID", min_value=1, value=1, step=1, key="segment_id")
        project_id = st.number_input("Project ID", min_value=1, value=1, step=1, key="project_id")
        
        # Session management
        st.markdown("### Session Management")
        col_session1, col_session2 = st.columns(2)
        
        with col_session1:
            if st.button("Create New Session", key="new_session"):
                # Create a new transformation session
                with st.spinner("Creating new session..."):
                    st.session_state['transformation_session_id'] = None
                    st.session_state['transformation_history'] = []
                    st.success("New session created!")
                    st.rerun()
        
        with col_session2:
            if st.button("Clear Current Session", key="clear_session") and st.session_state['transformation_session_id']:
                st.session_state['transformation_session_id'] = None
                st.session_state['transformation_history'] = []
                st.success("Session cleared!")
                st.rerun()
        
        # Display current session info
        if st.session_state['transformation_session_id']:
            session_info = tablellm.get_session_info(st.session_state['transformation_session_id'])
            
            st.markdown("### Current Session")
            st.markdown(f"""
            <div class="context-info">
                <strong>Session ID:</strong> {session_info['session_id']}<br>
                <strong>Transformations:</strong> {len(session_info['transformation_history'])}<br>
                <strong>Populated Fields:</strong> {', '.join(session_info['target_table_state'].get('populated_fields', ['None']))}
            </div>
            """, unsafe_allow_html=True)
            
            # Display transformation history
            if session_info['transformation_history']:
                with st.expander("Transformation History", expanded=False):
                    for i, tx in enumerate(session_info['transformation_history']):
                        st.markdown(f"""
                        <div class="history-item">
                            <strong>#{i+1}:</strong> {tx.get('description', 'Unknown transformation')}<br>
                            <strong>Modified:</strong> {', '.join(tx.get('fields_modified', []))}<br>
                            <strong>Filters:</strong> {json.dumps(tx.get('filter_conditions', {}))}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Display sample tables from database if connected
        if sqlite_conn:
            with st.expander("Available Database Tables", expanded=False):
                try:
                    cursor = sqlite_conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    for table in tables:
                        if st.button(table[0], key=f"table_{table[0]}"):
                            # Show sample data for this table
                            sample_df = pd.read_sql_query(f"SELECT * FROM '{table[0]}' LIMIT 5", sqlite_conn)
                            st.dataframe(sample_df)
                except Exception as e:
                    st.error(f"Error fetching tables: {e}")
        else:
            st.warning("SQLite database connection is required for data transformations")
    
    with col2:
        st.markdown('<p class="sub-header">Data Transformation Query</p>', unsafe_allow_html=True)
        
        # Query input
        query = st.text_area(
            "What transformation would you like to perform?",
            height=100,
            key="transformation_query",
            placeholder="Example: Bring Material Number with Material Type = ROH from MARA Table"
        )
        
        # Submit button and progress indicator
        col_button1, col_button2, col_button3 = st.columns([1, 1, 1])
        with col_button3:
            submit = st.button("🔄 Transform", key="transform_button", use_container_width=True)
        
        # Process the query for data transformation
        if submit and query and sqlite_conn:
            with st.spinner("Processing transformation..."):
                try:
                    # Process query with context awareness
                    code, result, session_id = tablellm.process_sequential_query(
                        query, 
                        object_id, 
                        segment_id, 
                        project_id,
                        st.session_state['transformation_session_id']
                    )
                    
                    # Update session ID if this is a new session
                    if not st.session_state['transformation_session_id']:
                        st.session_state['transformation_session_id'] = session_id
                    
                    # Get updated session info
                    session_info = tablellm.get_session_info(session_id)
                    
                    # Add to history
                    st.session_state['transformation_history'].append({
                        'query': query,
                        'code': code,
                        'timestamp': datetime.now().isoformat(),
                        'description': session_info['transformation_history'][-1]['description'] 
                            if session_info['transformation_history'] else "Unknown transformation"
                    })
                    
                    # Display results
                    st.success("Transformation complete!")
                    
                    # Display the transformation results
                    st.markdown("### Result")
                    
                    # Display code
                    if st.session_state['show_code']:
                        with st.expander("Generated Python Code", expanded=True):
                            st.code(code, language="python")
                    
                    # Display different result types appropriately
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result, use_container_width=True)
                    elif str(type(result)).find('matplotlib') != -1:
                        st.pyplot(result)
                    elif isinstance(result, (list, dict)):
                        st.json(result)
                    else:
                        st.write(result)
                    
                    # Display summary of what happened
                    latest_tx = session_info['transformation_history'][-1] if session_info['transformation_history'] else {}
                    st.markdown(f"""
                    <div class="context-info">
                        <strong>Transformation Summary:</strong><br>
                        {latest_tx.get('description', 'Transformation completed')}<br>
                        <strong>Fields Modified:</strong> {', '.join(latest_tx.get('fields_modified', []))}<br>
                        <strong>Filter Conditions:</strong> {json.dumps(latest_tx.get('filter_conditions', {}))}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error processing transformation: {e}")
        
        # Display previous transformations in this session
        if st.session_state['transformation_history']:
            st.markdown("### Previous Transformations")
            for i, tx in enumerate(st.session_state['transformation_history']):
                with st.expander(f"Transformation #{i+1}: {tx['description']}", expanded=False):
                    st.write(f"Query: {tx['query']}")
                    st.code(tx['code'], language="python")
        
        # Example transformation queries
        with st.expander("Example Transformation Queries"):
            example_queries = [
                "Bring Material Number with Material Type = ROH from MARA Table",
                "Insert Material Description where Material Number already exists",
                "Fill Material Group where Material Type = FERT",
                "Add Plant = 1000 to materials with group CHEM",
                "Set Status = Active for all plants with Region = EU"
            ]
            
            for i, query in enumerate(example_queries):
                if st.button(query, key=f"example_tx_{i}"):
                    st.session_state["transformation_query"] = query
                    st.rerun()

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    
    # Code visibility toggle
    st.session_state['show_code'] = st.checkbox("Show generated code by default", value=st.session_state['show_code'])
    
    # Context information
    if st.session_state['transformation_session_id']:
        st.markdown("---")
        st.markdown("### Context Information")
        session_info = tablellm.get_session_info(st.session_state['transformation_session_id'])
        
        # Display target table state
        st.markdown("#### Target Table State")
        table_state = session_info['target_table_state']
        st.markdown(f"**Populated Fields:** {', '.join(table_state.get('populated_fields', ['None']))}")
        st.markdown(f"**Remaining Fields:** {', '.join(table_state.get('remaining_mandatory_fields', ['None']))}")
        
        # Display token usage if available
        try:
            from token_tracker import get_token_usage_stats
            token_stats = get_token_usage_stats()
            
            st.markdown("---")
            st.markdown("### Token Usage")
            st.markdown(f"**Total API Calls:** {token_stats['total_api_calls']}")
            st.markdown(f"**Input Tokens:** {token_stats['total_input_tokens']:,}")
            st.markdown(f"**Output Tokens:** {token_stats['total_output_tokens']:,}")
            st.markdown(f"**Total Tokens:** {token_stats['total_tokens']:,}")
        except:
            pass
    
    # About section
    st.markdown("---")
    st.markdown("### About Context-Aware Transformations")
    st.markdown("""
    This system maintains context across sequential data transformations to:
    
    - Preserve previously populated data
    - Track transformation history
    - Understand relationships between transformations
    - Maintain consistency in the target dataset
    """)

# Cleanup connections when app is closed
def cleanup():
    if sqlite_conn:
        sqlite_conn.close()

# Register the cleanup function
import atexit
atexit.register(cleanup)