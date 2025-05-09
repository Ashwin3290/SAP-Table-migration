import os
import pandas as pd
import streamlit as st
import sqlite3
import json
from datetime import datetime

from code_exec import save_file
from tablellm import TableLLM
from tablellm import get_session_context

# Page configuration
st.set_page_config(
    page_title="TableLLM", 
    layout="wide",
    page_icon="📊"
)

sqlite_conn = None
# try:
sqlite_conn = sqlite3.connect('db.sqlite3')
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
st.markdown('<p class="main-header">Gen-AI Data Migration Tool</p>', unsafe_allow_html=True)
# st.markdown("Process sequential data transformations with context awareness, preserving state between queries.")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown('<p class="sub-header">Configuration</p>', unsafe_allow_html=True)
    
    # Object and segment selection widgets
    object_id = st.number_input("Object ID", min_value=1, value=41, step=1, key="object_id")
    segment_id = st.number_input("Segment ID", min_value=1, value=577, step=1, key="segment_id")
    project_id = st.number_input("Project ID", min_value=1, value=24, step=1, key="project_id")
    
    # Session management
    st.markdown("### Session Management")
    col_session1, col_session2 = st.columns(2)

    
    with col_session1:
        if st.button("Clear Current Session", key="clear_session") and st.session_state['transformation_session_id']:
            st.session_state['transformation_session_id'] = None
            st.session_state['transformation_history'] = []
            st.success("Session cleared!")
            st.rerun()
    
    # Display current session info
    if st.session_state['transformation_session_id']:
        session_info = get_session_context(st.session_state['transformation_session_id'])
        
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
        try:
            cursor = sqlite_conn.cursor()
            
            # First get the target SAP table name from connection_segments
            cursor.execute("""
                SELECT table_name 
                FROM connection_segments 
                WHERE obj_id_id = ? 
                AND project_id_id = ? 
                AND segment_id = ?
            """, (object_id, project_id, segment_id))
            
            target_table_result = cursor.fetchone()
            
            if target_table_result:
                target_table = target_table_result[0]
                
                # Get columns from the target table
                cursor.execute(f"PRAGMA table_info('{target_table}')")
                columns = cursor.fetchall()
                
                if columns:
                    # Create a searchable select box for columns
                    column_options = [col[1] for col in columns]
                    print(column_options)  # col[1] is the column name
                    selected_columns = st.selectbox(
                        f"Select columns from {target_table}",
                        options=column_options, # Show all columns selected by default
                        key=f"columns_{target_table}"
                    )
                    

                    # Show sample data button
                    if st.button(f"Preview data from {target_table}"):
                        if selected_columns:
                            columns_str = ", ".join([f'"{col}"' for col in selected_columns])
                            sample_df = pd.read_sql_query(
                                f"SELECT {columns_str} FROM '{target_table}' LIMIT 10", 
                                sqlite_conn
                            )
                            st.dataframe(sample_df)
                        else:
                            st.warning("Please select at least one column to preview")
                else:
                    st.warning(f"No columns found in table {target_table}")
            else:
                st.warning("No target SAP table found for the given parameters")
                
        except Exception as e:
            st.error(f"Error fetching table information: {e}")

with col2:
    st.markdown('<p class="sub-header">Data Transformation Query</p>', unsafe_allow_html=True)
    
    # Query input
    query = st.text_area(
        "What transformation would you like to perform?",
        height=100,
        key="query_area",
        placeholder="Example: Bring Material Number with Material Type = ROH from MARA Table"
    )
    
    # Submit button and progress indicator
    col_button1, col_button2, col_button3 = st.columns([1, 1, 1])
    with col_button3:
        submit = st.button("🔄 Transform", key="transform_button", use_container_width=True)
    
    # Process the query for data transformation
    if submit and query and sqlite_conn:
        with st.spinner("Processing transformation..."):
            # try:
            # Process query with context awareness
            code, result, session_id = tablellm.process_sequential_query(
                query, 
                object_id, 
                segment_id, 
                project_id,
                st.session_state['transformation_session_id'],
                selected_columns
            )

            if code is None:
                st.error(result)
            else:
                # Update session ID if this is a new session
                if not st.session_state['transformation_session_id']:
                    st.session_state['transformation_session_id'] = session_id
                
                # Get updated session info
                session_info = get_session_context(session_id)
                
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
                    with st.expander("Generated Python Code", expanded=False):
                        st.code(code, language="python")
                
                # Display different result types appropriately
                if isinstance(result, pd.DataFrame):
                    filtered_result = result.dropna(axis=1, how='all') 
                    st.dataframe(filtered_result, use_container_width=True)
                    st.write("Number of rows:", len(result))
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
                    
                # except Exception as e:
                    # st.error(f"Error processing transformation: {e}")
    
    # Display previous transformations in this session
    if st.session_state['transformation_history']:
        st.markdown("### Previous Transformations")
        for i, tx in enumerate(st.session_state['transformation_history']):
            with st.expander(f"Transformation #{i+1}: {tx['description']}", expanded=False):
                st.write(f"Query: {tx['query']}")
                st.code(tx['code'], language="python")
    
# Cleanup connections when app is closed
def cleanup():
    if sqlite_conn:
        sqlite_conn.close()

# Register the cleanup function
import atexit
atexit.register(cleanup)
# Run the app
