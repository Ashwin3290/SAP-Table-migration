"""
Streamlit application for TableLLM
"""

import os
import pandas as pd
import streamlit as st
from pymongo import MongoClient

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

# Initialize TableLLM
tablellm = TableLLM()
session_id = 0
# Initialize session state variables
if 'show_code' not in st.session_state:
    st.session_state['show_code'] = True

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
        'double': {
            'question': None,
            'code': None,
            'result': None,
            'session_id': 0
        }
    }

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
</style>
""", unsafe_allow_html=True)

# Helper function to update vote in database
def update_vote(session_id, vote):
    """Update the vote for a session"""
    if not db_client or not session_id:
        return False
    
    try:
        db_client.chat.update_one(
            {'session_id': session_id},
            {'$set': {'vote': vote}}
        )
        return True
    except Exception as e:
        st.error(f"Error updating vote: {e}")
        return False

# Main app header
st.markdown('<p class="main-header">TableLLM: Interactive Data Analysis</p>', unsafe_allow_html=True)
st.markdown("Ask questions about your data in natural language and get immediate insights.")

# Create tabs for different operations
tab1, tab2 = st.tabs(["Single Table Analysis", "Two-Table Analysis"])

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
        
        # Mode selection
        analysis_mode = st.radio(
            "Select Analysis Mode",
            ["Code (generates & runs Python)", "QA (direct answers)"],
            key="single_mode",
            horizontal=True
        )
        mode = "Code" if "Code" in analysis_mode else "QA"
        
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
                
                # # Save to history
                # session_id = tablellm.save_interaction(
                #     query, code, result, 
                #     st.session_state['tables']['single']['details'],
                #     db_client
                # )
                st.session_state['history']['single'] = {
                    'question': query,
                    'code': code,
                    'result': result,
                    'session_id': session_id
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
            
            # Voting buttons
            if session_id:
                vote_col1, vote_col2 = st.columns(2)
                with vote_col1:
                    if st.button("👍 Helpful", key="single_upvote"):
                        update_vote(session_id, 1)
                        st.success("Thanks for your feedback!")
                
                with vote_col2:
                    if st.button("👎 Not Helpful", key="single_downvote"):
                        update_vote(session_id, -1)
                        st.success("Thanks for your feedback!")
        
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

# Two-Table Analysis Tab
with tab2:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<p class="sub-header">Upload Your Data</p>', unsafe_allow_html=True)
        
        # File uploaders
        uploaded_file1 = st.file_uploader(
            "Choose first CSV or Excel file", 
            type=["csv", "xlsx", "xls"], 
            key="double_file_uploader1"
        )
        
        uploaded_file2 = st.file_uploader(
            "Choose second CSV or Excel file", 
            type=["csv", "xlsx", "xls"], 
            key="double_file_uploader2"
        )
        
        # Process first uploaded file
        if uploaded_file1:
            if uploaded_file1 != st.session_state['tables']['first']['file']:
                try:
                    with st.spinner("Processing first file..."):
                        df, text, details = save_file(uploaded_file1)
                        st.session_state['tables']['first'] = {
                            'file': uploaded_file1,
                            'df': df,
                            'text': text,
                            'details': details
                        }
                except Exception as e:
                    st.error(f"Error processing first file: {e}")
            
            # Display the first DataFrame
            st.subheader("First Table")
            st.dataframe(
                st.session_state['tables']['first']['df'],
                height=200,
                use_container_width=True
            )
        
        # Process second uploaded file
        if uploaded_file2:
            if uploaded_file2 != st.session_state['tables']['second']['file']:
                try:
                    with st.spinner("Processing second file..."):
                        df, text, details = save_file(uploaded_file2)
                        st.session_state['tables']['second'] = {
                            'file': uploaded_file2,
                            'df': df,
                            'text': text,
                            'details': details
                        }
                except Exception as e:
                    st.error(f"Error processing second file: {e}")
            
            # Display the second DataFrame
            st.subheader("Second Table")
            st.dataframe(
                st.session_state['tables']['second']['df'],
                height=200,
                use_container_width=True
            )
            
        # Generate exploration code for both tables
        if (uploaded_file1 and uploaded_file2 and 
            st.session_state['tables']['first']['df'] is not None and 
            st.session_state['tables']['second']['df'] is not None):
            
            if st.button("Generate Dual Table Exploration", key="generate_exploration_dual"):
                try:
                    from code_generator import generate_exploration_code
                    with st.spinner("Generating exploration code..."):
                        code_file = generate_exploration_code(
                            st.session_state['tables']['first']['df'],
                            f"{uploaded_file1.name}_{uploaded_file2.name}",
                            is_double=True
                        )
                    st.success(f"Dual table exploration code generated: {code_file}")
                    
                    # Add option to run the exploration code
                    if st.button("Run Dual Exploration", key="run_exploration_dual"):
                        with st.spinner("Running exploration..."):
                            from code_exec import execute_code
                            result = execute_code(
                                code_file, 
                                (st.session_state['tables']['first']['df'], 
                                 st.session_state['tables']['second']['df']),
                                is_double=True
                            )
                        
                        # Display various types of results
                        if isinstance(result, dict):
                            for key, value in result.items():
                                if 'plot' in key or 'figure' in key or 'heatmap' in key:
                                    st.pyplot(value)
                                elif isinstance(value, pd.DataFrame):
                                    st.dataframe(value)
                                else:
                                    st.write(f"**{key}:**", value)
                        else:
                            st.write(result)
                except Exception as e:
                    st.error(f"Error generating dual exploration code: {e}")
        
        if not uploaded_file1 or not uploaded_file2:
            st.info("Please upload both files to begin analysis")
    
    with col2:
        st.markdown('<p class="sub-header">Ask a Question About Both Tables</p>', unsafe_allow_html=True)
        
        # Query input
        query = st.text_area(
            "What would you like to know about these tables?",
            height=100,
            key="double_query"
        )
        
        # Submit button and progress indicator
        col_button1, col_button2, col_button3 = st.columns([1, 1, 1])
        with col_button3:
            submit = st.button("📊 Analyze", key="double_analyze_button", use_container_width=True)
        
        # Process the query
        if (submit and query and 
            st.session_state['tables']['first']['df'] is not None and 
            st.session_state['tables']['second']['df'] is not None):
            
            with st.spinner("Analyzing data..."):
                # Get response from TableLLM
                code, result = tablellm.process_query(
                    query,
                    (st.session_state['tables']['first']['text'], st.session_state['tables']['second']['text']),
                    (st.session_state['tables']['first']['df'], st.session_state['tables']['second']['df']),
                    mode="Code"  # Only Code mode for two tables
                )
                
                # Save to history
                # session_id = tablellm.save_interaction(
                #     query, code, result, 
                #     [st.session_state['tables']['first']['details'], st.session_state['tables']['second']['details']],
                #     db_client
                # )
                st.session_state['history']['double'] = {
                    'question': query,
                    'code': code,
                    'result': result,
                    'session_id': session_id
                }
            
            # Display results
            st.success("Analysis complete!")
            st.markdown("### Result")
            
            # Display code
            if st.session_state['show_code']:
                with st.expander("Generated Python Code", expanded=True):
                    st.code(code, language="python")
            else:
                if st.button("Show Code", key="show_code_button_double"):
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
            
            # Voting buttons
            if session_id:
                vote_col1, vote_col2 = st.columns(2)
                with vote_col1:
                    if st.button("👍 Helpful", key="double_upvote"):
                        update_vote(session_id, 1)
                        st.success("Thanks for your feedback!")
                
                with vote_col2:
                    if st.button("👎 Not Helpful", key="double_downvote"):
                        update_vote(session_id, -1)
                        st.success("Thanks for your feedback!")
        
        # Example queries
        with st.expander("Example Queries"):
            example_queries = [
                "Merge these tables on common column names",
                "Join the tables and show all records from both tables",
                "Count the total records after merging the tables",
                "What are the common columns between these tables?",
                "Compare summary statistics between both tables"
            ]
            
            for i, query in enumerate(example_queries):
                if st.button(query, key=f"double_example_{i}"):
                    st.session_state["double_query"] = query
                    st.rerun()

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    
    # Code visibility toggle
    st.session_state['show_code'] = st.checkbox("Show generated code by default", value=st.session_state['show_code'])
    
    # About section
    st.markdown("---")
    st.markdown("### About TableLLM")
    st.markdown("""
    TableLLM enables table data analysis through natural language.
    
    **Features:**
    - Upload CSV, Excel, or DOCX files
    - Query your data using natural language
    - Automatic code generation and execution
    - Support for table merging and comparison
    """)
    
    # Add information about the LLM being used
    st.markdown("---")
    st.markdown("### Model Information")
    if tablellm.use_gemini:
        st.markdown("Using **Google Gemini** for code generation")
    else:
        st.markdown(f"Using Ollama for code generation")