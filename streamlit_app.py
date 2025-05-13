"""
Streamlit UI for TableLLM
This module provides a web interface for the SAP Query Processor
"""
import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime

from sap_query_processor import SAPQueryProcessor
from utils.token_utils import get_token_usage_stats
import config

# Set page configuration
st.set_page_config(
    page_title="TableLLM - SAP Query Processor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "processor" not in st.session_state:
    st.session_state.processor = None
if "history" not in st.session_state:
    st.session_state.history = []
if "workspaces" not in st.session_state:
    st.session_state.workspaces = {}
if "workspace_selection" not in st.session_state:
    st.session_state.workspace_selection = None
if "object_id" not in st.session_state:
    st.session_state.object_id = 41
if "segment_id" not in st.session_state:
    st.session_state.segment_id = 577
if "project_id" not in st.session_state:
    st.session_state.project_id = 24

# Helper functions
def initialize_processor():
    """Initialize the SAP Query Processor"""
    try:
        if st.session_state.processor is None:
            with st.spinner("Initializing SAP Query Processor..."):
                st.session_state.processor = SAPQueryProcessor()
                # Get available workspaces
                st.session_state.workspaces = st.session_state.processor.get_workspaces()
                st.success("SAP Query Processor initialized successfully!")
        return st.session_state.processor
    except Exception as e:
        st.error(f"Failed to initialize SAP Query Processor: {e}")
        return None

def create_new_session():
    """Create a new transformation session"""
    try:
        processor = initialize_processor()
        if processor:
            with st.spinner("Creating new session..."):
                st.session_state.session_id = processor.session_manager.create_session()
                st.session_state.history = []
                st.success(f"Created new session: {st.session_state.session_id}")
    except Exception as e:
        st.error(f"Failed to create new session: {e}")

def load_session(session_id):
    """Load an existing session"""
    try:
        processor = initialize_processor()
        if processor:
            with st.spinner(f"Loading session {session_id}..."):
                session_info = processor.get_session_info(session_id)
                if session_info:
                    st.session_state.session_id = session_id
                    st.session_state.history = session_info.get("transformations", [])
                    st.success(f"Loaded session: {session_id}")
                else:
                    st.error(f"Session not found: {session_id}")
    except Exception as e:
        st.error(f"Failed to load session: {e}")

def process_query(query, target_sap_fields=None):
    """Process a transformation query"""
    try:
        processor = initialize_processor()
        if not processor:
            st.error("Processor not initialized")
            return
        
        if not st.session_state.session_id:
            create_new_session()
        
        with st.spinner("Processing query..."):
            start_time = datetime.now()
            
            code, result, session_id = processor.process_query(
                query,
                session_id=st.session_state.session_id,
                object_id=st.session_state.object_id,
                segment_id=st.session_state.segment_id,
                project_id=st.session_state.project_id,
                target_sap_fields=target_sap_fields
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update session ID if changed
            if session_id != st.session_state.session_id:
                st.session_state.session_id = session_id
            
            # Update history
            session_info = processor.get_session_info(session_id)
            if session_info:
                st.session_state.history = session_info.get("transformations", [])
            
            return code, result, duration
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return None, f"Error: {e}", 0

def display_result(result, duration):
    """Display the transformation result"""
    if isinstance(result, pd.DataFrame):
        st.success(f"Query processed in {duration:.2f} seconds")
        st.write(f"Result contains {len(result)} rows and {len(result.columns)} columns")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Data Table", "Data Statistics", "Visualization"])
        
        with tab1:
            st.dataframe(result, use_container_width=True)
            
            # Add download button
            csv = result.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="transformation_result.csv",
                mime="text/csv"
            )
        
        with tab2:
            if not result.empty:
                st.write("### Data Summary")
                st.write(result.describe())
                
                st.write("### Column Information")
                col_info = pd.DataFrame({
                    "Column": result.columns,
                    "Type": result.dtypes.values,
                    "Non-Null Count": result.count().values,
                    "Null Count": result.isna().sum().values,
                    "Null Percentage": (result.isna().sum() / len(result) * 100).values.round(2),
                    "Unique Values": [result[col].nunique() for col in result.columns]
                })
                st.dataframe(col_info, use_container_width=True)
        
        with tab3:
            if not result.empty:
                st.write("### Data Visualization")
                
                # Only attempt visualization if there's data
                try:
                    # Check if there are numeric columns
                    numeric_cols = result.select_dtypes(include=["number"]).columns.tolist()
                    if numeric_cols:
                        # Let user select columns to visualize
                        x_col = st.selectbox("Select X-axis column", result.columns)
                        y_col = st.selectbox("Select Y-axis column", numeric_cols)
                        
                        # Select chart type
                        chart_type = st.selectbox(
                            "Select chart type",
                            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram"]
                        )
                        
                        # Generate the selected chart
                        if chart_type == "Bar Chart":
                            fig = px.bar(result, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        elif chart_type == "Line Chart":
                            fig = px.line(result, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        elif chart_type == "Scatter Plot":
                            fig = px.scatter(result, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        elif chart_type == "Histogram":
                            fig = px.histogram(result, x=x_col, title=f"Distribution of {x_col}")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No numeric columns available for visualization")
                except Exception as e:
                    st.error(f"Error creating visualization: {e}")
    else:
        # Display error or text result
        st.error(f"Error: {result}")

def display_code(code):
    """Display the generated Python code"""
    if code:
        st.write("### Generated Python Code")
        st.code(code, language="python")

def display_token_usage():
    """Display token usage statistics"""
    try:
        processor = initialize_processor()
        if processor:
            token_stats = processor.get_token_usage()
            
            st.write("### Token Usage Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Tokens", token_stats.get("total_tokens", 0))
            
            with col2:
                st.metric("Input Tokens", token_stats.get("total_input_tokens", 0))
            
            with col3:
                st.metric("Output Tokens", token_stats.get("total_output_tokens", 0))
            
            # Display token usage by function
            functions = token_stats.get("calls_by_function", {})
            if functions:
                function_data = []
                for func, stats in functions.items():
                    function_data.append({
                        "Function": func,
                        "Calls": stats.get("calls", 0),
                        "Input Tokens": stats.get("input_tokens", 0),
                        "Output Tokens": stats.get("output_tokens", 0),
                        "Total Tokens": stats.get("input_tokens", 0) + stats.get("output_tokens", 0)
                    })
                
                function_df = pd.DataFrame(function_data)
                if not function_df.empty:
                    function_df = function_df.sort_values("Total Tokens", ascending=False)
                    st.dataframe(function_df, use_container_width=True)
                    
                    # Create token usage chart
                    fig = px.bar(
                        function_df,
                        x="Function",
                        y=["Input Tokens", "Output Tokens"],
                        title="Token Usage by Function",
                        barmode="stack"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying token usage: {e}")

def display_history():
    """Display transformation history for the current session"""
    if st.session_state.history:
        st.write("### Transformation History")
        
        # Create DataFrame from history
        history_data = []
        for item in st.session_state.history:
            history_data.append({
                "Timestamp": item.get("timestamp", ""),
                "Query": item.get("query", ""),
                "Intent": item.get("intent_type", ""),
                "Source Tables": ", ".join(item.get("source_tables", [])),
                "Target Table": item.get("target_table", ""),
                "Segment": item.get("segment", ""),
                "Plan Summary": item.get("plan_summary", "")
            })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No transformation history available")

# Main Streamlit app
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/ashwinch/tablellm-logo/main/logo.png", width=200)
        st.title("TableLLM")
        st.write("SAP Data Transformation Assistant")
        
        # Session management
        st.subheader("Session Management")
        
        # Initialize processor button
        if st.button("Initialize Processor"):
            initialize_processor()
        
        # Session options
        session_option = st.radio(
            "Session",
            ["Create New Session", "Load Existing Session", "Current Session"],
            index=2 if st.session_state.session_id else 0
        )
        
        if session_option == "Create New Session":
            if st.button("Create Session"):
                create_new_session()
        
        elif session_option == "Load Existing Session":
            try:
                processor = initialize_processor()
                if processor:
                    sessions = processor.list_sessions()
                    if sessions:
                        session_ids = [s["session_id"] for s in sessions]
                        selected_session = st.selectbox("Select Session", session_ids)
                        if st.button("Load Session"):
                            load_session(selected_session)
                    else:
                        st.info("No existing sessions found")
            except Exception as e:
                st.error(f"Error loading sessions: {e}")
        
        elif session_option == "Current Session":
            if st.session_state.session_id:
                st.info(f"Current Session ID: {st.session_state.session_id}")
                if st.button("Clear Session"):
                    st.session_state.session_id = None
                    st.session_state.history = []
                    st.success("Session cleared")
            else:
                st.info("No active session")
        
        # Workspace selection
        st.subheader("Workspace")
        if st.session_state.workspaces:
            workspace_names = list(st.session_state.workspaces.keys())
            st.session_state.workspace_selection = st.selectbox(
                "Select Workspace",
                workspace_names,
                index=0
            )
        else:
            st.info("No workspaces available")
        
        # Advanced settings
        st.subheader("Advanced Settings")
        with st.expander("ID Settings"):
            st.session_state.object_id = st.number_input("Object ID", value=st.session_state.object_id)
            st.session_state.segment_id = st.number_input("Segment ID", value=st.session_state.segment_id)
            st.session_state.project_id = st.number_input("Project ID", value=st.session_state.project_id)
    
    # Main content
    st.title("SAP Data Transformation Assistant")
    
    if not st.session_state.processor:
        st.warning("Processor not initialized. Please click 'Initialize Processor' in the sidebar.")
        return
    
    # Query input
    st.write("### Enter Your Transformation Query")
    query = st.text_area(
        "Describe the transformation you want to perform",
        height=100,
        placeholder="Example: Extract material type where material type equals ROH from MARA table"
    )
    target_field = st.text_input(
        "Target SAP Field (optional)",
        placeholder="Enter target field name if applicable"
    )
    
    # Process button
    if st.button("Process Query", type="primary"):
        if query:
            code, result, duration = process_query(query, target_field if target_field else None)
            if code and result is not None:
                display_result(result, duration)
                with st.expander("Show Generated Code", expanded=False):
                    display_code(code)
        else:
            st.warning("Please enter a query")
    
    # Tabs for additional information
    tab1, tab2, tab3 = st.tabs(["History", "Token Usage", "Help"])
    
    with tab1:
        display_history()
    
    with tab2:
        display_token_usage()
    
    with tab3:
        st.write("### How to Use TableLLM")
        st.write("""
        TableLLM helps you transform SAP data using natural language queries. 
        
        #### Query Examples:
        
        1. Basic Extraction:
           - "Extract material type from MARA where material type equals ROH"
           - "Bring customer details from KNA1"
        
        2. Field Mapping:
           - "Map material number from MARA to MATNR in target table"
           - "Copy name field from customer table to NAME1 field"
        
        3. Conditional Logic:
           - "If material group in (L002, ZMGRP1) then set value to GENAI01, else GENAI02"
           - "For materials with type ROH, extract description from MAKT"
        
        4. Data Cleaning:
           - "Remove special characters from material descriptions"
           - "Calculate length of descriptions and add to BISMT field"
        
        5. Multiple Tables:
           - "Join material data from MARA with plant data from MARC"
           - "For materials in MARA, get descriptions from MAKT where language is EN"
        
        6. Tiered Lookup:
           - "Check material in MARA_500 first, if not found check in MARA_700, then in MARA"
        """)
        
        st.write("#### Target SAP Field")
        st.write("""
        If you need to specify a particular target field, enter it in the 'Target SAP Field' input. 
        This is useful when your transformation affects a specific field in the target table.
        """)

# Run the app
if __name__ == "__main__":
    main()
