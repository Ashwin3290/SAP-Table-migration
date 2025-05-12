"""
Streamlit UI for TableLLM Relational Model
This module provides a web interface for the TableLLM Relational Data Transformation System
"""
import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime

from relational_integration import TableLLMRelational
from utils.token_utils import get_token_usage_stats
import config

# Set page configuration
st.set_page_config(
    page_title="TableLLM - Relational Transformation",
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
if "object_id" not in st.session_state:
    st.session_state.object_id = 29
if "segment_id" not in st.session_state:
    st.session_state.segment_id = 336
if "project_id" not in st.session_state:
    st.session_state.project_id = 24
if "current_diagram" not in st.session_state:
    st.session_state.current_diagram = None
if "show_diagram" not in st.session_state:
    st.session_state.show_diagram = False
if "segment_history" not in st.session_state:
    st.session_state.segment_history = {}

# Helper functions
def initialize_processor():
    """Initialize the TableLLM Relational processor"""
    try:
        if st.session_state.processor is None:
            with st.spinner("Initializing TableLLM Relational processor..."):
                st.session_state.processor = TableLLMRelational()
                st.success("TableLLM Relational processor initialized successfully!")
        return st.session_state.processor
    except Exception as e:
        st.error(f"Failed to initialize TableLLM Relational processor: {e}")
        return None

def load_session(session_id):
    """Load an existing session"""
    try:
        processor = initialize_processor()
        if processor:
            with st.spinner(f"Loading session {session_id}..."):
                session_info = processor.get_session_info(session_id)
                if session_info:
                    st.session_state.session_id = session_id
                    # Get session relationships and diagram
                    update_session_diagram(session_id)
                    st.success(f"Loaded session: {session_id}")
                else:
                    st.error(f"Session not found: {session_id}")
    except Exception as e:
        st.error(f"Failed to load session: {e}")

def update_session_diagram(session_id):
    """Update the session diagram"""
    try:
        processor = initialize_processor()
        if processor and session_id:
            # Get session diagram
            st.session_state.current_diagram = processor.get_session_diagram(session_id)
            
            # Get relationships
            relationships = processor.get_session_relationships(session_id)
            if relationships:
                st.session_state.relationships = relationships
            
            # Get session info
            session_info = processor.get_session_info(session_id)
            if session_info:
                segments = session_info.get("segments", [])
                if segments:
                    # Update segment history
                    for segment_id in segments:
                        if segment_id not in st.session_state.segment_history:
                            st.session_state.segment_history[segment_id] = []
    except Exception as e:
        st.error(f"Failed to update session diagram: {e}")

def process_query(query):
    """Process a transformation query"""
    try:
        processor = initialize_processor()
        if not processor:
            st.error("Processor not initialized")
            return
        
        with st.spinner("Processing query..."):
            start_time = datetime.now()
            
            code, result, session_id = processor.process_query(
                query,
                object_id=st.session_state.object_id,
                segment_id=st.session_state.segment_id,
                project_id=st.session_state.project_id,
                session_id=st.session_state.session_id
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update session ID if changed
            if session_id != st.session_state.session_id:
                st.session_state.session_id = session_id
            
            # Update session diagram and relationships
            if session_id:
                update_session_diagram(session_id)
                
                # Add query to segment history
                segment_id = st.session_state.segment_id
                if segment_id not in st.session_state.segment_history:
                    st.session_state.segment_history[segment_id] = []
                    
                st.session_state.segment_history[segment_id].append({
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "duration": duration
                })
            
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

def display_session_diagram():
    """Display the session diagram"""
    if st.session_state.current_diagram:
        st.write("### Session Relationship Diagram")
        
        # Display the Mermaid diagram
        st.markdown(st.session_state.current_diagram)
        
        # Display relationships in a table
        if hasattr(st.session_state, 'relationships') and st.session_state.relationships:
            st.write("### Table Relationships")
            
            # Create DataFrame from relationships
            rel_data = []
            for rel in st.session_state.relationships:
                rel_data.append({
                    "From Table": rel.get("from_table", ""),
                    "To Table": rel.get("to_table", ""),
                    "Type": rel.get("type", ""),
                    "Mapping": str(rel.get("mapping", {}))
                })
            
            if rel_data:
                rel_df = pd.DataFrame(rel_data)
                st.dataframe(rel_df, use_container_width=True)
    else:
        st.info("No session diagram available")

def display_segment_history():
    """Display query history for each segment"""
    if st.session_state.segment_history:
        st.write("### Segment Query History")
        
        # Create tabs for each segment
        segment_ids = sorted(st.session_state.segment_history.keys())
        tabs = st.tabs([f"Segment {segment_id}" for segment_id in segment_ids])
        
        for i, segment_id in enumerate(segment_ids):
            with tabs[i]:
                history = st.session_state.segment_history[segment_id]
                if history:
                    # Create DataFrame from history
                    history_data = []
                    for item in history:
                        history_data.append({
                            "Timestamp": item.get("timestamp", ""),
                            "Query": item.get("query", ""),
                            "Duration (s)": item.get("duration", 0)
                        })
                    
                    if history_data:
                        history_df = pd.DataFrame(history_data)
                        st.dataframe(history_df, use_container_width=True)
                else:
                    st.info(f"No queries executed for Segment {segment_id}")
    else:
        st.info("No segment history available")

# Main Streamlit app
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/ashwinch/tablellm-logo/main/logo.png", width=200)
        st.title("TableLLM Relational")
        st.write("Multi-Segment Data Transformation System")
        
        # Session management
        st.subheader("Session Management")
        
        # Initialize processor button
        if st.button("Initialize Processor"):
            initialize_processor()
        
        # Session options
        session_option = st.radio(
            "Session",
            ["Current Session", "Load Existing Session"],
            index=0 if st.session_state.session_id else 0
        )
        
        if session_option == "Current Session":
            if st.session_state.session_id:
                st.info(f"Current Session ID: {st.session_state.session_id}")
                if st.button("Clear Session"):
                    st.session_state.session_id = None
                    st.session_state.current_diagram = None
                    st.session_state.segment_history = {}
                    if hasattr(st.session_state, 'relationships'):
                        del st.session_state.relationships
                    st.success("Session cleared")
            else:
                st.info("No active session")
        
        elif session_option == "Load Existing Session":
            try:
                processor = initialize_processor()
                if processor:
                    # Get sessions directory
                    sessions_dir = "sessions"
                    if os.path.exists(sessions_dir):
                        session_ids = [name for name in os.listdir(sessions_dir) 
                                      if os.path.isdir(os.path.join(sessions_dir, name))]
                        if session_ids:
                            selected_session = st.selectbox("Select Session", session_ids)
                            if st.button("Load Session"):
                                load_session(selected_session)
                        else:
                            st.info("No existing sessions found")
                    else:
                        st.info("Sessions directory not found")
            except Exception as e:
                st.error(f"Error loading sessions: {e}")
        
        # Segment settings
        st.subheader("Segment Settings")
        st.session_state.segment_id = st.number_input("Segment ID", value=st.session_state.segment_id)
        
        # Toggle diagram visibility
        st.session_state.show_diagram = st.checkbox("Show Session Diagram", value=st.session_state.show_diagram)
        
        # Other settings
        st.subheader("Advanced Settings")
        with st.expander("ID Settings"):
            st.session_state.object_id = st.number_input("Object ID", value=st.session_state.object_id)
            st.session_state.project_id = st.number_input("Project ID", value=st.session_state.project_id)
    
    # Main content
    st.title("TableLLM Relational Transformation System")
    
    if not st.session_state.processor:
        st.warning("Processor not initialized. Please click 'Initialize Processor' in the sidebar.")
        return
    
    # Query input
    st.write("### Enter Your Transformation Query")
    query = st.text_area(
        "Describe the transformation you want to perform",
        height=100,
        placeholder="Example: Extract material number with material type = ROH from MARA Table"
    )
    
    # Process button
    if st.button("Process Query", type="primary"):
        if query:
            code, result, duration = process_query(query)
            if code and result is not None:
                display_result(result, duration)
                with st.expander("Show Generated Code", expanded=False):
                    display_code(code)
        else:
            st.warning("Please enter a query")
    
    # Diagram and History
    if st.session_state.show_diagram and st.session_state.session_id:
        display_session_diagram()
    
    # Tabs for additional information
    tab1, tab2 = st.tabs(["Segment History", "Help"])
    
    with tab1:
        display_segment_history()
    
    with tab2:
        st.write("### How to Use TableLLM Relational")
        st.write("""
        TableLLM Relational helps you transform data across multiple segments using natural language queries. 
        The system automatically detects target fields from your query.
        
        #### Query Examples:
        
        1. Basic Extraction:
           - "Bring Material Number with Material Type = ROH from MARA Table"
           - "Get customer details from KNA1"
        
        2. Multi-Segment Operations:
           - "Join Material from Basic Segment with Material from MARC segment and Bring Material and Plant field from MARC Table for the plants (1710, 9999)"
           - "Create a new table with vendor information from LFA1 and get their names"
        
        3. Field Mapping:
           - "Map material number from MARA to MATNR in the target table"
           - "Copy name field from customer table to NAME1 field"
        
        4. Conditional Logic:
           - "If material group in (L002, ZMGRP1) then set value to GENAI01, else GENAI02"
           - "For materials with type ROH, extract description from MAKT"
        
        5. Multiple Tables:
           - "Join material data from MARA with plant data from MARC"
           - "For materials in MARA, get descriptions from MAKT where language is EN"
        """)
        
        st.write("#### Working with Segments")
        st.write("""
        Segments represent different tables or groups of related data. You can:
        
        1. Change the Segment ID in the sidebar to work with a different segment
        2. Mention segments explicitly in your query: "Join Material from Basic Segment with Material from MARC segment"
        3. View the relationships between segments in the Session Diagram
        """)

# Run the app
if __name__ == "__main__":
    main()
