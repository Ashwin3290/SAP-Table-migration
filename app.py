import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import uuid
from DMtool.dmtool import DMTool

# Page configuration
st.set_page_config(
    page_title="DMTool - Data Transformation Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for card-based layout
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #6f42c1, #007bff);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 2px solid #e9ecef;
        margin-bottom: 20px;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .card-icon {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        margin-right: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .success { background-color: #28a745; }
    .warning { background-color: #ffc107; }
    .info { background-color: #17a2b8; }
    .error { background-color: #dc3545; }
    .pending { background-color: #6c757d; }
    
    .query-examples {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 6px;
        border-left: 4px solid #007bff;
        margin: 10px 0;
    }
    
    .result-summary {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
    }
    
    .bottom-status {
        background-color: #e9ecef;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
        color: #6c757d;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dm_tool' not in st.session_state:
    st.session_state.dm_tool = DMTool()

if 'sessions' not in st.session_state:
    st.session_state.sessions = {}

if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None

if 'query_results' not in st.session_state:
    st.session_state.query_results = None

if 'execution_status' not in st.session_state:
    st.session_state.execution_status = "Ready to execute query"

if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

if 'generated_sql' not in st.session_state:
    st.session_state.generated_sql = ""

if 'query_metadata' not in st.session_state:
    st.session_state.query_metadata = {}

# Helper functions
def create_new_session():
    """Create a new session ID and add it to sessions"""
    new_session_id = str(uuid.uuid4())
    st.session_state.sessions[new_session_id] = {
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'query_count': 0,
        'last_used': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return new_session_id

def get_session_options():
    """Get session options for dropdown"""
    options = ["Create New Session"]
    for session_id, session_info in st.session_state.sessions.items():
        created_time = session_info['created_at']
        query_count = session_info['query_count']
        label = f"{session_id[:8]}... (Created: {created_time}, Queries: {query_count})"
        options.append(label)
    return options

def extract_session_id_from_option(option):
    """Extract session ID from dropdown option"""
    if option == "Create New Session":
        return None
    return option.split("...")[0]

def update_session_usage(session_id):
    """Update session usage statistics"""
    if session_id in st.session_state.sessions:
        st.session_state.sessions[session_id]['query_count'] += 1
        st.session_state.sessions[session_id]['last_used'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_query_type_color(query_type):
    """Get color for query type badge"""
    colors = {
        'SIMPLE_TRANSFORMATION': '#1976d2',
        'JOIN_OPERATION': '#388e3c',
        'CROSS_SEGMENT': '#f57c00',
        'VALIDATION_OPERATION': '#7b1fa2',
        'AGGREGATION_OPERATION': '#d32f2f'
    }
    return colors.get(query_type, '#6c757d')

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>DMTool - Data Transformation Platform</h1>
        <p>Natural Language to SQL Transformation</p>
    </div>
    """, unsafe_allow_html=True)

    # Create layout columns
    col1, col2, col3 = st.columns([2, 5, 1])

    # Card 1: Configuration & Session Management
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h3>Configuration & Session</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Database Configuration")
        
        object_id = st.number_input("Object ID", value=41, min_value=1, step=1, key="object_id")
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            segment_id = st.number_input("Segment ID", value=577, min_value=1, step=1, key="segment_id")
        with col1_2:
            project_id = st.number_input("Project ID", value=24, min_value=1, step=1, key="project_id")

        st.subheader("Session Management")
        
        session_options = get_session_options()
        selected_session_option = st.selectbox(
            "Session",
            options=session_options,
            index=0,
            help="Create a new session or select an existing one"
        )

        # Handle session selection
        if selected_session_option == "Create New Session":
            if st.button("Create New Session", use_container_width=True):
                new_session_id = create_new_session()
                st.session_state.current_session_id = new_session_id
                st.success(f"New session created: {new_session_id[:8]}...")
                st.rerun()
        else:
            session_id = extract_session_id_from_option(selected_session_option)
            if session_id != st.session_state.current_session_id:
                st.session_state.current_session_id = session_id
                st.info(f"Selected session: {session_id[:8]}...")

        # Display current session info
        if st.session_state.current_session_id:
            session_info = st.session_state.sessions.get(st.session_state.current_session_id, {})
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 10px; border-radius: 6px; margin-top: 10px;">
                <small>
                <strong>Active Session:</strong> {st.session_state.current_session_id[:8]}...<br>
                <strong>Created:</strong> {session_info.get('created_at', 'Unknown')}<br>
                <strong>Queries:</strong> {session_info.get('query_count', 0)}
                </small>
            </div>
            """, unsafe_allow_html=True)

    # Card 2: Query Input
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h3>Natural Language Query</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Enter Your Transformation Request")
        
        # Query input
        query = st.text_area(
            "Query",
            height=120,
            placeholder="Type your natural language query here...",
            label_visibility="hidden",
            key="query_input"
        )

        if query:
            # Simple query type detection based on keywords
            query_lower = query.lower()
            if any(word in query_lower for word in ['join', 'merge', 'combine']):
                detected_type = 'JOIN_OPERATION'
            elif any(word in query_lower for word in ['validate', 'check', 'verify']):
                detected_type = 'VALIDATION_OPERATION'
            elif any(word in query_lower for word in ['count', 'sum', 'average', 'group']):
                detected_type = 'AGGREGATION_OPERATION'
            elif any(word in query_lower for word in ['segment', 'previous', 'transformation']):
                detected_type = 'CROSS_SEGMENT'
            else:
                detected_type = 'SIMPLE_TRANSFORMATION'
            
            color = get_query_type_color(detected_type)
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <strong>Detected Query Type:</strong>
                <span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 10px; font-size: 0.8em;">
                    {detected_type}
                </span>
            </div>
            """, unsafe_allow_html=True)

        execute_clicked = st.button("🚀 Execute", use_container_width=True, type="primary")
        

    # Card 3: Processing Status
    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h3>Processing Status</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Current Status")
        
        # Status indicator
        status_color = "success" if "successfully" in st.session_state.execution_status.lower() else "warning"
        st.markdown(f"""
        <div style="margin: 15px 0;">
            <span class="status-indicator {status_color}"></span>
            <span>{st.session_state.execution_status}</span>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Process Steps")
        # Session info
        if st.session_state.current_session_id:
            st.markdown(f"""
            <div style="margin-top: 20px; font-size: 0.9em; color: #6c757d;">
                <strong>Session ID:</strong> {st.session_state.current_session_id[:13]}...
            </div>
            """, unsafe_allow_html=True)

    # Execute query logic
    if execute_clicked and query.strip() and st.session_state.current_session_id:
        with st.spinner("Processing your query..."):
            try:
                st.session_state.execution_status = "Processing query..."
                st.session_state.last_query = query
                
                # Call the DMTool
                result, session_id = st.session_state.dm_tool.process_sequential_query(
                    query=query,
                    object_id=object_id,
                    segment_id=segment_id,
                    project_id=project_id,
                    session_id=st.session_state.current_session_id
                )
                
                # Update session usage
                update_session_usage(st.session_state.current_session_id)
                
                # Store results
                st.session_state.query_results = result
                st.session_state.current_session_id = session_id
                
                # Extract metadata if result is a DataFrame with attrs
                if isinstance(result, pd.DataFrame) and hasattr(result, 'attrs'):
                    st.session_state.query_metadata = result.attrs
                
                st.session_state.execution_status = "Query executed successfully"
                st.success("Query executed successfully!")
                st.rerun()
                
            except Exception as e:
                st.session_state.execution_status = f"Error: {str(e)}"
                st.error(f"Error executing query: {str(e)}")

    elif execute_clicked and not st.session_state.current_session_id:
        st.error("Please create or select a session first.")
    elif execute_clicked and not query.strip():
        st.error("Please enter a query.")

    # Card 4: Results & Output (Full Width)
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon" style="background-color: #6f42c1;">📊</div>
            <h3>Results & Output</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Results tabs
    if st.session_state.query_results is not None:
        tab1, tab2, tab3 = st.tabs(["📋 Data Table", "💻 SQL Query", "ℹ️ Metadata"])
        
        with tab1:
            st.subheader("Query Results")
            
            if isinstance(st.session_state.query_results, pd.DataFrame):
                if not st.session_state.query_results.empty:
                    # Display the dataframe
                    st.dataframe(
                        st.session_state.query_results,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Results summary
                    st.markdown("""
                    <div class="result-summary">
                        <h4>Summary:</h4>
                    """, unsafe_allow_html=True)
                    
                    summary_info = []
                    summary_info.append(f"• **Rows returned:** {len(st.session_state.query_results)}")
                    summary_info.append(f"• **Columns:** {len(st.session_state.query_results.columns)}")
                    
                    # Add transformation summary if available in attrs
                    if hasattr(st.session_state.query_results, 'attrs') and 'transformation_summary' in st.session_state.query_results.attrs:
                        trans_summary = st.session_state.query_results.attrs['transformation_summary']
                        summary_info.append(f"• **Target table:** {trans_summary.get('target_table', 'Unknown')}")
                        summary_info.append(f"• **Query type:** {trans_summary.get('query_type', 'Unknown')}")
                        if 'populated_fields' in trans_summary:
                            summary_info.append(f"• **Populated fields:** {', '.join(trans_summary['populated_fields'])}")
                    
                    for info in summary_info:
                        st.markdown(info)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Export options
                    col_exp1, col_exp2, col_exp3, col_exp4 = st.columns([1, 1, 1, 2])
                    
                    with col_exp1:
                        csv_data = st.session_state.query_results.to_csv(index=False)
                        st.download_button(
                            "📥 Export CSV",
                            data=csv_data,
                            file_name=f"dmtool_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col_exp2:
                        json_data = st.session_state.query_results.to_json(orient='records', indent=2)
                        st.download_button(
                            "📄 Export JSON",
                            data=json_data,
                            file_name=f"dmtool_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                else:
                    st.info("Query executed successfully but returned no data.")
                    if hasattr(st.session_state.query_results, 'attrs') and 'message' in st.session_state.query_results.attrs:
                        st.info(st.session_state.query_results.attrs['message'])
            
            else:
                st.write("**Result:**")
                st.write(st.session_state.query_results)
        
        with tab2:
            st.subheader("Generated SQL Query")
            if st.session_state.generated_sql:
                st.code(st.session_state.generated_sql, language='sql')
                
                # Copy button
                if st.button("📋 Copy SQL to Clipboard"):
                    st.info("SQL copied to clipboard! (Note: Actual clipboard functionality requires additional setup)")
            else:
                st.info("SQL query information not available. This feature will show the generated SQL when available.")
        
        with tab3:
            st.subheader("Query Metadata")
            if st.session_state.query_metadata:
                st.json(st.session_state.query_metadata)
            else:
                metadata = {
                    "last_query": st.session_state.last_query,
                    "session_id": st.session_state.current_session_id,
                    "executed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "object_id": object_id,
                    "segment_id": segment_id,
                    "project_id": project_id
                }
                st.json(metadata)
    
    else:
        st.info("Execute a query to see results here.")

    # Bottom status bar
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db_status = "Connected" if st.session_state.dm_tool else "Disconnected"
    session_status = "Active" if st.session_state.current_session_id else "No Session"
    
    st.markdown(f"""
    <div class="bottom-status">
        Ready | Database: {db_status} | Session: {session_status} | Last Update: {current_time}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()