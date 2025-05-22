# import os
# import pandas as pd
# import streamlit as st
# import sqlite3
# import json
# from datetime import datetime

# from code_exec import save_file
# from dmtool import DMTool
# from dmtool import get_session_context

# # Page configuration
# st.set_page_config(
#     page_title="DMTool", 
#     layout="wide",
#     page_icon="📊"
# )

# sqlite_conn = None
# # try:
# sqlite_conn = sqlite3.connect('db.sqlite3')
# # Initialize DMTool
# DMTool = DMTool()

# # Initialize session state variables
# if 'show_code' not in st.session_state:
#     st.session_state['show_code'] = True

# if 'transformation_session_id' not in st.session_state:
#     st.session_state['transformation_session_id'] = None

# if 'tables' not in st.session_state:
#     st.session_state['tables'] = {
#         'single': {
#             'file': None,
#             'df': None,
#             'text': None,
#             'details': None
#         },
#         'first': {
#             'file': None,
#             'df': None,
#             'text': None,
#             'details': None
#         },
#         'second': {
#             'file': None,
#             'df': None,
#             'text': None,
#             'details': None
#         }
#     }

# if 'history' not in st.session_state:
#     st.session_state['history'] = {
#         'single': {
#             'question': None,
#             'code': None,
#             'result': None,
#             'session_id': 0
#         },
#         'transformation': {
#             'questions': [],
#             'codes': [],
#             'results': [],
#             'session_id': None
#         }
#     }

# if 'transformation_history' not in st.session_state:
#     st.session_state['transformation_history'] = []

# # Custom CSS for better appearance
# st.markdown("""
# <style>
# /* Modern CSS for Gen-AI Data Migration Tool */

# /* Base styles and variables */
# :root {
#     --primary: #4361ee;
#     --primary-light: #4895ef;
#     --primary-dark: #3f37c9;
#     --secondary: #4cc9f0;
#     --success: #4ade80;
#     --warning: #f59e0b;
#     --danger: #ef4444;
#     --gray-100: #f3f4f6;
#     --gray-200: #e5e7eb;
#     --gray-300: #d1d5db;
#     --gray-400: #9ca3af;
#     --gray-700: #374151;
#     --gray-900: #111827;
#     --radius: 8px;
#     --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
#     --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
#     --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
#     --transition: all 0.2s ease;
# }

# /* Global styling */
# .main-header {
#     font-size: 2.25rem !important;
#     font-weight: 700 !important;
#     background: linear-gradient(120deg, var(--primary), var(--secondary));
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     margin-bottom: 1.5rem;
#     letter-spacing: -0.5px;
# }

# .sub-header {
#     font-size: 1.5rem !important;
#     font-weight: 600 !important;
#     color: var(--gray-900);
#     margin-bottom: 1rem;
#     border-bottom: 2px solid var(--gray-200);
#     padding-bottom: 0.5rem;
# }

# /* Card styling */
# .card {
#     background-color: white;
#     border-radius: var(--radius);
#     box-shadow: var(--shadow);
#     padding: 1.5rem;
#     margin-bottom: 1.5rem;
#     transition: var(--transition);
# }

# .card:hover {
#     box-shadow: var(--shadow-lg);
# }

# /* Form elements */
# .stTextInput > div > div > input, 
# .stNumberInput > div > div > input,
# .stTextArea > div > div > textarea,
# .stSelectbox > div > div > div {
#     border-radius: var(--radius) !important;
#     border: 1px solid var(--gray-300) !important;
#     padding: 0.75rem !important;
#     box-shadow: var(--shadow-sm) !important;
#     transition: var(--transition) !important;
# }

# .stTextInput > div > div > input:focus, 
# .stNumberInput > div > div > input:focus,
# .stTextArea > div > div > textarea:focus,
# .stSelectbox > div > div > div:focus {
#     border-color: var(--primary) !important;
#     box-shadow: 0 0 0 2px rgba(67, 97, 238, 0.15) !important;
# }

# /* Buttons */
# .stButton > button {
#     border-radius: var(--radius) !important;
#     font-weight: 600 !important;
#     text-transform: none !important;
#     padding: 0.5rem 1rem !important;
#     transition: var(--transition) !important;
#     border: none !important;
#     color: white !important;
#     background-color: var(--primary) !important;
# }

# .stButton > button:hover {
#     background-color: var(--primary-dark) !important;
#     transform: translateY(-1px);
# }

# .stButton > button:active {
#     transform: translateY(0);
# }

# /* Success and other buttons can be styled differently */
# .success-button > button {
#     background-color: var(--success) !important;
# }

# .warning-button > button {
#     background-color: var(--warning) !important;
# }

# .danger-button > button {
#     background-color: var(--danger) !important;
# }

# /* Code box styling */
# .code-box {
#     background-color: #1e1e1e;
#     color: #d4d4d4;
#     border-radius: var(--radius);
#     padding: 1rem;
#     margin-bottom: 1.5rem;
#     font-family: 'Consolas', 'Monaco', monospace;
#     font-size: 0.9rem;
#     overflow-x: auto;
#     border-left: 4px solid var(--primary);
# }

# /* Context info and history items */
# .context-info {
#     padding: 1.25rem;
#     background-color: rgba(67, 97, 238, 0.05);
#     border-radius: var(--radius);
#     margin-bottom: 1.5rem;
#     border-left: 4px solid var(--primary);
#     box-shadow: var(--shadow-sm);
# }

# .history-item {
#     padding: 1rem;
#     background-color: white;
#     border-radius: var(--radius);
#     margin-bottom: 1rem;
#     border-left: 4px solid var(--primary);
#     box-shadow: var(--shadow-sm);
#     transition: var(--transition);
# }

# .history-item:hover {
#     box-shadow: var(--shadow);
# }

# /* Example queries */
# .example-query {
#     padding: 0.75rem 1rem;
#     background-color: white;
#     border: 1px solid var(--gray-200);
#     border-radius: var(--radius);
#     cursor: pointer;
#     margin-bottom: 0.75rem;
#     transition: var(--transition);
#     box-shadow: var(--shadow-sm);
# }

# .example-query:hover {
#     background-color: var(--gray-100);
#     border-color: var(--gray-300);
#     box-shadow: var(--shadow);
# }

# /* Tables and dataframes */
# .dataframe {
#     border-collapse: collapse;
#     width: 100%;
#     border-radius: var(--radius);
#     overflow: hidden;
#     margin-bottom: 1.5rem;
#     box-shadow: var(--shadow);
# }

# .dataframe th {
#     background-color: var(--primary);
#     color: white;
#     font-weight: 600;
#     text-align: left;
#     padding: 0.75rem 1rem;
# }

# .dataframe td {
#     padding: 0.75rem 1rem;
#     border-bottom: 1px solid var(--gray-200);
# }

# .dataframe tr:nth-child(even) {
#     background-color: var(--gray-100);
# }

# .dataframe tr:hover {
#     background-color: rgba(67, 97, 238, 0.05);
# }

# /* Expanders */
# .streamlit-expanderHeader {
#     font-weight: 600 !important;
#     color: var(--gray-700) !important;
#     background-color: var(--gray-100) !important;
#     border-radius: var(--radius) !important;
#     padding: 0.75rem 1rem !important;
# }

# .streamlit-expanderHeader:hover {
#     background-color: var(--gray-200) !important;
# }

# .streamlit-expanderContent {
#     border: 1px solid var(--gray-200) !important;
#     border-top: none !important;
#     border-radius: 0 0 var(--radius) var(--radius) !important;
#     padding: 1rem !important;
# }

# /* Sidebar */
# [data-testid="stSidebar"] {
#     background-color: #f8f9fa;
#     border-right: 1px solid var(--gray-200);
# }

# [data-testid="stSidebarNav"] li {
#     border-radius: var(--radius);
#     margin-bottom: 0.25rem;
# }

# [data-testid="stSidebarNav"] a {
#     border-radius: var(--radius) !important;
#     font-weight: 500 !important;
# }

# /* Loader */
# .stSpinner > div {
#     border-top-color: var(--primary) !important;
# }

# /* Success/Error messages */
# .element-container .stAlert {
#     border-radius: var(--radius);
#     border: none !important;
#     box-shadow: var(--shadow-sm);
# }

# .element-container .stAlert > div:first-child {
#     padding: 1rem !important;
# }

# /* Apply styles to the main content area */
# .main .block-container {
#     padding-top: 2rem !important;
#     padding-bottom: 2rem !important;
#     max-width: 1200px !important;
# }

# /* Increase height of selectbox */
# .stSelectbox > div > div > div {
#     height: 60px !important;  /* Increased from default */
#     min-height: 60px !important;
#     display: flex !important;
#     align-items: center !important;
# }

# /* Increase height of button */
# .stButton > button {
#     min-height: 48px !important;  /* Increased from default */
#     height: auto !important;
#     padding: 0.75rem 1rem !important;  /* More vertical padding */
# }

# /* Ensure text is vertically centered */
# .stButton > button > div {
#     display: flex !important;
#     align-items: center !important;
#     justify-content: center !important;
# }

# /* Media queries for responsiveness */
# @media (max-width: 768px) {
#     .main-header {
#         font-size: 1.75rem !important;
#     }
    
#     .sub-header {
#         font-size: 1.25rem !important;
#     }
    
#     .card {
#         padding: 1rem;
#     }
# }
# </style>
# """, unsafe_allow_html=True)

# # Main app header
# st.markdown('<p class="main-header">Gen-AI Data Migration Tool</p>', unsafe_allow_html=True)
# # st.markdown("Process sequential data transformations with context awareness, preserving state between queries.")

# col1, col2 = st.columns([1, 1.5])

# with col1:
#     st.markdown('<p class="sub-header">Configuration</p>', unsafe_allow_html=True)
    
#     # Object and segment selection widgets
#     object_id = st.number_input("Object ID", min_value=1, value=41, step=1, key="object_id")
#     segment_id = st.number_input("Segment ID", min_value=1, value=577, step=1, key="segment_id")
#     project_id = st.number_input("Project ID", min_value=1, value=24, step=1, key="project_id")
    
#     # Session management
#     st.markdown("### Session Management")
#     col_session1, col_session2 = st.columns(2)

    
#     with col_session1:
#         if st.button("Clear Current Session", key="clear_session") and st.session_state['transformation_session_id']:
#             st.session_state['transformation_session_id'] = None
#             st.session_state['transformation_history'] = []
#             st.success("Session cleared!")
#             st.rerun()
    
#     # Display current session info
#     if st.session_state['transformation_session_id']:
#         session_info = get_session_context(st.session_state['transformation_session_id'])
        
#         st.markdown("### Current Session")
#         st.markdown(f"""
#         <div class="context-info">
#             <strong>Session ID:</strong> {session_info['session_id']}<br>
#             <strong>Transformations:</strong> {len(session_info['transformation_history'])}<br>
#             <strong>Populated Fields:</strong> {', '.join(session_info['target_table_state'].get('populated_fields', ['None']))}
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Display transformation history
#         if session_info['transformation_history']:
#             with st.expander("Transformation History", expanded=False):
#                 for i, tx in enumerate(session_info['transformation_history']):
#                     st.markdown(f"""
#                     <div class="history-item">
#                         <strong>#{i+1}:</strong> {tx.get('description', 'Unknown transformation')}<br>
#                         <strong>Modified:</strong> {', '.join(tx.get('fields_modified', []))}<br>
#                         <strong>Filters:</strong> {json.dumps(tx.get('filter_conditions', {}))}
#                     </div>
#                     """, unsafe_allow_html=True)
    
#     # Display sample tables from database if connected

# with col2:
#     st.markdown('<p class="sub-header">Data Transformation Query</p>', unsafe_allow_html=True)
    
#     # Query input
#     query = st.text_area(
#         "What transformation would you like to perform?",
#         height=100,
#         key="query_area",
#         placeholder="Example: Bring Material Number with Material Type = ROH from MARA Table"
#     )
    

#     submit = st.button("🔄 Transform", key="transform_button", use_container_width=True)

#     if submit and query and sqlite_conn:
#         with st.spinner("Processing transformation..."):
#             # Process query with context awareness and automatic query classification
#             code, result, session_id = DMTool.process_sequential_query(
#                 query, 
#                 object_id, 
#                 segment_id, 
#                 project_id,
#                 st.session_state['transformation_session_id']
#             )

#         if code is None:
#             st.error(result)
#         else:
#             # Update session ID if this is a new session
#             if not st.session_state['transformation_session_id']:
#                 st.session_state['transformation_session_id'] = session_id
            
#             # Get updated session info
#             session_info = get_session_context(session_id)
            
#             # Add to history
#             st.session_state['transformation_history'].append({
#                 'query': query,
#                 'code': code,
#                 'timestamp': datetime.now().isoformat(),
#                 'description': session_info['transformation_history'][-1]['description'] 
#                     if session_info['transformation_history'] else "Unknown transformation"
#             })
            
#             # Display results
#             st.success("Transformation complete!")
            
#             # Display the transformation results
#             st.markdown("### Result")
            
#             # Display code
#             if st.session_state['show_code']:
#                 with st.expander("Generated Python Code", expanded=False):
#                     st.code(code, language="python")
            
#             # Display different result types appropriately
#             if isinstance(result, pd.DataFrame):
#                 filtered_result = result.dropna(axis=1, how='all') 
#                 st.dataframe(filtered_result, use_container_width=True)
#                 st.write("Number of rows:", len(result))
#             elif str(type(result)).find('matplotlib') != -1:
#                 st.pyplot(result)
#             elif isinstance(result, (list, dict)):
#                 st.json(result)
#             else:
#                 st.write(result)
            
#             # Display summary of what happened
#             latest_tx = session_info['transformation_history'][-1] if session_info['transformation_history'] else {}
#             st.markdown(f"""
#             <div class="context-info">
#                 <strong>Transformation Summary:</strong><br>
#                 {latest_tx.get('description', 'Transformation completed')}<br>
#                 <strong>Fields Modified:</strong> {', '.join(latest_tx.get('fields_modified', []))}<br>
#                 <strong>Filter Conditions:</strong> {json.dumps(latest_tx.get('filter_conditions', {}))}
#             </div>
#             """, unsafe_allow_html=True)
                
#             # except Exception as e:
#                 # st.error(f"Error processing transformation: {e}")
    
#     # Display previous transformations in this session
#     if st.session_state['transformation_history']:
#         st.markdown("### Previous Transformations")
#         for i, tx in enumerate(st.session_state['transformation_history']):
#             with st.expander(f"Transformation #{i+1}: {tx['description']}", expanded=False):
#                 st.write(f"Query: {tx['query']}")
#                 st.code(tx['code'], language="python")
    
# # Cleanup connections when app is closed
# def cleanup():
#     if sqlite_conn:
#         sqlite_conn.close()

# # Register the cleanup function
# import atexit
# atexit.register(cleanup)
# # Run the app
