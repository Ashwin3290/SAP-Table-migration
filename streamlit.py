# Install `pip3 install streamlit`
# Run `streamlit run streamlit.py --server.port 8501`
# For more, check the docs for streamlit

import warnings
warnings.filterwarnings('ignore')
import time
import pandas as pd
import streamlit as st

from code_exec import save_file, run_code,preprocess_code
from tablellm import get_tllm_response_pure
from mongodb import update_vote_by_session_id

st.set_page_config(page_title="TableLLM", layout = "wide")

if 'chat' not in st.session_state:
    st.session_state['chat'] = {
        "user_input_single": None,
        "user_input_double": None,
        "bot_response_single_1": None, # single table operation, show text or code
        "bot_response_single_2": None, # single table operation, show code execution result
        "bot_response_double_1": None, # multiple table operation, show text or code
        "bot_response_double_2": None, # multiple table operation, show code execution result
    }

# chat mode: QA or Code
if 'chat_mode' not in st.session_state:
    st.session_state['chat_mode'] = 'Code'  # Default to Code mode

# Initialize empty tables instead of using default_table
if 'table0' not in st.session_state:
    st.session_state['table0'] = {
        "uploadFile": None,
        "table": None,
        "dataframe": None,
        "file_detail": None,
    }

if 'table1' not in st.session_state:
    st.session_state['table1'] = {
        "uploadFile": None,
        "table": None,
        "dataframe": None,
        "file_detail": None,
    }

if 'table2' not in st.session_state:
    st.session_state['table2'] = {
        "uploadFile": None,
        "table": None,
        "dataframe": None,
        "file_detail": None,
    }

# session id
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = None
if 'session_id_double' not in st.session_state:
    st.session_state['session_id_double'] = None

text_style = """
    <style>
        .mytext {
            border:1px solid black;
            border-radius:10px;
            border-color: #D6D6D8;
            padding:10px;
            height:auto;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin-bottom: 15px;
        }
    </style>
"""

st.markdown(text_style, unsafe_allow_html=True)

st.markdown('## TableLLM: Enabling Tabular Data Manipulation by LLMs in Real Office Usage Scenarios')

tab1, tab2 = st.tabs(["Single Table Operation", "Double Table Operation"])
with tab1:
    left_column, right_column = st.columns(2, gap = "large")
    
    with right_column.container():
        with st.chat_message(name="user", avatar="user"):
            user_input_single_placeholder = st.empty()
        with st.chat_message(name="assistant", avatar="assistant"):
            bot_response_single_1_placeholder = st.empty()
            bot_response_single_2_placeholder = st.empty()

        user_input = st.text_area("Enter your query:", key = "single tab1 user input")

        # buttons (send, upvote, downvote)
        button_column = st.columns(3)
        button_info = st.empty()
        with button_column[2]:
            send_button = st.button("✉️ Send", key = "single tab1 button", use_container_width=True)
            if send_button and len(user_input) != 0:
                if st.session_state['table0']['table'] is not None:
                    user_input_single_placeholder.markdown(user_input)
                    with st.spinner('loading...'):
                        bot_response, session_id = get_tllm_response_pure(user_input, 
                            st.session_state['table0']['table'],
                            st.session_state['table0']['file_detail'],
                            st.session_state['chat_mode'])

                    st.session_state['chat']['user_input_single'] = user_input
                    st.session_state['chat']['bot_response_single_1'] = bot_response
                    st.session_state['session_id'] = session_id

                    if st.session_state['chat_mode'] == 'QA':
                        bot_response_single_1_placeholder.markdown(bot_response)
                        st.session_state['chat']['bot_response_single_2'] = None
                    else:
                        bot_response_single_1_placeholder.code(bot_response, language='python')
                        
                        # run code and display result
                        code_res = run_code(bot_response, st.session_state['table0']['file_detail']['local_path'])
                        st.session_state['chat']['bot_response_single_2'] = code_res
                        
                        # change the type of code_res
                        if isinstance(code_res, pd.Series):
                            code_res = pd.DataFrame(code_res)

                        if isinstance(code_res, pd.DataFrame):
                            bot_response_single_2_placeholder.dataframe(code_res, height=min(250, len(st.session_state['table0']['dataframe'])*50), use_container_width=True)
                        elif bot_response.find('plt') != -1 and not isinstance(code_res, str):
                            bot_response_single_2_placeholder.pyplot(code_res)
                        else:
                            bot_response_single_2_placeholder.markdown(code_res)
                else:
                    st.error("Please upload a file to start")

        with button_column[1]:
            upvote_button = st.button("👍 Upvote", key = "single upvote button", use_container_width=True)
            if upvote_button and st.session_state['table0']['table'] is not None and st.session_state['session_id'] is not None:
                if st.session_state['chat']['user_input_single'] is not None:
                    user_input_single_placeholder.markdown(st.session_state['chat']['user_input_single'])
                if st.session_state['chat']['bot_response_single_1'] is not None:
                    if st.session_state['chat_mode'] == 'QA':
                        bot_response_single_1_placeholder.markdown(st.session_state['chat']['bot_response_single_1'])
                    else:
                        bot_response_single_1_placeholder.code(st.session_state['chat']['bot_response_single_1'], language='python')
                if st.session_state['chat']['bot_response_single_2'] is not None:
                    if st.session_state['chat_mode'] == 'Code':
                        code_res = st.session_state['chat']['bot_response_single_2']
                        if isinstance(code_res, pd.DataFrame):
                            bot_response_single_2_placeholder.dataframe(code_res, height=min(250, len(st.session_state['table0']['dataframe'])*50), use_container_width=True)
                        elif st.session_state['chat']['bot_response_single_1'].find('plt') != -1:
                            code_res = run_code(st.session_state['chat']['bot_response_single_1'], st.session_state['table0']['file_detail']['local_path'])
                            bot_response_single_2_placeholder.pyplot(code_res)
                        else:
                            bot_response_single_2_placeholder.markdown(code_res)
                # update vote
                update_vote_by_session_id(1, st.session_state['session_id'])
                button_info.success("Your upvote has been uploaded")
            elif upvote_button:
                button_info.info("Please start a conversation before voting.")

        with button_column[0]:
            downvote_button = st.button("👎 Downvote", key = "single downvote button", use_container_width=True)
            if downvote_button and st.session_state['table0']['table'] is not None and st.session_state['session_id'] is not None:
                if st.session_state['chat']['user_input_single'] is not None:
                    user_input_single_placeholder.markdown(st.session_state['chat']['user_input_single'])
                if st.session_state['chat']['bot_response_single_1'] is not None:
                    if st.session_state['chat_mode'] == 'QA':
                        bot_response_single_1_placeholder.markdown(st.session_state['chat']['bot_response_single_1'])
                    else:
                        bot_response_single_1_placeholder.code(st.session_state['chat']['bot_response_single_1'], language='python')
                if st.session_state['chat']['bot_response_single_2'] is not None:
                    if st.session_state['chat_mode'] == 'Code':
                        code_res = st.session_state['chat']['bot_response_single_2']
                        if isinstance(code_res, pd.DataFrame):
                            bot_response_single_2_placeholder.dataframe(code_res, height=min(250, len(st.session_state['table0']['dataframe'])*50), use_container_width=True)
                        elif st.session_state['chat']['bot_response_single_1'].find('plt') != -1:
                            code_res = run_code(st.session_state['chat']['bot_response_single_1'], st.session_state['table0']['file_detail']['local_path'])
                            bot_response_single_2_placeholder.pyplot(code_res)
                        else:
                            bot_response_single_2_placeholder.markdown(code_res)
                # update vote
                update_vote_by_session_id(-1, st.session_state['session_id'])
                button_info.success("Your downvote has been uploaded")
            elif downvote_button:
                button_info.info("Please start a conversation before voting.")

    with left_column:
        illustration0 = st.markdown('- Upload a CSV or Excel file to analyze it with natural language queries.\n\n- The system will generate Python code based on your queries and execute it.\n\n- You can ask questions like "What is the average value?" or "Plot the distribution of values."')
        uploadFile = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls", "docx"])

        # Display mode selection
        st.radio("Select Mode", ["Code", "QA"], key="chat_mode_selector", index=0, 
                 on_change=lambda: setattr(st.session_state, 'chat_mode', st.session_state.mode_selector))

        # Show table if file is uploaded
        if uploadFile is not None:
            if uploadFile == st.session_state['table0']['uploadFile']:
                st.dataframe(st.session_state['table0']['dataframe'], height=min(500, len(st.session_state['table0']['dataframe'])*50), use_container_width=True)
            else:
                try:
                    # if upload csv or xlsx, use Code mode
                    if uploadFile.name.endswith(('.csv', '.xlsx', '.xls')):
                        st.session_state['chat_mode'] = 'Code'
                        st.session_state.mode_selector = 'Code'
                    elif uploadFile.name.endswith(('.docx')):
                        st.session_state['chat_mode'] = 'QA'
                        st.session_state.mode_selector = 'QA'

                    # save file
                    with st.spinner('loading...'):
                        df, tabular, file_detail = save_file(uploadFile)
                        st.session_state['table0'] = {
                            "uploadFile": uploadFile,
                            "table": tabular,
                            "dataframe": df,
                            "file_detail": file_detail
                        }
                    st.dataframe(df, height=min(500, len(df)*50), use_container_width=True)
                    st.session_state['chat']['user_input_single'] = None
                    st.session_state['chat']['bot_response_single_1'] = None
                    st.session_state['chat']['bot_response_single_2'] = None

                except Exception as e:
                    st.error(f"Error Saving File: {str(e)}")
                st.rerun()
        else:
            # Display instructions if no file is uploaded
            st.info("Please upload a file to get started")

with tab2:
    left_column2, right_column2= st.columns(2, gap = "large")
    with left_column2:
        illustration1 = st.markdown('- Upload two CSV or Excel files to merge or analyze them together.\n\n- You can specify how to join the tables (e.g., inner join, outer join).\n\n- Ask questions like "Merge these tables on customer_id" or "Count unique values across both tables."')

        uploadFile1 = st.file_uploader("Upload your first table", type=["csv", "xlsx", "xls"])
        uploadFile2 = st.file_uploader("Upload your second table", type=["csv", "xlsx", "xls"])

        # Show tables if files are uploaded
        if uploadFile1 is not None:
            if uploadFile1 != st.session_state['table1']['uploadFile']:
                try:
                    with st.spinner('loading table1...'):
                        df, tabular, file_detail = save_file(uploadFile1)
                        st.session_state['table1'] = {
                            "uploadFile": uploadFile1,
                            "table": tabular,
                            "dataframe": df,
                            "file_detail": file_detail
                        }
                        st.session_state['chat']['user_input_double'] = None
                        st.session_state['chat']['bot_response_double_1'] = None
                        st.session_state['chat']['bot_response_double_2'] = None
                except Exception as e:
                    st.error(f"Error Saving File: {str(e)}")
                    
        if uploadFile2 is not None:
            if uploadFile2 != st.session_state['table2']['uploadFile']:
                try:
                    with st.spinner('loading table2...'):
                        df, tabular, file_detail = save_file(uploadFile2)
                        st.session_state['table2'] = {
                            "uploadFile": uploadFile2,
                            "table": tabular,
                            "dataframe": df,
                            "file_detail": file_detail
                        }
                        st.session_state['chat']['user_input_double'] = None
                        st.session_state['chat']['bot_response_double_1'] = None
                        st.session_state['chat']['bot_response_double_2'] = None
                except Exception as e:
                    st.error(f"Error Saving File: {str(e)}")
            
        if (st.session_state['table1']['dataframe'] is not None) and (st.session_state['table2']['dataframe'] is not None):
            st.subheader("Table 1")
            st.dataframe(st.session_state['table1']['dataframe'], height=min(250, len(st.session_state['table1']['dataframe'])*50), use_container_width=True)
            st.subheader("Table 2")
            st.dataframe(st.session_state['table2']['dataframe'], height=min(250, len(st.session_state['table2']['dataframe'])*50), use_container_width=True)
        else:
            st.info("Please upload both files to get started with table merging")

    with right_column2:
        with st.chat_message(name="user", avatar="user"):
            user_input_double_placeholder = st.empty()
        with st.chat_message(name="assistant", avatar="assistant"):
            bot_response_double_1_placeholder = st.empty()
            bot_response_double_2_placeholder = st.empty()

        user_input = st.text_area("Enter your query:", key = "double tab2 user input")

        button_column = st.columns(3)
        button_info2 = st.empty()
        with button_column[2]:
            send_button = st.button("✉️ Send", key = "double tab1 button", use_container_width=True)
            if send_button and len(user_input) != 0:
                if (st.session_state['table1']['dataframe'] is not None) and (st.session_state['table2']['dataframe'] is not None):
                    user_input_double_placeholder.markdown(user_input)
                    with st.spinner('loading...'):
                        bot_response, session_id = get_tllm_response_pure(question=user_input,
                            table=(st.session_state['table1']['table'], st.session_state['table2']['table']),
                            file_detail=[st.session_state['table1']['file_detail'], st.session_state['table2']['file_detail']],
                            mode='Code_Merge')

                    st.session_state['chat']['user_input_double'] = user_input
                    st.session_state['chat']['bot_response_double_1'] = bot_response
                    st.session_state['session_id_double'] = session_id

                    bot_response_double_1_placeholder.code(bot_response, language='python')

                    
                    # run code and display result
                    code_res = run_code(bot_response,
                        (st.session_state['table1']['file_detail']['local_path'], st.session_state['table2']['file_detail']['local_path']),
                        is_merge=True)
                    st.session_state['chat']['bot_response_double_2'] = code_res
                    bot_response_double_2_placeholder.dataframe(code_res, height=min(250, len(st.session_state['table1']['dataframe'])*50), use_container_width=True)
                else:
                    button_info2.error("Please upload both files to start")

        with button_column[1]:
            upvote_button = st.button("👍 Upvote", key = "double upvote button", use_container_width=True)
            if upvote_button and (st.session_state['table1']['dataframe'] is not None) and (st.session_state['table2']['dataframe'] is not None) and st.session_state['session_id_double'] is not None:
                if st.session_state['chat']['user_input_double'] is not None:
                    user_input_double_placeholder.markdown(st.session_state['chat']['user_input_double'])
                if st.session_state['chat']['bot_response_double_1'] is not None:
                    bot_response_double_1_placeholder.code(st.session_state['chat']['bot_response_double_1'], language='python')
                if st.session_state['chat']['bot_response_double_2'] is not None:
                    bot_response_double_2_placeholder.dataframe(st.session_state['chat']['bot_response_double_2'], height=min(250, len(st.session_state['table2']['dataframe'])*50), use_container_width=True)
                # update vote
                update_vote_by_session_id(1, st.session_state['session_id_double'])
                button_info2.success("Your upvote has been uploaded")
            elif upvote_button:
                button_info2.info("Please start a conversation before voting.")

        with button_column[0]:
            downvote_button = st.button("👎 Downvote", key = "double downvote button", use_container_width=True)
            if downvote_button and (st.session_state['table1']['dataframe'] is not None) and (st.session_state['table2']['dataframe'] is not None) and st.session_state['session_id_double'] is not None:
                if st.session_state['chat']['user_input_double'] is not None:
                    user_input_double_placeholder.markdown(st.session_state['chat']['user_input_double'])
                if st.session_state['chat']['bot_response_double_1'] is not None:
                    bot_response_double_1_placeholder.code(st.session_state['chat']['bot_response_double_1'], language='python')
                if st.session_state['chat']['bot_response_double_2'] is not None:
                    bot_response_double_2_placeholder.dataframe(st.session_state['chat']['bot_response_double_2'], height=min(250, len(st.session_state['table2']['dataframe'])*50), use_container_width=True)
                # update vote
                update_vote_by_session_id(-1, st.session_state['session_id_double'])
                button_info2.success("Your downvote has been uploaded")
            elif downvote_button:
                button_info2.info("Please start a conversation before voting.")

        # Example queries
        st.markdown("##### Example queries you can try:")
        example_queries = [
            "Merge these tables on common column names",
            "Join the tables and show all records from both tables",
            "Count the total records after merging the tables",
            "What are the common columns between these tables?",
            "Merge the tables and calculate summary statistics"
        ]
        
        for i, query in enumerate(example_queries):
            cols = st.columns([7,1])
            with cols[0]:
                st.markdown(f"<div class='mytext'>{query}</div>", unsafe_allow_html=True)
            with cols[1]:
                if st.button("Send", key=f"example_query_{i}", use_container_width=True):
                    if (st.session_state['table1']['dataframe'] is not None) and (st.session_state['table2']['dataframe'] is not None):
                        # Set query text and trigger similar flow as Send button
                        st.session_state["double tab2 user input"] = query
                        st.rerun()
                    else:
                        button_info2.error("Please upload both files first")