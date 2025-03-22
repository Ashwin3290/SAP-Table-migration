import json
import requests
import base64
import os
from dotenv import load_dotenv
import google.generativeai as genai

from code_exec import check_code
from mongodb import insert_chat
from prompt_format import QA_PROMPT, CODE_PROMPT, CODE_MERGE_PROMPT

load_dotenv()
with open('config.json', 'r') as f:
    config = json.load(f)
URL = config['model_url']
MODEL_NAME = config.get('model_name', 'codellama')  # Default to codellama if not specified

def get_qa_prompt(question, table, description):
    prompt = QA_PROMPT.format_map({
        'table_descriptions': description,
        'table_in_csv': table,
        'question': question
    })
    return prompt

def get_code_prompt(question, table):
    # Use a simple approach without tokenizer
    table = table.split('\n')
    head = table[0]
    body = table[1:6]  # Take first 5 lines
    body_str = "\n".join(body)
    
    prompt = CODE_PROMPT.format_map({
        'csv_data': f'{head}\n{body_str}',
        'question': question
    })
    return prompt

def get_code_merge_prompt(question, table):
    # Use a simple approach without tokenizer
    table1 = table[0].split('\n')
    head1, body1 = table1[0], table1[1:6]  # Take first 5 lines
    table2 = table[1].split('\n')
    head2, body2 = table2[0], table2[1:6]  # Take first 5 lines
    
    body1_str = "\n".join(body1)
    body2_str = "\n".join(body2)
    
    prompt = CODE_MERGE_PROMPT.format_map({
        'csv_data1': f'{head1}\n{body1_str}',
        'csv_data2': f'{head2}\n{body2_str}',
        'question': question
    })
    return prompt

def get_tablellm_response(question, table, file_detail, mode):
    # get prompt
    if mode == 'QA':
        prompt = get_qa_prompt(question, table, file_detail['description'] if isinstance(file_detail, dict) and 'description' in file_detail else '')
    elif mode == 'Code':
        prompt = get_code_prompt(question, table)
    elif mode == 'Code_Merge':
        prompt = get_code_merge_prompt(question, table)
    
    # get LLM response using Ollama API
    data = {
        'model': MODEL_NAME,
        'prompt': prompt,
        'stream': False,
        'temperature': 0.8,
        'top_p': 0.95,
        'max_tokens': 512
    }
    
    try:
        res = requests.post(url=URL, json=data)
        if res.status_code == 200:
            response = res.json()['response']
            
            # For code mode, check if code runs successfully
            if mode != 'QA':
                local_path = file_detail['local_path'] if mode == 'Code' else [file_detail[0]['local_path'], file_detail[1]['local_path']]
                if not check_code(code=response, local_path=local_path, is_merge=(mode == 'Code_Merge')):
                    # Could implement retry logic here
                    pass
        else:
            response = f'Error occur when generating response. Status code: {res.status_code}'
            print(response)
    except Exception as e:
        response = f'Error connecting to LLM service: {str(e)}'
        print(response)
    
    # save log
    session_id = insert_chat(question=question, answer=response, file_detail=file_detail)
    
    return response, session_id

def get_tllm_response_pure(question, table, file_detail, mode):
    # get prompt
    if mode == 'QA':
        prompt = get_qa_prompt(question, table, file_detail['description'] if isinstance(file_detail, dict) and 'description' in file_detail else '')
    elif mode == 'Code':
        prompt = get_code_prompt(question, table)
    elif mode == 'Code_Merge':
        prompt = get_code_merge_prompt(question, table)
    
    try:
        prompt=prompt+'\n'+"the file name is "+file_detail['name']+" and use pandas library for handling the data, and present the result in tabular format"
        res = generate(prompt)
        session_id = insert_chat(question=question, answer=res, file_detail=file_detail)
        if mode != 'QA':
            local_path = file_detail['local_path'] if mode == 'Code' else [file_detail[0]['local_path'], file_detail[1]['local_path']]
            if not check_code(code=res, local_path=local_path, is_merge=(mode == 'Code_Merge')):
                pass
        return res, session_id
    except Exception as e:
        return f"Error connecting to LLM service: {str(e)}", None
    

def generate(prompt):
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model.generate_content(prompt).text


