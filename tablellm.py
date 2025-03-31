import os
import uuid
import logging
import json
import requests
import pandas as pd    
from dotenv import load_dotenv
import google.generativeai as genai

from prompt_format import SINGLE_TABLE_TEMPLATE, DOUBLE_TABLE_TEMPLATE
from code_exec import create_code_file, execute_code

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load configuration
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    OLLAMA_URL = config.get('model_url', 'http://localhost:11434/api/generate')
    OLLAMA_MODEL = config.get('model_name', 'llama3')
except Exception as e:
    logger.warning(f"Error loading config: {e}. Using default values.")
    OLLAMA_URL = 'http://localhost:11434/api/generate'
    OLLAMA_MODEL = 'llama3'

table_desc = pd.read_excel('MARA_desc.xlsx')


class TableLLM:
    """TableLLM handles generating and executing code for table analysis"""
    
    def __init__(self):
        """Initialize the TableLLM instance"""
        # Configure Gemini if API key is available
        self.use_gemini = False
        if 'GEMINI_API_KEY' in os.environ:
            try:
                genai.configure(api_key=os.environ['GEMINI_API_KEY'])
                self.use_gemini = True
                logger.info("Using Gemini API for code generation")
            except Exception as e:
                logger.warning(f"Error configuring Gemini: {e}. Falling back to Ollama.")
    
    def _format_single_table_prompt(self, question, table):
        """Format a single table prompt"""
        # Extract header and first few rows
        table_lines = table.strip().split('\n')
        header = table_lines[0]
        sample_rows = '\n'.join(table_lines[1:6])  # First 5 data rows
        
        return SINGLE_TABLE_TEMPLATE.format(
            csv_data=f"{header}\n{sample_rows}",
            question=question
        )
    
    def _format_double_table_prompt(self, question, tables):
        """Format a double table prompt"""
        # Extract header and first few rows for each table
        table1_lines = tables[0].strip().split('\n')
        header1 = table1_lines[0]
        sample_rows1 = '\n'.join(table1_lines[1:6])
        
        table2_lines = tables[1].strip().split('\n')
        header2 = table2_lines[0]
        sample_rows2 = '\n'.join(table2_lines[1:6])
        
        return DOUBLE_TABLE_TEMPLATE.format(
            csv_data1=f"{header1}\n{sample_rows1}",
            csv_data2=f"{header2}\n{sample_rows2}",
            question=question
        )

    
    def _generate_with_gemini(self, prompt):
        """Generate response using Gemini API"""
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error with Gemini: {e}")
            return None
    
    def _generate_with_ollama(self, prompt):
        """Generate response using Ollama API"""
        try:
            data = {
                'model': OLLAMA_MODEL,
                'prompt': prompt,
                'stream': False,
                'temperature': 0.2,  # Lower temperature for more deterministic code
                'top_p': 0.95,
                'max_tokens': 2048
            }
            
            response = requests.post(url=OLLAMA_URL, json=data, timeout=60)
            if response.status_code == 200:
                return response.json()['response']
            else:
                logger.error(f"Error from Ollama API: {response.status_code}")
                return f"Error from LLM service. Status code: {response.status_code}"
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return f"Error connecting to LLM service: {str(e)}"
    
    def generate(self, prompt):
        """Generate a response using the available LLM service"""
        if self.use_gemini:
            response = self._generate_with_gemini(prompt)
            if response:
                return response
            
        # Fall back to Ollama if Gemini fails or isn't configured
        return self._generate_with_ollama(prompt)
    
    def process_query(self, question, tables, dataframes, mode='Code'):
        """Process a query and execute code or generate QA response"""
        
        is_double = isinstance(tables, tuple) and len(tables) == 2
        
        # Generate the appropriate prompt
        if mode == 'QA':
            description = ''
            if isinstance(tables, dict) and 'description' in tables:
                description = tables['description']
            prompt = self._format_qa_prompt(question, tables, description)
            response = self.generate(prompt)
            return response, None  # No code to execute for QA
        
        elif mode == 'Code':
            if is_double:
                prompt = self._format_double_table_prompt(question, tables)
            else:
                prompt = self._format_single_table_prompt(question, tables)
            
            prompt = prompt + "\n\nTable Description:\n" + table_desc.to_csv(index=False)
                
            logger.info(f"Generating code for query: {question}")
            
            # Generate code content and extract only the body
            raw_code = self.generate(prompt)
            
            # Clean the code - extract only what would go inside the function
            if "# CODE STARTS HERE" in raw_code and "# CODE ENDS HERE" in raw_code:
                start_marker = "# CODE STARTS HERE"
                end_marker = "# CODE ENDS HERE"
                start_idx = raw_code.find(start_marker) + len(start_marker)
                end_idx = raw_code.find(end_marker)
                if start_idx != -1 and end_idx != -1:
                    code_content = raw_code[start_idx:end_idx].strip()
                else:
                    code_content = raw_code
            else:
                code_content = raw_code

            
            if "```python" in code_content:
                code_content = code_content.replace("```python", "").replace("```", "")
            if "```" in code_content:
                code_content = code_content.replace("```", "")
            # Create a code file with the generated content
            code_file = create_code_file(code_content, question, is_double=is_double)
            
            # Execute the code
            result = execute_code(code_file, dataframes, is_double=is_double)
            
            return code_content, result
        
        else:
            return f"Unknown mode: {mode}", None
            
    def save_interaction(self, question, code, result, file_details, db_client=None):
        """Save the interaction to database if a client is provided"""
        if not db_client:
            return None
            
        session_id = str(uuid.uuid4())
        try:
            db_client.chat.insert_one({
                'session_id': session_id,
                'question': question,
                'code': code,
                'result': str(result)[:1000],  # Limit result size
                'file_details': file_details,
                'vote': 0
            })
            return session_id
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            return None