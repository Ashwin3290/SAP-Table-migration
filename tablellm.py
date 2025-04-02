import os
import uuid
import logging
import json
import requests
import pandas as pd
import sqlite3
from io import StringIO
from dotenv import load_dotenv
from google import genai
from google.genai import types
from token_tracker import track_token_usage, get_token_usage_stats

# Import planner functions 
from planner import process_query as planner_process_query
from planner import get_session_context, get_or_create_session_target_df, save_session_target_df
from code_exec import create_code_file, execute_code

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

api_key = os.environ.get('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)


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


class TableLLM:
    """TableLLM handles generating and executing code for table analysis with context awareness"""
    
    def __init__(self):
        """Initialize the TableLLM instance"""
        # Configure Gemini if API key is available
        self.use_gemini = False
        if 'GEMINI_API_KEY' in os.environ:
            try:
                self.use_gemini = True
                logger.info("Using Gemini API for code generation")
            except Exception as e:
                logger.warning(f"Error configuring Gemini: {e}. Falling back to Ollama.")
    
    def _format_single_table_prompt(self, question, table):
        """Format a single table prompt"""
        # Implementation remains the same
        pass
    
    def _format_context_aware_prompt(self, resolved_data):
        """Format a code generation prompt that includes context"""
        
        # Extract context information
        context = resolved_data.get("context", {})
        history = context.get("transformation_history", [])
        table_state = context.get("target_table_state", {})
        
        # Build context section for the prompt
        context_section = """
Previous Transformations:
"""
        if not history:
            context_section += "None (this is the first transformation)"
        else:
            for i, tx in enumerate(history):
                context_section += f"""
{i+1}. {tx.get('description', 'Unknown transformation')}
   - Fields modified: {', '.join(tx.get('fields_modified', []))}
   - Filter conditions: {json.dumps(tx.get('filter_conditions', {}))}
"""
        
        context_section += f"""
Current Target Table State:
- Populated fields: {', '.join(table_state.get('populated_fields', []))}
- Remaining mandatory fields: {', '.join(table_state.get('remaining_mandatory_fields', []))}
"""
        
        # Handle insertion fields - ensure it's properly formatted
        insertion_fields_str = ""
        if "insertion_fields" in resolved_data and resolved_data["insertion_fields"]:
            if isinstance(resolved_data["insertion_fields"], list) and len(resolved_data["insertion_fields"]) > 0:
                insertion_fields_str = resolved_data["insertion_fields"][0]["source_field"]
                target_field_str = resolved_data["insertion_fields"][0]["target_field"]
            else:
                logger.warning(f"Unexpected insertion_fields format: {resolved_data['insertion_fields']}")
                insertion_fields_str = str(resolved_data["insertion_fields"])
                target_field_str = "Unknown"
        else:
            insertion_fields_str = "None"
            target_field_str = "None"
            
        # Combine with the regular prompt
        prompt = f"""
I need ONLY Python code - DO NOT include any explanations, markdown, or comments outside the code.

Source table info and description:
{resolved_data['source_info']}
{resolved_data['source_describe']}


{"There can also be more than one source table. In that case these are the Additional source table info and description:" if resolved_data["additional_source_table"] else ""}
{resolved_data["additional_source_tables"] if resolved_data["additional_source_table"] else ""}

Target table info and description:
{resolved_data['target_info']}
{resolved_data['target_describe']}

Columns that will be used for filtering:
{resolved_data['filtering_fields']}

Source column from where data has to be picked:
{insertion_fields_str}

Target column where data will be inserted:
{target_field_str}
{context_section}

Question: {resolved_data['restructured_question']}

I want your code in this exact function template:
df1 will be the source table and df2 will be the target table.
The function must update df2 (target table) WITHOUT replacing any previously populated data.

{"If we have additional tables then you will find them in additional_tables dictionary with the table names as the keys. df1 has the " + resolved_data['source_table_name'] if resolved_data["additional_source_table"] else ""}


def analyze_data(df1, df2, additional_tables=None):
    # Your Code comes here
    return result

REQUIREMENTS:
1. Output ONLY the code that goes between the comments
2. Assign your final output to a variable named 'result'
3. PRESERVE any existing data in df2 that was populated by previous transformations
4. Handle data errors and empty dataframes
5. NEVER use print() statements
6. NEVER reference file paths
7. Include comments inside your code
8. Use the latest Python syntax and libraries
9. Use efficient data processing techniques
10. Insert Data into the target table while keeping the rest of the columns and data constant, so if there are any other columns already in the data then make sure to maintain the relationship that existed in the source table like the id mapping
11. When having context, make sure to use the context information in the code generation
12. Use the schema of the target table to properly add the data to the target table
13. Use the schema of the source table to properly filter the data from the source table
14. Based on the current df1 table add the values while keeping the current structure constant, you would have to use key matching to accurately add the data
"""
        
        return prompt
    
    @track_token_usage()
    def _generate_with_gemini(self, prompt):
        """Generate response using Gemini API"""
        try:
            response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents = prompt,
        )
            # Log token usage statistics after call
            logger.info(f"Current token usage: {get_token_usage_stats()}")
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
    
    def process_sequential_query(self, query, object_id=29, segment_id=336, project_id=24, session_id=None):
        """
        Process a query as part of a sequential transformation
        
        Parameters:
        query (str): The user's query
        object_id (int): Object ID for mapping
        segment_id (int): Segment ID for mapping
        project_id (int): Project ID for mapping
        session_id (str): Optional session ID, creates new session if None
        
        Returns:
        tuple: (code, result, session_id)
        """
        # Process query with context awareness
        resolved_data = planner_process_query(object_id, segment_id, project_id, query, session_id)
        if not resolved_data:
            return None, "Failed to resolve query", session_id
        
        # Get session ID from the results
        session_id = resolved_data.get("session_id")
        
        # Connect to database
        conn = sqlite3.connect('db.sqlite3')
        print(resolved_data)
        # Extract table names and field names
        source_table = resolved_data['source_table_name']
        target_table = resolved_data['target_table_name']
        source_fields = resolved_data['source_field_names']
        target_fields = resolved_data['target_sap_fields']
        additional_tables = resolved_data['additional_source_table']
        
        
        # Get source dataframe
        source_df = pd.read_sql_query(f"SELECT * FROM {source_table}", conn)
        
        # Get target dataframe (either existing or new)
        target_df = get_or_create_session_target_df(session_id, target_table , conn)
        
        if additional_tables:
            additional_source_tables = {}
            for table in additional_tables:
                additional_source_tables[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        
        else:
            additional_source_tables = None
        

        # Generate code with context awareness
        code_prompt = self._format_context_aware_prompt(resolved_data)
        code = self.generate(code_prompt)
        
        # Clean the code
        if "# CODE STARTS HERE" in code and "# CODE ENDS HERE" in code:
            start_marker = "# CODE STARTS HERE"
            end_marker = "# CODE ENDS HERE"
            start_idx = code.find(start_marker) + len(start_marker)
            end_idx = code.find(end_marker)
            if start_idx != -1 and end_idx != -1:
                code_content = code[start_idx:end_idx].strip()
            else:
                code_content = code
        else:
            code_content = code

        if "```python" in code_content:
            code_content = code_content.replace("```python", "").replace("```", "")
        if "```" in code_content:
            code_content = code_content.replace("```", "")
        
        # Create code file
        code_file = create_code_file(code_content, query, is_double=True)
        
        # Execute code
        result = execute_code(code_file, (source_df, target_df),additional_tables=additional_source_tables, is_double=True)
        
        # Save the updated target dataframe
        if isinstance(result, pd.DataFrame):
            save_session_target_df(session_id, result)
        
        conn.close()
        return code_content, result, session_id
    
    def process_query(self, question, tables, dataframes, mode='Code', session_id=None):
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
                # Use the context-aware sequential processing
                return self.process_sequential_query(question, session_id=session_id)
            else:
                # Single table case remains the same
                prompt = self._format_single_table_prompt(question, tables)
                
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
    
    def get_session_info(self, session_id):
        """
        Get information about a session
        
        Parameters:
        session_id (str): The session ID
        
        Returns:
        dict: Session information
        """
        context = get_session_context(session_id)
        return {
            "session_id": session_id,
            "context": context,
            "transformation_history": context.get("context", {}).get("transformation_history", []) if context else [],
            "target_table_state": context.get("context", {}).get("target_table_state", {}) if context else {}
        }
            
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