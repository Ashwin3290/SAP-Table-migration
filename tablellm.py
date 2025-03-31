import os
import uuid
import logging
import json
import requests
import pandas as pd    
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from prompt_format import SINGLE_TABLE_TEMPLATE, DOUBLE_TABLE_TEMPLATE, QA_TEMPLATE, DOUBLE_TABLE_TEMPLATE_PRE
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

# Try to load table descriptions if available
try:
    table_desc = pd.read_excel('MARA_desc.xlsx')
except Exception as e:
    logger.warning(f"Error loading table descriptions: {e}")
    table_desc = pd.DataFrame()


class TableLLM:
    """TableLLM handles generating and executing code for table analysis with DMC support"""
    
    def __init__(self):
        """Initialize the TableLLM instance with LLM support"""
        # Configure Gemini if API key is available
        self.use_gemini = False
        self.api_key = os.environ.get('GEMINI_API_KEY')
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.use_gemini = True
                logger.info("Using Gemini API for code generation")
                
                # Configure safety settings for DMC functionality
                self.safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            except Exception as e:
                logger.warning(f"Error configuring Gemini: {e}. Falling back to Ollama.")
    
    # ==============================
    # Original TableLLM functionality
    # ==============================
    
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
    
    def _format_double_table_prompt_pre(self, question, tables):
        """Format a double table prompt"""
        # Extract header and first few rows for each table
        table1_lines = tables[0].strip().split('\n')
        header1 = table1_lines[0]
        
        table2_lines = tables[1].strip().split('\n')
        header2 = table2_lines[0]
        
        return DOUBLE_TABLE_TEMPLATE_PRE.format(
            csv_data1=f"{header1}",
            csv_data2=f"{header2}",
            question=question,
            table_desc=table_desc.to_csv(index=False),
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
    
    def _format_qa_prompt(self, question, table, description=''):
        """Format a QA prompt"""
        return QA_TEMPLATE.format(
            table_descriptions=description,
            table_in_csv=table,
            question=question
        )
    
    def _generate_with_gemini(self, prompt, temperature=0.2, model='gemini-2.0-flash'):
        """Generate response using Gemini API"""
        try:
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 2048,
                    "top_p": 0.95,
                    "top_k": 40,
                },
                safety_settings=getattr(self, 'safety_settings', None)
            )
            response = model_instance.generate_content(prompt)
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
    
    def generate(self, prompt, temperature=0.2):
        """Generate a response using the available LLM service"""
        if self.use_gemini:
            response = self._generate_with_gemini(prompt, temperature=temperature)
            if response:
                return response
            
        # Fall back to Ollama if Gemini fails or isn't configured
        return self._generate_with_ollama(prompt)
    
    def pre_process_query(self, question, tables, dataframes, mode='Code'):
        """Preprocess the query to return structured response"""

        is_double = isinstance(tables,tuple) and len(tables) == 2

        # generate appropriate prompt

        if mode=='Code':
            if is_double:
                prompt = self._format_double_table_prompt_pre(question, tables)

            prompt = prompt 
            logger.info(f"Generating structured query for user query:{question}")

            raw_prompt = self.generate(prompt)

        return raw_prompt
    
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
            
            if not table_desc.empty:
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
    
    # ==============================
    # DMC Integration functionality
    # ==============================
    
    def load_dmc_mappings(self, excel_path, project_id, object_id, segment_id):
        """
        Load DMC mappings from an Excel file and create structured mapping objects.
        
        Args:
            excel_path: Path to the Excel file containing DMC mappings
            project_id: Project ID
            object_id: Object ID
            segment_id: Segment ID
            
        Returns:
            List of mapping dictionaries
        """
        try:
            # Load Excel file directly
            df = pd.read_excel(excel_path)
            
            # Create mapping objects
            mappings = []
            for _, row in df.iterrows():
                mapping = {
                    "project_id": project_id,
                    "object_id": object_id,
                    "segment_id": segment_id,
                    "source_table": row.get('source_table', ''),
                    "source_field_name": row.get('source_field_name', ''),
                    "target_table": row.get('target_sap_table', ''),
                    "target_field": row.get('target_sap_field', ''),
                    "description": row.get('description', f"{row.get('source_field_name', '')} mapping")
                }
                mappings.append(mapping)
            
            return mappings
        except Exception as e:
            logger.error(f"Error loading DMC Excel file: {e}")
            return []
    
    def generate_first_dmc_prompt(self, user_query, mappings, project_info):
        """
        Generate the first LLM prompt to bridge user intent to technical schema.
        
        Args:
            user_query: The natural language query from the user
            mappings: The DMC mappings loaded from Excel
            project_info: Dictionary with project_id, object_id, segment_id
            
        Returns:
            Prompt for the first LLM call
        """
        # Extract and format field mappings for the prompt
        field_descriptions = []
        for mapping in mappings:
            field_descriptions.append(
                f"{mapping['description']} maps to {mapping['source_table']}.{mapping['source_field_name']} "
                f"(target: {mapping['target_table']}.{mapping['target_field']})"
            )
        
        field_mappings_text = "\n".join(field_descriptions)
        
        # Get unique source and target tables
        source_tables = set(m["source_table"] for m in mappings if m["source_table"])
        target_tables = set(m["target_table"] for m in mappings if m["target_table"])
        
        # Create the prompt
        prompt = f"""
        You are an expert in translating business requests into precise technical requirements.
        
        USER QUERY: "{user_query}"
        
        PROJECT CONTEXT:
        - Project ID: {project_info.get('project_id')}
        - Object ID: {project_info.get('object_id')}
        - Segment ID: {project_info.get('segment_id')}
        
        AVAILABLE TABLES:
        - Source Tables: {', '.join(source_tables)}
        - Target Tables: {', '.join(target_tables)}
        
        FIELD MAPPINGS:
        {field_mappings_text}
        
        INSTRUCTIONS:
        1. Analyze the user query to identify key operations and fields.
        2. Map any user-friendly terms to the actual technical field names.
        3. Identify which source and target tables are needed.
        4. Do NOT generate code yet.
        5. Create a precise technical specification in JSON format.
        
        RETURN A JSON OBJECT WITH THESE FIELDS:
        - "operation": The main operation (e.g., SELECT, UPDATE, INSERT, DELETE, TRANSFORM)
        - "source_tables": List of needed source tables with their column names
        - "target_tables": List of needed target tables with their column names
        - "conditions": Any filtering/selection conditions in technical field names
        - "transformations": Any data transformations needed
        - "technical_description": A precise technical description using actual table and column names
        
        JSON FORMAT ONLY.
        """
        
        return prompt
    
    def generate_second_dmc_prompt(self, first_llm_output, user_query):
        """
        Generate the second LLM prompt to create Python code based on first LLM output
        
        Args:
            first_llm_output: The parsed output from the first LLM call
            user_query: The original natural language query
            
        Returns:
            Prompt for the second LLM call
        """
        prompt = f"""
        You are an expert Python code generator. Based on the technical specification below,
        generate Python code that implements the requirement without using SQL queries.
        
        ORIGINAL USER QUERY: "{user_query}"
        
        TECHNICAL SPECIFICATION:
        {first_llm_output}
        
        REQUIREMENTS:
        1. Generate Python code that implements the operation directly using pandas or similar libraries.
        2. Do NOT use any SQL or database connections.
        3. Do NOT use CSV files for data storage.
        4. Handle the data operations in memory.
        5. Use actual table and column names as specified in the technical specification.
        6. Include clear comments to explain your code.
        7. Implement robust error handling.
        8. Return only the Python code with no surrounding explanation.
        
        The code should follow this template:
        ```python
        def process_data(data_sources, project_info):
            '''
            Process data according to the user query
            
            Args:
                data_sources: Dictionary of dataframes where keys are table names
                project_info: Dictionary with project_id, object_id, segment_id
                
            Returns:
                Processed data (dataframe or other appropriate format)
            '''
            # Your implementation here
            
            # Return the result
            return result
        ```
        
        RETURN ONLY THE PYTHON CODE.
        """
        
        return prompt
    
    def generate_dmc_code(self, user_query, dmc_excel_path, project_id, object_id, segment_id):
        """
        Main method that implements the two-step LLM process to generate Python code
        
        Args:
            user_query: The natural language query from the user
            dmc_excel_path: Path to the Excel file with DMC mappings
            project_id: Project ID
            object_id: Object ID
            segment_id: Segment ID
            
        Returns:
            Generated Python code as a string
        """
        if not self.use_gemini:
            return "Error: DMC functionality requires Gemini API. Please set GEMINI_API_KEY environment variable."
            
        # Step 1: Load DMC mappings
        mappings = self.load_dmc_mappings(dmc_excel_path, project_id, object_id, segment_id)
        if not mappings:
            return "Error: Could not load DMC mappings from Excel file."
        
        # Project info dictionary
        project_info = {
            "project_id": project_id,
            "object_id": object_id,
            "segment_id": segment_id
        }
        
        # Step 2: Generate first prompt and call LLM
        first_prompt = self.generate_first_dmc_prompt(user_query, mappings, project_info)
        first_llm_output = self._generate_with_gemini(first_prompt)
        if not first_llm_output:
            return "Error: First LLM call failed to generate output."
        
        # Step 3: Generate second prompt and call LLM
        second_prompt = self.generate_second_dmc_prompt(first_llm_output, user_query)
        generated_code = self._generate_with_gemini(second_prompt, temperature=0.1)  # Lower temperature for code
        if not generated_code:
            return "Error: Second LLM call failed to generate code."
        
        # Step 4: Clean up the generated code
        # Remove any extra markdown or text around the Python code
        if "```python" in generated_code:
            code_start = generated_code.find("```python") + 9
            code_end = generated_code.rfind("```")
            if code_end > code_start:
                generated_code = generated_code[code_start:code_end].strip()
        elif "```" in generated_code:
            code_start = generated_code.find("```") + 3
            code_end = generated_code.rfind("```")
            if code_end > code_start:
                generated_code = generated_code[code_start:code_end].strip()
        
        return generated_code
    
    def load_table_data(self, table_paths):
        """
        Load table data from various sources
        
        Args:
            table_paths: Dictionary mapping table names to file paths
            
        Returns:
            Dictionary of dataframes
        """
        data_sources = {}
        
        for table_name, file_path in table_paths.items():
            try:
                # Determine file type and load accordingly
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    logger.warning(f"Unsupported file type for table {table_name}: {file_path}")
                    continue
                
                data_sources[table_name] = df
            except Exception as e:
                logger.error(f"Error loading table {table_name} from {file_path}: {e}")
        
        return data_sources
    
    def process_dmc_query(self, query, dmc_file_path, table_paths, project_id, object_id, segment_id):
        """
        Process a DMC query with the two-step LLM approach
        
        Args:
            query: Natural language query from the user
            dmc_file_path: Path to DMC Excel file with field mappings
            table_paths: Dictionary mapping table names to file paths 
            project_id: Project ID
            object_id: Object ID
            segment_id: Segment ID
            
        Returns:
            tuple: (code_content, result)
        """
        # Check if Gemini is available
        if not self.use_gemini:
            return "Error: DMC functionality requires Gemini API. Please set GEMINI_API_KEY.", None
        
        # Generate code using the two-step LLM approach
        code_content = self.generate_dmc_code(
            query, dmc_file_path, project_id, object_id, segment_id
        )
        
        # Create a file with the generated code
        code_file = create_code_file(
            code_content=code_content,
            query=query,
            is_double=False  # DMC queries typically work with single dataframe result
        )
        
        # Load table data
        data_sources = self.load_table_data(table_paths)
        
        # Project info
        project_info = {
            'project_id': project_id,
            'object_id': object_id,
            'segment_id': segment_id
        }
        
        # Execute the code
        result = execute_code(code_file, data_sources)
        
        return code_content, result