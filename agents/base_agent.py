"""
Base agent implementation for TableLLM
"""
import os
import json
import re
import traceback
from google import genai
from google.genai import types
from utils.logging_utils import agent_logger as logger
from utils.token_utils import track_token_usage
import config

class BaseAgent:
    """
    Base class for all agents with common functionality
    """
    
    def __init__(self, client=None):
        """
        Initialize the base agent
        
        Parameters:
        client (genai.Client, optional): LLM client, creates new one if None
        """
        try:
            # Initialize client if not provided
            if client is None:
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not found in environment variables")
                self.client = genai.Client(api_key=api_key)
            else:
                self.client = client
                
            # Set agent name
            self.agent_name = self.__class__.__name__
            
            logger.info(f"Initialized {self.agent_name}")
        except Exception as e:
            logger.error(f"Error initializing {self.__class__.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    @track_token_usage()
    def _call_llm(self, prompt, model=None, temperature=None, top_p=None, top_k=None):
        """
        Call the LLM with given prompt and parameters
        
        Parameters:
        prompt (str): The prompt to send to the LLM
        model (str, optional): Model to use, defaults to config.GENERATION_MODEL
        temperature (float, optional): Temperature parameter for generation
        top_p (float, optional): Top p parameter for generation
        top_k (int, optional): Top k parameter for generation
        
        Returns:
        str: The generated text response
        """
        try:
            # Set model and parameters
            if model is None:
                model = config.GENERATION_MODEL
                
            if temperature is None:
                temperature = config.TEMPERATURE
                
            if top_p is None:
                top_p = config.TOP_P
                
            if top_k is None:
                top_k = config.TOP_K
            
            # Log the call
            logger.debug(f"{self.agent_name} calling LLM with prompt: {prompt[:100]}...")
            
            # Call the LLM
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
            )
            
            # Check if response is valid
            if not response or not hasattr(response, "text") or not response.text:
                logger.warning(f"{self.agent_name} received invalid response from LLM")
                return None
            
            return response.text
        except Exception as e:
            logger.error(f"Error in {self.agent_name}._call_llm: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _parse_json_response(self, response_text):
        """
        Parse a JSON response from the LLM
        
        Parameters:
        response_text (str): The response text to parse
        
        Returns:
        dict: The parsed JSON data or None if parsing fails
        """
        if not response_text:
            return None
        
        try:
            # First try to extract JSON between triple backticks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.MULTILINE)
            if json_match:
                json_text = json_match.group(1).strip()
                return json.loads(json_text)
            
            # If that fails, try to parse the whole response as JSON
            return json.loads(response_text.strip())
        except Exception as e:
            logger.error(f"Error parsing JSON response in {self.agent_name}: {e}")
            logger.error(f"Response text: {response_text}")
            return None
    
    def _extract_code(self, response_text):
        """
        Extract Python code from the LLM response
        
        Parameters:
        response_text (str): The response text to parse
        
        Returns:
        str: The extracted code or None if extraction fails
        """
        if not response_text:
            return None
        
        try:
            # Try to extract code between triple backticks
            code_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", response_text, re.MULTILINE)
            if code_match:
                return code_match.group(1).strip()
            
            # If that fails, try to extract just a function definition
            function_match = re.search(r"def\s+\w+\s*\([\s\S]*?return\s+[\s\S]*?(?:\n\n|$)", response_text, re.MULTILINE)
            if function_match:
                return function_match.group(0).strip()
            
            # If all else fails, return the whole response
            return response_text.strip()
        except Exception as e:
            logger.error(f"Error extracting code in {self.agent_name}: {e}")
            return None
    
    def _validate_response(self, response_data, required_keys=None):
        """
        Validate that a response contains all required keys
        
        Parameters:
        response_data (dict): The response data to validate
        required_keys (list, optional): List of required keys, defaults to []
        
        Returns:
        bool: True if valid, False otherwise
        """
        if response_data is None:
            return False
        
        if required_keys is None:
            return True
        
        return all(key in response_data for key in required_keys)
    
    def process(self, *args, **kwargs):
        """
        Process the request - to be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement process()")
