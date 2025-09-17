
import logging
import os
from DMtool.llm_query_gen.llm_config import LLMManager
from DMtool.errors import APIError
import json


logger = logging.getLogger(__name__)

class QueryTemplateRepository:
    """Repository of query templates for common transformation patterns"""
    
    def __init__(self, template_file="DMtool/query_templates.json"):
        """
        Initialize the template repository
        
        Parameters:
        template_file (str): Path to the JSON file containing templates
        """
        self.template_file = template_file
        self.templates = self._load_templates()
        api_key = os.environ.get("API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise APIError("Gemini API key not configured")
        self.llm = LLMManager(
            provider="google",
            model="gemini/gemini-2.5-flash",
            api_key=api_key
        )
        
    def _load_templates(self):
        """
        Load templates from the JSON file
        
        Returns:
        list: List of template dictionaries
        """
        try:
            if os.path.exists(self.template_file):
                with open(self.template_file, 'r') as f:
                    templates = json.load(f)
                return templates
            else:
                logger.warning(f"Template file {self.template_file} not found.")
                return []
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            return []
    
    def find_matching_template(self, query):
        """
        Find the best matching template for a given query using LLM
        
        Parameters:
        query (str): The natural language query
        
        Returns:
        dict: The best matching template or None if no good match
        """
        try:

            if not self.templates:
                logger.warning("No templates available")
                return None
            

            template_options = []
            for i, template in enumerate(self.templates):
                template_options.append(f"{i+1}. ID: {template['id']}\n   Pattern: {template['prompt']}")
            

            llm_prompt = f"""You are an expert at matching user queries to data transformation templates.

    USER QUERY: "{query}"

    AVAILABLE TEMPLATES:
    {chr(10).join(template_options)}

    INSTRUCTIONS:
    Analyze the user query and determine which template pattern best matches the intent and structure.
    Properly understand what the template is performing for and how it relates to the query.

    Consider:
    - Query operations (bring, add, delete, update, check, join, etc.)
    - Data sources (tables, fields, segments)
    - Conditional logic (IF/ELSE, CASE statements)
    - Filtering conditions (WHERE clauses)
    - Transformations (date formatting, string operations, etc.)

    Respond with ONLY the template ID (nothing else).


    Examples:
    - "Bring Material Number from MARA where Material Type = ROH" → simple_filter_transformation
    - "If Plant is 1000 then 'Domestic' else 'International'" → conditional_value_assignment  
    - "Add new column for current date" → get_current_date
    - "Join data from Basic segment with Sales segment" → join_segment_data

    Template ID:"""

            try:
                
                response = self.llm.generate(llm_prompt, temperature=0.05, max_tokens=50)
                
                if response :

                    template_id = response.strip().strip('"').strip("'").lower()
                    

                    best_match = None
                    for template in self.templates:
                        if template['id'].lower() == template_id:
                            best_match = template
                            break
                    
                    if best_match:
                        return best_match
                    else:
                        logger.warning(f"Template ID '{template_id}' not found in available templates")

                        for template in self.templates:
                            if template_id in template['id'].lower() or template['id'].lower() in template_id:
                                return template
                        
                        return {}
                else:
                    logger.warning("Invalid response from LLM")
                    return {}
                    
            except Exception as llm_error:
                logger.error(f"Error calling LLM for template matching: {llm_error}")
                return {}
                
        except Exception as e:
            logger.error(f"Error finding matching template: {e}")
            return None    
