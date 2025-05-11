"""
Segment Agent for TableLLM
This agent is responsible for identifying segments and their order of processing
"""
import json
import re
from utils.logging_utils import agent_logger as logger
from agents.base_agent import BaseAgent
import config

class SegmentAgent(BaseAgent):
    """
    Agent for identifying segments and their relationships
    """
    
    def __init__(self, client=None):
        """
        Initialize the segment agent
        
        Parameters:
        client (genai.Client, optional): LLM client, creates new one if None
        """
        super().__init__(client)
        
        # Default segment hierarchy
        self.default_segments = {
            "Material_Management": [
                "Material_Basic_Segment",
                "Material_Plant_Segment",
                "Material_Description_Segment"
            ],
            "Customer_Management": [
                "Customer",
                "Customer_Company",
                "Customer_Sales"
            ]
        }
    
    def process(self, query, table_info, intent_info=None):
        """
        Identify segments and their processing order
        
        Parameters:
        query (str): The natural language query
        table_info (dict): Table selection information
        intent_info (dict, optional): Intent information
        
        Returns:
        dict: {
            "primary_segment": Primary segment name
            "dependent_segments": List of dependent segment names
            "segment_operation_order": List of segments in processing order
            "confidence": Confidence score
        }
        """
        logger.info(f"Identifying segments for query: {query[:100]}...")
        
        # Try to extract segments from the query directly
        extracted_segments = self._extract_segments_from_query(query)
        
        # Get workspace from intent info
        workspace = None
        if intent_info and "workspace" in intent_info:
            workspace = intent_info["workspace"]
        
        # Create segment identification prompt
        prompt = f"""
You are an expert SAP data transformation analyst. Your task is to identify the primary and dependent segments for this transformation.

QUERY: {query}

TABLE INFORMATION:
{json.dumps(table_info, indent=2)}

INTENT INFORMATION:
{json.dumps(intent_info, indent=2) if intent_info else "Not available"}

SEGMENTS MENTIONED IN QUERY:
{json.dumps(extracted_segments, indent=2) if extracted_segments else "None detected"}

COMMON SAP SEGMENT HIERARCHIES:
- Material_Basic_Segment: Contains basic material information
- Material_Plant_Segment: Contains plant-specific material data
- Material_Description_Segment: Contains material descriptions
- Customer: Contains basic customer data
- Customer_Company: Contains company-specific customer data
- Customer_Sales: Contains sales-related customer data

Based on the query and tables, identify:

1. Primary Segment - The main segment this transformation operates on
2. Dependent Segments - Other segments that are affected or needed
3. Operation Order - The order in which segments should be processed
4. Confidence - Your confidence level in this selection (0.0-1.0)

Return ONLY a JSON object with these fields and no additional explanation:
{{
  "primary_segment": "PRIMARY_SEGMENT_NAME",
  "dependent_segments": ["DEPENDENT_SEGMENT_1", "DEPENDENT_SEGMENT_2", ...],
  "segment_operation_order": ["SEGMENT_1", "SEGMENT_2", ...],
  "confidence": CONFIDENCE_SCORE
}}
"""
        
        # Call the LLM
        response_text = self._call_llm(prompt)
        if not response_text:
            logger.warning("Failed to get response from LLM for segment identification")
            return self._get_default_segments(workspace, table_info)
        
        # Parse the response
        segment_data = self._parse_json_response(response_text)
        
        # Validate the response
        required_keys = ["primary_segment", "dependent_segments", "segment_operation_order", "confidence"]
        if not self._validate_response(segment_data, required_keys):
            logger.warning(f"Invalid segment identification response: {segment_data}")
            return self._get_default_segments(workspace, table_info)
        
        # Log the result
        logger.info(f"Identified primary segment: {segment_data['primary_segment']}")
        dependent_segments = ", ".join(segment_data["dependent_segments"])
        logger.info(f"Dependent segments: {dependent_segments}")
        
        return segment_data
    
    def _extract_segments_from_query(self, query):
        """
        Extract segment names directly from the query
        
        Parameters:
        query (str): The natural language query
        
        Returns:
        list: List of segment names found in the query
        """
        # List of common segment names to look for
        common_segments = [
            "Material_Basic_Segment", "Material_Plant_Segment", "Material_Description_Segment",
            "Customer", "Customer_Company", "Customer_Sales"
        ]
        
        # Also look for variations
        segment_patterns = [
            r"[Mm]aterial[\s\-_]+[Bb]asic",
            r"[Mm]aterial[\s\-_]+[Pp]lant",
            r"[Mm]aterial[\s\-_]+[Dd]escription",
            r"[Cc]ustomer[\s\-_]+[Bb]asic",
            r"[Cc]ustomer[\s\-_]+[Cc]ompany",
            r"[Cc]ustomer[\s\-_]+[Ss]ales"
        ]
        
        found_segments = []
        
        # Check for exact segment names
        for segment in common_segments:
            if segment.lower() in query.lower():
                found_segments.append(segment)
        
        # Check for variations using regex
        for pattern in segment_patterns:
            if re.search(pattern, query):
                # Convert to standard format
                match = re.search(pattern, query).group(0)
                # Standardize the format
                standardized = match.replace(" ", "_").replace("-", "_").title()
                if standardized not in found_segments:
                    found_segments.append(standardized)
        
        return found_segments
    
    def _get_default_segments(self, workspace, table_info):
        """
        Get default segments when identification fails
        
        Parameters:
        workspace (str): Workspace name
        table_info (dict): Table selection information
        
        Returns:
        dict: Default segment information
        """
        # Try to determine default segments based on workspace
        default_primary = None
        default_dependents = []
        
        if workspace and workspace in self.default_segments:
            # Use default hierarchy for this workspace
            segments = self.default_segments[workspace]
            if segments:
                default_primary = segments[0]
                default_dependents = segments[1:] if len(segments) > 1 else []
        
        # If no defaults from workspace, try to infer from tables
        if not default_primary and table_info and "target_table" in table_info:
            target_table = table_info["target_table"]
            
            # Common target table to segment mappings
            if target_table.startswith("MARA") or "MTRL" in target_table:
                default_primary = "Material_Basic_Segment"
                default_dependents = ["Material_Plant_Segment", "Material_Description_Segment"]
            elif target_table.startswith("MARC") or "PLANT" in target_table:
                default_primary = "Material_Plant_Segment"
                default_dependents = ["Material_Basic_Segment"]
            elif target_table.startswith("MAKT") or "DESC" in target_table:
                default_primary = "Material_Description_Segment"
                default_dependents = ["Material_Basic_Segment"]
            elif target_table.startswith("KNA1") or "CUST" in target_table:
                default_primary = "Customer"
                default_dependents = ["Customer_Company", "Customer_Sales"]
        
        # If still no defaults, use generic ones
        if not default_primary:
            default_primary = "Material_Basic_Segment"
            default_dependents = []
        
        # Create operation order (primary first, then dependents)
        operation_order = [default_primary] + default_dependents
        
        return {
            "primary_segment": default_primary,
            "dependent_segments": default_dependents,
            "segment_operation_order": operation_order,
            "confidence": 0.5
        }
