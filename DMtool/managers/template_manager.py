# managers/template_manager.py
import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from DMtool.enums.database_types import DatabaseType, TemplateType

logger = logging.getLogger(__name__)

class TemplateManager:
    """Manages database-specific query templates using JSON format"""
    
    def __init__(self, db_type: DatabaseType, templates_base_path: str = "templates"):
        """Initialize template manager for specific database type"""
        self.db_type = db_type
        self.templates_base_path = Path(templates_base_path)
        self.templates: List[Dict[str, Any]] = []  # Keep as list to match your format
        self.templates_dict: Dict[str, Dict[str, Any]] = {}  # For quick lookup
        self.query_patterns: Dict[str, Any] = {}
        
        # Load templates and patterns
        self._load_templates()
        
        logger.info(f"Template manager initialized for {db_type.value} with {len(self.templates)} templates")
        
    def _load_templates(self) -> None:
        """Load templates from JSON files in your format"""
        try:
            templates_loaded = False
            
            # Try multiple locations for templates
            possible_paths = [
                # Your current location
                Path("DMtool/query_templates.json"),
                Path("query_templates.json"),
                # Standard structure
                self.templates_base_path / self.db_type.value / "templates.json",
                self.templates_base_path / "base" / "templates.json",
                # Fallback locations
                Path("templates") / f"{self.db_type.value}_templates.json",
                Path("templates.json")
            ]
            
            for path in possible_paths:
                if path.exists():
                    logger.info(f"Loading templates from: {path}")
                    self._load_json_templates(path)
                    templates_loaded = True
                    break
                    
            if not templates_loaded:
                logger.warning("No template files found, using fallback templates")
                self._load_fallback_templates()
                
            # Load base templates and merge
            base_templates = self._load_base_templates()
            if base_templates:
                # Add base templates that don't exist in db-specific templates
                existing_ids = {t['id'] for t in self.templates}
                for base_template in base_templates:
                    if base_template['id'] not in existing_ids:
                        self.templates.append(base_template)
                        
            # Create dictionary for quick lookup
            self.templates_dict = {t['id']: t for t in self.templates}
            
            logger.info(f"Loaded {len(self.templates)} total templates")
            
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
            self._load_fallback_templates()
            
    def _load_json_templates(self, file_path: Path) -> None:
        """Load templates from JSON file in your exact format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                templates_data = json.load(file)
                
            if isinstance(templates_data, list):
                # Your format - list of template objects
                self.templates = templates_data
                logger.info(f"Loaded {len(templates_data)} templates from {file_path}")
            else:
                logger.error(f"Invalid template format in {file_path} - expected list")
                raise ValueError("Template file must contain a list of templates")
                
        except Exception as e:
            logger.error(f"Error loading JSON templates from {file_path}: {e}")
            raise
            
    def _load_base_templates(self) -> List[Dict[str, Any]]:
        """Load base templates that work across all databases"""
        base_path = self.templates_base_path / "base" / "templates.json"
        
        if base_path.exists():
            try:
                with open(base_path, 'r', encoding='utf-8') as file:
                    base_templates = json.load(file)
                    logger.info(f"Loaded {len(base_templates)} base templates")
                    return base_templates
            except Exception as e:
                logger.warning(f"Error loading base templates: {e}")
                
        return self._get_default_base_templates()
        
    def _get_default_base_templates(self) -> List[Dict[str, Any]]:
        """Get default base templates that work across databases"""
        return [
            {
                "id": "simple_select",
                "prompt": "Select {fields} from {table}",
                "query": "SELECT {fields} FROM {table}",
                "plan": [
                    "1. Identify the table {table}",
                    "2. Identify the fields {fields} to select",
                    "3. Execute SELECT query"
                ]
            },
            {
                "id": "simple_insert",
                "prompt": "Insert data into {target_table} from {source_table}",
                "query": "INSERT INTO {target_table} ({target_fields}) SELECT {source_fields} FROM {source_table}",
                "plan": [
                    "1. Identify source table {source_table}",
                    "2. Identify target table {target_table}",
                    "3. Map source fields to target fields",
                    "4. Execute INSERT operation"
                ]
            },
            {
                "id": "simple_update",
                "prompt": "Update {field} in {table} with {value}",
                "query": "UPDATE {table} SET {field} = '{value}'",
                "plan": [
                    "1. Identify the table {table}",
                    "2. Identify the field {field} to update",
                    "3. Set the new value {value}",
                    "4. Execute UPDATE operation"
                ]
            }
        ]
        
    def _load_fallback_templates(self) -> None:
        """Load hardcoded fallback templates in your format"""
        self.templates = [
            {
                "id": "simple_transformation",
                "prompt": "Bring {field} from {table}",
                "query": "INSERT INTO {target_table} ({target_field}) SELECT {field} FROM {table}",
                "plan": [
                    "1. Identify source table {table}",
                    "2. Identify field {field} to extract",
                    "3. Insert into target table {target_table}"
                ]
            },
            {
                "id": "filtered_transformation",
                "prompt": "Bring {field} from {table} where {condition}",
                "query": "INSERT INTO {target_table} ({target_field}) SELECT {field} FROM {table} WHERE {condition}",
                "plan": [
                    "1. Identify source table {table}",
                    "2. Identify field {field} to extract",
                    "3. Apply filter condition {condition}",
                    "4. Insert filtered data into target table"
                ]
            }
        ]
        
        self.templates_dict = {t['id']: t for t in self.templates}
        logger.warning("Using fallback templates")
        
    def get_template_by_id(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get template by ID"""
        return self.templates_dict.get(template_id)
        
    def find_matching_template(self, query: str, extracted_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Find the best matching template for a query using your template format"""
        try:
            query_lower = query.lower()
            best_match = None
            best_score = 0
            
            # Check each template's prompt pattern
            for template in self.templates:
                prompt_pattern = template.get('prompt', '').lower()
                
                if prompt_pattern:
                    score = self._calculate_pattern_similarity(query_lower, prompt_pattern)
                    
                    if score > best_score:
                        best_score = score
                        best_match = template
                        
            # If no good match, return a default template
            if best_score < 0.3:
                # Look for simple transformation templates
                for template in self.templates:
                    if 'simple' in template.get('id', '').lower():
                        return template
                        
                # Fallback to first template
                return self.templates[0] if self.templates else None
                
            return best_match
            
        except Exception as e:
            logger.error(f"Error finding matching template: {e}")
            return self.templates[0] if self.templates else None
            
    def _calculate_pattern_similarity(self, query: str, pattern: str) -> float:
        """Calculate similarity between query and template prompt pattern"""
        # Remove template variables for comparison
        import re
        clean_pattern = re.sub(r'\{[^}]+\}', '', pattern)
        
        # Simple keyword matching
        query_words = set(query.split())
        pattern_words = set(clean_pattern.split())
        
        if not pattern_words:
            return 0.0
            
        common_words = query_words.intersection(pattern_words)
        similarity = len(common_words) / len(pattern_words)
        
        # Boost score for exact phrase matches
        for word in pattern_words:
            if word in query and len(word) > 3:
                similarity += 0.1
                
        return min(similarity, 1.0)
        
    def format_template_query(self, template: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Format template query with parameters using your format"""
        try:
            query_template = template.get('query', '')
            if not query_template:
                logger.warning(f"No query template found in template: {template.get('id')}")
                return ""
                
            # Extract required parameters from template
            import re
            required_params = re.findall(r'\{([^}]+)\}', query_template)
            
            # Handle missing parameters gracefully
            safe_params = params.copy()
            for param in required_params:
                if param not in safe_params:
                    safe_params[param] = f"MISSING_{param.upper()}"
                    logger.warning(f"Missing required parameter: {param}")
                    
            return query_template.format(**safe_params)
            
        except Exception as e:
            logger.error(f"Error formatting template query: {e}")
            return ""
            
    def get_template_plan(self, template: Dict[str, Any]) -> List[str]:
        """Get execution plan steps from template"""
        return template.get('plan', [])
        
    def get_relevant_templates(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant templates based on extracted data"""
        relevant = []
        
        # Primary template
        main_template = self.find_matching_template("", extracted_data)
        if main_template:
            relevant.append(main_template)
            
        # Add additional templates based on query characteristics
        query_type = extracted_data.get('query_type', '').lower()
        
        for template in self.templates:
            template_id = template.get('id', '').lower()
            
            # Match by keywords
            if any(keyword in template_id for keyword in [query_type, 'join', 'conditional', 'date']):
                if template not in relevant:
                    relevant.append(template)
                    
        return relevant
        
    def list_available_templates(self) -> List[Dict[str, str]]:
        """List all available templates"""
        templates_list = []
        for template in self.templates:
            templates_list.append({
                'id': template.get('id', 'unknown'),
                'prompt': template.get('prompt', ''),
                'description': template.get('prompt', '')[:100] + '...' if len(template.get('prompt', '')) > 100 else template.get('prompt', '')
            })
        return templates_list
        
    def add_template(self, template: Dict[str, Any]) -> bool:
        """Add a new template to the collection"""
        try:
            if 'id' not in template:
                logger.error("Template must have an 'id' field")
                return False
                
            # Check if template already exists
            if template['id'] in self.templates_dict:
                logger.warning(f"Template {template['id']} already exists, updating...")
                
            self.templates.append(template)
            self.templates_dict[template['id']] = template
            
            logger.info(f"Added template: {template['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding template: {e}")
            return False
            
    def save_templates(self, output_path: str = None) -> bool:
        """Save current templates to JSON file"""
        try:
            if not output_path:
                output_path = f"templates/{self.db_type.value}/templates.json"
                
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.templates, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Templates saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")
            return False
            
    def validate_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template structure"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        # Required fields
        required_fields = ['id', 'prompt', 'query', 'plan']
        for field in required_fields:
            if field not in template:
                result['valid'] = False
                result['errors'].append(f"Missing required field: {field}")
                
        # Validate query has parameters
        query = template.get('query', '')
        if query and '{' not in query:
            result['warnings'].append("Query template has no parameters")
            
        # Validate plan is a list
        plan = template.get('plan', [])
        if not isinstance(plan, list):
            result['valid'] = False
            result['errors'].append("Plan must be a list of strings")
            
        return result
        
    def get_template_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded templates"""
        return {
            'total_templates': len(self.templates),
            'database_type': self.db_type.value,
            'template_ids': [t.get('id') for t in self.templates],
            'templates_with_plans': len([t for t in self.templates if t.get('plan')]),
            'average_plan_steps': sum(len(t.get('plan', [])) for t in self.templates) / len(self.templates) if self.templates else 0
        }