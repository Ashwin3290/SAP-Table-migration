# managers/prompt_manager.py
import os
import json
import logging
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

from DMtool.enums.database_types import DatabaseType, PromptType

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages database-specific prompts and templates"""
    
    def __init__(self, db_type: DatabaseType, prompts_base_path: str = "prompts"):
        """Initialize prompt manager for specific database type"""
        self.db_type = db_type
        self.prompts_base_path = Path(prompts_base_path)
        self.prompts: Dict[str, str] = {}
        self.syntax_rules: Dict[str, Any] = {}
        
        # Load prompts and syntax rules
        self._load_prompts()
        
        logger.info(f"Prompt manager initialized for {db_type.value}")
        
    def _load_prompts(self) -> None:
        """Load prompts from YAML/JSON files or use fallback"""
        try:
            if True:
                # Try YAML first
                base_prompts = self._load_yaml_file(self.prompts_base_path / "base" / "prompts.yaml")
                db_prompts = self._load_yaml_file(self.prompts_base_path / self.db_type.value / "prompts.yaml")
                self.syntax_rules = self._load_yaml_file(self.prompts_base_path / self.db_type.value / "syntax_rules.yaml")
            else:
                # Fallback to JSON
                base_prompts = self._load_json_file(self.prompts_base_path / "base" / "prompts.json")
                db_prompts = self._load_json_file(self.prompts_base_path / self.db_type.value / "prompts.json")
                self.syntax_rules = self._load_json_file(self.prompts_base_path / self.db_type.value / "syntax_rules.json")
            
            # Merge prompts (db-specific overrides base)
            self.prompts = {**base_prompts, **db_prompts}
            
            if not self.prompts:
                raise Exception("No prompts loaded from files")
                
            logger.debug(f"Loaded {len(self.prompts)} prompts for {self.db_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            # Use fallback prompts
            self._load_fallback_prompts()
            
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file and return contents"""
            
        if not file_path.exists():
            logger.warning(f"Prompt file not found: {file_path}")
            return {}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file) or {}
        except Exception as e:
            logger.error(f"Error loading YAML file {file_path}: {e}")
            return {}
            
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file and return contents"""
        if not file_path.exists():
            logger.warning(f"Prompt file not found: {file_path}")
            return {}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file) or {}
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return {}
            
    def _load_fallback_prompts(self) -> None:
        """Load hardcoded fallback prompts"""
        self.prompts = self._get_sqlite_prompts() if self.db_type == DatabaseType.SQLITE else self._get_generic_prompts()
        self.syntax_rules = self._get_sqlite_syntax_rules() if self.db_type == DatabaseType.SQLITE else self._get_generic_syntax_rules()
        logger.warning("Using hardcoded fallback prompts")
        
    def _get_sqlite_prompts(self) -> Dict[str, str]:
        """Get SQLite-specific prompts"""
        return {
            PromptType.QUERY_EXTRACTION.value: """
You are analyzing a data transformation query for SQLite database.

QUERY: "{query}"
CONTEXT: {context}
AVAILABLE TABLES: {available_tables}
SCHEMA INFO: {schema_info}

Extract and return JSON with:
{{
  "query_type": "SIMPLE_TRANSFORMATION|JOIN_OPERATION|CROSS_SEGMENT|VALIDATION_OPERATION|AGGREGATION_OPERATION",
  "confidence": 0.95,
  "source_tables": ["table1", "table2"],
  "target_tables": ["target_table"],
  "fields_mapping": {{
    "source_fields": ["field1", "field2"],
    "target_fields": ["target_field1", "target_field2"],
    "filtering_fields": ["filter_field1"],
    "insertion_fields": ["insert_field1"]
  }},
  "conditions": {{
    "field1": "value1",
    "field2": ["value2", "value3"]
  }},
  "joins": [
    {{
      "left_table": "table1",
      "right_table": "table2", 
      "join_field": "common_field",
      "join_type": "inner"
    }}
  ],
  "semantic_understanding": {{
    "business_intent": "description",
    "transformation_goal": "goal",
    "data_flow": "source -> target description"
  }}
}}

SQLite Specific Rules:
- No RIGHT JOIN or FULL JOIN support
- Use IFNULL instead of ISNULL
- Use substr() for string operations
- Date functions: date(), datetime(), strftime()
- Limit syntax: LIMIT n OFFSET m
""",
            
            PromptType.CODE_GENERATION.value: """
Generate SQLite execution plan and queries.

EXTRACTED DATA: {extracted_data}
DB SYNTAX RULES: {db_syntax}
AVAILABLE TEMPLATES: {templates}

Use these templates as reference for generating appropriate SQL queries. Each template has:
- id: Template identifier
- prompt: Pattern description  
- query: SQL template with parameters
- plan: Step-by-step execution plan

Select the most appropriate template based on the extracted data and customize it for the specific requirements.

Return JSON with:
{{
  "execution_plan": {{
    "steps": ["step1", "step2", "step3"],
    "dependencies": ["dep1", "dep2"],
    "rollback_strategy": "strategy"
  }},
  "sql_queries": [
    "INSERT INTO target_table (col1, col2) SELECT col1, col2 FROM source_table WHERE condition;"
  ],
  "validation_checks": [
    "SELECT COUNT(*) FROM target_table;"
  ],
  "expected_outcome": {{
    "rows_affected": "estimated_number",
    "new_columns": ["col1", "col2"],
    "data_quality_impact": "description"
  }}
}}

CRITICAL SQLite Rules:
- No RIGHT JOIN or FULL JOIN
- Use IFNULL not ISNULL  
- String functions: substr(), replace(), trim()
- UPDATE with JOIN: UPDATE table SET col = (SELECT col FROM other WHERE condition)
""",

            PromptType.QUERY_VALIDATION.value: """
Validate the following SQLite query for syntax and logical errors:

QUERY: {query}
CONTEXT: {context}

Check for SQLite compatibility:
1. No RIGHT JOIN or FULL JOIN
2. Use IFNULL instead of ISNULL
3. Use substr() instead of SUBSTRING()
4. Proper table and column references

Return JSON with:
{{
  "is_valid": true,
  "syntax_errors": [],
  "warnings": [],
  "suggestions": []
}}
""",

            PromptType.ERROR_FIXING.value: """
Fix the following SQLite query that failed:

ORIGINAL QUERY: {query}
ERROR MESSAGE: {error}
CONTEXT: {context}

Common SQLite fixes:
- Replace RIGHT JOIN with LEFT JOIN (swap table order)
- Replace ISNULL with IFNULL
- Use substr() instead of SUBSTRING()

Return JSON with:
{{
  "fixed_query": "corrected SQL query",
  "changes_made": ["change1", "change2"],
  "explanation": "explanation of fixes"
}}
"""
        }
        
    def _get_generic_prompts(self) -> Dict[str, str]:
        """Get generic database prompts"""
        return {
            PromptType.QUERY_EXTRACTION.value: """
Analyze this data transformation query for {database_type} database.

QUERY: "{query}"
CONTEXT: {context}

Extract and return JSON with query analysis.
""",
            PromptType.CODE_GENERATION.value: """
Generate {database_type} SQL queries for the extracted data.

EXTRACTED DATA: {extracted_data}
AVAILABLE TEMPLATES: {templates}

Use the provided templates as reference to generate appropriate SQL.
""",
            PromptType.QUERY_VALIDATION.value: """
Validate this {database_type} query for syntax errors.
""",
            PromptType.ERROR_FIXING.value: """
Fix this {database_type} query that failed with error: {error}
"""
        }
        
    def _get_sqlite_syntax_rules(self) -> Dict[str, Any]:
        """Get SQLite syntax rules"""
        return {
            'supports_right_join': False,
            'supports_full_join': False,
            'supports_window_functions': True,
            'supports_cte': True,
            'null_function': 'IFNULL',
            'string_functions': ['substr', 'replace', 'trim', 'upper', 'lower'],
            'date_functions': ['date', 'datetime', 'strftime', 'julianday'],
            'limit_syntax': 'LIMIT',
            'quote_identifier': '[]'
        }
        
    def _get_generic_syntax_rules(self) -> Dict[str, Any]:
        """Get generic syntax rules"""
        return {
            'supports_right_join': True,
            'supports_full_join': True,
            'null_function': 'COALESCE',
            'string_functions': ['SUBSTRING', 'REPLACE', 'TRIM'],
            'date_functions': ['NOW', 'DATE'],
            'limit_syntax': 'LIMIT'
        }
        
    def get_prompt(self, prompt_type: PromptType, **kwargs) -> str:
        """Get formatted prompt for specific type and database"""
        try:
            template = self.prompts.get(prompt_type.value, "")
            
            if not template:
                logger.warning(f"No prompt found for type: {prompt_type.value}")
                return ""
                
            kwargs['database_type'] = self.db_type.value
            kwargs['syntax_rules'] = self._format_syntax_rules()
            
            return template.format(**kwargs)
            
        except Exception as e:
            logger.error(f"Error formatting prompt {prompt_type.value}: {e}")
            return ""
            
    def _format_syntax_rules(self) -> str:
        """Format syntax rules for inclusion in prompts"""
        if not self.syntax_rules:
            return "No specific syntax rules"
            
        rules = []
        for key, value in self.syntax_rules.items():
            if isinstance(value, bool):
                status = "supported" if value else "not supported"
                rules.append(f"- {key.replace('_', ' ').title()}: {status}")
            elif isinstance(value, list):
                rules.append(f"- {key.replace('_', ' ').title()}: {', '.join(value)}")
            else:
                rules.append(f"- {key.replace('_', ' ').title()}: {value}")
                
        return "\n".join(rules)
        
    def get_syntax_rules(self) -> Dict[str, Any]:
        """Get syntax rules for the database type"""
        return self.syntax_rules.copy()
        
    def update_prompt(self, prompt_type: PromptType, template: str) -> None:
        """Update a specific prompt template"""
        self.prompts[prompt_type.value] = template
        logger.info(f"Updated prompt for {prompt_type.value}")
        
    def list_available_prompts(self) -> List[str]:
        """List all available prompt types"""
        return list(self.prompts.keys())
        
    def validate_prompt_variables(self, prompt_type: PromptType, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that all required variables are provided for a prompt"""
        template = self.prompts.get(prompt_type.value, "")
        
        if not template:
            return {'valid': False, 'error': 'Prompt template not found'}
            
        try:
            # Try to format with provided variables to check for missing ones
            template.format(**variables)
            return {'valid': True}
        except KeyError as e:
            missing_var = str(e).strip("'\"")
            return {
                'valid': False, 
                'error': f'Missing required variable: {missing_var}',
                'missing_variables': [missing_var]
            }
        except Exception as e:
            return {'valid': False, 'error': f'Template formatting error: {str(e)}'}
            
    def get_example_variables(self, prompt_type: PromptType) -> Dict[str, str]:
        """Get example variables for a prompt type"""
        examples = {
            PromptType.QUERY_EXTRACTION.value: {
                'query': 'Bring material number from MARA table where material type = ROH',
                'context': '{"session_id": "123", "tables": ["MARA", "MAKT"]}',
                'available_tables': '["MARA", "MAKT", "MARC"]',
                'schema_info': '{"MARA": {"columns": ["MATNR", "MTART"]}}'
            },
            PromptType.CODE_GENERATION.value: {
                'extracted_data': '{"query_type": "SIMPLE_TRANSFORMATION", "source_tables": ["MARA"]}',
                'db_syntax': '{"supports_right_join": false}',
                'templates': '[{"id": "simple_filter_transformation", "prompt": "Bring field from table", "query": "INSERT INTO..."}]'
            },
            PromptType.QUERY_VALIDATION.value: {
                'query': 'SELECT * FROM MARA WHERE MTART = "ROH"',
                'context': '{"table_schemas": {"MARA": ["MATNR", "MTART"]}}'
            },
            PromptType.ERROR_FIXING.value: {
                'query': 'SELECT * FROM MARA RIGHT JOIN MAKT',
                'error': 'RIGHT JOIN not supported in SQLite',
                'context': '{"available_functions": ["IFNULL", "SUBSTR"]}'
            }
        }
        
        return examples.get(prompt_type.value, {})
        
    def reload_prompts(self) -> bool:
        """Reload prompts from files"""
        try:
            self._load_prompts()
            logger.info("Prompts reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload prompts: {e}")
            return False
            
    def export_prompts_to_json(self, output_path: str) -> bool:
        """Export current prompts to JSON file (YAML alternative)"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                'database_type': self.db_type.value,
                'prompts': self.prompts,
                'syntax_rules': self.syntax_rules
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Prompts exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export prompts: {e}")
            return False