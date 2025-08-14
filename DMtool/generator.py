# generator.py
import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from google import genai
from google.genai import types

from DMtool.enums.database_types import DatabaseType, PromptType, CodeGenerationResult
from DMtool.managers.prompt_manager import PromptManager
from DMtool.managers.template_manager import TemplateManager
from DMtool.database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class CodeGeneratorExecutor:
    """Generates SQL code and execution plans"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.prompt_manager = PromptManager(db_manager.db_type)
        self.template_manager = TemplateManager(db_manager.db_type)
        
        # Initialize LLM client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.llm_client = genai.Client(api_key=api_key)
        
    def generate_and_plan(self, extracted_data: Dict[str, Any]) -> CodeGenerationResult:
        """Generate execution plan and SQL code in single LLM call"""
        try:
            # Get relevant templates
            relevant_templates = self.template_manager.get_relevant_templates(extracted_data)
            
            # Get database syntax rules
            syntax_rules = self.db_manager.get_syntax_rules()
            
            # Build generation prompt
            generation_prompt = self.prompt_manager.get_prompt(
                PromptType.CODE_GENERATION,
                extracted_data=json.dumps(extracted_data, indent=2),
                db_syntax=json.dumps(syntax_rules, indent=2),
                templates=json.dumps([template for template in relevant_templates], indent=2)
            )
            
            # Execute LLM call
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=generation_prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            
            if not response or not hasattr(response, "text"):
                raise ValueError("Invalid response from LLM")
                
            # Parse response
            parsed_response = self._parse_generation_response(response.text)
            
            # Create structured result
            return self._create_generation_result(parsed_response, extracted_data)
            
        except Exception as e:
            logger.error(f"Error in generate_and_plan: {e}")
            # Return fallback result
            return self._create_fallback_result(extracted_data)
            
    def _parse_generation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM generation response"""
        try:
            import re
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to parse entire response as JSON
                json_str = response_text.strip()
                
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing generation response JSON: {e}")
            return {
                'execution_plan': {
                    'steps': ['Parse error occurred'],
                    'dependencies': [],
                    'rollback_strategy': 'Full rollback on any error'
                },
                'sql_queries': [],
                'validation_checks': [],
                'expected_outcome': {
                    'rows_affected': 'unknown',
                    'new_columns': [],
                    'data_quality_impact': 'Unable to determine due to parse error'
                }
            }
            
    def _create_generation_result(self, parsed_response: Dict[str, Any], 
                                 extracted_data: Dict[str, Any]) -> CodeGenerationResult:
        """Create structured generation result"""
        return CodeGenerationResult(
            execution_plan=parsed_response.get('execution_plan', {}),
            sql_queries=parsed_response.get('sql_queries', []),
            validation_checks=parsed_response.get('validation_checks', []),
            expected_outcome=parsed_response.get('expected_outcome', {})
        )
        
    def _create_fallback_result(self, extracted_data: Dict[str, Any]) -> CodeGenerationResult:
        """Create fallback result when generation fails"""
        # Generate basic query using templates
        fallback_queries = self._generate_fallback_queries(extracted_data)
        
        return CodeGenerationResult(
            execution_plan={
                'steps': [
                    'Execute fallback transformation',
                    'Validate results',
                    'Commit if successful'
                ],
                'dependencies': [],
                'rollback_strategy': 'Full rollback on error'
            },
            sql_queries=fallback_queries,
            validation_checks=[
                f"SELECT COUNT(*) FROM {table}" 
                for table in extracted_data.get('target_tables', [])
            ],
            expected_outcome={
                'rows_affected': 'unknown',
                'new_columns': extracted_data.get('fields_mapping', {}).get('target_fields', []),
                'data_quality_impact': 'Fallback generation - limited analysis'
            }
        )
        
    def _generate_fallback_queries(self, extracted_data: Dict[str, Any]) -> List[str]:
        """Generate fallback SQL queries using templates"""
        queries = []
        
        try:
            query_type = extracted_data.get('query_type', 'SIMPLE_TRANSFORMATION')
            
            # Get appropriate template
            from enums.database_types import TemplateType
            template_type = TemplateType(query_type)
            template = self.template_manager.get_template(template_type)
            
            if template:
                # Prepare template parameters
                params = self._prepare_template_params(extracted_data)
                
                # Format query using template
                query = self.template_manager.format_template_query(template, params)
                if query:
                    queries.append(query)
                    
        except Exception as e:
            logger.error(f"Error generating fallback queries: {e}")
            # Basic fallback
            source_tables = extracted_data.get('source_tables', [])
            target_tables = extracted_data.get('target_tables', [])
            
            if source_tables and target_tables:
                queries.append(f"-- Fallback query for {source_tables[0]} -> {target_tables[0]}")
                queries.append(f"SELECT * FROM {source_tables[0]} LIMIT 1")
                
        return queries
        
    def _prepare_template_params(self, extracted_data: Dict[str, Any]) -> Dict[str, str]:
        """Prepare parameters for template formatting"""
        fields_mapping = extracted_data.get('fields_mapping', {})
        
        return {
            'source_table': ', '.join(extracted_data.get('source_tables', [])),
            'target_table': ', '.join(extracted_data.get('target_tables', [])),
            'source_fields': ', '.join(fields_mapping.get('source_fields', [])),
            'target_fields': ', '.join(fields_mapping.get('target_fields', [])),
            'conditions': self._format_conditions(extracted_data.get('conditions', {})),
            'join_clauses': self._format_joins(extracted_data.get('joins', []))
        }
        
    def _format_conditions(self, conditions: Dict[str, Any]) -> str:
        """Format conditions for SQL WHERE clause"""
        if not conditions:
            return "1=1"
            
        condition_parts = []
        for field, value in conditions.items():
            if isinstance(value, list):
                values_str = ', '.join(f"'{v}'" for v in value)
                condition_parts.append(f"{field} IN ({values_str})")
            else:
                condition_parts.append(f"{field} = '{value}'")
                
        return ' AND '.join(condition_parts)
        
    def _format_joins(self, joins: List[Dict[str, str]]) -> str:
        """Format joins for SQL JOIN clauses"""
        if not joins:
            return ""
            
        join_parts = []
        for join in joins:
            join_type = join.get('join_type', 'INNER').upper()
            left_table = join.get('left_table', '')
            right_table = join.get('right_table', '')
            join_field = join.get('join_field', '')
            
            if left_table and right_table and join_field:
                join_parts.append(
                    f"{join_type} JOIN {right_table} ON {left_table}.{join_field} = {right_table}.{join_field}"
                )
                
        return ' '.join(join_parts)

class QueryValidator:
    """Validates SQL queries for syntax and logical correctness"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.prompt_manager = PromptManager(db_manager.db_type)
        
        # Initialize LLM client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.llm_client = genai.Client(api_key=api_key)
        
    def validate_queries(self, queries: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate multiple SQL queries"""
        validation_results = {
            'overall_valid': True,
            'query_results': [],
            'warnings': [],
            'suggestions': []
        }
        
        for i, query in enumerate(queries):
            result = self.validate_single_query(query, context)
            result['query_index'] = i
            result['query'] = query
            
            validation_results['query_results'].append(result)
            
            if not result['is_valid']:
                validation_results['overall_valid'] = False
                
        return validation_results
        
    def validate_single_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single SQL query"""
        try:
            # Basic syntax validation
            syntax_check = self._validate_syntax(query)
            
            # Database-specific validation
            db_check = self._validate_database_compatibility(query)
            
            # Schema validation
            schema_check = self._validate_schema_references(query, context)
            
            # Combine results
            is_valid = (syntax_check['valid'] and 
                       db_check['valid'] and 
                       schema_check['valid'])
                       
            return {
                'is_valid': is_valid,
                'syntax_errors': syntax_check.get('errors', []),
                'compatibility_errors': db_check.get('errors', []),
                'schema_errors': schema_check.get('errors', []),
                'warnings': (syntax_check.get('warnings', []) + 
                           db_check.get('warnings', []) + 
                           schema_check.get('warnings', [])),
                'suggestions': (syntax_check.get('suggestions', []) + 
                              db_check.get('suggestions', []) + 
                              schema_check.get('suggestions', []))
            }
            
        except Exception as e:
            logger.error(f"Error validating query: {e}")
            return {
                'is_valid': False,
                'syntax_errors': [f"Validation error: {str(e)}"],
                'compatibility_errors': [],
                'schema_errors': [],
                'warnings': [],
                'suggestions': []
            }
            
    def _validate_syntax(self, query: str) -> Dict[str, Any]:
        """Validate basic SQL syntax"""
        result = {'valid': True, 'errors': [], 'warnings': [], 'suggestions': []}
        
        if not query or not query.strip():
            result['valid'] = False
            result['errors'].append("Empty query")
            return result
            
        # Basic SQL keyword validation
        query_upper = query.upper().strip()
        
        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'TRUNCATE', 'DELETE FROM']
        for keyword in dangerous_keywords:
            if keyword in query_upper and 'WHERE' not in query_upper:
                result['warnings'].append(f"Potentially dangerous operation without WHERE clause: {keyword}")
                
        # Check for common syntax issues
        if query_upper.count('(') != query_upper.count(')'):
            result['valid'] = False
            result['errors'].append("Mismatched parentheses")
            
        return result
        
    def _validate_database_compatibility(self, query: str) -> Dict[str, Any]:
        """Validate database-specific compatibility"""
        result = {'valid': True, 'errors': [], 'warnings': [], 'suggestions': []}
        
        dialect_info = self.db_manager.get_dialect_info()
        query_upper = query.upper()
        
        # Check for unsupported JOIN types
        if not dialect_info.get('supports_right_join', True) and 'RIGHT JOIN' in query_upper:
            result['valid'] = False
            result['errors'].append("RIGHT JOIN not supported in this database")
            result['suggestions'].append("Use LEFT JOIN with table order swapped")
            
        if not dialect_info.get('supports_full_join', True) and 'FULL JOIN' in query_upper:
            result['valid'] = False
            result['errors'].append("FULL JOIN not supported in this database")
            result['suggestions'].append("Use UNION of LEFT and RIGHT JOINs")
            
        # Check for database-specific functions
        if self.db_manager.db_type == DatabaseType.SQLITE:
            if 'ISNULL(' in query_upper:
                result['valid'] = False
                result['errors'].append("ISNULL function not supported in SQLite")
                result['suggestions'].append("Use IFNULL or COALESCE instead")
                
        return result
        
    def _validate_schema_references(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate table and column references against schema"""
        result = {'valid': True, 'errors': [], 'warnings': [], 'suggestions': []}
        
        try:
            # Extract table names from query (basic regex approach)
            import re
            
            # Look for table references in FROM and JOIN clauses
            table_pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            table_matches = re.findall(table_pattern, query, re.IGNORECASE)
            
            available_tables = self.db_manager.get_tables()
            
            for table in table_matches:
                if table not in available_tables:
                    result['valid'] = False
                    result['errors'].append(f"Table '{table}' does not exist")
                    
                    # Suggest similar table names
                    suggestions = self._find_similar_tables(table, available_tables)
                    if suggestions:
                        result['suggestions'].append(f"Did you mean: {', '.join(suggestions)}")
                        
            # Validate column references (simplified - would need proper SQL parsing for accuracy)
            schema_context = context.get('schema_knowledge', {}).get('schemas', {})
            
            for table in table_matches:
                if table in schema_context:
                    columns = [col['name'] for col in schema_context[table].get('columns', [])]
                    
                    # Look for column references (basic pattern)
                    column_pattern = rf'{table}\.([a-zA-Z_][a-zA-Z0-9_]*)'
                    column_matches = re.findall(column_pattern, query, re.IGNORECASE)
                    
                    for column in column_matches:
                        if column not in columns:
                            result['warnings'].append(f"Column '{table}.{column}' may not exist")
                            
        except Exception as e:
            logger.warning(f"Error in schema validation: {e}")
            result['warnings'].append("Could not fully validate schema references")
            
        return result
        
    def _find_similar_tables(self, target_table: str, available_tables: List[str]) -> List[str]:
        """Find similar table names for suggestions"""
        suggestions = []
        target_lower = target_table.lower()
        
        for table in available_tables:
            table_lower = table.lower()
            
            # Simple similarity check
            if target_lower in table_lower or table_lower in target_lower:
                suggestions.append(table)
            elif self._calculate_similarity(target_lower, table_lower) > 0.6:
                suggestions.append(table)
                
        return suggestions[:3]  # Return top 3 suggestions
        
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (simple character-based approach)"""
        if not str1 or not str2:
            return 0.0
            
        # Character-based similarity
        common_chars = set(str1) & set(str2)
        total_chars = set(str1) | set(str2)
        
        return len(common_chars) / len(total_chars) if total_chars else 0.0

class QueryFixer:
    """Automatically fixes SQL queries using LLM"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.prompt_manager = PromptManager(db_manager.db_type)
        
        # Initialize LLM client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.llm_client = genai.Client(api_key=api_key)
        
    def fix_query(self, query: str, error_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fix a failed SQL query using LLM"""
        try:
            # Build fixing prompt
            fixing_prompt = self.prompt_manager.get_prompt(
                PromptType.ERROR_FIXING,
                query=query,
                error=error_message,
                context=json.dumps(context, indent=2)
            )
            
            # Execute LLM call
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=fixing_prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            
            if not response or not hasattr(response, "text"):
                raise ValueError("Invalid response from LLM")
                
            # Parse response
            parsed_response = self._parse_fixing_response(response.text)
            
            return {
                'success': True,
                'fixed_query': parsed_response.get('fixed_query', query),
                'changes_made': parsed_response.get('changes_made', []),
                'explanation': parsed_response.get('explanation', 'No explanation provided')
            }
            
        except Exception as e:
            logger.error(f"Error fixing query: {e}")
            return {
                'success': False,
                'fixed_query': query,
                'changes_made': [],
                'explanation': f"Failed to fix query: {str(e)}"
            }
            
    def _parse_fixing_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM fixing response"""
        try:
            import re
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                # Try to parse entire response as JSON
                return json.loads(response_text.strip())
                
        except json.JSONDecodeError:
            # Fallback: extract fixed query from code blocks
            import re
            
            sql_match = re.search(r'```sql\s*(.*?)\s*```', response_text, re.DOTALL)
            if sql_match:
                return {
                    'fixed_query': sql_match.group(1).strip(),
                    'changes_made': ['Extracted from code block'],
                    'explanation': 'Query extracted from response'
                }
            else:
                return {
                    'fixed_query': response_text.strip(),
                    'changes_made': ['Full response as query'],
                    'explanation': 'Used full response as fixed query'
                }

class SQLGenerator:
    """Main SQL generator that orchestrates the generation process"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.code_generator = CodeGeneratorExecutor(db_manager)
        self.validator = QueryValidator(db_manager)
        self.fixer = QueryFixer(db_manager)
        
    def generate_sql_with_plan(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL with execution plan and validation"""
        try:
            # Step 1: Generate code and plan
            generation_result = self.code_generator.generate_and_plan(extracted_data)
            
            # Step 2: Validate generated queries
            validation_result = self.validator.validate_queries(
                generation_result.sql_queries, 
                extracted_data
            )
            
            # Step 3: Fix queries if needed
            if not validation_result['overall_valid']:
                fixed_queries = []
                
                for query_result in validation_result['query_results']:
                    if not query_result['is_valid']:
                        # Attempt to fix the query
                        error_messages = (query_result['syntax_errors'] + 
                                        query_result['compatibility_errors'] + 
                                        query_result['schema_errors'])
                        
                        fix_result = self.fixer.fix_query(
                            query_result['query'],
                            '; '.join(error_messages),
                            extracted_data
                        )
                        
                        if fix_result['success']:
                            fixed_queries.append(fix_result['fixed_query'])
                            logger.info(f"Successfully fixed query: {fix_result['explanation']}")
                        else:
                            fixed_queries.append(query_result['query'])  # Keep original if fix failed
                            logger.warning(f"Failed to fix query: {fix_result['explanation']}")
                    else:
                        fixed_queries.append(query_result['query'])
                        
                # Re-validate fixed queries
                final_validation = self.validator.validate_queries(fixed_queries, extracted_data)
                
                return {
                    'success': final_validation['overall_valid'],
                    'execution_plan': generation_result.execution_plan,
                    'sql_queries': fixed_queries,
                    'validation_checks': generation_result.validation_checks,
                    'expected_outcome': generation_result.expected_outcome,
                    'validation_result': final_validation,
                    'fixes_applied': True,
                    'original_queries': generation_result.sql_queries
                }
            else:
                return {
                    'success': True,
                    'execution_plan': generation_result.execution_plan,
                    'sql_queries': generation_result.sql_queries,
                    'validation_checks': generation_result.validation_checks,
                    'expected_outcome': generation_result.expected_outcome,
                    'validation_result': validation_result,
                    'fixes_applied': False
                }
                
        except Exception as e:
            logger.error(f"Error in generate_sql_with_plan: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_plan': {},
                'sql_queries': [],
                'validation_checks': [],
                'expected_outcome': {},
                'validation_result': {},
                'fixes_applied': False
            }
            
    def generate_simple_query(self, query_type: str, params: Dict[str, Any]) -> str:
        """Generate a simple query using templates"""
        try:
            from enums.database_types import TemplateType
            template_type = TemplateType(query_type)
            template = self.code_generator.template_manager.get_template(template_type)
            
            if template:
                return self.code_generator.template_manager.format_template_query(template, params)
            else:
                logger.warning(f"No template found for query type: {query_type}")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating simple query: {e}")
            return ""