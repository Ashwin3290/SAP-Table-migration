# planner.py
import logging
import json
import re
import os
import datetime
from typing import Dict, List, Any, Optional
from google import genai
from google.genai import types

from DMtool.enums.database_types import DatabaseType, PromptType, QueryExtractionResult, QueryType
from DMtool.managers.prompt_manager import PromptManager
from DMtool.database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class ClassificationEnhancer:
    """Enhances LLM classification with fuzzy matching and validation"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def enhance_classification_details(self, classification_details: Dict[str, Any], 
                                     segment_id: Optional[int] = None) -> Dict[str, Any]:
        """Enhance classification with database validation and fuzzy matching"""
        try:
            enhanced_details = classification_details.copy()
            
            # Get available tables from database
            available_tables = self.db_manager.get_tables()
            
            # Validate and enhance table references
            mentioned_tables = classification_details.get('source_tables', [])
            validated_tables = self._validate_tables(mentioned_tables, available_tables)
            enhanced_details['validated_tables'] = validated_tables
            
            # Validate and enhance column references
            mentioned_columns = classification_details.get('fields_mapping', {}).get('source_fields', [])
            validated_columns = self._validate_columns(mentioned_columns, validated_tables['valid_tables'])
            enhanced_details['validated_columns'] = validated_columns
            
            # Add schema context
            schema_context = self._get_schema_context(validated_tables['valid_tables'])
            enhanced_details['schema_context'] = schema_context
            
            return enhanced_details
            
        except Exception as e:
            logger.error(f"Error enhancing classification details: {e}")
            return classification_details
            
    def _validate_tables(self, mentioned_tables: List[str], available_tables: List[str]) -> Dict[str, Any]:
        """Validate table names against database schema"""
        valid_tables = []
        invalid_tables = []
        suggestions = {}
        
        for table in mentioned_tables:
            if table in available_tables:
                valid_tables.append(table)
            else:
                invalid_tables.append(table)
                # Find closest match
                suggestion = self._find_closest_match(table, available_tables)
                if suggestion:
                    suggestions[table] = suggestion
                    
        return {
            'valid_tables': valid_tables,
            'invalid_tables': invalid_tables,
            'suggestions': suggestions
        }
        
    def _validate_columns(self, mentioned_columns: List[str], valid_tables: List[str]) -> Dict[str, Any]:
        """Validate column names against table schemas"""
        column_validation = {}
        
        for table in valid_tables:
            try:
                columns = self.db_manager.get_columns(table)
                available_columns = [col['name'] for col in columns]
                
                table_validation = {
                    'valid_columns': [],
                    'invalid_columns': [],
                    'suggestions': {}
                }
                
                for column in mentioned_columns:
                    if column in available_columns:
                        table_validation['valid_columns'].append(column)
                    else:
                        table_validation['invalid_columns'].append(column)
                        suggestion = self._find_closest_match(column, available_columns)
                        if suggestion:
                            table_validation['suggestions'][column] = suggestion
                            
                column_validation[table] = table_validation
                
            except Exception as e:
                logger.warning(f"Error validating columns for table {table}: {e}")
                
        return column_validation
        
    def _get_schema_context(self, tables: List[str]) -> Dict[str, Any]:
        """Get schema context for tables"""
        schema_context = {}
        
        for table in tables:
            try:
                schema_context[table] = self.db_manager.get_schema(table)
            except Exception as e:
                logger.warning(f"Error getting schema for table {table}: {e}")
                
        return schema_context
        
    def _find_closest_match(self, query: str, candidates: List[str], threshold: float = 0.6) -> Optional[str]:
        """Find closest string match using simple similarity"""
        if not query or not candidates:
            return None
            
        query_lower = query.lower()
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            candidate_lower = candidate.lower()
            
            # Simple similarity based on common characters
            if query_lower in candidate_lower or candidate_lower in query_lower:
                score = min(len(query_lower), len(candidate_lower)) / max(len(query_lower), len(candidate_lower))
            else:
                # Character-based similarity
                common_chars = set(query_lower) & set(candidate_lower)
                total_chars = set(query_lower) | set(candidate_lower)
                score = len(common_chars) / len(total_chars) if total_chars else 0
                
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate
                
        return best_match

class QueryDataExtractor:
    """Extracts comprehensive query data in a single LLM call"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.prompt_manager = PromptManager(db_manager.db_type)
        self.classification_enhancer = ClassificationEnhancer(db_manager)
        
        # Initialize LLM client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.llm_client = genai.Client(api_key=api_key)
        
    def extract_all_query_data(self, query: str, context: Dict[str, Any]) -> QueryExtractionResult:
        """Extract comprehensive query data in single LLM call"""
        try:
            # Get available database context
            db_context = self._prepare_database_context(context)
            
            # Build extraction prompt
            extraction_prompt = self.prompt_manager.get_prompt(
                PromptType.QUERY_EXTRACTION,
                query=query,
                context=json.dumps(context, indent=2),
                available_tables=db_context.get('tables', []),
                schema_info=json.dumps(db_context.get('schemas', {}), indent=2)
            )
            
            # Execute LLM call
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=extraction_prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            
            if not response or not hasattr(response, "text"):
                raise ValueError("Invalid response from LLM")
                
            # Parse response
            raw_response = self._parse_llm_response(response.text)
            
            # Enhance with classification enhancer
            enhanced_response = self.classification_enhancer.enhance_classification_details(
                raw_response,
                context.get('segment_id')
            )
            
            # Convert to structured result
            return self._create_extraction_result(enhanced_response)
            
        except Exception as e:
            logger.error(f"Error in extract_all_query_data: {e}")
            # Return fallback result
            return self._create_fallback_result(query)
            
    def _prepare_database_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare database context for extraction"""
        try:
            # Get available tables
            tables = self.db_manager.get_tables()
            
            # Get schemas for relevant tables (limit to avoid token overflow)
            schemas = {}
            relevant_tables = context.get('relevant_tables', tables[:10])  # Limit to first 10
            
            for table in relevant_tables:
                try:
                    schemas[table] = self.db_manager.get_schema(table)
                except Exception as e:
                    logger.warning(f"Error getting schema for {table}: {e}")
                    
            return {
                'tables': tables,
                'schemas': schemas,
                'dialect_info': self.db_manager.get_dialect_info()
            }
            
        except Exception as e:
            logger.error(f"Error preparing database context: {e}")
            return {'tables': [], 'schemas': {}, 'dialect_info': {}}
            
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response JSON"""
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to parse entire response as JSON
                json_str = response_text.strip()
                
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response JSON: {e}")
            # Return basic fallback structure
            return {
                'query_type': 'SIMPLE_TRANSFORMATION',
                'confidence': 0.5,
                'source_tables': [],
                'target_tables': [],
                'fields_mapping': {
                    'source_fields': [],
                    'target_fields': [],
                    'filtering_fields': [],
                    'insertion_fields': []
                },
                'conditions': {},
                'joins': [],
                'semantic_understanding': {
                    'business_intent': 'Parse error occurred',
                    'transformation_goal': 'Unknown',
                    'data_flow': 'Unable to determine'
                }
            }
            
    def _create_extraction_result(self, enhanced_response: Dict[str, Any]) -> QueryExtractionResult:
        """Create structured extraction result"""
        try:
            query_type_str = enhanced_response.get('query_type', 'SIMPLE_TRANSFORMATION')
            query_type = QueryType(query_type_str)
        except ValueError:
            query_type = QueryType.SIMPLE_TRANSFORMATION
            
        return QueryExtractionResult(
            query_type=query_type,
            confidence=enhanced_response.get('confidence', 0.8),
            source_tables=enhanced_response.get('source_tables', []),
            target_tables=enhanced_response.get('target_tables', []),
            fields_mapping=enhanced_response.get('fields_mapping', {}),
            conditions=enhanced_response.get('conditions', {}),
            joins=enhanced_response.get('joins', []),
            semantic_understanding=enhanced_response.get('semantic_understanding', {})
        )
        
    def _create_fallback_result(self, query: str) -> QueryExtractionResult:
        """Create fallback result when extraction fails"""
        return QueryExtractionResult(
            query_type=QueryType.SIMPLE_TRANSFORMATION,
            confidence=0.3,
            source_tables=[],
            target_tables=[],
            fields_mapping={
                'source_fields': [],
                'target_fields': [],
                'filtering_fields': [],
                'insertion_fields': []
            },
            conditions={},
            joins=[],
            semantic_understanding={
                'business_intent': f'Process query: {query[:100]}...',
                'transformation_goal': 'Data transformation',
                'data_flow': 'Unknown flow due to extraction error'
            }
        )

class ContextualSessionManager:
    """Manages session context and state for transformations"""
    
    def __init__(self, storage_path: str = "sessions"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
    def create_session(self) -> str:
        """Create new session and return session ID"""
        import uuid
        from datetime import datetime
        
        session_id = str(uuid.uuid4())
        session_data = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'transformations': [],
            'segments_visited': {},
            'key_mappings': []
        }
        
        session_path = os.path.join(self.storage_path, f"{session_id}.json")
        with open(session_path, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        logger.info(f"Created new session: {session_id}")
        return session_id
        
    def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session context"""
        if not session_id:
            return None
            
        session_path = os.path.join(self.storage_path, f"{session_id}.json")
        if not os.path.exists(session_path):
            return None
            
        try:
            with open(session_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading session context: {e}")
            return None
            
    def update_context(self, session_id: str, context_data: Dict[str, Any]) -> bool:
        """Update session context"""
        try:
            session_path = os.path.join(self.storage_path, f"{session_id}.json")
            
            # Load existing context
            if os.path.exists(session_path):
                with open(session_path, 'r') as f:
                    existing_context = json.load(f)
            else:
                existing_context = {
                    'session_id': session_id,
                    'created_at': datetime.now().isoformat(),
                    'transformations': [],
                    'segments_visited': {},
                    'key_mappings': []
                }
                
            # Update with new data
            existing_context.update(context_data)
            existing_context['updated_at'] = datetime.now().isoformat()
            
            # Save updated context
            with open(session_path, 'w') as f:
                json.dump(existing_context, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating session context: {e}")
            return False
            
    def add_transformation_record(self, session_id: str, transformation_data: Dict[str, Any]) -> bool:
        """Add transformation record to session"""
        try:
            context = self.get_context(session_id)
            if not context:
                context = {
                    'session_id': session_id,
                    'transformations': [],
                    'segments_visited': {},
                    'key_mappings': []
                }
                
            # Add transformation with timestamp and ID
            from datetime import datetime
            transformation_record = {
                'id': len(context['transformations']) + 1,
                'timestamp': datetime.now().isoformat(),
                **transformation_data
            }
            
            context['transformations'].append(transformation_record)
            
            return self.update_context(session_id, context)
            
        except Exception as e:
            logger.error(f"Error adding transformation record: {e}")
            return False

class Planner:
    """Enhanced planner for query processing and context management"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.query_extractor = QueryDataExtractor(db_manager)
        self.session_manager = ContextualSessionManager()
        
    def process_query(self, object_id: int, segment_id: int, project_id: int, 
                     query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process query and return comprehensive extraction data"""
        try:
            # Create session if not provided
            if not session_id:
                session_id = self.session_manager.create_session()
                
            # Prepare context
            context = self._prepare_query_context(
                object_id, segment_id, project_id, session_id
            )
            
            # Extract query data
            extraction_result = self.query_extractor.extract_all_query_data(query, context)
            
            # Enrich with additional context
            enriched_data = self._enrich_extraction_data(extraction_result, context, session_id)
            
            # Update session with transformation record
            self.session_manager.add_transformation_record(session_id, {
                'original_query': query,
                'query_type': extraction_result.query_type.value,
                'object_id': object_id,
                'segment_id': segment_id,
                'project_id': project_id,
                'confidence': extraction_result.confidence
            })
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            raise
            
    def _prepare_query_context(self, object_id: int, segment_id: int, 
                              project_id: int, session_id: str) -> Dict[str, Any]:
        """Prepare context for query processing"""
        context = {
            'object_id': object_id,
            'segment_id': segment_id,
            'project_id': project_id,
            'session_id': session_id
        }
        
        # Add session context if available
        session_context = self.session_manager.get_context(session_id)
        if session_context:
            context['session_history'] = session_context.get('transformations', [])
            context['segments_visited'] = session_context.get('segments_visited', {})
            context['key_mappings'] = session_context.get('key_mappings', [])
            
        # Add database context
        try:
            context['available_tables'] = self.db_manager.get_tables()
            context['database_type'] = self.db_manager.db_type.value
            context['dialect_info'] = self.db_manager.get_dialect_info()
        except Exception as e:
            logger.warning(f"Error adding database context: {e}")
            
        return context
        
    def _enrich_extraction_data(self, extraction_result: QueryExtractionResult, 
                               context: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Enrich extraction data with comprehensive context"""
        
        # Convert extraction result to dict
        enriched_data = {
            'query_type': extraction_result.query_type.value,
            'confidence': extraction_result.confidence,
            'source_tables': extraction_result.source_tables,
            'target_tables': extraction_result.target_tables,
            'fields_mapping': extraction_result.fields_mapping,
            'conditions': extraction_result.conditions,
            'joins': extraction_result.joins,
            'semantic_understanding': extraction_result.semantic_understanding,
            'session_id': session_id
        }
        
        # Add schema knowledge
        enriched_data['schema_knowledge'] = self._get_schema_knowledge(
            extraction_result.source_tables + extraction_result.target_tables
        )
        
        # Add contextual knowledge
        enriched_data['contextual_knowledge'] = {
            'session_history': context.get('session_history', []),
            'segments_visited': context.get('segments_visited', {}),
            'key_mappings': context.get('key_mappings', []),
            'database_context': {
                'type': context.get('database_type'),
                'dialect_info': context.get('dialect_info', {})
            }
        }
        
        # Add semantic knowledge
        enriched_data['semantic_knowledge'] = {
            'business_rules': self._extract_business_rules(enriched_data),
            'data_lineage': self._trace_data_lineage(enriched_data),
            'optimization_suggestions': self._suggest_optimizations(enriched_data)
        }
        
        return enriched_data
        
    def _get_schema_knowledge(self, table_names: List[str]) -> Dict[str, Any]:
        """Get comprehensive schema knowledge for tables"""
        schema_knowledge = {
            'schemas': {},
            'relationships': {},
            'constraints': {}
        }
        
        for table_name in table_names:
            if not table_name:
                continue
                
            try:
                schema_knowledge['schemas'][table_name] = self.db_manager.get_schema(table_name)
            except Exception as e:
                logger.warning(f"Error getting schema for {table_name}: {e}")
                schema_knowledge['schemas'][table_name] = {'error': str(e)}
                
        return schema_knowledge
        
    def _extract_business_rules(self, data: Dict[str, Any]) -> List[str]:
        """Extract business rules from query data"""
        rules = []
        
        # Extract rules from conditions
        conditions = data.get('conditions', {})
        for field, value in conditions.items():
            rules.append(f"Field '{field}' must equal '{value}'")
            
        # Extract rules from semantic understanding
        semantic = data.get('semantic_understanding', {})
        business_intent = semantic.get('business_intent', '')
        if business_intent:
            rules.append(f"Business intent: {business_intent}")
            
        return rules
        
    def _trace_data_lineage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Trace data lineage through transformation"""
        lineage = {
            'source_to_target_mapping': {},
            'transformation_flow': [],
            'data_dependencies': []
        }
        
        # Map source fields to target fields
        fields_mapping = data.get('fields_mapping', {})
        source_fields = fields_mapping.get('source_fields', [])
        target_fields = fields_mapping.get('target_fields', [])
        
        for i, source_field in enumerate(source_fields):
            if i < len(target_fields):
                lineage['source_to_target_mapping'][source_field] = target_fields[i]
                
        # Document transformation flow
        source_tables = data.get('source_tables', [])
        target_tables = data.get('target_tables', [])
        
        if source_tables and target_tables:
            lineage['transformation_flow'] = [
                f"Extract from: {', '.join(source_tables)}",
                f"Transform: Apply business rules and mappings",
                f"Load into: {', '.join(target_tables)}"
            ]
            
        return lineage
        
    def _suggest_optimizations(self, data: Dict[str, Any]) -> List[str]:
        """Suggest optimization opportunities"""
        suggestions = []
        
        # Check for potential optimizations
        joins = data.get('joins', [])
        if len(joins) > 2:
            suggestions.append("Consider optimizing multiple joins with proper indexing")
            
        conditions = data.get('conditions', {})
        if len(conditions) > 5:
            suggestions.append("Multiple filter conditions detected - consider compound indexes")
            
        source_tables = data.get('source_tables', [])
        if len(source_tables) > 3:
            suggestions.append("Multiple source tables - consider materialized views for performance")
            
        return suggestions