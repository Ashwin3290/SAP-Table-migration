import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

from DMtool.enums.database_types import DatabaseType, ExecutionStatus,ExecutionResult
from DMtool.database.database_manager import DatabaseManager
from DMtool.planner import Planner
from DMtool.generator import SQLGenerator
from DMtool.executor import Executor
from DMtool.logging_config import setup_logging

setup_logging(log_to_file=True,
    log_to_console=True)
logger = logging.getLogger(__name__)
load_dotenv()

class DMToolError(Exception):
    """Base exception for DMTool operations"""
    pass

class DMTool:
    """
    Main DMTool interface for natural language to SQL transformation
    
    This is the primary interface that orchestrates the entire pipeline:
    1. Query Planning and Data Extraction
    2. SQL Generation with Validation
    3. Transaction-based Execution
    """
    
    def __init__(self, db_type: Optional[DatabaseType] = None, 
                 connection_params: Optional[Dict[str, Any]] = None):
        """
        Initialize DMTool with database configuration
        
        Args:
            db_type: Database type (defaults to environment variable DB_TYPE)
            connection_params: Database connection parameters
        """
        try:
            # Initialize database manager
            self.db_manager = DatabaseManager(db_type, connection_params)
            
            # Initialize core components
            self.planner = Planner(self.db_manager)
            self.generator = SQLGenerator(self.db_manager)
            self.executor = Executor(self.db_manager)
            
            # Tool state
            self.current_session = None
            self.transformation_history = []
            
            logger.info(f"DMTool initialized successfully with {self.db_manager.db_type.value} database")
            
        except Exception as e:
            logger.error(f"Failed to initialize DMTool: {e}")
            raise DMToolError(f"Initialization failed: {e}")
            
    def process_query(self, query: str, object_id: int, segment_id: int, 
                     project_id: int, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main method to process a natural language query
        
        Args:
            query: Natural language query
            object_id: Object identifier
            segment_id: Segment identifier  
            project_id: Project identifier
            session_id: Optional session ID for context
            
        Returns:
            Dictionary containing results and metadata
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Step 1: Plan and extract query data
            logger.info("Step 1: Planning and data extraction")
            planning_result = self.planner.process_query(
                object_id, segment_id, project_id, query, session_id
            )
            
            if not planning_result:
                raise DMToolError("Planning phase failed")
                
            # Update session tracking
            if not session_id:
                session_id = planning_result.get('session_id')
            self.current_session = session_id
            
            # Step 2: Generate SQL with validation and fixing
            logger.info("Step 2: SQL generation and validation")
            generation_result = self.generator.generate_sql_with_plan(planning_result)
            
            if not generation_result.get('success'):
                raise DMToolError(f"SQL generation failed: {generation_result.get('error', 'Unknown error')}")
                
            # Step 3: Execute with transaction support
            logger.info("Step 3: Transaction-based execution")
            execution_result = self.executor.execute_with_transaction(
                generation_result, 
                planning_result
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive result
            result = self._create_comprehensive_result(
                query, planning_result, generation_result, execution_result, processing_time
            )
            logger.info(f"Query executed:{query}")
            # Update transformation history
            self._update_transformation_history(result)
            
            logger.info(f"Query processed successfully in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Query processing failed after {processing_time:.2f} seconds: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'session_id': session_id,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
    def _create_comprehensive_result(self, query: str, planning_result: Dict[str, Any],
                                   generation_result: Dict[str, Any], 
                                   execution_result: ExecutionResult,
                                   processing_time: float) -> Dict[str, Any]:
        """Create comprehensive result combining all phases"""
        
        success = execution_result.status == ExecutionStatus.SUCCESS
        
        result = {
            # Overall status
            'success': success,
            'query': query,
            'session_id': planning_result.get('session_id'),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            
            # Planning phase results
            'planning': {
                'query_type': planning_result.get('query_type'),
                'confidence': planning_result.get('confidence'),
                'source_tables': planning_result.get('source_tables', []),
                'target_tables': planning_result.get('target_tables', []),
                'fields_mapping': planning_result.get('fields_mapping', {}),
                'semantic_understanding': planning_result.get('semantic_understanding', {})
            },
            
            # Generation phase results
            'generation': {
                'execution_plan': generation_result.get('execution_plan', {}),
                'sql_queries': generation_result.get('sql_queries', []),
                'validation_checks': generation_result.get('validation_checks', []),
                'fixes_applied': generation_result.get('fixes_applied', False),
                'validation_result': generation_result.get('validation_result', {})
            },
            
            # Execution phase results
            'execution': {
                'status': execution_result.status.value,
                'queries_executed': execution_result.queries_executed or [],
                'rows_affected': execution_result.rows_affected,
                'error_message': execution_result.error_message
            },
            
            # Data results (if available)
            'data': self._extract_data_results(execution_result),
            
            # Context and metadata
            'context': {
                'database_type': self.db_manager.db_type.value,
                'schema_knowledge': planning_result.get('schema_knowledge', {}),
                'contextual_knowledge': planning_result.get('contextual_knowledge', {}),
                'semantic_knowledge': planning_result.get('semantic_knowledge', {})
            }
        }
        
        return result
        
    def _extract_data_results(self, execution_result: ExecutionResult) -> Optional[pd.DataFrame]:
        """Extract data results from execution"""
        try:
            if execution_result.results and len(execution_result.results) > 0:
                # For SELECT queries, return the data
                last_result = execution_result.results[-1]
                
                if hasattr(last_result, 'fetchall'):
                    # Convert database result to DataFrame
                    rows = last_result.fetchall()
                    if rows:
                        columns = [desc[0] for desc in last_result.description]
                        return pd.DataFrame([dict(zip(columns, row)) for row in rows])
                elif isinstance(last_result, pd.DataFrame):
                    return last_result
                    
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting data results: {e}")
            return None
            
    def _update_transformation_history(self, result: Dict[str, Any]):
        """Update transformation history"""
        try:
            history_entry = {
                'timestamp': result['timestamp'],
                'query': result['query'],
                'success': result['success'],
                'query_type': result['planning']['query_type'],
                'processing_time': result['processing_time'],
                'rows_affected': result['execution']['rows_affected'],
                'session_id': result['session_id']
            }
            
            self.transformation_history.append(history_entry)
            
            # Keep only last 100 transformations
            if len(self.transformation_history) > 100:
                self.transformation_history = self.transformation_history[-100:]
                
        except Exception as e:
            logger.warning(f"Error updating transformation history: {e}")
            
    def create_session(self) -> str:
        """Create a new session for tracking transformations"""
        try:
            session_id = self.planner.session_manager.create_session()
            self.current_session = session_id
            logger.info(f"Created new session: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise DMToolError(f"Session creation failed: {e}")
            
    def get_session_context(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get context for a session"""
        try:
            target_session = session_id or self.current_session
            if not target_session:
                return None
                
            return self.planner.session_manager.get_context(target_session)
        except Exception as e:
            logger.error(f"Failed to get session context: {e}")
            return None
            
    def test_query_generation(self, query: str, object_id: int, segment_id: int, 
                            project_id: int) -> Dict[str, Any]:
        """Test query generation without execution"""
        try:
            # Only run planning and generation phases
            planning_result = self.planner.process_query(
                object_id, segment_id, project_id, query
            )
            
            generation_result = self.generator.generate_sql_with_plan(planning_result)
            
            # Test the generated queries
            test_result = self.executor.test_queries(generation_result.get('sql_queries', []))
            
            return {
                'success': True,
                'planning': planning_result,
                'generation': generation_result,
                'test_result': test_result
            }
            
        except Exception as e:
            logger.error(f"Query generation test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and health status"""
        try:
            health_check = self.db_manager.health_check()
            tables = self.db_manager.get_tables()
            
            return {
                'database_type': self.db_manager.db_type.value,
                'health': health_check,
                'tables_count': len(tables),
                'tables': tables[:20],  # First 20 tables
                'dialect_info': self.db_manager.get_dialect_info()
            }
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {
                'error': str(e)
            }
            
    def get_transformation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent transformation history"""
        return self.transformation_history[-limit:] if self.transformation_history else []
        
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        try:
            total_transformations = len(self.transformation_history)
            successful_transformations = sum(1 for t in self.transformation_history if t['success'])
            
            stats = {
                'total_transformations': total_transformations,
                'successful_transformations': successful_transformations,
                'success_rate': successful_transformations / total_transformations if total_transformations > 0 else 0,
                'current_session': self.current_session,
                'database_stats': self.executor.get_execution_stats()
            }
            
            if self.transformation_history:
                avg_processing_time = sum(t['processing_time'] for t in self.transformation_history) / len(self.transformation_history)
                stats['average_processing_time'] = avg_processing_time
                
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get execution stats: {e}")
            return {'error': str(e)}
            
    def validate_connection(self) -> Dict[str, Any]:
        """Validate database connection"""
        try:
            health_check = self.db_manager.health_check()
            
            return {
                'connected': health_check['connection_active'],
                'database_type': health_check['database_type'],
                'status': health_check['status'],
                'tables_accessible': health_check.get('tables_count', 0) > 0
            }
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return {
                'connected': False,
                'error': str(e)
            }
            
    def close(self):
        """Close DMTool and cleanup resources"""
        try:
            if self.db_manager:
                self.db_manager.close()
            logger.info("DMTool closed successfully")
        except Exception as e:
            logger.error(f"Error closing DMTool: {e}")
            
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup

# Convenience functions for easy usage
def create_dmtool(db_type: str = None, **connection_params) -> DMTool:
    """
    Create DMTool instance with simplified interface
    
    Args:
        db_type: Database type ('sqlite', 'postgresql', 'mysql', etc.)
        **connection_params: Database connection parameters
        
    Returns:
        DMTool instance
    """
    try:
        if db_type:
            db_type_enum = DatabaseType(db_type.lower())
        else:
            db_type_enum = None
            
        return DMTool(db_type_enum, connection_params)
        
    except Exception as e:
        raise DMToolError(f"Failed to create DMTool: {e}")
        
def process_natural_language_query(query: str, object_id: int, segment_id: int, 
                                 project_id: int, db_type: str = None, 
                                 **connection_params) -> Dict[str, Any]:
    """
    Process a natural language query with automatic DMTool creation
    
    Args:
        query: Natural language query
        object_id: Object identifier
        segment_id: Segment identifier
        project_id: Project identifier
        db_type: Database type
        **connection_params: Database connection parameters
        
    Returns:
        Processing results
    """
    with create_dmtool(db_type, **connection_params) as dmtool:
        return dmtool.process_query(query, object_id, segment_id, project_id)

# Example usage and testing
if __name__ == "__main__":
    try:
        # Example usage
        dmtool = create_dmtool('sqlite', database_path='example.db')
        
        # Test database connection
        connection_status = dmtool.validate_connection()
        print(f"Database connection: {connection_status}")
        
        # Get database info
        db_info = dmtool.get_database_info()
        print(f"Database info: {db_info}")
        
        # Example query processing
        result = dmtool.process_query(
            query="Bring material number from MARA table where material type equals ROH",
            object_id=1,
            segment_id=1,
            project_id=1
        )
        
        print(f"Query result: {result}")
        
        # Get execution stats
        stats = dmtool.get_execution_stats()
        print(f"Execution stats: {stats}")
        
        dmtool.close()
        
    except Exception as e:
        print(f"Example execution failed: {e}")
        import traceback
        traceback.print_exc()