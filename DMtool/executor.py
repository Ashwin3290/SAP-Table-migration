# executor.py
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from DMtool.enums.database_types import ExecutionStatus, ExecutionResult
from DMtool.database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class ExecutionContext:
    """Manages execution context and state"""
    
    def __init__(self, session_id: str, transformation_id: str):
        self.session_id = session_id
        self.transformation_id = transformation_id
        self.start_time = datetime.now()
        self.execution_steps = []
        self.current_step = 0
        self.total_steps = 0
        
    def add_step(self, step_name: str, description: str = ""):
        """Add execution step"""
        step_info = {
            'step_number': len(self.execution_steps) + 1,
            'name': step_name,
            'description': description,
            'status': 'pending',
            'start_time': None,
            'end_time': None,
            'result': None,
            'error': None
        }
        self.execution_steps.append(step_info)
        self.total_steps = len(self.execution_steps)
        
    def start_step(self, step_number: int):
        """Start execution of a step"""
        if 0 <= step_number - 1 < len(self.execution_steps):
            step = self.execution_steps[step_number - 1]
            step['status'] = 'running'
            step['start_time'] = datetime.now()
            self.current_step = step_number
            logger.info(f"Starting step {step_number}: {step['name']}")
            
    def complete_step(self, step_number: int, result: Any = None):
        """Complete execution of a step"""
        if 0 <= step_number - 1 < len(self.execution_steps):
            step = self.execution_steps[step_number - 1]
            step['status'] = 'completed'
            step['end_time'] = datetime.now()
            step['result'] = result
            logger.info(f"Completed step {step_number}: {step['name']}")
            
    def fail_step(self, step_number: int, error: str):
        """Mark step as failed"""
        if 0 <= step_number - 1 < len(self.execution_steps):
            step = self.execution_steps[step_number - 1]
            step['status'] = 'failed'
            step['end_time'] = datetime.now()
            step['error'] = error
            logger.error(f"Failed step {step_number}: {step['name']} - {error}")
            
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        completed_steps = sum(1 for step in self.execution_steps if step['status'] == 'completed')
        failed_steps = sum(1 for step in self.execution_steps if step['status'] == 'failed')
        
        return {
            'session_id': self.session_id,
            'transformation_id': self.transformation_id,
            'total_steps': self.total_steps,
            'completed_steps': completed_steps,
            'failed_steps': failed_steps,
            'current_step': self.current_step,
            'start_time': self.start_time.isoformat(),
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'steps': self.execution_steps
        }

class TransactionManager:
    """Manages database transactions with savepoints and rollback"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.savepoints = []
        self.transaction_active = False
        
    def begin_transaction(self) -> bool:
        """Begin a new transaction"""
        try:
            self.db_manager.connector.begin_transaction()
            self.transaction_active = True
            logger.debug("Transaction started")
            return True
        except Exception as e:
            logger.error(f"Failed to begin transaction: {e}")
            return False
            
    def create_savepoint(self, name: str) -> bool:
        """Create a savepoint"""
        try:
            if not self.transaction_active:
                logger.warning("Cannot create savepoint - no active transaction")
                return False
                
            # SQLite doesn't support named savepoints, so we track them manually
            self.savepoints.append({
                'name': name,
                'timestamp': datetime.now(),
                'queries_count': len(getattr(self, 'executed_queries', []))
            })
            
            logger.debug(f"Created savepoint: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create savepoint {name}: {e}")
            return False
            
    def rollback_to_savepoint(self, name: str) -> bool:
        """Rollback to a specific savepoint"""
        try:
            savepoint = next((sp for sp in self.savepoints if sp['name'] == name), None)
            if not savepoint:
                logger.warning(f"Savepoint {name} not found")
                return False
                
            # For SQLite, we need to rollback the entire transaction
            # In a more sophisticated implementation, we'd track and replay queries
            self.rollback_transaction()
            
            logger.debug(f"Rolled back to savepoint: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to rollback to savepoint {name}: {e}")
            return False
            
    def commit_transaction(self) -> bool:
        """Commit the transaction"""
        try:
            if not self.transaction_active:
                logger.warning("No active transaction to commit")
                return False
                
            self.db_manager.connector.commit_transaction()
            self.transaction_active = False
            self.savepoints.clear()
            logger.debug("Transaction committed")
            return True
        except Exception as e:
            logger.error(f"Failed to commit transaction: {e}")
            return False
            
    def rollback_transaction(self) -> bool:
        """Rollback the entire transaction"""
        try:
            if not self.transaction_active:
                logger.warning("No active transaction to rollback")
                return False
                
            self.db_manager.connector.rollback_transaction()
            self.transaction_active = False
            self.savepoints.clear()
            logger.debug("Transaction rolled back")
            return True
        except Exception as e:
            logger.error(f"Failed to rollback transaction: {e}")
            return False

class QueryExecutor:
    """Executes individual SQL queries with monitoring and error handling"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def execute_query_with_monitoring(self, query: str, params: Optional[Dict[str, Any]] = None,
                                    timeout: float = 30.0) -> Dict[str, Any]:
        """Execute query with performance monitoring"""
        start_time = time.time()
        
        try:
            # Validate query before execution
            if not self._is_safe_query(query):
                return {
                    'success': False,
                    'error': 'Query failed safety validation',
                    'execution_time': 0,
                    'result': None
                }
                
            # Execute query
            result = self.db_manager.execute_query(query, params)
            
            execution_time = time.time() - start_time
            
            # Check for timeout
            if execution_time > timeout:
                logger.warning(f"Query execution took {execution_time:.2f}s (timeout: {timeout}s)")
                
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'rows_affected': getattr(result, 'rowcount', 0) if hasattr(result, 'rowcount') else 0,
                'query': query
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed after {execution_time:.2f}s: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'result': None,
                'query': query
            }
            
    def execute_batch_queries(self, queries: List[str], 
                            params: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Execute multiple queries with individual monitoring"""
        results = []
        
        for i, query in enumerate(queries):
            query_params = params[i] if params and i < len(params) else None
            result = self.execute_query_with_monitoring(query, query_params)
            result['query_index'] = i
            results.append(result)
            
            # Stop on first failure if needed
            if not result['success']:
                logger.warning(f"Stopping batch execution at query {i} due to failure")
                break
                
        return results
        
    def _is_safe_query(self, query: str) -> bool:
        """Basic safety validation for queries"""
        if not query or not query.strip():
            return False
            
        # Check for potentially dangerous operations
        dangerous_patterns = [
            r'\bDROP\s+DATABASE\b',
            r'\bDROP\s+SCHEMA\b',
            r'\bTRUNCATE\s+TABLE\b.*WITHOUT\s+WHERE',
            r'\bDELETE\s+FROM\b.*(?!WHERE)',
            r'\bALTER\s+TABLE\b.*\bDROP\b'
        ]
        
        import re
        query_upper = query.upper()
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query_upper):
                logger.warning(f"Potentially dangerous query pattern detected: {pattern}")
                return False
                
        return True

class Executor:
    """Main executor that orchestrates SQL execution with transactions and monitoring"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.transaction_manager = TransactionManager(db_manager)
        self.query_executor = QueryExecutor(db_manager)
        
    def execute_with_transaction(self, generation_result: Dict[str, Any], 
                                context: Dict[str, Any]) -> ExecutionResult:
        """Execute SQL queries with full transaction support"""
        
        session_id = context.get('session_id', 'unknown')
        transformation_id = f"transform_{int(time.time())}"
        
        # Create execution context
        exec_context = ExecutionContext(session_id, transformation_id)
        
        try:
            # Prepare execution plan
            execution_plan = generation_result.get('execution_plan', {})
            sql_queries = generation_result.get('sql_queries', [])
            validation_checks = generation_result.get('validation_checks', [])
            
            # Add execution steps
            exec_context.add_step("begin_transaction", "Start database transaction")
            
            for i, query in enumerate(sql_queries):
                exec_context.add_step(f"execute_query_{i+1}", f"Execute: {query[:50]}...")
                
            for i, check in enumerate(validation_checks):
                exec_context.add_step(f"validate_{i+1}", f"Validate: {check[:50]}...")
                
            exec_context.add_step("commit_transaction", "Commit all changes")
            
            # Begin transaction
            exec_context.start_step(1)
            if not self.transaction_manager.begin_transaction():
                exec_context.fail_step(1, "Failed to begin transaction")
                return self._create_failure_result(exec_context, "Transaction start failed")
            exec_context.complete_step(1, "Transaction started")
            
            # Execute queries
            query_results = []
            for i, query in enumerate(sql_queries):
                step_num = i + 2  # +2 because step 1 is begin_transaction
                exec_context.start_step(step_num)
                
                # Create savepoint before each query
                savepoint_name = f"before_query_{i+1}"
                self.transaction_manager.create_savepoint(savepoint_name)
                
                # Execute query
                result = self.query_executor.execute_query_with_monitoring(query)
                
                if result['success']:
                    exec_context.complete_step(step_num, result)
                    query_results.append(result)
                else:
                    exec_context.fail_step(step_num, result['error'])
                    
                    # Rollback to savepoint and decide whether to continue
                    self.transaction_manager.rollback_to_savepoint(savepoint_name)
                    
                    # For now, fail the entire execution on any query failure
                    self.transaction_manager.rollback_transaction()
                    return self._create_failure_result(exec_context, f"Query {i+1} failed: {result['error']}")
                    
            # Execute validation checks
            validation_results = []
            validation_step_start = len(sql_queries) + 2
            
            for i, check in enumerate(validation_checks):
                step_num = validation_step_start + i
                exec_context.start_step(step_num)
                
                result = self.query_executor.execute_query_with_monitoring(check)
                
                if result['success']:
                    exec_context.complete_step(step_num, result)
                    validation_results.append(result)
                else:
                    exec_context.fail_step(step_num, result['error'])
                    logger.warning(f"Validation check {i+1} failed: {result['error']}")
                    # Continue with other validations even if one fails
                    
            # Commit transaction
            final_step = exec_context.total_steps
            exec_context.start_step(final_step)
            
            if self.transaction_manager.commit_transaction():
                exec_context.complete_step(final_step, "Transaction committed")
                
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    results=query_results,
                    queries_executed=[r['query'] for r in query_results],
                    rows_affected=sum(r.get('rows_affected', 0) for r in query_results)
                )
            else:
                exec_context.fail_step(final_step, "Failed to commit transaction")
                return self._create_failure_result(exec_context, "Transaction commit failed")
                
        except Exception as e:
            logger.error(f"Execution failed with exception: {e}")
            
            # Rollback transaction on any exception
            try:
                self.transaction_manager.rollback_transaction()
            except:
                pass  # Ignore rollback errors
                
            return self._create_failure_result(exec_context, f"Execution exception: {str(e)}")
            
    def _create_failure_result(self, exec_context: ExecutionContext, error_message: str) -> ExecutionResult:
        """Create failure result with execution context"""
        summary = exec_context.get_summary()
        
        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            results=[],
            error_message=error_message,
            queries_executed=[step.get('result', {}).get('query', '') 
                            for step in summary['steps'] 
                            if step.get('result') and step['result'].get('query')]
        )
        
    def execute_single_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Execute a single query without transaction management"""
        try:
            result = self.query_executor.execute_query_with_monitoring(query, params)
            
            if result['success']:
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    results=[result['result']],
                    queries_executed=[query],
                    rows_affected=result.get('rows_affected', 0)
                )
            else:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    results=[],
                    error_message=result['error'],
                    queries_executed=[query]
                )
                
        except Exception as e:
            logger.error(f"Single query execution failed: {e}")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                results=[],
                error_message=str(e),
                queries_executed=[query]
            )
            
    def test_queries(self, queries: List[str]) -> Dict[str, Any]:
        """Test queries without executing them (syntax check only)"""
        test_results = {
            'all_valid': True,
            'query_tests': []
        }
        
        for i, query in enumerate(queries):
            test_result = {
                'query_index': i,
                'query': query,
                'valid': True,
                'issues': []
            }
            
            # Basic syntax validation
            if not query.strip():
                test_result['valid'] = False
                test_result['issues'].append("Empty query")
                test_results['all_valid'] = False
                
            # Check for dangerous operations
            elif not self.query_executor._is_safe_query(query):
                test_result['valid'] = False
                test_result['issues'].append("Query failed safety validation")
                test_results['all_valid'] = False
                
            test_results['query_tests'].append(test_result)
            
        return test_results
        
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        # This would be enhanced with actual metrics collection
        return {
            'database_type': self.db_manager.db_type.value,
            'connection_active': self.db_manager.connector.is_connected,
            'transaction_active': self.transaction_manager.transaction_active,
            'savepoints_count': len(self.transaction_manager.savepoints)
        }