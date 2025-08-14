# enums/database_types.py
from enum import Enum
from typing import Dict, List, Any

class DatabaseType(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MSSQL = "mssql"
    ORACLE = "oracle"

class PromptType(Enum):
    QUERY_EXTRACTION = "query_extraction"
    CODE_GENERATION = "code_generation"
    QUERY_VALIDATION = "query_validation"
    ERROR_FIXING = "error_fixing"

class TemplateType(Enum):
    SIMPLE_TRANSFORMATION = "simple_transformation"
    JOIN_OPERATION = "join_operation"
    CROSS_SEGMENT = "cross_segment"
    VALIDATION_OPERATION = "validation_operation"
    AGGREGATION_OPERATION = "aggregation_operation"

class QueryType(Enum):
    SIMPLE_TRANSFORMATION = "SIMPLE_TRANSFORMATION"
    JOIN_OPERATION = "JOIN_OPERATION"
    CROSS_SEGMENT = "CROSS_SEGMENT"
    VALIDATION_OPERATION = "VALIDATION_OPERATION"
    AGGREGATION_OPERATION = "AGGREGATION_OPERATION"

class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    ROLLBACK = "rollback"

# Data classes for type safety
from dataclasses import dataclass
from typing import Optional

@dataclass
class QueryExtractionResult:
    query_type: QueryType
    confidence: float
    source_tables: List[str]
    target_tables: List[str]
    fields_mapping: Dict[str, List[str]]
    conditions: Dict[str, Any]
    joins: List[Dict[str, str]]
    semantic_understanding: Dict[str, str]
    
@dataclass
class CodeGenerationResult:
    execution_plan: Dict[str, Any]
    sql_queries: List[str]
    validation_checks: List[str]
    expected_outcome: Dict[str, Any]
    
@dataclass
class ExecutionResult:
    status: ExecutionStatus
    results: List[Any]
    error_message: Optional[str] = None
    queries_executed: List[str] = None
    rows_affected: int = 0