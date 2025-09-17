class CodeGenerationError(Exception):
    """Exception raised for code generation errors."""
    pass

class ExecutionError(Exception):
    """Exception raised for code execution errors."""
    pass



class SQLInjectionError(Exception):
    """Exception raised for potential SQL injection attempts."""
    pass

class SessionError(Exception):
    """Exception raised for session-related errors."""
    pass


class APIError(Exception):
    """Exception raised for API-related errors."""
    pass


class DataProcessingError(Exception):
    """Exception raised for data processing errors."""
    pass

