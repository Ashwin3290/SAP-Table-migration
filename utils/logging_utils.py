"""
Logging utilities for TableLLM
"""
import os
import logging
from datetime import datetime
import config

def setup_logger(name, log_file=None, level=None):
    """
    Set up a logger with given name, file, and level
    """
    if level is None:
        level = getattr(logging, config.LOG_LEVEL)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create loggers
main_logger = setup_logger('tablellm')
agent_logger = setup_logger('tablellm.agent', config.AGENT_LOG_PATH)
token_logger = setup_logger('tablellm.token', config.TOKEN_LOG_PATH)

class LoggingContext:
    """
    Context manager to track and log function execution details
    """
    def __init__(self, logger, function_name, log_args=False):
        self.logger = logger
        self.function_name = function_name
        self.log_args = log_args
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.function_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type:
            self.logger.error(f"Error in {self.function_name}: {exc_val}")
            return False
        else:
            self.logger.info(f"Completed {self.function_name} in {duration.total_seconds():.2f}s")
            return True
