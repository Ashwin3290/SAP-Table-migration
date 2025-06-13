import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.INFO, log_to_file=True, log_to_console=False):
    """
    Setup logging configuration for the entire application
    
    Parameters:
    log_level: Logging level (default: INFO)
    log_to_file: Whether to log to file (default: True)
    log_to_console: Whether to log to console (default: False)
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"dmtool_{timestamp}.log")
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set the root logger level
    root_logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handlers = []
    
    if log_to_file:
        # File handler with rotation (max 10MB, keep 5 files)
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    if log_to_console:
        # Console handler (optional)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Prevent duplicate logging from individual module configurations
    root_logger.propagate = False
    
    logging.info(f"Logging initialized. Log file: {log_file}")

setup_logging(
    log_level=logging.INFO,
    log_to_file=True,
    log_to_console=False  # Set to True if you want both file and console
)