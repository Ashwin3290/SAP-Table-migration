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
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"dmtool_{timestamp}.log")
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handlers = []
    
    if log_to_file:
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger.propagate = False
    
    logging.info(f"Logging initialized. Log file: {log_file}")

setup_logging(
    log_level=logging.INFO,
    log_to_file=True,
    log_to_console=False
)