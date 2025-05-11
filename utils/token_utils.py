"""
Token tracking utilities for TableLLM
Enhanced version based on the original token_tracker.py
"""
import os
import json
import functools
import time
from datetime import datetime
from utils.logging_utils import token_logger
import config

# Store token usage
_token_usage = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_calls": 0,
    "calls_by_function": {},
}

def estimate_tokens(text):
    """
    Estimate the number of tokens in a text using a simple tokenizer
    More accurate than character count but less accurate than a full tokenizer
    """
    if not text:
        return 0
    
    # Simple approximation based on words and punctuation
    # Average words per token is ~0.75, so we use that as a multiplier
    words = text.split()
    num_words = len(words)
    
    # Add tokens for punctuation and whitespace
    punctuation_count = sum(1 for c in text if c in ".,;:!?()[]{}-_=\"'")
    
    # Estimate total tokens
    estimated_tokens = int((num_words + punctuation_count) / 0.75)
    return max(1, estimated_tokens)  # Return at least 1 token

def track_token_usage(log_to_file=True, log_path=None):
    """
    Decorator for tracking token usage in functions that call the LLM API
    """
    if log_path is None:
        log_path = config.TOKEN_LOG_PATH
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Record function name and start time
            function_name = func.__name__
            start_time = time.time()
            call_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Try to extract prompt and response
            input_text = ""
            output_text = ""
            
            # Try to extract prompt from function arguments
            if "prompt" in kwargs:
                input_text = kwargs["prompt"]
            elif "contents" in kwargs:
                input_text = kwargs["contents"]
            elif len(args) > 0 and isinstance(args[0], str):
                input_text = args[0]
            
            # Try to extract response from result
            if hasattr(result, "text"):
                output_text = result.text
            elif isinstance(result, dict) and "text" in result:
                output_text = result["text"]
            elif isinstance(result, str):
                output_text = result
            
            # Estimate token counts
            input_tokens = estimate_tokens(input_text)
            output_tokens = estimate_tokens(output_text)
            
            # Update global token usage
            _token_usage["total_input_tokens"] += input_tokens
            _token_usage["total_output_tokens"] += output_tokens
            _token_usage["total_calls"] += 1
            
            # Update function-specific token usage
            if function_name not in _token_usage["calls_by_function"]:
                _token_usage["calls_by_function"][function_name] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            
            _token_usage["calls_by_function"][function_name]["calls"] += 1
            _token_usage["calls_by_function"][function_name]["input_tokens"] += input_tokens
            _token_usage["calls_by_function"][function_name]["output_tokens"] += output_tokens
            
            # Log to file if enabled
            if log_to_file:
                duration = time.time() - start_time
                log_entry = {
                    "timestamp": call_time,
                    "function": function_name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "duration_seconds": duration,
                    "total_input_tokens": _token_usage["total_input_tokens"],
                    "total_output_tokens": _token_usage["total_output_tokens"],
                    "total_calls": _token_usage["total_calls"],
                }
                token_logger.info(json.dumps(log_entry))
            
            token_logger.debug(f"Function: {function_name}, Input tokens: {input_tokens}, Output tokens: {output_tokens}")
            
            return result
        return wrapper
    return decorator

def get_token_usage_stats():
    """
    Get the current token usage statistics
    """
    total_tokens = _token_usage["total_input_tokens"] + _token_usage["total_output_tokens"]
    
    return {
        "total_tokens": total_tokens,
        "total_input_tokens": _token_usage["total_input_tokens"],
        "total_output_tokens": _token_usage["total_output_tokens"],
        "total_calls": _token_usage["total_calls"],
        "calls_by_function": _token_usage["calls_by_function"],
    }

def reset_token_usage_stats():
    """
    Reset all token usage statistics to zero
    """
    global _token_usage
    _token_usage = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_calls": 0,
        "calls_by_function": {},
    }
    return True
