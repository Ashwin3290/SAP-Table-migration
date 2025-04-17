import time
import functools
import json
import logging
import os
from typing import Callable, Any, Dict, Optional
import tiktoken

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables to track token usage
TOTAL_INPUT_TOKENS = 0
TOTAL_OUTPUT_TOKENS = 0
TOTAL_API_CALLS = 0

# Function to estimate token count using tiktoken
def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string using tiktoken
    """
    try:
        # Use cl100k_base encoding (used by many models including GPT-4)
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = len(enc.encode(text))
        return tokens
    except Exception as e:
        logger.warning(f"Error estimating tokens: {e}")
        # Fallback method: rough approximation
        return len(text) // 4

# Decorator to track token usage
def track_token_usage(log_to_file: bool = True, log_path: str = 'token_usage.log'):
    """
    Decorator to track token usage for Gemini API calls
    
    Parameters:
    - log_to_file: Whether to log usage to a file
    - log_path: Path to the log file
    
    Usage:
    @track_token_usage()
    def your_function_that_calls_gemini(prompt, ...):
        ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_API_CALLS
            
            # Extract prompt from args or kwargs based on function signature
            prompt = None
            
            # For planner's parse_data function
            if func.__name__ == 'parse_data' and len(args) >= 2:
                prompt = args[1]  # Second argument is the query
            
            # For tablellm's _generate_with_gemini function
            elif func.__name__ == '_generate_with_gemini' and len(args) >= 2:
                prompt = args[1]  # Second argument is the prompt
            
            # If it's in kwargs
            elif 'prompt' in kwargs:
                prompt = kwargs['prompt']
            elif 'contents' in kwargs:
                prompt = kwargs['contents']
            elif 'question' in kwargs:
                prompt = kwargs['question']
            
            # Get input token count
            input_tokens = estimate_tokens(prompt) if prompt else 0
            TOTAL_INPUT_TOKENS += input_tokens
            
            # Record start time
            start_time = time.time()
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Record end time
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Get output token count
            output_text = ""
            if result and hasattr(result, 'text'):
                output_text = result.text
            elif isinstance(result, str):
                output_text = result
            
            output_tokens = estimate_tokens(output_text) if output_text else 0
            TOTAL_OUTPUT_TOKENS += output_tokens
            
            # Increment API call counter
            TOTAL_API_CALLS += 1
            
            # Prepare usage data
            usage_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "function": func.__name__,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "elapsed_time": elapsed_time,
                "total_input_tokens_so_far": TOTAL_INPUT_TOKENS,
                "total_output_tokens_so_far": TOTAL_OUTPUT_TOKENS,
                "total_calls_so_far": TOTAL_API_CALLS
            }
            
            # Log usage data
            # logger.info(f"API Call - {usage_data}")
            
            # Log to file if requested
            if log_to_file:
                try:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps(usage_data) + "\n")
                except Exception as e:
                    logger.warning(f"Error writing to token usage log: {e}")
            
            return result
        return wrapper
    return decorator

# Function to get current token usage statistics
def get_token_usage_stats() -> Dict[str, int]:
    """
    Get the current token usage statistics
    
    Returns:
    Dict with usage statistics
    """
    return {
        "total_input_tokens": TOTAL_INPUT_TOKENS,
        "total_output_tokens": TOTAL_OUTPUT_TOKENS,
        "total_tokens": TOTAL_INPUT_TOKENS + TOTAL_OUTPUT_TOKENS,
        "total_api_calls": TOTAL_API_CALLS
    }

# Function to reset token usage statistics
def reset_token_usage_stats() -> None:
    """Reset all token usage statistics to zero"""
    global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_API_CALLS
    TOTAL_INPUT_TOKENS = 0
    TOTAL_OUTPUT_TOKENS = 0
    TOTAL_API_CALLS = 0
    # logger.info("Token usage statistics have been reset")

# Function to estimate cost based on token usage (if pricing is available)
def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gemini-pro") -> float:
    """
    Estimate the cost of API usage based on token count
    
    Note: Pricing is subject to change. This is just an example.
    
    Parameters:
    - input_tokens: Number of input tokens
    - output_tokens: Number of output tokens
    - model: Model name
    
    Returns:
    Estimated cost in USD
    """
    # Example pricing (subject to change)
    pricing = {
        "gemini-pro": {"input": 0.00025, "output": 0.0005},  # per 1K tokens
        "gemini-ultra": {"input": 0.0008, "output": 0.0024}, # per 1K tokens
    }
    
    if model not in pricing:
        return 0.0
    
    input_cost = (input_tokens / 1000) * pricing[model]["input"]
    output_cost = (output_tokens / 1000) * pricing[model]["output"]
    
    return input_cost + output_cost