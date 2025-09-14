"""
Simple LLM Manager for DMTool - Direct LiteLLM Wrapper

A simple class that wraps LiteLLM to make it easy to use any LLM provider
with a unified interface.
"""

import os
import logging
from typing import Optional, Any
import litellm
from dotenv import load_dotenv
import time

load_dotenv()
logger = logging.getLogger(__name__)

class LLMManager:
    """
    Simple LLM wrapper that handles any provider through LiteLLM
    """
    
    def __init__(self, 
                 provider: Optional[str] = None,
                 model: str = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize LLM Manager
        
        Args:
            provider (str, optional): Provider name (only for reference, not used)
            model (str): Model name in LiteLLM format (e.g., "gemini/gemini-2.5-flash", "gpt-3.5-turbo", "claude-3-sonnet-20240229")
            api_key (str, optional): API key for the provider
            base_url (str, optional): Base URL for custom APIs (OpenAI-compatible)
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        litellm.set_verbose = os.getenv("LITELLM_VERBOSE", "false").lower() == "true"
        litellm.drop_params = True  # Drop unsupported parameters automatically
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate completion from prompt
        
        Args:
            prompt (str): The prompt to send to the LLM
            **kwargs: Any LiteLLM parameters (temperature, max_tokens, top_p, top_k, etc.)
            
        Returns:
            str: Generated response text or None if failed
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            completion_params = {
                "model": self.model,
                "messages": messages,
            }
            if self.api_key:
                completion_params["api_key"] = self.api_key
            if self.base_url:
                completion_params["api_base"] = self.base_url
            time_start = time.time()
            response = litellm.completion(**completion_params)
            logger.info(f"Reponse Latency {time.time() - time_start:.2f}s")
            logger.info(f"Token usage: {response.usage}")
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error in LLM generate: {e}")
            return None
def create_gemini_llm():
    return LLMManager(
        provider="gemini",
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY")
    )
def create_openai_llm():
    return LLMManager(
        provider="openai", 
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )
def create_anthropic_llm():
    return LLMManager(
        provider="anthropic",
        model="claude-3-sonnet-20240229", 
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
def create_deepseek_llm():
    return LLMManager(
        provider="deepseek",
        model="deepseek-chat",  # or whatever their model name is
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # whatever key name you want
        base_url="https://api.deepseek.com/v1"  # their endpoint
    )
def create_ollama_llm():
    return LLMManager(
        provider="ollama",
        model="llama3",  # local model name
        api_key="not-needed",  # Ollama doesn't need API key
        base_url="http://localhost:11434/v1"
    )

# Global instance for backward compatibility
_global_llm = None

def get_global_llm() -> LLMManager:
    """Get global LLM instance (creates default Gemini if none exists)"""
    global _global_llm
    if _global_llm is None:
        _global_llm = create_gemini_llm()
    return _global_llm

def set_global_llm(llm: LLMManager):
    """Set global LLM instance"""
    global _global_llm
    _global_llm = llm