"""
Lightweight, reusable Python utility for accurate token counting.

This module provides efficient token counting functionality using Hugging Face's
transformers library with caching for optimal performance.
"""

import logging
from functools import lru_cache
from typing import Union, List, Optional
from transformers import AutoTokenizer

# Set up logging
logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL = "gpt2"


class TokenCounter:
    """
    A lightweight token counter using Hugging Face tokenizers with caching.
    
    This class provides efficient token counting with automatic caching of
    tokenizers to avoid repeated model loading.
    """
    
    def __init__(self, model_id: str = DEFAULT_MODEL):
        """
        Initialize the TokenCounter with a specific model.
        
        Args:
            model_id (str): The Hugging Face model identifier.
                          Defaults to "gpt2".
        """
        self.model_id = model_id
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        """Lazy loading of tokenizer with caching."""
        if self._tokenizer is None:
            self._tokenizer = self._get_tokenizer(self.model_id)
        return self._tokenizer
    
    @staticmethod
    @lru_cache(maxsize=8)
    def _get_tokenizer(model_id: str):
        """
        Get a tokenizer for the specified model with LRU caching.
        
        Args:
            model_id (str): The Hugging Face model identifier.
            
        Returns:
            AutoTokenizer: The loaded tokenizer.
            
        Raises:
            Exception: If the tokenizer cannot be loaded.
        """
        try:
            logger.info(f"Loading tokenizer for model: {model_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Ensure tokenizer has a pad token (required for some models)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer for model {model_id}: {e}")
            raise
    
    def count_tokens(self, text: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Count tokens in the given text(s).
        
        Args:
            text (Union[str, List[str]]): Text or list of texts to tokenize.
            
        Returns:
            Union[int, List[int]]: Token count(s) for the input text(s).
        """
        if isinstance(text, str):
            return len(self.tokenizer.encode(text, add_special_tokens=True))
        elif isinstance(text, list):
            return [len(self.tokenizer.encode(t, add_special_tokens=True)) for t in text]
        else:
            raise ValueError("Input must be a string or list of strings")
    
    def get_tokens(self, text: str) -> List[str]:
        """
        Get the actual tokens for debugging/inspection purposes.
        
        Args:
            text (str): Text to tokenize.
            
        Returns:
            List[str]: List of tokens.
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        return self.tokenizer.convert_ids_to_tokens(token_ids)
    
    def clear_cache(self):
        """Clear the tokenizer cache."""
        TokenCounter._get_tokenizer.cache_clear()
        logger.info("Tokenizer cache cleared")


# Convenience function for quick token counting
@lru_cache(maxsize=128)
def count_tokens(text: str, model_id: str = DEFAULT_MODEL) -> int:
    """
    Count tokens in a text string using the specified model.
    
    This is a convenience function that provides caching for both the tokenizer
    and the results of token counting for repeated calls.
    
    Args:
        text (str): The text to count tokens for.
        model_id (str): The Hugging Face model identifier.
                       Defaults to "gpt2".
    
    Returns:
        int: The number of tokens in the text.
        
    Example:
        >>> count_tokens("Hello, world!")
        4
        >>> count_tokens("This is a test", "gpt2")
        5
    """
    counter = TokenCounter(model_id)
    return counter.count_tokens(text)


def get_available_cache_info() -> dict:
    """
    Get information about the current cache state.
    
    Returns:
        dict: Cache information including hits, misses, and current size.
    """
    tokenizer_cache = TokenCounter._get_tokenizer.cache_info()
    count_cache = count_tokens.cache_info()
    
    return {
        "tokenizer_cache": {
            "hits": tokenizer_cache.hits,
            "misses": tokenizer_cache.misses,
            "current_size": tokenizer_cache.currsize,
            "max_size": tokenizer_cache.maxsize
        },
        "count_cache": {
            "hits": count_cache.hits,
            "misses": count_cache.misses,
            "current_size": count_cache.currsize,
            "max_size": count_cache.maxsize
        }
    }


def clear_all_caches():
    """Clear all token counting caches."""
    TokenCounter._get_tokenizer.cache_clear()
    count_tokens.cache_clear()
    logger.info("All token counting caches cleared") 