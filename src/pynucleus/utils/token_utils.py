"""
Token utilities for PyNucleus system using HuggingFace tokenizers.
"""

import logging
from typing import Optional, Union, List

class TokenCounter:
    """Efficient token counting using HuggingFace tokenizers."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)
        
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.logger.info(f"TokenCounter initialized with model: {model_name}")
        except ImportError:
            self.logger.warning("Transformers library not available, using fallback token counting")
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer {model_name}, using fallback: {e}")
    
    def count_tokens(self, text: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Count tokens in text or list of texts.
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Token count(s)
        """
        if isinstance(text, list):
            return [self._count_single_text(t) for t in text]
        else:
            return self._count_single_text(text)
    
    def _count_single_text(self, text: str) -> int:
        """Count tokens in a single text string."""
        if not text or text is None:
            return 0
            
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
            except Exception as e:
                self.logger.warning(f"Tokenizer failed, using fallback: {e}")
                return self._fallback_count(text)
        else:
            return self._fallback_count(text)
    
    def _fallback_count(self, text: str) -> int:
        """Fallback token counting method."""
        # Simple word-based approximation
        # Typical rule: ~0.75 tokens per word for English text
        words = len(text.split())
        return max(1, int(words * 0.75))
    
    def estimate_cost(self, text: str, cost_per_1k_tokens: float = 0.002) -> float:
        """
        Estimate API cost for text processing.
        
        Args:
            text: Text to estimate cost for
            cost_per_1k_tokens: Cost per 1000 tokens (default: $0.002)
            
        Returns:
            Estimated cost in dollars
        """
        token_count = self.count_tokens(text)
        return (token_count / 1000.0) * cost_per_1k_tokens
    
    def chunk_text_by_tokens(self, text: str, max_tokens: int = 2000, overlap: int = 100) -> List[str]:
        """
        Chunk text by token count with overlap.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap: Token overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Simple sentence-based chunking for fallback
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

# Convenience function for direct usage
def count_tokens(text: Union[str, List[str]], model_name: str = "microsoft/DialoGPT-medium") -> Union[int, List[int]]:
    """
    Count tokens in text using HuggingFace tokenizer.
    
    Args:
        text: Text or list of texts to count tokens for
        model_name: HuggingFace model name for tokenizer
        
    Returns:
        Token count(s)
    """
    counter = TokenCounter(model_name)
    return counter.count_tokens(text) 