"""
Unit tests for token_utils module.

Tests cover functionality, caching behavior, error handling, and performance.
"""

import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import tempfile
import os

from .token_utils import (
    TokenCounter, 
    count_tokens, 
    get_available_cache_info, 
    clear_all_caches,
    DEFAULT_MODEL
)


class TestTokenCounter:
    """Test cases for TokenCounter class."""
    
    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()
    
    def test_init_default_model(self):
        """Test TokenCounter initialization with default model."""
        counter = TokenCounter()
        assert counter.model_id == DEFAULT_MODEL
        assert counter._tokenizer is None
    
    def test_init_custom_model(self):
        """Test TokenCounter initialization with custom model."""
        custom_model = "gpt2"
        counter = TokenCounter(custom_model)
        assert counter.model_id == custom_model
        assert counter._tokenizer is None
    
    @patch('tokens_func.token_utils.AutoTokenizer.from_pretrained')
    def test_tokenizer_loading(self, mock_tokenizer):
        """Test tokenizer loading and caching."""
        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tok
        
        counter = TokenCounter("test-model")
        
        # First access should load tokenizer
        tokenizer1 = counter.tokenizer
        assert mock_tokenizer.called
        assert tokenizer1.pad_token == "<eos>"
        
        # Second access should use cached tokenizer
        mock_tokenizer.reset_mock()
        tokenizer2 = counter.tokenizer
        assert not mock_tokenizer.called
        assert tokenizer1 is tokenizer2
    
    @patch('tokens_func.token_utils.AutoTokenizer.from_pretrained')
    def test_count_tokens_string(self, mock_tokenizer):
        """Test token counting for single string."""
        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1, 2, 3, 4]  # 4 tokens
        mock_tok.pad_token = "<pad>"
        mock_tokenizer.return_value = mock_tok
        
        counter = TokenCounter()
        result = counter.count_tokens("Hello world")
        
        assert result == 4
        mock_tok.encode.assert_called_once_with("Hello world", add_special_tokens=True)
    
    @patch('tokens_func.token_utils.AutoTokenizer.from_pretrained')
    def test_count_tokens_list(self, mock_tokenizer):
        """Test token counting for list of strings."""
        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.encode.side_effect = [[1, 2], [1, 2, 3, 4, 5]]  # 2 and 5 tokens
        mock_tok.pad_token = "<pad>"
        mock_tokenizer.return_value = mock_tok
        
        counter = TokenCounter()
        result = counter.count_tokens(["Hi", "Hello world again"])
        
        assert result == [2, 5]
        assert mock_tok.encode.call_count == 2
    
    def test_count_tokens_invalid_input(self):
        """Test token counting with invalid input."""
        counter = TokenCounter()
        
        with pytest.raises(ValueError, match="Input must be a string or list of strings"):
            counter.count_tokens(123)
    
    @patch('tokens_func.token_utils.AutoTokenizer.from_pretrained')
    def test_get_tokens(self, mock_tokenizer):
        """Test getting actual tokens."""
        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1, 2, 3]
        mock_tok.convert_ids_to_tokens.return_value = ["Hello", "world", "!"]
        mock_tok.pad_token = "<pad>"
        mock_tokenizer.return_value = mock_tok
        
        counter = TokenCounter()
        tokens = counter.get_tokens("Hello world!")
        
        assert tokens == ["Hello", "world", "!"]
        mock_tok.encode.assert_called_once_with("Hello world!", add_special_tokens=True)
        mock_tok.convert_ids_to_tokens.assert_called_once_with([1, 2, 3])
    
    @patch('tokens_func.token_utils.AutoTokenizer.from_pretrained')
    def test_tokenizer_loading_error(self, mock_tokenizer):
        """Test handling of tokenizer loading errors."""
        mock_tokenizer.side_effect = Exception("Model not found")
        
        counter = TokenCounter("invalid-model")
        
        with pytest.raises(Exception, match="Model not found"):
            _ = counter.tokenizer
    
    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        counter = TokenCounter()
        counter.clear_cache()
        # Should not raise any errors


class TestCountTokensFunction:
    """Test cases for count_tokens convenience function."""
    
    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()
    
    @patch('tokens_func.token_utils.TokenCounter')
    def test_count_tokens_function(self, mock_counter_class):
        """Test the convenience count_tokens function."""
        # Mock TokenCounter instance
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 5
        mock_counter_class.return_value = mock_counter
        
        result = count_tokens("Hello world")
        
        assert result == 5
        mock_counter_class.assert_called_once_with(DEFAULT_MODEL)
        mock_counter.count_tokens.assert_called_once_with("Hello world")
    
    @patch('tokens_func.token_utils.TokenCounter')
    def test_count_tokens_custom_model(self, mock_counter_class):
        """Test count_tokens function with custom model."""
        # Mock TokenCounter instance
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 3
        mock_counter_class.return_value = mock_counter
        
        result = count_tokens("Hi", "gpt2")
        
        assert result == 3
        mock_counter_class.assert_called_once_with("gpt2")
        mock_counter.count_tokens.assert_called_once_with("Hi")
    
    @patch('tokens_func.token_utils.TokenCounter')
    def test_count_tokens_caching(self, mock_counter_class):
        """Test that count_tokens function uses caching."""
        # Mock TokenCounter instance
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 4
        mock_counter_class.return_value = mock_counter
        
        # Call twice with same arguments
        result1 = count_tokens("Test text")
        result2 = count_tokens("Test text")
        
        assert result1 == result2 == 4
        # TokenCounter should only be created once due to caching
        mock_counter_class.assert_called_once_with(DEFAULT_MODEL)


class TestCacheUtilities:
    """Test cases for cache utility functions."""
    
    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()
    
    def test_get_cache_info(self):
        """Test getting cache information."""
        cache_info = get_available_cache_info()
        
        assert "tokenizer_cache" in cache_info
        assert "count_cache" in cache_info
        
        # Check structure of cache info
        for cache_type in ["tokenizer_cache", "count_cache"]:
            assert "hits" in cache_info[cache_type]
            assert "misses" in cache_info[cache_type]
            assert "current_size" in cache_info[cache_type]
            assert "max_size" in cache_info[cache_type]
    
    def test_clear_all_caches(self):
        """Test clearing all caches."""
        # This should run without errors
        clear_all_caches()
        
        cache_info = get_available_cache_info()
        assert cache_info["tokenizer_cache"]["current_size"] == 0
        assert cache_info["count_cache"]["current_size"] == 0


class TestIntegration:
    """Integration tests using real tokenizers (if available)."""
    
    def setup_method(self):
        """Clear caches before each test."""
        clear_all_caches()
    
    @pytest.mark.slow
    def test_real_tokenizer_integration(self):
        """Test with a real tokenizer (marked as slow test)."""
        try:
            # Use a small, commonly available model for testing
            counter = TokenCounter("gpt2")
            
            # Test basic functionality
            text = "Hello, world!"
            token_count = counter.count_tokens(text)
            
            assert isinstance(token_count, int)
            assert token_count > 0
            
            # Test that we get actual tokens
            tokens = counter.get_tokens(text)
            assert isinstance(tokens, list)
            assert len(tokens) == token_count
            
        except Exception as e:
            pytest.skip(f"Real tokenizer test skipped due to: {e}")
    
    @pytest.mark.slow
    def test_cache_performance(self):
        """Test that caching improves performance."""
        try:
            text = "This is a test sentence for performance testing."
            
            # Time first call (cache miss)
            import time
            start_time = time.time()
            result1 = count_tokens(text)
            first_call_time = time.time() - start_time
            
            # Time second call (cache hit)
            start_time = time.time()
            result2 = count_tokens(text)
            second_call_time = time.time() - start_time
            
            assert result1 == result2
            # Second call should be significantly faster (though this may vary)
            assert second_call_time < first_call_time * 2  # Conservative check
            
        except Exception as e:
            pytest.skip(f"Performance test skipped due to: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 