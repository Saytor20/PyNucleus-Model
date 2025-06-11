"""
Unit tests for LLMRunner class.

This module provides comprehensive tests for the LLMRunner class including
initialization, text generation, error handling, and edge cases.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import pytest
from src.pynucleus.llm.llm_runner import LLMRunner


class TestLLMRunner(unittest.TestCase):
    """Test cases for LLMRunner class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock transformers to avoid actual model loading in tests
        self.tokenizer_mock = MagicMock()
        self.model_mock = MagicMock()
        
        # Set up tokenizer mock
        self.tokenizer_mock.pad_token = '[PAD]'
        self.tokenizer_mock.eos_token = '</s>'
        self.tokenizer_mock.pad_token_id = 1
        self.tokenizer_mock.eos_token_id = 2
        self.tokenizer_mock.model_max_length = 512
        self.tokenizer_mock.__len__ = Mock(return_value=50000)
        
        # Set up model mock
        param1 = torch.tensor([1, 2, 3])
        param2 = torch.tensor([[1, 2], [3, 4]])
        self.model_mock.parameters.return_value = [param1, param2]
    
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    @patch('src.pynucleus.llm.llm_runner.AutoModelForCausalLM')
    def test_init_default_parameters(self, mock_model_class, mock_tokenizer_class):
        """Test LLMRunner initialization with default parameters."""
        mock_tokenizer_class.from_pretrained.return_value = self.tokenizer_mock
        mock_model_class.from_pretrained.return_value = self.model_mock
        
        runner = LLMRunner()
        
        self.assertEqual(runner.model_id, "tiiuae/Falcon3-1B-Base")
        self.assertEqual(runner.device, "cpu")
        self.assertIsNotNone(runner.tokenizer)
        self.assertIsNotNone(runner.model)
    
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    @patch('src.pynucleus.llm.llm_runner.AutoModelForCausalLM')
    def test_init_custom_parameters(self, mock_model_class, mock_tokenizer_class):
        """Test LLMRunner initialization with custom parameters."""
        mock_tokenizer_class.from_pretrained.return_value = self.tokenizer_mock
        mock_model_class.from_pretrained.return_value = self.model_mock
        
        custom_model = "gpt2"
        runner = LLMRunner(model_id=custom_model, device="cpu")
        
        self.assertEqual(runner.model_id, custom_model)
        self.assertEqual(runner.device, "cpu")
    
    def test_init_invalid_device(self):
        """Test LLMRunner initialization with invalid device."""
        with self.assertRaises(ValueError) as context:
            LLMRunner(device="invalid")
        
        self.assertIn("Device must be either 'cpu' or 'cuda'", str(context.exception))
    
    @patch('src.pynucleus.llm.llm_runner.torch.cuda.is_available')
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    @patch('src.pynucleus.llm.llm_runner.AutoModelForCausalLM')
    def test_init_cuda_fallback_to_cpu(self, mock_model_class, mock_tokenizer_class, mock_cuda_available):
        """Test CUDA fallback to CPU when CUDA is not available."""
        mock_cuda_available.return_value = False
        mock_tokenizer_class.from_pretrained.return_value = self.tokenizer_mock
        mock_model_class.from_pretrained.return_value = self.model_mock
        
        with patch('warnings.warn') as mock_warn:
            runner = LLMRunner(device="cuda")
            mock_warn.assert_called_once()
            self.assertEqual(runner.device, "cpu")
    
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    @patch('src.pynucleus.llm.llm_runner.AutoModelForCausalLM')
    def test_tokenizer_no_pad_token(self, mock_model_class, mock_tokenizer_class):
        """Test handling of tokenizer without pad token."""
        # Mock tokenizer without pad token
        tokenizer_no_pad = MagicMock()
        tokenizer_no_pad.pad_token = None
        tokenizer_no_pad.eos_token = '</s>'
        tokenizer_no_pad.model_max_length = 512
        tokenizer_no_pad.__len__ = Mock(return_value=50000)
        
        mock_tokenizer_class.from_pretrained.return_value = tokenizer_no_pad
        mock_model_class.from_pretrained.return_value = self.model_mock
        
        runner = LLMRunner()
        
        # Should set pad_token to eos_token
        self.assertEqual(runner.tokenizer.pad_token, runner.tokenizer.eos_token)
    
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    @patch('src.pynucleus.llm.llm_runner.AutoModelForCausalLM')
    def test_ask_basic_functionality(self, mock_model_class, mock_tokenizer_class):
        """Test basic ask functionality."""
        mock_tokenizer_class.from_pretrained.return_value = self.tokenizer_mock
        mock_model_class.from_pretrained.return_value = self.model_mock
        
        # Mock tokenizer call
        self.tokenizer_mock.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock model generate
        self.model_mock.generate.return_value = [torch.tensor([1, 2, 3, 4, 5])]
        
        # Mock decode
        self.tokenizer_mock.decode.return_value = "generated text"
        
        runner = LLMRunner()
        result = runner.ask("Hello, how are you?")
        
        self.assertEqual(result, "generated text")
        self.model_mock.generate.assert_called_once()
    
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    @patch('src.pynucleus.llm.llm_runner.AutoModelForCausalLM')
    def test_ask_multiple_sequences(self, mock_model_class, mock_tokenizer_class):
        """Test ask with multiple return sequences."""
        mock_tokenizer_class.from_pretrained.return_value = self.tokenizer_mock
        mock_model_class.from_pretrained.return_value = self.model_mock
        
        # Mock tokenizer call
        self.tokenizer_mock.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock model generate for multiple sequences
        self.model_mock.generate.return_value = [
            torch.tensor([1, 2, 3, 4, 5]),
            torch.tensor([1, 2, 3, 6, 7])
        ]
        
        # Mock decode calls
        self.tokenizer_mock.decode.side_effect = ["first response", "second response"]
        
        runner = LLMRunner()
        result = runner.ask("Hello", num_return_sequences=2)
        
        self.assertEqual(result, ["first response", "second response"])
        self.assertEqual(len(result), 2)
    
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    @patch('src.pynucleus.llm.llm_runner.AutoModelForCausalLM')
    def test_ask_empty_prompt(self, mock_model_class, mock_tokenizer_class):
        """Test ask with empty prompt."""
        mock_tokenizer_class.from_pretrained.return_value = self.tokenizer_mock
        mock_model_class.from_pretrained.return_value = self.model_mock
        
        runner = LLMRunner()
        
        with self.assertRaises(ValueError) as context:
            runner.ask("")
        
        self.assertIn("Prompt cannot be empty", str(context.exception))
        
        with self.assertRaises(ValueError):
            runner.ask("   ")  # Only whitespace
    
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    @patch('src.pynucleus.llm.llm_runner.AutoModelForCausalLM')
    def test_ask_invalid_parameters(self, mock_model_class, mock_tokenizer_class):
        """Test ask with invalid parameters."""
        mock_tokenizer_class.from_pretrained.return_value = self.tokenizer_mock
        mock_model_class.from_pretrained.return_value = self.model_mock
        
        runner = LLMRunner()
        
        with self.assertRaises(ValueError):
            runner.ask("Hello", max_length=0)
        
        with self.assertRaises(ValueError):
            runner.ask("Hello", max_length=-1)
        
        with self.assertRaises(ValueError):
            runner.ask("Hello", num_return_sequences=0)
    
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    @patch('src.pynucleus.llm.llm_runner.AutoModelForCausalLM')
    def test_ask_with_custom_parameters(self, mock_model_class, mock_tokenizer_class):
        """Test ask with custom generation parameters."""
        mock_tokenizer_class.from_pretrained.return_value = self.tokenizer_mock
        mock_model_class.from_pretrained.return_value = self.model_mock
        
        # Mock tokenizer call
        self.tokenizer_mock.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock model generate
        self.model_mock.generate.return_value = [torch.tensor([1, 2, 3, 4, 5])]
        self.tokenizer_mock.decode.return_value = "custom response"
        
        runner = LLMRunner()
        result = runner.ask(
            "Test prompt",
            max_length=150,
            temperature=0.5,
            do_sample=False,
            top_p=0.8,
            top_k=40
        )
        
        # Verify generate was called with custom parameters
        call_args = self.model_mock.generate.call_args
        self.assertEqual(call_args.kwargs['max_length'], 150)
        self.assertEqual(call_args.kwargs['temperature'], 0.5)
        self.assertEqual(call_args.kwargs['do_sample'], False)
        self.assertEqual(call_args.kwargs['top_p'], None)  # Should be None when do_sample=False
        self.assertEqual(call_args.kwargs['top_k'], None)  # Should be None when do_sample=False
    
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    @patch('src.pynucleus.llm.llm_runner.AutoModelForCausalLM')
    def test_get_model_info(self, mock_model_class, mock_tokenizer_class):
        """Test get_model_info method."""
        mock_tokenizer_class.from_pretrained.return_value = self.tokenizer_mock
        mock_model_class.from_pretrained.return_value = self.model_mock
        
        runner = LLMRunner()
        info = runner.get_model_info()
        
        expected_keys = [
            'model_id', 'device', 'vocab_size', 'parameters', 
            'parameters_human', 'pad_token', 'eos_token', 'max_length'
        ]
        
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['model_id'], "gpt2")
        self.assertEqual(info['device'], "cpu")
        self.assertEqual(info['vocab_size'], 50257)
        self.assertEqual(info['parameters'], 7)
        self.assertEqual(info['parameters_human'], "7.0M")
    
    def test_get_model_info_no_model(self):
        """Test get_model_info when model is not loaded."""
        # Create runner without loading model
        runner = LLMRunner.__new__(LLMRunner)
        runner.model = None
        
        info = runner.get_model_info()
        self.assertEqual(info, {"status": "Model not loaded"})
    
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    @patch('src.pynucleus.llm.llm_runner.AutoModelForCausalLM')
    def test_repr(self, mock_model_class, mock_tokenizer_class):
        """Test string representation of LLMRunner."""
        mock_tokenizer_class.from_pretrained.return_value = self.tokenizer_mock
        mock_model_class.from_pretrained.return_value = self.model_mock
        
        runner = LLMRunner()
        repr_str = repr(runner)
        
        self.assertEqual(repr_str, "LLMRunner(model_id='gpt2', device='cpu')")
    
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    def test_model_loading_failure(self, mock_tokenizer_class):
        """Test handling of model loading failure."""
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Model not found")
        
        with self.assertRaises(RuntimeError) as context:
            LLMRunner()
        
        self.assertIn("Failed to load model", str(context.exception))
    
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    @patch('src.pynucleus.llm.llm_runner.AutoModelForCausalLM')
    def test_ask_unloaded_model(self, mock_model_class, mock_tokenizer_class):
        """Test ask method when model is not loaded."""
        mock_tokenizer_class.from_pretrained.return_value = self.tokenizer_mock
        mock_model_class.from_pretrained.return_value = self.model_mock
        
        runner = LLMRunner()
        # Simulate unloaded model
        runner.model = None
        runner.tokenizer = None
        
        with self.assertRaises(RuntimeError) as context:
            runner.ask("Hello")
        
        self.assertIn("Model not loaded", str(context.exception))
    
    @patch('src.pynucleus.llm.llm_runner.AutoTokenizer')
    @patch('src.pynucleus.llm.llm_runner.AutoModelForCausalLM')
    def test_ask_generation_failure(self, mock_model_class, mock_tokenizer_class):
        """Test ask method when generation fails."""
        mock_tokenizer_class.from_pretrained.return_value = self.tokenizer_mock
        mock_model_class.from_pretrained.return_value = self.model_mock
        
        # Mock tokenizer call
        self.tokenizer_mock.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock model generate to raise exception
        self.model_mock.generate.side_effect = Exception("Generation failed")
        
        runner = LLMRunner()
        
        with self.assertRaises(RuntimeError) as context:
            runner.ask("Hello")
        
        self.assertIn("Text generation failed", str(context.exception))


class TestLLMRunnerIntegration(unittest.TestCase):
    """Integration tests for LLMRunner (these require actual model downloads)."""
    
    @unittest.skip("Requires model download - enable for integration testing")
    def test_real_model_loading(self):
        """Test loading a real small model (requires internet connection)."""
        # Use a very small model for testing
        runner = LLMRunner(model_id="sshleifer/tiny-gpt2", device="cpu")
        
        self.assertIsNotNone(runner.model)
        self.assertIsNotNone(runner.tokenizer)
        
        # Test basic generation
        result = runner.ask("Hello", max_length=20)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
    
    @unittest.skip("Requires model download - enable for integration testing")
    def test_cuda_device(self):
        """Test CUDA device usage (requires CUDA-capable device)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        runner = LLMRunner(model_id="sshleifer/tiny-gpt2", device="cuda")
        
        self.assertEqual(runner.device, "cuda")
        self.assertTrue(next(runner.model.parameters()).is_cuda)


if __name__ == '__main__':
    unittest.main() 