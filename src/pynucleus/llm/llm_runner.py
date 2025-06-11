"""
LLMRunner: A minimal class for streamlined querying of Hugging Face LLM models.

This module provides the LLMRunner class that simplifies text generation using
pre-trained language models from Hugging Face's transformers library.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Union
import warnings


class LLMRunner:
    """
    A minimal class to streamline querying Hugging Face LLM models.
    
    This class provides an easy-to-use interface for text generation with proper
    handling of padding tokens, end-of-sequence tokens, and device selection.
    
    Attributes:
        model_id (str): The Hugging Face model identifier
        device (str): Device to run the model on ('cpu' or 'cuda')
        tokenizer: The loaded tokenizer
        model: The loaded model
    """
    
    def __init__(self, 
                 model_id: str = "gpt2",
                 device: str = "cpu"):
        """
        Initialize LLMRunner with specified model and device.
        
        Args:
            model_id (str): HuggingFace model identifier. 
                          Defaults to "gpt2" (lightweight, widely available)
            device (str): Device to run model on ("cpu" or "cuda").
                        Defaults to "cpu"
        
        Raises:
            ValueError: If device is not "cpu" or "cuda"
            RuntimeError: If model fails to load
        """
        if device not in ['cpu', 'cuda']:
            raise ValueError("Device must be either 'cpu' or 'cuda'")
        
        if device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
        
        self.model_id = model_id
        self.device = device
        self.tokenizer = None
        self.model = None
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """
        Load the tokenizer and model from Hugging Face.
        
        This method handles the initialization of both the tokenizer and model,
        with proper device placement and error handling.
        """
        try:
            print(f"Loading tokenizer for {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Handle models without pad token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    # Add a pad token if neither exists
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            print(f"Loading model {self.model_id} on {self.device}...")
            
            # Load model with appropriate torch dtype for efficiency
            torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map=self.device if self.device == 'cuda' else None,
                trust_remote_code=True  # Some models may require this
            )
            
            if self.device == 'cpu':
                self.model = self.model.to(self.device)
            
            # Resize model embeddings if we added a pad token
            if self.tokenizer.pad_token == '[PAD]':
                try:
                    vocab_size = self.tokenizer.vocab_size
                except AttributeError:
                    vocab_size = len(self.tokenizer.get_vocab())
                self.model.resize_token_embeddings(vocab_size)
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_id}: {str(e)}")
    
    def ask(self, 
            prompt: str, 
            max_length: int = 100, 
            temperature: float = 0.7,
            do_sample: bool = True,
            top_p: float = 0.9,
            top_k: int = 50,
            num_return_sequences: int = 1) -> Union[str, List[str]]:
        """
        Generate text response given a prompt.
        
        This method provides an easy-to-use interface for text generation with
        sensible defaults and proper handling of special tokens.
        
        Args:
            prompt (str): Input text prompt for generation
            max_length (int): Maximum length of generated text including prompt.
                            Defaults to 100
            temperature (float): Sampling temperature (0.0 = greedy, higher = more random).
                               Defaults to 0.7
            do_sample (bool): Whether to use sampling or greedy decoding.
                            Defaults to True
            top_p (float): Nucleus sampling parameter. Defaults to 0.9
            top_k (int): Top-k sampling parameter. Defaults to 50
            num_return_sequences (int): Number of sequences to generate.
                                      Defaults to 1
        
        Returns:
            Union[str, List[str]]: Generated text(s). Returns a single string if
                                 num_return_sequences=1, otherwise returns a list
        
        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt is empty or parameters are invalid
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Please reinitialize LLMRunner.")
        
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        
        if num_return_sequences <= 0:
            raise ValueError("num_return_sequences must be positive")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length
            )
            
            # Move inputs to the same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p if do_sample else None,
                    top_k=top_k if do_sample else None,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode generated text
            generated_texts = []
            input_length = inputs['input_ids'].shape[1]
            
            for output in outputs:
                # Extract only the newly generated tokens (exclude input prompt)
                generated_tokens = output[input_length:]
                generated_text = self.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                generated_texts.append(generated_text.strip())
            
            # Return single string if only one sequence, otherwise return list
            if num_return_sequences == 1:
                return generated_texts[0]
            else:
                return generated_texts
                
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including model_id, device, vocabulary size,
                 and model parameters count
        """
        if self.model is None:
            return {"status": "Model not loaded"}
        
        param_count = sum(p.numel() for p in self.model.parameters())
        
        # Safely get vocabulary size
        vocab_size = None
        if self.tokenizer:
            try:
                vocab_size = self.tokenizer.vocab_size
            except AttributeError:
                try:
                    vocab_size = len(self.tokenizer.get_vocab())
                except:
                    vocab_size = "Unknown"
        
        return {
            "model_id": self.model_id,
            "device": self.device,
            "vocab_size": vocab_size,
            "parameters": param_count,
            "parameters_human": f"{param_count / 1e6:.1f}M" if param_count < 1e9 else f"{param_count / 1e9:.1f}B",
            "pad_token": self.tokenizer.pad_token if self.tokenizer else None,
            "eos_token": self.tokenizer.eos_token if self.tokenizer else None,
            "max_length": self.tokenizer.model_max_length if self.tokenizer else None
        }
    
    def __repr__(self) -> str:
        """String representation of the LLMRunner."""
        return f"LLMRunner(model_id='{self.model_id}', device='{self.device}')" 