"""
LLM runner for PyNucleus system using HuggingFace models.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

class LLMRunner:
    """Run LLM models using HuggingFace transformers."""
    
    def __init__(self, model_id: str = "microsoft/DialoGPT-medium", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the LLM model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.logger.info(f"Loading model: {self.model_id}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device if self.device != "auto" else None
            )
            
            # Move to device if specified
            if self.device != "auto" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.logger.info(f"Model loaded successfully on device: {self.model.device}")
            
        except ImportError as e:
            self.logger.error(f"Required dependencies not available: {e}")
            self.logger.error("Please install: pip install torch transformers")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_id}: {e}")
            raise
    
    def ask(
        self,
        question: str,
        max_length: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Simple interface to ask the LLM a question and get back just the response text.
        
        Args:
            question: The question to ask
            max_length: Maximum length of response
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Response text string
        """
        try:
            result = self.generate_response(
                prompt=question,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample
            )
            return result.get('response', 'No response generated')
        except Exception as e:
            self.logger.error(f"Ask method failed: {e}")
            return f"Error: {e}"

    def generate_response(
        self, 
        prompt: str, 
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Dict[str, Any]:
        """
        Generate response from the LLM.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not properly initialized")
        
        try:
            import torch
            
            start_time = datetime.now()
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Move to same device as model
            inputs = inputs.to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from response
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):].strip()
            
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            result = {
                "response": response_text,
                "model_id": self.model_id,
                "prompt": prompt,
                "generation_time": generation_time,
                "timestamp": end_time.isoformat(),
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "do_sample": do_sample,
                    "top_p": top_p,
                    "top_k": top_k
                }
            }
            
            self.logger.info(f"Generated response in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return {
                "response": f"Error generating response: {e}",
                "model_id": self.model_id,
                "prompt": prompt,
                "generation_time": 0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def batch_generate(
        self, 
        prompts: List[str], 
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **generation_kwargs: Arguments for generate_response
            
        Returns:
            List of response dictionaries
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.generate_response(prompt, **generation_kwargs)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {"error": "Model not initialized"}
        
        try:
            import torch
            
            return {
                "model_id": self.model_id,
                "device": str(self.model.device),
                "dtype": str(self.model.dtype),
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
                "memory_footprint": self.model.get_memory_footprint() if hasattr(self.model, 'get_memory_footprint') else "Unknown"
            }
        except Exception as e:
            return {"error": f"Failed to get model info: {e}"} 