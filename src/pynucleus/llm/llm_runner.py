"""
LLM runner for PyNucleus system using HuggingFace models.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

class LLMRunner:
    """Run LLM models using HuggingFace transformers."""
    
    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct", device: str = "auto"):
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
            
            # Improved tokenization with attention mask
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=min(max_length - 50, 1024),  # Leave room for generation
                return_attention_mask=True
            )
            
            # Move to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response with proper attention mask
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2  # Reduce repetition
                )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from response
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):].strip()
            
            # Enhanced cleaning to remove conversation history contamination
            response_text = self._clean_conversation_artifacts(response_text)
            
            # Clean up response - remove training artifacts and contamination
            response_lines = response_text.split('\n')
            clean_lines = []
            
            for line in response_lines:
                line = line.strip()
                # Skip common training artifacts and conversation patterns
                if any(artifact in line.lower() for artifact in [
                    'you are an ai assistant',
                    'human:',
                    'assistant:',
                    'answer:',
                    'you will be given a task',
                    'must complete the task',
                    'i am an ai',
                    'as an ai',
                    'user:',
                    'system:'
                ]):
                    break  # Stop at first training artifact
                
                if line and not line.startswith('#'):  # Keep non-empty, non-comment lines
                    clean_lines.append(line)
            
            response_text = ' '.join(clean_lines)
            
            # Additional cleaning for conversation contamination
            response_text = self._remove_conversation_contamination(response_text)
            
            # Limit response length to avoid contamination
            if len(response_text) > 1000:
                sentences = response_text.split('.')
                response_text = '. '.join(sentences[:3]) + '.'
            
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
    
    def _clean_conversation_artifacts(self, text: str) -> str:
        """
        Clean conversation artifacts from model response.
        
        Args:
            text: Raw response text
            
        Returns:
            Cleaned text with conversation artifacts removed
        """
        import re
        
        # Remove conversation patterns at the start of text
        # Match patterns like "Human: ... Assistant: ..." or similar
        conversation_pattern = r'^.*?(?:Human|User|System):\s*.*?(?:Assistant|AI|Bot):\s*'
        text = re.sub(conversation_pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove conversation patterns that appear anywhere in the text
        # This catches mid-text and end-text contamination
        mid_conversation_pattern = r'\s+Human:\s*[^.]*(?:Answer:|Assistant:)\s*[^.]*\.?'
        text = re.sub(mid_conversation_pattern, '', text, flags=re.IGNORECASE)
        
        # Remove standalone conversation markers
        conversation_markers = [
            r'\bHuman:\s*[^.!?]*[.!?]?\s*',
            r'\bAssistant:\s*', 
            r'\bUser:\s*[^.!?]*[.!?]?\s*',
            r'\bSystem:\s*[^.!?]*[.!?]?\s*',
            r'\bAI:\s*',
            r'\bBot:\s*',
            r'\bAnswer:\s*(?:Sure|Yes|Certainly)?\s*'
        ]
        
        for marker in conversation_markers:
            text = re.sub(marker, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _remove_conversation_contamination(self, text: str) -> str:
        """
        Remove conversation contamination that may appear mid-response.
        
        Args:
            text: Response text to clean
            
        Returns:
            Cleaned text with conversation contamination removed
        """
        import re
        
        # First, try to find where conversation contamination starts and cut there
        contamination_start = re.search(
            r'\s+(?:Human|User|Assistant|AI):\s*', 
            text, 
            re.IGNORECASE
        )
        
        if contamination_start:
            # Cut off everything from the contamination point
            clean_text = text[:contamination_start.start()].strip()
            # Make sure we have substantial content before cutting
            if len(clean_text) > 50:
                return clean_text
        
        # If no contamination found or cut would be too aggressive, do line-by-line cleaning
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Check if line contains conversation patterns
            if re.search(r'\b(?:Human|User|Assistant|AI):\s*', line, re.IGNORECASE):
                # If we find conversation pattern, stop processing here
                break
            
            # Check for mid-sentence conversation contamination
            # Look for patterns like "text Human: more text" 
            conversation_contamination = re.search(
                r'(.+?)\s+(?:Human|User|Assistant|AI):\s+(.+)', 
                line, 
                re.IGNORECASE
            )
            
            if conversation_contamination:
                # Keep only the part before the contamination
                clean_part = conversation_contamination.group(1).strip()
                if clean_part and len(clean_part) > 10:  # Only keep substantial content
                    clean_lines.append(clean_part)
                break  # Stop processing after contamination found
            else:
                clean_lines.append(line)
        
        cleaned_text = ' '.join(clean_lines).strip()
        
        # Final cleanup - remove any remaining conversation artifacts at the end
        cleaned_text = re.sub(r'\s+(?:Human|User|Assistant|AI):\s*.*$', '', cleaned_text, flags=re.IGNORECASE)
        
        return cleaned_text
    
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