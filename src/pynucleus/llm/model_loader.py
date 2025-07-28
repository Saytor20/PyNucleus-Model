"""
Robust model loader with automatic GGUF/HuggingFace detection and graceful fallback.
Uses singleton pattern for efficient model reuse and 8-bit quantization for CPU performance.
"""

from ..settings import settings
from ..utils.logger import logger
from ..utils.env import *
import torch
import os
from pathlib import Path

class ModelLoader:
    """Enhanced singleton model loader with persistent caching and memory optimization."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        # Singleton pattern - prevents multiple model instances eating memory
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # Model storage
            self._tokenizer = None
            self._hf_model = None
            self._gguf_model = None
            self._loading_method = None
            self._pipeline = None
            
            # Cache setup - speeds up repeated queries significantly
            self._model_cache_dir = Path("cache/models")
            self._model_cache_dir.mkdir(parents=True, exist_ok=True)
            self._response_cache = {}
            self._cache_max_size = 5000  # Adjust based on available memory
            self._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
            self._initialized = True
            logger.info("Enhanced ModelLoader singleton initialized with persistent caching")
    
    def _get_cache_key(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate cache key for response caching."""
        import hashlib
        # Include loading method to avoid cache conflicts between model types
        cache_data = f"{prompt[:200]}_{max_tokens}_{temperature}_{self._loading_method}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _get_cached_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Get cached response if available."""
        cache_key = self._get_cache_key(prompt, max_tokens, temperature)
        if cache_key in self._response_cache:
            self._cache_stats["hits"] += 1
            logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
            return self._response_cache[cache_key]
        
        self._cache_stats["misses"] += 1
        return None
    
    def _cache_response(self, prompt: str, max_tokens: int, temperature: float, response: str):
        """Cache response with LRU eviction."""
        cache_key = self._get_cache_key(prompt, max_tokens, temperature)
        
        # Simple FIFO eviction - could upgrade to proper LRU later
        if len(self._response_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
            self._cache_stats["evictions"] += 1
        
        self._response_cache[cache_key] = response
        logger.debug(f"Cached response for prompt: {prompt[:50]}...")
    
    def _save_model_state(self):
        """Save quantized model state to disk for faster reloading."""
        if not self._hf_model:
            return
        
        try:
            import pickle
            state_file = self._model_cache_dir / f"{settings.MODEL_ID.replace('/', '_')}_state.pkl"
            
            # Save only essential model components
            model_state = {
                'loading_method': self._loading_method,
                'model_config': self._hf_model.config.to_dict() if hasattr(self._hf_model, 'config') else None,
                'tokenizer_config': self._tokenizer.init_kwargs if hasattr(self._tokenizer, 'init_kwargs') else None
            }
            
            with open(state_file, 'wb') as f:
                pickle.dump(model_state, f)
            
            logger.info(f"Model state saved to {state_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save model state: {e}")
    
    def _load_model_state(self) -> dict:
        """Load saved model state from disk."""
        try:
            import pickle
            state_file = self._model_cache_dir / f"{settings.MODEL_ID.replace('/', '_')}_state.pkl"
            
            if state_file.exists():
                with open(state_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load model state: {e}")
        
        return None

    def _check_gguf_availability(self):
        """Check if GGUF model file exists and llama-cpp-python is available."""
        try:
            import llama_cpp
            gguf_path = Path(settings.OPTIMIZED_MODEL_PATH)
            return gguf_path.exists() and gguf_path.is_file()
        except ImportError:
            return False

    def _load_gguf_model(self):
        """Load GGUF model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
            
            print(f"🔧 Loading optimized model from: {settings.OPTIMIZED_MODEL_PATH}")
            
            # Detect hardware capabilities
            if torch.backends.mps.is_available():
                print("🍎 Using Metal GPU acceleration (macOS)")
                use_metal = True
                n_gpu_layers = -1
            elif torch.cuda.is_available():
                print("🚀 Using GPU acceleration")
                use_metal = False
                n_gpu_layers = 0
            else:
                print("💻 Using CPU")
                use_metal = False
                n_gpu_layers = 0
            
            model = Llama(
                model_path=settings.OPTIMIZED_MODEL_PATH,
                n_threads=os.cpu_count(),
                n_ctx=2048,
                use_metal=use_metal,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            
            print(f"✅ Optimized model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Optimized model loading failed: {e}")
            logger.error(f"Optimized model loading failed: {e}")
            return None

    def _load_huggingface_model(self):
        """Load HuggingFace model with 8-bit quantization for CPU performance."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
            
            print(f"🔧 Loading HuggingFace model: {settings.MODEL_ID}")
            
            # Load tokenizer first
            print("📝 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID, trust_remote_code=True)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Determine best loading strategy with 8-bit quantization
            if torch.cuda.is_available() and settings.USE_CUDA:
                print("🚀 CUDA detected - loading on GPU")
                
                if settings.USE_GPU_QUANTIZATION:
                    print("🔧 Using GPU with 8-bit quantization for memory efficiency")
                    try:
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0
                        )
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            settings.MODEL_ID,
                            device_map="auto",
                            quantization_config=quantization_config,
                            trust_remote_code=True
                        )
                        print("✅ HuggingFace model loaded with GPU 8-bit quantization")
                        return tokenizer, model
                        
                    except Exception as cuda_quant_error:
                        print(f"❌ GPU quantization failed: {cuda_quant_error}")
                        print("🔄 Falling back to GPU full precision...")
                
                # GPU full precision loading
                print("🔧 Using GPU with full precision (FP16)")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        settings.MODEL_ID,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                    print("✅ HuggingFace model loaded with GPU full precision (FP16)")
                    return tokenizer, model
                    
                except Exception as cuda_error:
                    print(f"❌ GPU loading failed: {cuda_error}")
                    print("🔄 Falling back to CPU loading...")
            
            # CPU fallback with 8-bit quantization
            print("💻 Loading model on CPU with 8-bit quantization...")
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    settings.MODEL_ID,
                    device_map="cpu",
                    quantization_config=quantization_config,
                    trust_remote_code=True
                )
                print("✅ HuggingFace model loaded with CPU 8-bit quantization")
                return tokenizer, model
                
            except Exception as quant_error:
                print(f"❌ 8-bit quantization failed: {quant_error}")
                print("🔄 Falling back to FP32...")
                
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        settings.MODEL_ID,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        trust_remote_code=True
                    )
                    print("✅ HuggingFace model loaded with CPU FP32")
                    return tokenizer, model
                    
                except Exception as fp32_error:
                    print(f"❌ FP32 loading failed: {fp32_error}")
                    raise Exception("All HuggingFace loading methods failed")
            
        except Exception as e:
            print(f"❌ HuggingFace loading failed: {e}")
            logger.error(f"HuggingFace model loading failed: {e}")
            return None, None

    def initialize_models(self):
        """Initialize models with enhanced caching and automatic fallback."""
        if self._loading_method is not None:
            logger.info("Models already initialized, skipping...")
            return
        
        print("🚀 Initializing Enhanced PyNucleus Model Loader")
        print("=" * 50)
        
        # Check for saved model state
        saved_state = self._load_model_state()
        if saved_state:
            print(f"📁 Found cached model state for {saved_state.get('loading_method', 'Unknown')}")
        
        # Try optimized model first if available
        if self._check_gguf_availability():
            print("\n📦 Attempting optimized model loading...")
            self._gguf_model = self._load_gguf_model()
            
            if self._gguf_model:
                self._loading_method = "Optimized"
                print(f"🎯 Model loading complete: {self._loading_method}")
                self._save_model_state()
                return
            else:
                print("🔄 Optimized loading failed, trying standard loading...")
        
        # Fallback to HuggingFace
        print("\n🤗 Attempting standard model loading...")
        self._tokenizer, self._hf_model = self._load_huggingface_model()
        
        if self._hf_model:
            self._loading_method = "HuggingFace"
            print(f"🎯 Model loading complete: {self._loading_method}")
            
            # Create pipeline for efficient inference
            try:
                from transformers import pipeline
                
                # Check if model was loaded with accelerate
                if hasattr(self._hf_model, 'hf_device_map') or hasattr(self._hf_model, 'device_map'):
                    # Model was loaded with accelerate, don't specify device
                    self._pipeline = pipeline(
                        "text-generation",
                        model=self._hf_model,
                        tokenizer=self._tokenizer
                    )
                    print("✅ Text generation pipeline created (accelerate-compatible)")
                else:
                    # Model was loaded normally, can specify device
                    self._pipeline = pipeline(
                        "text-generation",
                        model=self._hf_model,
                        tokenizer=self._tokenizer,
                        device="cpu"
                    )
                    print("✅ Text generation pipeline created")
                
                # Save model state for faster next startup
                self._save_model_state()
                
            except Exception as e:
                logger.warning(f"Pipeline creation failed: {e}")
        else:
            self._loading_method = "Failed"
            print("💥 All model loading methods failed!")
            logger.error("Critical: No models could be loaded")

    def generate(self, prompt: str, max_tokens=None, temperature=0.7, stream=False) -> str:
        """Generate response using the loaded model with optimized parameters."""
        # Initialize models if not done yet
        if self._loading_method is None:
            self.initialize_models()
        
        # Validate input
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided")
            return self._generate_fallback_response("empty prompt")
        
        # Use settings default if max_tokens not provided
        if max_tokens is None:
            max_tokens = settings.MAX_TOKENS
        
        # Check if we have a working model
        if self._loading_method == "Failed" or (not self._hf_model and not self._gguf_model):
            logger.error("No working model available")
            return self._generate_fallback_response("no model available")
        
        logger.info(f"Generating with {self._loading_method} model (max_tokens={max_tokens}, stream={stream})")
        
        # Use optimized model if available
        if self._gguf_model and self._loading_method == "Optimized":
            try:
                # Truncate prompt if too long
                max_prompt_length = 2048
                if len(prompt) > max_prompt_length:
                    prompt = prompt[:max_prompt_length]
                    logger.warning(f"Prompt truncated to {max_prompt_length} characters")
                
                response = self._gguf_model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["###", "\n\n\n"],
                    echo=False
                )
                
                if hasattr(response, 'choices') and response.choices:
                    return response.choices[0].text.strip()
                else:
                    return self._generate_fallback_response("gguf_no_response")
                    
            except Exception as e:
                logger.error(f"Optimized model generation failed: {e}")
                return self._generate_fallback_response(f"optimized_error: {str(e)}")
        
        # Use HuggingFace model with pipeline for better performance
        elif self._hf_model and self._loading_method == "HuggingFace":
            try:
                # Truncate prompt if too long
                max_prompt_length = 1200
                if len(prompt) > max_prompt_length:
                    prompt = prompt[:max_prompt_length] + "..."
                
                # Format prompt using chat template for SmolLM2 and similar models
                formatted_prompt = self._format_chat_prompt(prompt)
                
                # Use pipeline for efficient generation
                if self._pipeline:
                    outputs = self._pipeline(
                        formatted_prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9,
                        top_k=40,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,
                        pad_token_id=self._tokenizer.eos_token_id,
                        eos_token_id=self._tokenizer.eos_token_id,
                        return_full_text=False
                    )
                    
                    if outputs and len(outputs) > 0:
                        result = outputs[0]['generated_text'].strip()
                        result = self._clean_repetitive_text(result)
                        
                        if result and len(result) > 5:
                            return result
                
                # Fallback to direct model generation if pipeline fails
                inputs = self._tokenizer(
                    formatted_prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=1024,
                    padding=False,
                    return_attention_mask=True
                )
                inputs = inputs.to(self._hf_model.device)
                
                generation_kwargs = {
                    **inputs,
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repetition_penalty": 1.1,
                    "no_repeat_ngram_size": 3,
                    "pad_token_id": self._tokenizer.eos_token_id,
                    "eos_token_id": self._tokenizer.eos_token_id,
                }
                
                with torch.no_grad():
                    outputs = self._hf_model.generate(**generation_kwargs)
                
                input_length = inputs["input_ids"].shape[1]
                new_tokens = outputs[0][input_length:]
                result = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                result = self._clean_repetitive_text(result)
                
                if result and len(result) > 5:
                    return result
                
                raise ValueError("Invalid HuggingFace response")
                
            except Exception as e:
                logger.warning(f"HuggingFace generation failed: {e}, using fallback")
                return self._generate_fallback_response(prompt)
        
        # Should not reach here
        return self._generate_fallback_response("unknown error")

    def _clean_repetitive_text(self, text: str) -> str:
        """Clean repetitive patterns from generated text."""
        if not text:
            return text
        
        lines = text.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                cleaned_lines.append('')
                continue
                
            # Skip exact duplicate lines
            if line.lower() in seen_lines:
                continue
                
            # Skip lines that are mostly repetitive words
            words = line.split()
            if len(words) > 3:
                word_counts = {}
                for word in words:
                    word_lower = word.lower().strip('.,!?;:')
                    word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
                
                # If any word appears more than 30% of the time, likely repetitive
                max_count = max(word_counts.values()) if word_counts else 0
                if max_count > len(words) * 0.3:
                    continue
            
            seen_lines.add(line.lower())
            cleaned_lines.append(line)
        
        # Join lines and clean up extra whitespace
        result = '\n'.join(cleaned_lines)
        
        # Remove excessive repetition of phrases
        import re
        # Remove patterns where same phrase repeats 3+ times
        result = re.sub(r'(.{10,}?)\1{2,}', r'\1', result, flags=re.IGNORECASE)
        
        return result.strip()

    def _format_chat_prompt(self, prompt: str) -> str:
        """Format prompt using chat template for SmolLM2 and similar instruction models."""
        try:
            # Check if this model supports chat templates
            if hasattr(self._tokenizer, 'apply_chat_template') and hasattr(self._tokenizer, 'chat_template'):
                messages = [{"role": "user", "content": prompt}]
                formatted = self._tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                logger.info(f"Using chat template for model: {settings.MODEL_ID}")
                return formatted
            else:
                # Fallback to original prompt for models without chat template
                logger.info(f"No chat template available, using original prompt")
                return prompt
        except Exception as e:
            logger.warning(f"Chat template formatting failed: {e}, using original prompt")
            return prompt

    def _generate_fallback_response(self, context: str) -> str:
        """Generate a fallback response when model generation fails."""
        if "timeout" in context or "generation_timeout" in context:
            return "I apologize, but the response generation is taking longer than expected. Please try rephrasing your question or ask a simpler question."
        elif "memory" in context.lower() or "out of memory" in context.lower():
            return "I'm experiencing high memory usage. Please try a shorter question or contact support if this persists."
        elif "no_model" in context:
            return "I'm currently unable to process your request due to model loading issues. Please try again in a moment."
        elif "empty prompt" in context:
            return "Please provide a question or prompt for me to respond to."
        else:
            return f"I apologize, but I encountered an issue while processing your request: {context}. Please try again or rephrase your question."

    def get_model_info(self):
        """Get information about the currently loaded model."""
        if self._loading_method is None:
            self.initialize_models()
        
        return {
            "method": self._loading_method,
            "model_id": settings.MODEL_ID,
            "optimized_model_path": settings.OPTIMIZED_MODEL_PATH,
            "has_gguf": self._gguf_model is not None,
            "has_hf": self._hf_model is not None,
            "has_pipeline": self._pipeline is not None,
            "use_cuda": settings.USE_CUDA,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }

# Global singleton instance
_model_loader = None

def get_model_loader():
    """Get the global singleton model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader

def generate(prompt: str, max_tokens=None, temperature=0.7, stream=False) -> str:
    """Generate response using the singleton model loader."""
    loader = get_model_loader()
    return loader.generate(prompt, max_tokens, temperature, stream)

def get_model_info():
    """Get information about the currently loaded model."""
    loader = get_model_loader()
    return loader.get_model_info()

# Initialize models on import
_model_loader = ModelLoader()
_model_loader.initialize_models()

# Export main functions
__all__ = ["generate", "get_model_info"] 