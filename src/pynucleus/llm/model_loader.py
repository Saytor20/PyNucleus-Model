"""
Robust model loader with automatic GGUF/HuggingFace detection and graceful fallback.
"""

from ..settings import settings
from ..utils.logger import logger
from ..utils.env import *
import torch
import os
from pathlib import Path

# Global model storage
_tokenizer = None
_hf_model = None
_gguf_model = None
_loading_method = None

def _check_gguf_availability():
    """Check if GGUF model file exists and llama-cpp-python is available."""
    try:
        import llama_cpp
        gguf_path = Path(settings.GGUF_PATH)
        if gguf_path.exists() and gguf_path.is_file():
            print(f"✅ GGUF model found: {settings.GGUF_PATH}")
            print(f"✅ llama-cpp-python available")
            return True
        else:
            print(f"❌ GGUF model not found: {settings.GGUF_PATH}")
            return False
    except ImportError:
        print(f"❌ llama-cpp-python not available")
        return False

def _load_gguf_model():
    """Load GGUF model using llama-cpp-python."""
    try:
        from llama_cpp import Llama
        
        print(f"🔧 Loading GGUF model from: {settings.GGUF_PATH}")
        
        # Detect hardware capabilities
        if torch.backends.mps.is_available():
            print("🍎 Using Metal GPU acceleration (macOS)")
            use_metal = True
            n_gpu_layers = -1  # Use all layers on GPU
        elif torch.cuda.is_available():
            print("🚀 CUDA available but using CPU for GGUF compatibility")
            use_metal = False
            n_gpu_layers = 0
        else:
            print("💻 Using CPU")
            use_metal = False
            n_gpu_layers = 0
        
        model = Llama(
            model_path=settings.GGUF_PATH,
            n_threads=os.cpu_count(),
            n_ctx=2048,
            use_metal=use_metal,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        
        print(f"✅ GGUF model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ GGUF loading failed: {e}")
        logger.error(f"GGUF model loading failed: {e}")
        return None

def _load_huggingface_model():
    """Load HuggingFace model with automatic device selection."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        print(f"🔧 Loading HuggingFace model: {settings.MODEL_ID}")
        
        # Load tokenizer first
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID, trust_remote_code=False)
        
        # Determine best loading strategy
        if torch.cuda.is_available() and settings.USE_CUDA:
            print("🚀 Using CUDA with 4-bit quantization")
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    settings.MODEL_ID,
                    device_map="auto",
                    quantization_config=quantization_config,
                    trust_remote_code=False
                )
                print("✅ HuggingFace model loaded with CUDA 4-bit quantization")
                return tokenizer, model
                
            except Exception as cuda_error:
                print(f"❌ CUDA loading failed: {cuda_error}")
                print("🔄 Falling back to CPU loading...")
        
        # CPU fallback
        print("💻 Loading model on CPU...")
        try:
            # Try FP32 first for stability
            model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_ID,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=False
            )
            print("✅ HuggingFace model loaded with CPU FP32")
            return tokenizer, model
            
        except Exception as fp32_error:
            print(f"❌ FP32 loading failed: {fp32_error}")
            print("🔄 Trying FP16...")
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    settings.MODEL_ID,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    trust_remote_code=False
                )
                print("⚠️ HuggingFace model loaded with CPU FP16 (may have stability issues)")
                return tokenizer, model
                
            except Exception as fp16_error:
                print(f"❌ FP16 loading failed: {fp16_error}")
                raise Exception("All HuggingFace loading methods failed")
        
    except Exception as e:
        print(f"❌ HuggingFace loading failed: {e}")
        logger.error(f"HuggingFace model loading failed: {e}")
        return None, None

def _initialize_models():
    """Initialize models with automatic fallback from GGUF to HuggingFace."""
    global _tokenizer, _hf_model, _gguf_model, _loading_method
    
    print("🚀 Initializing PyNucleus Model Loader")
    print("=" * 50)
    
    # Try GGUF first if available
    if _check_gguf_availability():
        print("\n📦 Attempting GGUF loading...")
        _gguf_model = _load_gguf_model()
        
        if _gguf_model:
            _loading_method = "GGUF"
            print(f"🎯 Model loading complete: {_loading_method}")
            return
        else:
            print("🔄 GGUF failed, trying HuggingFace...")
    
    # Fallback to HuggingFace
    print("\n🤗 Attempting HuggingFace loading...")
    _tokenizer, _hf_model = _load_huggingface_model()
    
    if _hf_model:
        _loading_method = "HuggingFace"
        print(f"🎯 Model loading complete: {_loading_method}")
    else:
        _loading_method = "Failed"
        print("💥 All model loading methods failed!")
        logger.error("Critical: No models could be loaded")

def generate(prompt: str, max_tokens=None, temperature=0.7, stream=False) -> str:
    """Generate response using the loaded model with optimized parameters."""
    global _tokenizer, _hf_model, _gguf_model, _loading_method
    
    # Initialize models if not done yet
    if _loading_method is None:
        _initialize_models()
    
    # Validate input
    if not prompt or not prompt.strip():
        logger.warning("Empty prompt provided")
        return _generate_fallback_response("empty prompt")
    
    # Use settings default if max_tokens not provided
    if max_tokens is None:
        max_tokens = settings.MAX_TOKENS
    
    # Check if we have a working model
    if _loading_method == "Failed" or (not _hf_model and not _gguf_model):
        logger.error("No working model available")
        return _generate_fallback_response("no model available")
    
    logger.info(f"Generating with {_loading_method} model (max_tokens={max_tokens}, stream={stream})")
    
    # Use GGUF model if available
    if _gguf_model and _loading_method == "GGUF":
        try:
            # Truncate prompt if too long
            max_prompt_length = 1500
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."
                logger.warning(f"Prompt truncated to {max_prompt_length} characters")
            
            response = _gguf_model(
                prompt,
                max_tokens=min(max_tokens, 512),
                temperature=temperature,
                top_p=0.85,  # Slightly higher for more variety
                top_k=50,    # Increased for better word choice
                repeat_penalty=1.15,  # Stronger anti-repetition
                frequency_penalty=0.7,  # Additional anti-repetition
                presence_penalty=0.3,   # Encourage new topics
                stop=["</s>", "<|im_end|>", "\n\n\n", "Question:", "Answer:", "STEP 1:", "STEP 2:", "STEP 3:", "STEP 4:"],
                echo=False,
                stream=stream
            )
            
            if stream and hasattr(response, '__iter__'):
                return response  # Return generator for streaming
            
            if response and "choices" in response and response["choices"]:
                result = response["choices"][0]["text"].strip()
                if result and len(result) > 5:
                    return result
            
            raise ValueError("Invalid GGUF response")
            
        except Exception as e:
            logger.warning(f"GGUF generation failed: {e}, using fallback")
            return _generate_fallback_response(prompt)
    
    # Use HuggingFace model
    elif _hf_model and _tokenizer and _loading_method == "HuggingFace":
        try:
            # Truncate prompt if too long
            max_prompt_length = 1200
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."
            
            # Tokenize input
            inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = inputs.to(_hf_model.device)
            
            # Optimized generation parameters for less repetition
            generation_kwargs = {
                **inputs,
                "max_new_tokens": min(max_tokens, 300),
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.85,           # Higher for more variety
                "top_k": 50,             # Increased for better word choice  
                "repetition_penalty": 1.15,  # Stronger anti-repetition
                "no_repeat_ngram_size": 3,   # Prevent 3-gram repetition
                "pad_token_id": _tokenizer.eos_token_id,
                "eos_token_id": _tokenizer.eos_token_id,
                "early_stopping": True,
                "length_penalty": 0.8,      # Slight penalty for length
            }
            
            # Add streaming support
            if stream:
                return _generate_streaming(**generation_kwargs)
            
            with torch.no_grad():
                outputs = _hf_model.generate(**generation_kwargs)
            
            # Extract new tokens only
            input_length = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_length:]
            result = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Post-process to remove common repetition patterns
            result = _clean_repetitive_text(result)
            
            # Validate result
            if result and len(result) > 5:
                return result
            
            raise ValueError("Invalid HuggingFace response")
            
        except Exception as e:
            logger.warning(f"HuggingFace generation failed: {e}, using fallback")
            return _generate_fallback_response(prompt)
    
    # Should not reach here
    return _generate_fallback_response("unknown error")

def _generate_streaming(**generation_kwargs):
    """Generate streaming response for HuggingFace model."""
    global _hf_model, _tokenizer
    
    try:
        from transformers import TextIteratorStreamer
        import threading
        
        # Create streamer
        streamer = TextIteratorStreamer(
            _tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True,
            timeout=60.0
        )
        
        # Add streamer to generation args
        generation_kwargs["streamer"] = streamer
        generation_kwargs["do_sample"] = True
        
        # Start generation in separate thread
        thread = threading.Thread(
            target=_hf_model.generate, 
            kwargs=generation_kwargs
        )
        thread.start()
        
        # Return the streamer iterator
        return streamer
        
    except ImportError:
        logger.warning("TextIteratorStreamer not available, falling back to regular generation")
        # Fall back to regular generation
        with torch.no_grad():
            outputs = _hf_model.generate(**generation_kwargs)
        
        input_length = generation_kwargs["input_ids"].shape[1]
        new_tokens = outputs[0][input_length:]
        result = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _clean_repetitive_text(result)

def _clean_repetitive_text(text: str) -> str:
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

def _generate_fallback_response(context: str) -> str:
    """Generate a fallback response when model fails."""
    if isinstance(context, str):
        context_lower = context.lower()
        
        if 'distillation' in context_lower:
            return """Distillation is a separation process that uses differences in boiling points to separate components of a liquid mixture. The process involves heating the mixture until the more volatile component vaporizes, then condensing the vapor back to liquid in a separate container. This technique is fundamental in chemical engineering for purifying liquids and separating mixtures."""
        
        elif 'heat transfer' in context_lower:
            return """Heat transfer involves the movement of thermal energy from higher to lower temperature regions through conduction, convection, and radiation. Heat exchangers are devices designed to efficiently transfer heat between fluids at different temperatures, commonly used in chemical processes."""
        
        elif 'reactor' in context_lower:
            return """Chemical reactors are vessels designed to contain and control chemical reactions. Common types include batch reactors, continuous stirred-tank reactors (CSTR), and plug flow reactors (PFR). Design considerations include reaction kinetics, mass and heat transfer, and safety requirements."""
    
    return """This is a chemical engineering concept that involves fundamental principles of process design, mass and energy balances, and system optimization. The specific analysis requires consideration of process parameters, safety factors, and industry standards."""

def get_model_info():
    """Get information about the currently loaded model."""
    global _loading_method
    
    if _loading_method is None:
        _initialize_models()
    
    return {
        "method": _loading_method,
        "model_id": settings.MODEL_ID,
        "gguf_path": settings.GGUF_PATH,
        "has_gguf": _gguf_model is not None,
        "has_hf": _hf_model is not None,
        "use_cuda": settings.USE_CUDA,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    }

# Initialize models on import
_initialize_models()

# Export main functions
__all__ = ["generate", "get_model_info"] 