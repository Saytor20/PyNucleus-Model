# At top, before transformers imports:
from ..utils.env import *

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ..settings import settings
from ..utils.logger import logger

# Lazy loading - models will be loaded on first use
tokenizer = None
model = None

def _ensure_model_loaded():
    """Load model and tokenizer on first use"""
    global tokenizer, model
    
    if tokenizer is None or model is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID, trust_remote_code=False)
            
            # Try 4-bit quantization first
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype="float16"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    settings.MODEL_ID,
                    device_map="auto",
                    quantization_config=quant_config,
                    trust_remote_code=False
                )
                logger.info(f"Quantized 4-bit model loaded: {settings.MODEL_ID}")
            except ImportError:
                # Fallback to standard loading if bitsandbytes not available
                logger.warning("bitsandbytes not available, falling back to standard loading")
                model = AutoModelForCausalLM.from_pretrained(
                    settings.MODEL_ID,
                    device_map="auto",
                    trust_remote_code=False
                )
                logger.info(f"Standard model loaded: {settings.MODEL_ID}")
        except Exception as e:
            logger.error(f"Model load failure: {e}")
            raise

def generate(prompt: str, temp=0.7, max_tokens=None) -> str:
    _ensure_model_loaded()
    max_tokens = max_tokens or settings.MAX_TOKENS
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        # Move to same device as model if possible
        if hasattr(model, 'device'):
            inputs = inputs.to(model.device)
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            temperature=temp,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"Error generating response: {e}" 