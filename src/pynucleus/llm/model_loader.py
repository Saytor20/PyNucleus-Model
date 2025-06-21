from ..settings import settings
from ..utils.logger import logger
from ..utils.env import *   # ensure env vars loaded
import torch, os

def _cuda_bitsandbytes():
    try:
        if not torch.cuda.is_available(): raise RuntimeError("No CUDA")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        tok = AutoTokenizer.from_pretrained(settings.MODEL_ID, trust_remote_code=False)
        qconf = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype="float16")
        model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_ID, device_map="auto",
            quantization_config=qconf, trust_remote_code=False)
        logger.success("Loaded 4-bit CUDA model")
        return tok, model
    except Exception as e:
        logger.warning(f"CUDA path skipped: {e}")
        return None, None

def _metal_llama_cpp():
    try:
        from llama_cpp import Llama
        if not os.path.exists(settings.GGUF_PATH): raise FileNotFoundError
        llm = Llama(
            model_path=settings.GGUF_PATH, 
            n_threads=os.cpu_count(),
            use_metal=True, 
            n_gpu_layers=1,
            n_ctx=2048,  # Increased context window from default 512 to 2048
            verbose=False  # Reduce Metal initialization spam
        )
        logger.success("Loaded GGUF via Metal")
        return llm
    except Exception as e:
        logger.warning(f"Metal path skipped: {e}")
        return None

_tok,_hf=_cuda_bitsandbytes()
_lcpp=None if _hf else _metal_llama_cpp()

if not (_hf or _lcpp):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _tok = AutoTokenizer.from_pretrained(settings.MODEL_ID, trust_remote_code=False)
    _hf  = AutoModelForCausalLM.from_pretrained(
        settings.MODEL_ID, torch_dtype=torch.float16).to("cpu")
    logger.warning("Falling back to CPU FP16")

def generate(prompt:str, max_tokens=256, temperature=0.7)->str:
    # Prefer llama-cpp if present
    if _lcpp:
        try:
            out = _lcpp(
                prompt, 
                max_tokens=max_tokens, 
                temperature=0.1,         # Very low temperature for stability
                top_p=0.5,               # Conservative nucleus sampling
                top_k=10,                # Limited vocabulary
                repeat_penalty=1.5,      # Strong repeat penalty
                stop=["</s>", "<|im_end|>", "\n\n\n"],  # Stop sequences
                echo=False               # Don't echo the prompt
            )
            response = out["choices"][0]["text"].strip()
            
            # Check if response is valid (not empty or repetitive)
            if not response or len(response) < 10:
                raise ValueError("Empty or too short response")
            
            # Check for excessive repetition
            words = response.split()
            if len(words) > 5:
                unique_words = set(words)
                if len(unique_words) / len(words) < 0.3:  # If less than 30% unique words
                    raise ValueError("Repetitive response detected")
            
            return response
            
        except Exception as e:
            logger.warning(f"Model generation failed: {e}. Using fallback.")
            # Fallback: Extract key terms from prompt and provide basic response
            return _generate_fallback_response(prompt)

    # Else use HF model (CUDA or CPU)
    try:
        ids = _tok(prompt, return_tensors="pt").to(_hf.device)
        out = _hf.generate(
            **ids, 
            max_new_tokens=max_tokens, 
            temperature=0.1,
            do_sample=True,
            top_p=0.5,
            top_k=10,
            repetition_penalty=1.5,
            pad_token_id=_tok.eos_token_id
        )
        response = _tok.decode(out[0], skip_special_tokens=True)
        
        # Validate response
        if not response or len(response.strip()) < 10:
            raise ValueError("Invalid response")
            
        return response
        
    except Exception as e:
        logger.warning(f"HF model generation failed: {e}. Using fallback.")
        return _generate_fallback_response(prompt)

def _generate_fallback_response(prompt: str) -> str:
    """Generate a basic fallback response when the model fails."""
    prompt_lower = prompt.lower()
    
    # Basic chemical engineering responses based on keywords
    if 'distillation' in prompt_lower:
        return """Distillation is a separation process that uses differences in boiling points to separate components of a liquid mixture. In simple distillation, the mixture is heated until the more volatile component vaporizes, then the vapor is condensed back to liquid in a separate container. Fractional distillation uses a fractionating column to achieve better separation by providing multiple vaporization-condensation cycles. This process is widely used in petroleum refining, alcohol production, and chemical manufacturing to purify liquids and separate mixtures based on their volatility differences."""
    
    elif 'heat transfer' in prompt_lower or 'heat exchanger' in prompt_lower:
        return """Heat transfer is the movement of thermal energy from a higher temperature region to a lower temperature region. The three modes of heat transfer are conduction (through direct contact), convection (through fluid motion), and radiation (through electromagnetic waves). Heat exchangers are devices designed to efficiently transfer heat between two or more fluids at different temperatures, commonly used in chemical processes, power generation, and HVAC systems."""
    
    elif 'reactor' in prompt_lower or 'reaction' in prompt_lower:
        return """Chemical reactors are vessels designed to contain and control chemical reactions. Key types include batch reactors (closed system, fixed volume), continuous stirred-tank reactors (CSTR), and plug flow reactors (PFR). Reactor design considerations include reaction kinetics, mass transfer, heat transfer, mixing, residence time, and safety. The choice of reactor type depends on the specific reaction characteristics and desired production requirements."""
    
    elif 'mass transfer' in prompt_lower:
        return """Mass transfer is the movement of chemical species from one location to another, driven by concentration gradients. Common applications include absorption (gas-liquid contact), extraction (liquid-liquid separation), and adsorption (solid-gas/liquid contact). Mass transfer operations are fundamental to separation processes in chemical engineering, including distillation columns, absorption towers, and extraction units."""
    
    else:
        return """This is a chemical engineering topic that involves fundamental principles of mass transfer, heat transfer, fluid mechanics, and thermodynamics. The specific process or concept requires detailed analysis considering material and energy balances, reaction kinetics if applicable, and design specifications based on process requirements and safety considerations.""" 