from ..settings import settings
from ..utils.logger import logger
import os, torch

def _try_cuda_bitsandbytes():
    try:
        if not settings.USE_CUDA or not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                  BitsAndBytesConfig)

        tok = AutoTokenizer.from_pretrained(settings.MODEL_ID, trust_remote_code=False)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype="float16")
        mod = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_ID, device_map="auto",
            quantization_config=quant, trust_remote_code=False
        )
        logger.success("Loaded 4-bit model with bitsandbytes/CUDA")
        return tok, mod
    except Exception as e:
        logger.warning(f"CUDA/bitsandbytes path failed: {e}")
        return None, None

def _try_metal_llama_cpp():
    try:
        from llama_cpp import Llama
        if not os.path.exists(settings.GGUF_PATH):
            raise FileNotFoundError(f"GGUF file missing: {settings.GGUF_PATH}")
        llm = Llama(model_path=settings.GGUF_PATH, n_threads=os.cpu_count(),
                    use_metal=True, n_gpu_layers=1, n_ctx=4096)
        logger.success("Loaded GGUF via llama-cpp (Metal/CPU)")
        return llm
    except Exception as e:
        logger.warning(f"llama-cpp path failed: {e}")
        return None

# ---- public API --------------------------------------------------------
_tokenizer, _hf_model = _try_cuda_bitsandbytes()
_llama_cpp_model      = None if _hf_model else _try_metal_llama_cpp()

if not (_hf_model or _llama_cpp_model):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID, trust_remote_code=False)
    _hf_model  = AutoModelForCausalLM.from_pretrained(
        settings.MODEL_ID, torch_dtype=torch.float16, device_map=None,
        trust_remote_code=False
    ).to("cpu")
    logger.warning("Falling back to FP16 CPU model (slow)")

def generate(prompt: str, max_tokens=256, temperature=0.7) -> str:
    if _hf_model:
        inputs = _tokenizer(prompt, return_tensors="pt").to(_hf_model.device)
        ids = _hf_model.generate(**inputs, max_new_tokens=max_tokens,
                                 temperature=temperature)
        return _tokenizer.decode(ids[0], skip_special_tokens=True)

    # llama-cpp route
    out = _llama_cpp_model(prompt, max_tokens=max_tokens, temperature=temperature)
    return out["choices"][0]["text"].strip() 