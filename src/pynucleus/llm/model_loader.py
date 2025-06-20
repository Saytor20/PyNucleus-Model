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
        llm = Llama(model_path=settings.GGUF_PATH, n_threads=os.cpu_count(),
                    use_metal=True, n_gpu_layers=1)
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
        out = _lcpp(prompt, max_tokens=max_tokens, temperature=temperature)
        return out["choices"][0]["text"].strip()

    # Else use HF model (CUDA or CPU)
    ids = _tok(prompt, return_tensors="pt").to(_hf.device)
    out = _hf.generate(**ids, max_new_tokens=max_tokens, temperature=temperature)
    return _tok.decode(out[0], skip_special_tokens=True) 