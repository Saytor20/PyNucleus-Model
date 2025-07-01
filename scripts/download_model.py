#!/usr/bin/env python3
"""
Download and cache the Qwen 2.5 1.5B Instruct model with cross-platform compatibility.
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pynucleus.settings import settings
from pynucleus.utils.logger import logger

def download_model():
    """Download and cache the Qwen 2.5 1.5B Instruct model with cross-platform compatibility"""
    model_id = settings.MODEL_ID
    logger.info(f"Downloading Qwen 2.5 1.5B Instruct model: {model_id}")
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        logger.success("✓ Tokenizer downloaded successfully")
        
        # Download model with cross-platform settings
        logger.info("Downloading Qwen 2.5 1.5B Instruct model weights...")
        try:
            # Try FP32 first for cross-platform stability
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=False,
                device_map="cpu"
            )
            logger.success("✓ Qwen 2.5 1.5B Instruct model downloaded with FP32 (cross-platform stable)")
        except Exception as e:
            logger.warning(f"FP32 download failed: {e}, trying auto dtype")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                trust_remote_code=False,
                device_map="cpu"
            )
            logger.success("✓ Qwen 2.5 1.5B Instruct model downloaded with auto dtype")
        
        # Test basic functionality with stable parameters
        logger.info("Testing Qwen 2.5 1.5B Instruct model...")
        test_input = tokenizer("Hello world", return_tensors="pt")
        
        # Use stable generation parameters
        with torch.no_grad():
            output = model.generate(
                **test_input, 
                max_new_tokens=10, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        test_output = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.success(f"✓ Qwen 2.5 1.5B Instruct model test successful: {test_output}")
        
        logger.success("Qwen 2.5 1.5B Instruct model download and setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Qwen 2.5 1.5B Instruct model download failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("Downloading Qwen 2.5 1.5B Instruct model...")
    success = download_model()
    if success:
        print("\n✓ Qwen 2.5 1.5B Instruct model ready for use!")
    else:
        print("\n✗ Qwen 2.5 1.5B Instruct model download failed. Check logs for details.")
        sys.exit(1) 