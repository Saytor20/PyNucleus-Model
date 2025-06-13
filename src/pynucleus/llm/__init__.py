"""
PyNucleus LLM Module

Language model utilities and query interfaces.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from .llm_runner import LLMRunner
    from .query_llm import LLMQueryManager, quick_ask_llm
    
    __all__ = [
        "LLMRunner",
        "LLMQueryManager", 
        "quick_ask_llm"
    ]
    
except ImportError as e:
    print(f"Warning: Some LLM components not available: {e}")
__all__ = []