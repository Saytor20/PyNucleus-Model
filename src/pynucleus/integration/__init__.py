"""
PyNucleus Integration Module

Enhanced integration components for DWSIM-RAG integration and advanced analytics.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from .config_manager import ConfigManager
    from .dwsim_rag_integrator import DWSIMRAGIntegrator
    from .llm_output_generator import LLMOutputGenerator
    from .dwsim_data_integrator import DWSIMDataIntegrator
    from .settings import *
    
    __all__ = [
        "ConfigManager",
        "DWSIMRAGIntegrator", 
        "LLMOutputGenerator",
        "DWSIMDataIntegrator"
    ]
    
except ImportError as e:
    print(f"Warning: Some integration components not available: {e}")
    __all__ = [] 