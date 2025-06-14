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
    from pynucleus.integration.config_manager import ConfigManager
    from pynucleus.integration.dwsim_rag_integrator import DWSIMRAGIntegrator
    from pynucleus.integration.llm_output_generator import LLMOutputGenerator
    from pynucleus.integration.dwsim_data_integrator import DWSIMDataIntegrator
    from pynucleus.integration.settings import *
    
    __all__ = [
        "ConfigManager",
        "DWSIMRAGIntegrator", 
        "LLMOutputGenerator",
        "DWSIMDataIntegrator"
    ]
    
except ImportError as e:
    print(f"Warning: Some integration components not available: {e}")
    __all__ = []