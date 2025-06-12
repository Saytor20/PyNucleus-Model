"""
PyNucleus Pipeline Module

Main pipeline components for RAG, DWSIM, and enhanced integration.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from .pipeline_rag import RAGPipeline
    from .pipeline_dwsim import DWSIMPipeline  
    from .pipeline_export import ResultsExporter
    from .pipeline_utils import PipelineUtils
    from .enhanced_pipeline_utils import EnhancedPipelineUtils
    
    __all__ = [
        "RAGPipeline",
        "DWSIMPipeline", 
        "ResultsExporter",
        "PipelineUtils",
        "EnhancedPipelineUtils"
    ]
    
except ImportError as e:
    print(f"Warning: Some pipeline components not available: {e}")
    # Provide minimal exports for backward compatibility
    __all__ = [] 