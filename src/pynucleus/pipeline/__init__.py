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

# Import pipeline components with proper error handling
__all__ = []

try:
    from .pipeline_rag import RAGPipeline
    __all__.append("RAGPipeline")
except ImportError as e:
    print(f"Warning: RAG pipeline not available: {e}")

try:
    from .pipeline_dwsim import DWSIMPipeline
    __all__.append("DWSIMPipeline")
except ImportError as e:
    print(f"Warning: DWSIM pipeline not available: {e}")

try:
    from .pipeline_export import ResultsExporter
    __all__.append("ResultsExporter")
except ImportError as e:
    print(f"Warning: Results exporter not available: {e}")

try:
    from .pipeline_utils import PipelineUtils
    __all__.append("PipelineUtils")
except ImportError as e:
    print(f"Warning: Pipeline utilities not available: {e}")

try:
    from .enhanced_pipeline_utils import EnhancedPipelineUtils
    __all__.append("EnhancedPipelineUtils")
except ImportError as e:
    print(f"Warning: Enhanced pipeline utilities not available: {e}")