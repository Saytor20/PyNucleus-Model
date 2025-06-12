"""
PyNucleus Pipeline Package

This package contains modular pipeline components for the PyNucleus project.
"""

# Import components with error handling
try:
    from .pipeline_rag import RAGPipeline
except ImportError:
    RAGPipeline = None

try:
    from .pipeline_dwsim import DWSIMPipeline
except ImportError:
    DWSIMPipeline = None

try:
    from .pipeline_export import ResultsExporter
except ImportError:
    ResultsExporter = None

try:
    from .pipeline_utils import PipelineUtils
except ImportError:
    PipelineUtils = None

# Only export successfully imported components
__all__ = []
if RAGPipeline:
    __all__.append('RAGPipeline')
if DWSIMPipeline:
    __all__.append('DWSIMPipeline')
if ResultsExporter:
    __all__.append('ResultsExporter')
if PipelineUtils:
    __all__.append('PipelineUtils') 