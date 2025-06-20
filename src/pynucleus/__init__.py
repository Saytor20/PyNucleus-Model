"""
PyNucleus unified package.

A chemical process simulation and RAG (Retrieval-Augmented Generation) system
that integrates DWSIM simulations with LLM-powered analysis and querying.
"""

# Set environment variables early to prevent telemetry
import os
os.environ.setdefault("CHROMA_TELEMETRY", "FALSE")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("POSTHOG_DISABLED", "1")

__version__ = "0.1.0"
__author__ = "PyNucleus Contributors"
__email__ = ""
__description__ = "Chemical process simulation and RAG system"

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
] 