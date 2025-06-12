"""
Utilities Package

Contains utility modules for the PyNucleus system.
"""

# Import components with error handling
try:
    from .token_utils import count_tokens, estimate_cost, TokenCounter
except ImportError:
    count_tokens = None
    estimate_cost = None
    TokenCounter = None

try:
    from .logging_config import setup_logging
except ImportError:
    setup_logging = None

try:
    from .performance_analyzer import PerformanceAnalyzer
except ImportError:
    PerformanceAnalyzer = None

# Only export successfully imported components
__all__ = []
if count_tokens:
    __all__.extend(['count_tokens', 'estimate_cost'])
if TokenCounter:
    __all__.append('TokenCounter')
if setup_logging:
    __all__.append('setup_logging')
if PerformanceAnalyzer:
    __all__.append('PerformanceAnalyzer')
