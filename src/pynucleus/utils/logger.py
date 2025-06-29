import logging
import sys
from typing import Any
from ..settings import settings

# Simple rich-based logger that's more reliable than loguru
try:
    from rich.console import Console
    from rich.logging import RichHandler
    
    console = Console()
    
    # Configure standard logging with rich handler
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=True, show_path=False)]
    )
    
    logger = logging.getLogger("pynucleus")
    
    # Add success method for compatibility
    def success(message: str):
        logger.info(f"✅ {message}")
    
    logger.success = success
    
except ImportError:
    # Fallback to standard logging if rich is not available
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    logger = logging.getLogger("pynucleus") 
    
    # Add success method for compatibility
    def success(message: str):
        logger.info(f"✅ {message}")
    
    logger.success = success

# Export get_logger for compatibility

def get_logger(name: str = None) -> logging.Logger:
    """Return the configured logger instance (optionally with a custom name)."""
    if name:
        return logging.getLogger(name)
    return logger 