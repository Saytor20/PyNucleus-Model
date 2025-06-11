"""
Logging Configuration Module

Provides centralized logging setup for PyNucleus pipeline operations.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logging(
    debug: bool = False,
    log_file: Optional[Path] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Set up logging configuration for PyNucleus pipeline.
    
    Args:
        debug: If True, set logging level to DEBUG, otherwise INFO
        log_file: Optional file path to write logs to
        log_format: Format string for log messages
        
    Returns:
        Configured logger instance
    """
    
    # Set logging level
    level = logging.DEBUG if debug else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create PyNucleus specific logger
    logger = logging.getLogger('pynucleus')
    logger.setLevel(level)
    
    if debug:
        logger.info("🔧 Debug logging enabled")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'pynucleus.{name}')

def create_log_file_path(base_dir: str = "logs") -> Path:
    """
    Create a timestamped log file path.
    
    Args:
        base_dir: Base directory for log files
        
    Returns:
        Path to log file with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_dir)
    log_dir.mkdir(exist_ok=True)
    return log_dir / f"pynucleus_{timestamp}.log" 