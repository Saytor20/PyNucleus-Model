"""
Logging configuration for PyNucleus system.
"""

import logging
import logging.config
import sys
import platform
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logging(
    debug: bool = False,
    log_file: Optional[Path] = None,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging configuration for PyNucleus.
    
    Args:
        debug: Enable debug level logging
        log_file: Custom log file path
        console_output: Enable console logging
        file_output: Enable file logging
        
    Returns:
        Root logger instance
    """
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Set log file
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"pynucleus_{timestamp}.log"
    
    # Set logging level
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler
    if file_output:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    
    # Log initial setup message
    root_logger.info("=" * 60)
    root_logger.info("PyNucleus Logging System Initialized")
    root_logger.info(f"Log Level: {logging.getLevelName(log_level)}")
    root_logger.info(f"Console Output: {console_output}")
    root_logger.info(f"File Output: {file_output}")
    if file_output:
        root_logger.info(f"Log File: {log_file}")
    root_logger.info("=" * 60)
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_system_info(logger: logging.Logger):
    """
    Log comprehensive system information.
    
    Args:
        logger: Logger instance to use
    """
    try:
        logger.info("System Information:")
        logger.info(f"  Platform: {platform.platform()}")
        logger.info(f"  Python Version: {platform.python_version()}")
        logger.info(f"  Python Implementation: {platform.python_implementation()}")
        logger.info(f"  Architecture: {platform.architecture()[0]}")
        logger.info(f"  Processor: {platform.processor()}")
        logger.info(f"  Working Directory: {Path.cwd()}")
        
        # Log Python path
        logger.debug(f"  Python Path: {sys.path}")
        
        # Log memory info if available
        try:
            import psutil
            memory = psutil.virtual_memory()
            logger.info(f"  Total Memory: {memory.total // (1024**3)} GB")
            logger.info(f"  Available Memory: {memory.available // (1024**3)} GB")
        except ImportError:
            logger.debug("  psutil not available for memory information")
            
    except Exception as e:
        logger.warning(f"Failed to log system information: {e}")

def configure_module_logger(
    module_name: str,
    level: Optional[str] = None,
    file_path: Optional[Path] = None
) -> logging.Logger:
    """
    Configure a specific module logger.
    
    Args:
        module_name: Name of the module
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_path: Optional specific log file for this module
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(module_name)
    
    if level:
        numeric_level = getattr(logging, level.upper(), None)
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)
    
    if file_path:
        # Create module-specific file handler
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Module-specific logging configured for {module_name}")
        logger.info(f"Log file: {file_path}")
    
    return logger 