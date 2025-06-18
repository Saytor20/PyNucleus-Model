"""
Logging configuration for PyNucleus system.
Provides centralized logging setup to replace logging.basicConfig usage.
"""

import logging
import logging.config
import sys
import platform
from pathlib import Path
from typing import Optional
from datetime import datetime

def configure_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """
    Central logging configuration function to replace logging.basicConfig usage.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Root logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Set default log file if not provided
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"pynucleus_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger

def setup_diagnostic_logging(diagnostic_name: str, timestamp: Optional[str] = None) -> tuple[logging.Logger, logging.Logger, Path]:
    """
    Setup specialized logging for diagnostic tools with both file and console loggers.
    
    Args:
        diagnostic_name: Name of the diagnostic tool (e.g., 'system_diagnostic', 'system_validator')
        timestamp: Optional timestamp string for log file naming
        
    Returns:
        Tuple of (file_logger, console_logger, log_file_path)
    """
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create timestamp for unique log file
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{diagnostic_name}_{timestamp}.log"
    
    # Setup file logger (clean format without symbols)
    file_logger = logging.getLogger(f'{diagnostic_name}_file')
    file_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in file_logger.handlers[:]:
        file_logger.removeHandler(handler)
        
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_formatter)
    file_logger.addHandler(file_handler)
    
    # Setup console logger (with symbols for better UX)
    console_logger = logging.getLogger(f'{diagnostic_name}_console')
    console_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in console_logger.handlers[:]:
        console_logger.removeHandler(handler)
        
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    console_logger.addHandler(console_handler)
    
    return file_logger, console_logger, log_file

def clean_message_for_file(message: str) -> str:
    """
    Remove symbols and emojis from message for clean file logging.
    
    Args:
        message: Message with potential symbols/emojis
        
    Returns:
        Clean message without symbols
    """
    symbols_to_remove = [
        "✅", "❌", "⚠️", "🔍", "📊", "🎉", "🔧", "📁", "📋", "🚀", "🟢", "🟡", "🔴", 
        "ℹ️", "📄", "💾", "🐍", "─", "═", "•", "▶", "⏭️", "🧪", "1️⃣", "2️⃣", "3️⃣", 
        "4️⃣", "5️⃣", "6️⃣", "7️⃣", "🗑️", "💡"
    ]
    
    clean_msg = message
    for symbol in symbols_to_remove:
        clean_msg = clean_msg.replace(symbol, "")
    
    # Clean up extra spaces
    clean_msg = " ".join(clean_msg.split())
    return clean_msg.strip()

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