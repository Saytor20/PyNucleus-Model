"""
Robust Logging Configuration Module

Provides centralized, configurable logging setup for PyNucleus pipeline operations.
Supports simultaneous console and file logging with configurable debug modes.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

# Global logger instance to prevent reconfiguration
_logger_configured = False
_main_logger = None

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to level name for console
        if hasattr(record, 'color_enabled') and record.color_enabled:
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

def setup_logging(
    debug: bool = False,
    log_file: Optional[Path] = None,
    force_reconfigure: bool = False,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up robust logging configuration for PyNucleus pipeline.
    
    Configures logging to write simultaneously to:
    - Console (stdout) for quick debugging
    - File at logs/pipeline.log for persistent storage
    
    Args:
        debug: If True, set logging level to DEBUG, otherwise INFO
        log_file: Optional custom file path (defaults to logs/pipeline.log)
        force_reconfigure: Force reconfiguration even if already configured
        console_output: Enable console logging (default: True)
        file_output: Enable file logging (default: True)
        
    Returns:
        Configured PyNucleus logger instance
    """
    global _logger_configured, _main_logger
    
    # Prevent duplicate configuration unless forced
    if _logger_configured and not force_reconfigure:
        return _main_logger
    
    # Set logging level based on debug flag
    level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Default log file path
    if log_file is None:
        log_file = log_dir / "pipeline.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Define log formats
    # Detailed format: timestamp | level | module | message
    detailed_format = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    simple_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Date format for timestamps
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 1. CONSOLE HANDLER - for quick debugging
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Use colored formatter for console
        console_formatter = ColoredFormatter(
            fmt=simple_format,
            datefmt=date_format
        )
        console_handler.setFormatter(console_formatter)
        
        # Add color flag to records
        class ColorFilter(logging.Filter):
            def filter(self, record):
                record.color_enabled = True
                return True
        
        console_handler.addFilter(ColorFilter())
        root_logger.addHandler(console_handler)
    
    # 2. FILE HANDLER - for persistent storage
    if file_output:
        # Create rotating file handler to prevent huge log files
        from logging.handlers import RotatingFileHandler
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB max file size
            backupCount=5,          # Keep 5 backup files
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        
        # Use detailed format for file logging
        file_formatter = logging.Formatter(
            fmt=detailed_format,
            datefmt=date_format
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # 3. Create PyNucleus specific logger
    logger = logging.getLogger('pynucleus')
    logger.setLevel(level)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = True
    
    # Log configuration details
    if debug:
        logger.debug("🔧 Debug logging enabled")
        logger.debug(f"📁 Log file: {log_file}")
        logger.debug(f"📊 Console output: {console_output}")
        logger.debug(f"💾 File output: {file_output}")
    else:
        logger.info("📋 Logging configured (INFO level)")
        if file_output:
            logger.info(f"📁 Log file: {log_file}")
    
    # Mark as configured and store reference
    _logger_configured = True
    _main_logger = logger
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name under PyNucleus namespace.
    
    Args:
        name: Logger name (usually __name__ or module name)
        
    Returns:
        Logger instance with proper naming
    """
    # Clean up the name to avoid double 'pynucleus' prefix
    if name.startswith('pynucleus.'):
        clean_name = name
    elif name.startswith('__main__'):
        clean_name = 'pynucleus.main'
    else:
        clean_name = f'pynucleus.{name}'
    
    return logging.getLogger(clean_name)

def reset_logging():
    """Reset logging configuration (useful for testing)."""
    global _logger_configured, _main_logger
    
    # Remove all handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    # Reset global state
    _logger_configured = False
    _main_logger = None

def create_timestamped_log_file(base_dir: str = "logs", prefix: str = "pynucleus") -> Path:
    """
    Create a timestamped log file path for session-specific logging.
    
    Args:
        base_dir: Base directory for log files
        prefix: Prefix for log file name
        
    Returns:
        Path to timestamped log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_dir)
    log_dir.mkdir(exist_ok=True)
    return log_dir / f"{prefix}_{timestamp}.log"

def get_log_level_name(debug: bool) -> str:
    """Get human-readable log level name."""
    return "DEBUG" if debug else "INFO"

def log_system_info(logger: logging.Logger):
    """Log basic system information for debugging."""
    import platform
    import sys
    
    logger.info("🖥️ System Information:")
    logger.info(f"   • Platform: {platform.system()} {platform.release()}")
    logger.info(f"   • Python: {sys.version.split()[0]}")
    logger.info(f"   • Working Directory: {os.getcwd()}")
    logger.info(f"   • Process ID: {os.getpid()}")

# Convenience function for quick setup
def quick_setup(debug: bool = False) -> logging.Logger:
    """
    Quick logging setup with default configuration.
    
    Args:
        debug: Enable debug mode
        
    Returns:
        Configured logger
    """
    return setup_logging(debug=debug) 