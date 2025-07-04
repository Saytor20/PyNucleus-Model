#!/usr/bin/env python3
"""
Enhanced Error Handler for PyNucleus CLI

Provides comprehensive error handling with clear, actionable error messages
and proper debug information when needed.
"""

import logging
import os
import sys
from functools import wraps
from typing import Any, Callable, TypeVar, Union
from rich.console import Console
from rich.traceback import install as rich_install_traceback

# Configure rich traceback for better error display
rich_install_traceback(show_locals=True)

# Global console instance for error display
console = Console()

# Type hint for decorated functions
F = TypeVar('F', bound=Callable[..., Any])

# Check if we're in debug mode
DEBUG = os.getenv('DEBUG', '').lower() in ('true', '1', 'yes', 'on')

def error_handler(func: F) -> F:
    """
    Enhanced error handler decorator with clear, actionable error messages.
    
    Provides specific handling for common error types and maintains debug
    information when DEBUG mode is enabled.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Wrapped function with enhanced error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            # Handle both standard FileNotFoundError and custom ones
            if hasattr(e, 'filename') and e.filename:
                filename = e.filename
            else:
                # Extract filename from error message for custom FileNotFoundError
                error_msg = str(e)
                if "Configuration file not found:" in error_msg:
                    filename = error_msg.split("Configuration file not found:")[-1].strip()
                else:
                    filename = error_msg
            console.print(f"🚫 [red]File not found: {filename}[/red]")
            console.print(f"   [dim]Check that the file path is correct and the file exists[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            return None
        except PermissionError as e:
            console.print(f"🚫 [red]Permission denied: {e.filename}[/red]")
            console.print(f"   [dim]Check file permissions or run with appropriate privileges[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            return None
        except IsADirectoryError as e:
            console.print(f"🚫 [red]Expected a file but found a directory: {e.filename}[/red]")
            console.print(f"   [dim]Specify a file path instead of a directory[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            return None
        except NotADirectoryError as e:
            console.print(f"🚫 [red]Expected a directory but found a file: {e.filename}[/red]")
            console.print(f"   [dim]Specify a directory path instead of a file[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            return None
        except OSError as e:
            console.print(f"🚫 [red]System error: {e}[/red]")
            console.print(f"   [dim]Check system resources and file system permissions[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            return None
        except ValueError as e:
            console.print(f"🚫 [red]Invalid value: {e}[/red]")
            console.print(f"   [dim]Check input parameters and data format[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            return None
        except TypeError as e:
            console.print(f"🚫 [red]Type error: {e}[/red]")
            console.print(f"   [dim]Check parameter types and function arguments[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            return None
        except KeyError as e:
            console.print(f"🚫 [red]Missing required key: {e}[/red]")
            console.print(f"   [dim]Check configuration file or data structure[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            return None
        except ImportError as e:
            console.print(f"🚫 [red]Import error: {e}[/red]")
            console.print(f"   [dim]Check that all required dependencies are installed[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            return None
        except ConnectionError as e:
            console.print(f"🚫 [red]Connection error: {e}[/red]")
            console.print(f"   [dim]Check network connectivity and server availability[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            return None
        except KeyboardInterrupt:
            console.print("\n[yellow]⏹️  Operation cancelled by user[/yellow]")
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(130)  # Standard exit code for SIGINT
        except SystemExit as e:
            # Handle Typer's Exit exceptions and other system exits
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(e.code)
        except Exception as e:
            console.print(f"❗ [red]Unexpected error: {str(e)}[/red]")
            console.print(f"   [dim]If this error persists, please report it as a bug[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
                raise  # Re-raise in debug mode for full traceback
            return None
    
    return wrapper

def cli_error_handler(func: F) -> F:
    """
    CLI-specific error handler that exits with appropriate error codes.
    
    Similar to error_handler but designed for CLI commands that should
    exit with specific error codes instead of returning None.
    
    Args:
        func: Function to wrap with CLI error handling
        
    Returns:
        Wrapped function with CLI error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            # Handle both standard FileNotFoundError and custom ones
            if hasattr(e, 'filename') and e.filename:
                filename = e.filename
            else:
                # Extract filename from error message for custom FileNotFoundError
                error_msg = str(e)
                if "Configuration file not found:" in error_msg:
                    filename = error_msg.split("Configuration file not found:")[-1].strip()
                else:
                    filename = error_msg
            console.print(f"🚫 [red]File not found: {filename}[/red]")
            console.print(f"   [dim]Check that the file path is correct and the file exists[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(2)  # Standard exit code for file not found
        except PermissionError as e:
            console.print(f"🚫 [red]Permission denied: {e.filename}[/red]")
            console.print(f"   [dim]Check file permissions or run with appropriate privileges[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(13)  # Standard exit code for permission denied
        except IsADirectoryError as e:
            console.print(f"🚫 [red]Expected a file but found a directory: {e.filename}[/red]")
            console.print(f"   [dim]Specify a file path instead of a directory[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(21)  # Standard exit code for EISDIR
        except NotADirectoryError as e:
            console.print(f"🚫 [red]Expected a directory but found a file: {e.filename}[/red]")
            console.print(f"   [dim]Specify a directory path instead of a file[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(20)  # Standard exit code for ENOTDIR
        except OSError as e:
            console.print(f"🚫 [red]System error: {e}[/red]")
            console.print(f"   [dim]Check system resources and file system permissions[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(5)  # Standard exit code for I/O error
        except ValueError as e:
            console.print(f"🚫 [red]Invalid value: {e}[/red]")
            console.print(f"   [dim]Check input parameters and data format[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(22)  # Standard exit code for invalid argument
        except TypeError as e:
            console.print(f"🚫 [red]Type error: {e}[/red]")
            console.print(f"   [dim]Check parameter types and function arguments[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(22)  # Standard exit code for invalid argument
        except KeyError as e:
            console.print(f"🚫 [red]Missing required key: {e}[/red]")
            console.print(f"   [dim]Check configuration file or data structure[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(22)  # Standard exit code for invalid argument
        except ImportError as e:
            console.print(f"🚫 [red]Import error: {e}[/red]")
            console.print(f"   [dim]Check that all required dependencies are installed[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(1)  # General error
        except ConnectionError as e:
            console.print(f"🚫 [red]Connection error: {e}[/red]")
            console.print(f"   [dim]Check network connectivity and server availability[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(61)  # Standard exit code for connection refused
        except KeyboardInterrupt:
            console.print("\n[yellow]⏹️  Operation cancelled by user[/yellow]")
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(130)  # Standard exit code for SIGINT
        except SystemExit as e:
            # Handle Typer's Exit exceptions and other system exits
            if DEBUG:
                console.print_exception(show_locals=True)
            sys.exit(e.code)
        except Exception as e:
            console.print(f"❗ [red]Unexpected error: {str(e)}[/red]")
            console.print(f"   [dim]If this error persists, please report it as a bug[/dim]")
            if DEBUG:
                console.print_exception(show_locals=True)
                raise  # Re-raise in debug mode for full traceback
            sys.exit(1)  # General error
    
    return wrapper

def setup_error_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup error logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("pynucleus.errors")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create console handler with rich formatting
    from rich.logging import RichHandler
    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        rich_tracebacks=True
    )
    
    # Set formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

# Initialize error logger
error_logger = setup_error_logging()

# Export main decorator for backward compatibility
__all__ = ['error_handler', 'cli_error_handler', 'error_logger', 'setup_error_logging'] 