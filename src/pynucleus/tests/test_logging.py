#!/usr/bin/env python3
"""
Tests for logging configuration
"""

import os
import sys
import pytest
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pynucleus.utils.logging_config import setup_logging, get_logger, reset_logging


class TestLoggingConfiguration:
    """Test the robust logging configuration."""
    
    def setup_method(self):
        """Reset logging before each test."""
        reset_logging()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_logging()
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        logger = setup_logging(debug=False)
        
        assert logger is not None
        assert logger.name == 'pynucleus'
        assert logger.level == 20  # INFO level
    
    def test_setup_logging_debug(self):
        """Test debug logging setup."""
        logger = setup_logging(debug=True)
        
        assert logger is not None
        assert logger.name == 'pynucleus'
        assert logger.level == 10  # DEBUG level
    
    def test_log_file_creation(self):
        """Test that log file is created."""
        # Clean up any existing log file
        log_file = Path("logs/test_pipeline.log")
        if log_file.exists():
            log_file.unlink()
        
        # Setup logging with custom file
        logger = setup_logging(debug=True, log_file=log_file)
        
        # Write a test message
        logger.info("Test message")
        
        # Check file was created
        assert log_file.exists()
        
        # Check file content
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test message" in content
            assert "INFO" in content
    
    def test_get_logger(self):
        """Test getting named loggers."""
        # Setup main logging first
        setup_logging()
        
        # Get a named logger
        logger = get_logger("test_module")
        
        assert logger is not None
        assert "pynucleus.test_module" in logger.name
    
    def test_console_and_file_output(self):
        """Test that logging works to both console and file."""
        log_file = Path("logs/test_output.log")
        if log_file.exists():
            log_file.unlink()
        
        # Setup with both outputs
        logger = setup_logging(
            debug=True,
            log_file=log_file,
            console_output=True,
            file_output=True
        )
        
        # Log a test message
        test_message = "Test console and file output"
        logger.info(test_message)
        
        # Check file output
        assert log_file.exists()
        with open(log_file, 'r') as f:
            content = f.read()
            assert test_message in content
    
    def test_force_reconfigure(self):
        """Test force reconfiguration."""
        # Initial setup
        logger1 = setup_logging(debug=False)
        assert logger1.level == 20  # INFO
        
        # Reconfigure with force
        logger2 = setup_logging(debug=True, force_reconfigure=True)
        assert logger2.level == 10  # DEBUG
    
    def test_different_log_levels(self):
        """Test different log levels are properly set."""
        log_file = Path("logs/test_levels.log")
        if log_file.exists():
            log_file.unlink()
        
        # Test INFO level
        logger = setup_logging(debug=False, log_file=log_file)
        logger.debug("Debug message")  # Should not appear
        logger.info("Info message")    # Should appear
        logger.warning("Warning message")  # Should appear
        
        # Check file content
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Debug message" not in content  # DEBUG filtered out
            assert "Info message" in content
            assert "Warning message" in content
        
        # Reset and test DEBUG level  
        reset_logging()
        log_file.unlink()
        
        logger = setup_logging(debug=True, log_file=log_file)
        logger.debug("Debug message")  # Should appear now
        logger.info("Info message")    # Should appear
        
        # Check file content
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Debug message" in content  # DEBUG now included
            assert "Info message" in content


def test_logging_integration():
    """Test logging integration with the main application."""
    # This is an integration test to verify logging works end-to-end
    
    # Setup logging
    logger = setup_logging(debug=True)
    
    # Get different loggers
    cli_logger = get_logger("cli")
    pipeline_logger = get_logger("pipeline")
    
    # Test logging from different components
    cli_logger.info("CLI started")
    pipeline_logger.info("Pipeline initialized")
    
    # Should not raise any exceptions
    assert True 