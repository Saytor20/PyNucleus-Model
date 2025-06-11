#!/usr/bin/env python3
"""
Tests for document processing functionality
"""

import os
import pytest
from pathlib import Path
from pynucleus.rag.document_processor import process_documents


def test_document_processor_setup():
    """Test document processor setup and directory structure."""
    # Check required directories
    assert Path("data/raw").exists(), "Raw data directory not found"
    assert Path("data/processed").exists(), "Processed data directory not found"

    # Check if we have test documents
    raw_files = list(Path("data/raw").glob("*"))
    assert len(raw_files) > 0, "No test documents found in data/raw"


def test_document_processing():
    """Test document processing functionality."""
    # Process documents
    result = process_documents()

    # Check if processing was successful
    assert result is not None, "Document processing failed"

    # Check if output files exist
    processed_files = list(Path("data/processed").glob("*"))
    assert len(processed_files) > 0, "No processed files found"


def test_error_handling():
    """Test error handling in document processor."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        process_documents(input_file="non_existent_file.txt")

    # Test with invalid file type
    with pytest.raises(ValueError):
        process_documents(input_file="test.invalid")
