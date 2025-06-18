#!/usr/bin/env python3
"""
Basic CLI tests for PyNucleus commands.

Tests that --help works for all CLI commands.
"""

import subprocess
import sys
from pathlib import Path
import pytest

# Add src to Python path for testing
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def run_cli_command(command_args):
    """Helper function to run CLI commands and capture output."""
    try:
        result = subprocess.run(
            [sys.executable, "run_pipeline.py"] + command_args,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path(__file__).parent.parent
        )
        return result
    except subprocess.TimeoutExpired:
        pytest.fail(f"Command timed out: {command_args}")


def test_main_help():
    """Test that main CLI help works."""
    result = run_cli_command(["--help"])
    assert result.returncode == 0, f"Main help failed: {result.stderr}"
    assert "PyNucleus Pipeline CLI" in result.stdout
    

def test_ingest_help():
    """Test that ingest --help works."""
    result = run_cli_command(["ingest", "--help"])
    assert result.returncode == 0, f"Ingest help failed: {result.stderr}"
    assert "Ingest and process documents for RAG system" in result.stdout
    assert "--source-dir" in result.stdout
    assert "--output-dir" in result.stdout
    

def test_build_faiss_help():
    """Test that build-faiss --help works."""
    result = run_cli_command(["build-faiss", "--help"])
    assert result.returncode == 0, f"Build-faiss help failed: {result.stderr}"
    assert "Build FAISS vector index from processed chunks" in result.stdout
    assert "--chunk-dir" in result.stdout
    assert "--index-dir" in result.stdout
    assert "--force-rebuild" in result.stdout
    

def test_ask_help():
    """Test that ask --help works."""
    result = run_cli_command(["ask", "--help"])
    assert result.returncode == 0, f"Ask help failed: {result.stderr}"
    assert "Ask a question to the RAG system" in result.stdout
    assert "--question" in result.stdout
    assert "--model-id" in result.stdout
    assert "--top-k" in result.stdout
    

def test_run_help():
    """Test that existing run --help works."""
    result = run_cli_command(["run", "--help"])
    assert result.returncode == 0, f"Run help failed: {result.stderr}"
    assert "Execute the full PyNucleus pipeline" in result.stdout
    

def test_pipeline_and_ask_help():
    """Test that existing pipeline-and-ask --help works."""
    result = run_cli_command(["pipeline-and-ask", "--help"])
    assert result.returncode == 0, f"Pipeline-and-ask help failed: {result.stderr}"
    assert "Run the pipeline and then query an LLM" in result.stdout
    

def test_all_commands_listed_in_main_help():
    """Test that all commands are listed in main help."""
    result = run_cli_command(["--help"])
    assert result.returncode == 0
    
    # Check that all commands are listed
    commands = ["run", "pipeline-and-ask", "test-logging", "ingest", "build-faiss", "ask"]
    for command in commands:
        assert command in result.stdout, f"Command '{command}' not found in main help"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 