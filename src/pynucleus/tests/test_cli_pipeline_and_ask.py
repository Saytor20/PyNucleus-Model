"""
Unit tests for the pipeline-and-ask CLI command functionality.

Tests the integration between:
- Pipeline execution
- Report file discovery
- LLM querying
- CLI interface
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to Python path for testing
src_path = str(Path(__file__).parent.parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import CLI module - using relative import to avoid issues
try:
    import run_pipeline
    from run_pipeline import find_latest_report, pipeline_and_ask
except ImportError:
    # If running from different directory, try absolute import
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    import run_pipeline
    from run_pipeline import find_latest_report, pipeline_and_ask


class TestFindLatestReport(unittest.TestCase):
    """Test the find_latest_report function."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.test_dir) / "output"
        self.llm_reports_dir = self.output_dir / "llm_reports"
        self.llm_reports_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_no_reports_directory(self):
        """Test when llm_reports directory doesn't exist."""
        non_existent_dir = Path(self.test_dir) / "no_reports"
        result = find_latest_report(non_existent_dir)
        self.assertIsNone(result)
    
    def test_empty_reports_directory(self):
        """Test when llm_reports directory is empty."""
        result = find_latest_report(self.output_dir)
        self.assertIsNone(result)
    
    def test_single_report_file(self):
        """Test with a single report file."""
        report_file = self.llm_reports_dir / "test_report.md"
        report_file.write_text("Test report content")
        
        result = find_latest_report(self.output_dir)
        self.assertEqual(result, report_file)
    
    def test_multiple_report_files(self):
        """Test with multiple report files - should return most recent."""
        # Create first file
        report1 = self.llm_reports_dir / "report1.md"
        report1.write_text("Report 1")
        
        # Wait a moment to ensure different timestamps
        import time
        time.sleep(0.1)
        
        # Create second file (should be newer)
        report2 = self.llm_reports_dir / "report2.md"
        report2.write_text("Report 2")
        
        result = find_latest_report(self.output_dir)
        self.assertEqual(result, report2)
    
    def test_mixed_file_types(self):
        """Test with both .md and .txt files."""
        md_file = self.llm_reports_dir / "report.md"
        txt_file = self.llm_reports_dir / "report.txt"
        other_file = self.llm_reports_dir / "report.pdf"  # Should be ignored
        
        md_file.write_text("Markdown report")
        
        import time
        time.sleep(0.1)
        
        txt_file.write_text("Text report")
        other_file.write_text("PDF report")
        
        result = find_latest_report(self.output_dir)
        self.assertEqual(result, txt_file)  # Should be the newest .md or .txt file


class TestPipelineAndAskIntegration(unittest.TestCase):
    """Test the complete pipeline-and-ask integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = Path(self.test_dir) / "test_config.json"
        self.output_dir = Path(self.test_dir) / "output"
        
        # Create test configuration
        test_config = {
            "simulations": [
                {
                    "name": "test_simulation",
                    "type": "distillation",
                    "components": ["water", "ethanol"]
                }
            ],
            "rag": {
                "top_k": 3,
                "similarity_threshold": 0.5
            },
            "llm": {
                "summary_length": "medium"
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f, indent=2)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('run_pipeline.run_full_pipeline')
    @patch('run_pipeline.LLMQueryManager')
    @patch('run_pipeline.ConfigManager')
    def test_successful_pipeline_and_ask(self, mock_config_mgr, mock_llm_mgr, mock_pipeline):
        """Test successful execution of pipeline-and-ask."""
        
        # Mock configuration manager
        mock_cfg_instance = MagicMock()
        mock_config_mgr.return_value = mock_cfg_instance
        mock_cfg_instance.load.return_value = MagicMock()
        
        # Mock successful pipeline execution
        mock_pipeline.return_value = {
            'success': True,
            'rag_data': [],
            'dwsim_data': [],
            'exported_files': []
        }
        
        # Create test report file
        llm_reports_dir = self.output_dir / "llm_reports"
        llm_reports_dir.mkdir(parents=True, exist_ok=True)
        test_report = llm_reports_dir / "test_report.md"
        test_report.write_text("Test simulation report with analysis results.")
        
        # Mock LLM query manager
        mock_llm_instance = MagicMock()
        mock_llm_mgr.return_value = mock_llm_instance
        mock_llm_instance.ask_llm.return_value = "This is a test LLM response about the report."
        
        # Import and patch typer.Exit to avoid actual exits
        with patch('typer.Exit') as mock_exit:
            # Mock print to capture output instead of printing
            with patch('builtins.print') as mock_print:
                try:
                    # Call the function - this should work without crashing
                    pipeline_and_ask(
                        config_path=self.config_file,
                        question="What are the key findings?",
                        model_id="gpt2",
                        output_dir=self.output_dir,
                        verbose=False,
                        log_file=None
                    )
                    
                    # Verify pipeline was called
                    mock_pipeline.assert_called_once()
                    
                    # Verify LLM was called
                    mock_llm_instance.ask_llm.assert_called_once()
                    
                    # Verify response was printed
                    print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
                    response_printed = any("This is a test LLM response" in str(call) for call in print_calls)
                    self.assertTrue(response_printed, "LLM response should be printed")
                    
                except Exception as e:
                    # If typer.Exit was called, that's expected behavior
                    if "Exit" not in str(type(e)):
                        self.fail(f"Unexpected exception: {e}")
    
    @patch('run_pipeline.run_full_pipeline')
    @patch('run_pipeline.ConfigManager')
    def test_pipeline_failure(self, mock_config_mgr, mock_pipeline):
        """Test handling of pipeline execution failure."""
        
        # Mock configuration manager
        mock_cfg_instance = MagicMock()
        mock_config_mgr.return_value = mock_cfg_instance
        mock_cfg_instance.load.return_value = MagicMock()
        
        # Mock failed pipeline execution
        mock_pipeline.return_value = {
            'success': False,
            'error': 'Pipeline failed'
        }
        
        with patch('typer.Exit') as mock_exit:
            with patch('builtins.print'):
                try:
                    pipeline_and_ask(
                        config_path=self.config_file,
                        question="What are the key findings?",
                        model_id="gpt2",
                        output_dir=self.output_dir,
                        verbose=False,
                        log_file=None
                    )
                except:
                    pass  # Expected to exit
                
                # Verify exit was called due to pipeline failure
                mock_exit.assert_called_with(1)
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        non_existent_config = Path(self.test_dir) / "missing_config.json"
        
        with patch('typer.Exit') as mock_exit:
            with patch('builtins.print'):
                try:
                    pipeline_and_ask(
                        config_path=non_existent_config,
                        question="What are the key findings?",
                        model_id="gpt2",
                        output_dir=self.output_dir,
                        verbose=False,
                        log_file=None
                    )
                except:
                    pass  # Expected to exit
                
                # Verify exit was called due to missing config
                mock_exit.assert_called_with(1)


class TestCLIHelp(unittest.TestCase):
    """Test CLI help functionality."""
    
    def test_help_contains_pipeline_and_ask(self):
        """Test that help output contains the new command."""
        # This is more of an integration test, but we can check the typer app
        import run_pipeline
        
        # Check if the command is registered
        commands = run_pipeline.app.registered_commands
        
        # Handle different typer versions
        if hasattr(commands, 'values'):
            command_names = [cmd.name for cmd in commands.values()]
        else:
            # For newer typer versions, commands might be a list
            command_names = [cmd.name if hasattr(cmd, 'name') else str(cmd) for cmd in commands]
        
        # Check that pipeline-and-ask is in the registered commands
        self.assertTrue(any("pipeline-and-ask" in str(name) for name in command_names),
                       f"pipeline-and-ask not found in commands: {command_names}")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions used by the CLI."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_find_latest_report_with_subdirectories(self):
        """Test find_latest_report ignores subdirectories."""
        output_dir = Path(self.test_dir)
        llm_reports_dir = output_dir / "llm_reports"
        llm_reports_dir.mkdir(parents=True)
        
        # Create a subdirectory (should be ignored)
        subdir = llm_reports_dir / "subdir"
        subdir.mkdir()
        subdir_file = subdir / "report.md"
        subdir_file.write_text("Subdirectory report")
        
        # Create a file in the main directory
        main_file = llm_reports_dir / "main_report.md"
        main_file.write_text("Main report")
        
        result = find_latest_report(output_dir)
        self.assertEqual(result, main_file)


if __name__ == '__main__':
    # Configure test environment
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Run tests
    unittest.main(verbosity=2) 