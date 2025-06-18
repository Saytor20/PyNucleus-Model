#!/usr/bin/env python3
"""
PyNucleus Pipeline CLI

Command-line interface for the PyNucleus chemical process simulation and RAG system.
Provides commands to run the full pipeline, query LLMs, and test various components.

Usage:
    python run_pipeline.py run --config-path <config.json> --output-dir <output>
    python run_pipeline.py pipeline-and-ask --config-path <config.json> --question "..."
"""

############################################################
# 4 — Typer CLI entrypoint
############################################################
# Dependency: pip install typer[all]

from pathlib import Path
import sys
import typer
from typing import Optional
import glob
import os
from datetime import datetime

# Add src to Python path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pynucleus.integration.config_manager import ConfigManager
from pynucleus.pipeline.pipeline_utils import run_full_pipeline
from pynucleus.utils.logging_config import configure_logging, get_logger, log_system_info
from pynucleus.llm.query_llm import LLMQueryManager

app = typer.Typer(
    add_completion=False, 
    pretty_exceptions_enable=False,
    help="PyNucleus Pipeline CLI - RAG + DWSIM Chemical Process Simulation"
)

def find_latest_report(output_dir: Path) -> Optional[Path]:
    """
    Find the latest generated report file in the LLM reports directory.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Path to the latest report file, or None if no reports found
    """
    llm_reports_dir = output_dir / "llm_reports"
    
    if not llm_reports_dir.exists():
        return None
    
    # Look for markdown and text files (common report formats)
    patterns = ["*.md", "*.txt"]
    report_files = []
    
    for pattern in patterns:
        report_files.extend(llm_reports_dir.glob(pattern))
    
    if not report_files:
        return None
    
    # Return the most recently modified file
    latest_file = max(report_files, key=lambda f: f.stat().st_mtime)
    return latest_file

@app.command("run")
def run_pipeline(
    config_path: Path = typer.Option(..., "--config-path", help="JSON/CSV config file"),
    output_dir: Path = typer.Option("data/05_output", "--output-dir", help="Save location"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Verbose logging"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Custom log file path (default: logs/pipeline.log)"),
):
    """Execute the full PyNucleus pipeline from the command line."""
    
    # Initialize robust logging configuration
    try:
        # Setup logging with enhanced configuration
        logger = configure_logging(
            level="DEBUG" if verbose else "INFO",
            log_file=log_file
        )
        
        # Get CLI-specific logger
        cli_logger = get_logger(__name__)
        
        # Log startup information
        cli_logger.info("🚀 PyNucleus CLI started")
        cli_logger.info(f"📋 Configuration: {config_path}")
        cli_logger.info(f"📁 Output directory: {output_dir}")
        cli_logger.info(f"🔧 Verbose mode: {'ON' if verbose else 'OFF'}")
        
        if verbose:
            log_system_info(cli_logger)
        
        # Validate inputs
        if not config_path.exists():
            cli_logger.error(f"❌ Configuration file not found: {config_path}")
            raise typer.Exit(1)
        
        # Load configuration
        cli_logger.info("⚙️ Loading configuration...")
        cfg_mgr = ConfigManager(config_dir=config_path.parent)
        settings = cfg_mgr.load(config_path.name)
        cli_logger.info("✅ Configuration loaded successfully")
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cli_logger.info(f"📂 Output directory prepared: {output_dir}")
        
        # Run the pipeline
        cli_logger.info("🔄 Starting pipeline execution...")
        result = run_full_pipeline(settings=settings, output_dir=output_dir)
        
        if result:
            cli_logger.info("🎉 Pipeline completed successfully!")
            cli_logger.info(f"📊 Results: {result.get('summary', 'N/A')}")
        else:
            cli_logger.warning("⚠️ Pipeline completed with warnings")
            
    except KeyboardInterrupt:
        cli_logger.info("⏹️ Pipeline interrupted by user")
        raise typer.Exit(130)  # Standard exit code for Ctrl+C
        
    except Exception as e:
        # Use logger if available, otherwise print to stderr
        try:
            cli_logger.error(f"❌ Pipeline failed: {str(e)}")
            if verbose:
                cli_logger.exception("Full error details:")
        except:
            print(f"❌ Pipeline failed: {str(e)}", file=sys.stderr)
        
        raise typer.Exit(1)

@app.command("pipeline-and-ask")
def pipeline_and_ask(
    config_path: Path = typer.Option(..., "--config-path", help="JSON/CSV config file"),
    question: str = typer.Option(..., "--question", help="Question to ask the LLM about the report"),
    model_id: str = typer.Option("tiiuae/falcon-rw-0.3b", "--model-id", help="LLM model ID"),
    output_dir: Path = typer.Option("data/05_output", "--output-dir", help="Save location"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Verbose logging"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Custom log file path"),
):
    """Run the pipeline and then query an LLM about the latest generated report."""
    
    try:
        # Setup logging
        logger = configure_logging(
            level="DEBUG" if verbose else "INFO",
            log_file=log_file
        )
        
        cli_logger = get_logger(__name__)
        
        cli_logger.info("🚀 Starting Pipeline-and-Ask command")
        cli_logger.info(f"📋 Configuration: {config_path}")
        cli_logger.info(f"❓ Question: {question}")
        cli_logger.info(f"🤖 Model ID: {model_id}")
        cli_logger.info(f"📁 Output directory: {output_dir}")
        
        # Validate inputs
        if not config_path.exists():
            cli_logger.error(f"❌ Configuration file not found: {config_path}")
            raise typer.Exit(1)
        
        # Step 1: Run the pipeline
        print("=" * 60)
        print("🔄 STEP 1: Running PyNucleus Pipeline")
        print("=" * 60)
        
        # Load configuration
        cli_logger.info("⚙️ Loading configuration...")
        cfg_mgr = ConfigManager(config_dir=config_path.parent)
        settings = cfg_mgr.load(config_path.name)
        cli_logger.info("✅ Configuration loaded successfully")
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run the pipeline
        pipeline_result = run_full_pipeline(settings=settings, output_dir=output_dir)
        
        if not pipeline_result or not pipeline_result.get('success', True):
            cli_logger.error("❌ Pipeline execution failed")
            print("❌ Pipeline execution failed. Cannot proceed to LLM query.")
            raise typer.Exit(1)
        
        print("✅ Pipeline completed successfully!")
        
        # Step 2: Find the latest report
        print("\n" + "=" * 60)
        print("🔍 STEP 2: Finding Latest Report")
        print("=" * 60)
        
        latest_report = find_latest_report(output_dir)
        
        if not latest_report:
            cli_logger.error("❌ No reports found in the output directory")
            print(f"❌ No reports found in {output_dir}/llm_reports/")
            print("💡 Make sure the pipeline generates LLM reports before querying.")
            raise typer.Exit(1)
        
        print(f"✅ Found latest report: {latest_report}")
        cli_logger.info(f"📄 Using report: {latest_report}")
        
        # Step 3: Query the LLM
        print("\n" + "=" * 60)
        print("🤖 STEP 3: Querying LLM")
        print("=" * 60)
        
        print(f"🤖 Initializing LLM with model: {model_id}")
        print(f"❓ Question: {question}")
        print(f"📄 Report: {latest_report.name}")
        
        # Initialize LLM Query Manager
        try:
            llm_manager = LLMQueryManager(
                model_id=model_id,
                device="cpu",  # Default to CPU for compatibility
                max_tokens=8192
            )
            
            cli_logger.info(f"🤖 LLM Manager initialized with model: {model_id}")
            
            # Query the LLM
            print("\n🔄 Generating response...")
            response = llm_manager.ask_llm(
                user_query=question,
                report_file_path=latest_report,
                system_message="You are an expert chemical process engineer analyzing simulation reports.",
                generation_params={
                    'max_length': 500,  # Reasonable response length
                    'temperature': 0.7,
                    'do_sample': True,
                    'top_p': 0.9
                }
            )
            
            # Display results
            print("\n" + "=" * 60)
            print("📋 LLM RESPONSE")
            print("=" * 60)
            print(f"Question: {question}")
            print(f"Report: {latest_report.name}")
            print(f"Model: {model_id}")
            print("-" * 60)
            print(response)
            print("=" * 60)
            
            cli_logger.info("✅ LLM query completed successfully")
            
        except Exception as e:
            cli_logger.error(f"❌ LLM query failed: {str(e)}")
            print(f"❌ Failed to query LLM: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
            raise typer.Exit(1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Operation interrupted by user")
        raise typer.Exit(130)
        
    except Exception as e:
        print(f"❌ Command failed: {str(e)}")
        if 'cli_logger' in locals():
            cli_logger.error(f"❌ Command failed: {str(e)}")
        raise typer.Exit(1)

@app.command("test-logging")
def test_logging():
    """Test the logging configuration."""
    
    # Setup logging in test mode
    logger = configure_logging(level="DEBUG")
    cli_logger = get_logger("test")
    
    print("🧪 Testing logging configuration...")
    
    # Test different log levels
    cli_logger.debug("🔍 This is a DEBUG message")
    cli_logger.info("📋 This is an INFO message") 
    cli_logger.warning("⚠️ This is a WARNING message")
    cli_logger.error("❌ This is an ERROR message")
    
    # Log system info
    log_system_info(cli_logger)
    
    # Check log file
    log_file = Path("logs/pipeline.log")
    if log_file.exists():
        cli_logger.info(f"✅ Log file created: {log_file}")
        cli_logger.info(f"📏 Log file size: {log_file.stat().st_size} bytes")
    else:
        cli_logger.error("❌ Log file not found")
    
    print("✅ Logging test completed!")

if __name__ == "__main__":
    app() 