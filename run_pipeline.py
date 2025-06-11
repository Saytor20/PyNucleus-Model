############################################################
# 4 — Typer CLI entrypoint
############################################################
# Dependency: pip install typer[all]

from pathlib import Path
import sys
import typer
from typing import Optional

# Add src to Python path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pynucleus.integration.config_manager import ConfigManager
from pynucleus.pipeline.pipeline_utils import run_full_pipeline
from pynucleus.utils.logging_config import setup_logging, get_logger, log_system_info

app = typer.Typer(
    add_completion=False, 
    pretty_exceptions_enable=False,
    help="PyNucleus Pipeline CLI - RAG + DWSIM Chemical Process Simulation"
)

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
        logger = setup_logging(
            debug=verbose,
            log_file=log_file,
            console_output=True,
            file_output=True
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

@app.command("test-logging")
def test_logging():
    """Test the logging configuration."""
    
    # Setup logging in test mode
    logger = setup_logging(debug=True, force_reconfigure=True)
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