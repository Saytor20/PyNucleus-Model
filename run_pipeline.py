############################################################
# 4 — Typer CLI entrypoint
############################################################
# Dependency: pip install typer[all]

from pathlib import Path
import sys
import typer

# Add src to Python path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pynucleus.integration.config_manager import ConfigManager
from pynucleus.pipeline.pipeline_utils import run_full_pipeline
from pynucleus.utils.logging_config import setup_logging

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

@app.command()
def main(
    config_path: Path = typer.Option(..., "--config-path", help="JSON/CSV config file"),
    output_dir: Path = typer.Option("data/05_output", "--output-dir", help="Save location"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Verbose logging"),
):
    """Execute the full PyNucleus pipeline from the command line."""
    setup_logging(debug=verbose)
    cfg_mgr = ConfigManager(config_dir=config_path.parent)
    settings = cfg_mgr.load(config_path.name)
    run_full_pipeline(settings=settings, output_dir=output_dir)

if __name__ == "__main__":
    app() 