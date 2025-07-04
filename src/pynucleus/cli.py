#!/usr/bin/env python3
"""
PyNucleus Unified CLI

A comprehensive command-line interface for the PyNucleus chemical process simulation 
and RAG system. Provides commands for pipeline execution, chat interactions, 
system monitoring, and more.

Usage:
    pynucleus run --config configs/production_config.json
    pynucleus chat --model Qwen/Qwen2.5-1.5B-Instruct
    pynucleus build --template-id 1 --feedstock natural_gas
    pynucleus stats --mode system --live
    pynucleus auto-ingest --watch-dir data/01_raw
"""

# Apply telemetry patch before any imports
import sys
from pathlib import Path

# Add src to Python path if needed
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Apply telemetry patch before any ChromaDB imports
from pynucleus.utils.telemetry_patch import apply_telemetry_patch
apply_telemetry_patch()

import logging
import traceback
from typing import Optional, List
import signal
import time

import rich
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as rich_install_traceback
from typer import Typer, Option, Argument, Context, Exit, echo
from typer.rich_utils import _get_rich_console

# Configure rich traceback for better error display
rich_install_traceback(show_locals=True)
console = Console()

# Add src to Python path if needed
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ============================================================================
# TYPEWRITER EFFECT FUNCTION
# ============================================================================

def typewriter_effect(text, delay: float = 0.03, style: str = "bold green"):
    """
    Display text with a typewriter effect, showing each word with a slight delay.
    
    Args:
        text: The text to display (str, dict, or other object)
        delay (float): Delay between words in seconds (default: 0.03)
        style (str): Rich style to apply to the text (default: "bold green")
    """
    if not text:
        return
    
    # Convert to string if it's not already
    if isinstance(text, dict):
        # Try to extract answer from common response formats
        if 'answer' in text:
            text_str = str(text['answer'])
        elif 'response' in text:
            text_str = str(text['response'])
        elif 'text' in text:
            text_str = str(text['text'])
        else:
            text_str = str(text)
    else:
        text_str = str(text)
    
    # Split text into words but preserve formatting
    words = text_str.split()
    
    for i, word in enumerate(words):
        # Add space before word (except for the first word)
        if i > 0:
            console.print(" ", end="")
        
        # Print word with style
        console.print(word, end="", style=style)
        
        # Add slight delay between words
        if i < len(words) - 1:  # Don't delay after the last word
            time.sleep(delay)
    
    # Add final newline
    console.print()

def typewriter_effect_char(text, delay: float = 0.01, style: str = "bold green"):
    """
    Display text with a character-by-character typewriter effect.
    
    Args:
        text: The text to display (str, dict, or other object)
        delay (float): Delay between characters in seconds (default: 0.01)
        style (str): Rich style to apply to the text (default: "bold green")
    """
    if not text:
        return
    
    # Convert to string if it's not already
    if isinstance(text, dict):
        # Try to extract answer from common response formats
        if 'answer' in text:
            text_str = str(text['answer'])
        elif 'response' in text:
            text_str = str(text['response'])
        elif 'text' in text:
            text_str = str(text['text'])
        else:
            text_str = str(text)
    else:
        text_str = str(text)
    
    for char in text_str:
        console.print(char, end="", style=style)
        time.sleep(delay)
    
    # Add final newline
    console.print()

# Main CLI app
app = Typer(
    name="pynucleus",
    help="🧪 PyNucleus Chemical Process Simulation & RAG System",
    add_completion=True,
    pretty_exceptions_enable=True,
    rich_markup_mode="rich",
    no_args_is_help=False  # Allow running without args to show interactive menu
)

# Configure logging
def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup robust logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    if log_file is None:
        log_file = f"logs/pynucleus_{Path().cwd().name}.log"
    
    # Configure rich handler for console
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=verbose,
        rich_tracebacks=True
    )
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Setup root logger
    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler],
        format="%(message)s"
    )
    
    return logging.getLogger(__name__)

# Import the enhanced error handler
from pynucleus.utils.error_handler import cli_error_handler

# For backward compatibility, alias the new decorator
handle_errors = cli_error_handler

# ============================================================================
# RUN COMMAND - Pipeline Execution
# ============================================================================

@app.command("run")
@handle_errors
def run_pipeline(
    config_path: Path = Option(..., "--config", "-c", help="Configuration file path"),
    output_dir: Path = Option("data/05_output", "--output", "-o", help="Output directory"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging"),
    log_file: Optional[str] = Option(None, "--log-file", help="Custom log file path"),
    dry_run: bool = Option(False, "--dry-run", help="Validate configuration without execution")
):
    """🚀 Execute the full PyNucleus pipeline"""
    
    logger = setup_logging(verbose, log_file)
    
    console.print("🧪 [bold blue]PyNucleus Pipeline Execution[/bold blue]")
    console.print(f"📋 Config: {config_path}")
    console.print(f"📁 Output: {output_dir}")
    
    if dry_run:
        console.print("🔍 [yellow]Dry run mode - validating configuration only[/yellow]")
    
    # Import and run pipeline
    from pynucleus.integration.config_manager import ConfigManager
    from pynucleus.pipeline.pipeline_utils import run_full_pipeline
    
    # Load configuration - this will naturally raise FileNotFoundError if file doesn't exist
    console.print("⚙️  Loading configuration...")
    cfg_mgr = ConfigManager(config_dir=config_path.parent)
    settings = cfg_mgr.load(config_path.name)
    console.print("✅ Configuration loaded successfully")
    
    if dry_run:
        console.print("✅ Configuration validation complete")
        return
    
    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory prepared: {output_dir}")
    
    # Execute pipeline
    console.print("🔄 Starting pipeline execution...")
    with console.status("[bold green]Running pipeline..."):
        result = run_full_pipeline(settings=settings, output_dir=output_dir)
    
    if result and result.get('success', True):
        console.print("🎉 [bold green]Pipeline completed successfully![/bold green]")
        if 'summary' in result:
            console.print(f"📊 Summary: {result['summary']}")
    else:
        console.print("⚠️  [yellow]Pipeline completed with warnings[/yellow]")

# ============================================================================
# CHAT COMMAND - Interactive Chat Interface
# ============================================================================

@app.command("chat")
@handle_errors  
def interactive_chat(
    model_id: str = Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", "-m", help="LLM model ID"),
    top_k: int = Option(5, "--top-k", "-k", help="Number of RAG results to retrieve"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging"),
    temperature: float = Option(0.7, "--temperature", "-t", help="Model temperature"),
    max_tokens: int = Option(512, "--max-tokens", help="Maximum response tokens"),
    single: Optional[str] = Option(None, "--single", "-s", help="Ask a single question and exit"),
    pretty: bool = Option(True, "--pretty/--plain", help="Use enhanced formatting for output"),
    stream: bool = Option(True, "--stream/--no-stream", help="Enable streaming typewriter effect for responses"),
    stream_delay: float = Option(0.03, "--stream-delay", help="Delay between words in streaming mode (seconds)"),
    stream_style: str = Option("bold green", "--stream-style", help="Rich style for streaming text"),
    char_mode: bool = Option(False, "--char-mode", help="Use character-by-character streaming instead of word-by-word")
):
    """💬 Start interactive chat with PyNucleus RAG system with streaming typewriter effect"""
    
    logger = setup_logging(verbose)
    
    # Import chat functionality
    try:
        from pynucleus.rag.engine import ask as rag_ask
        from pynucleus.utils.pretty_formatter import format_for_terminal
        
        console.print("🔄 [yellow]Initializing RAG system...[/yellow]")
        
        # Test RAG system
        test_result = rag_ask("test connection")
        if not test_result:
            console.print("[red]❌ Failed to initialize RAG system[/red]")
            raise Exit(1)
            
        console.print("✅ [green]RAG system ready[/green]\n")
        
        # Handle single question mode
        if single:
            console.print("💭 [bold blue]Single Question Mode[/bold blue]")
            console.print(f"❓ Question: {single}")
            
            # Process single question
            with console.status("[bold green]Thinking..."):
                result = rag_ask(single)
            
            # Display response
            console.print("\n[bold green]Answer:[/bold green]")
            if stream:
                if char_mode:
                    typewriter_effect_char(result, delay=stream_delay, style=stream_style)
                else:
                    typewriter_effect(result, delay=stream_delay, style=stream_style)
            elif pretty:
                format_for_terminal(result)
            else:
                # Even in plain mode, use enhanced formatting for better readability
                format_for_terminal(result)
            return
        
        # Interactive chat mode
        console.print("💬 [bold blue]PyNucleus Interactive Chat[/bold blue]")
        console.print(f"🤖 Model: {model_id}")
        console.print(f"🔍 Top-K: {top_k}")
        console.print("💡 Type 'exit', 'quit', or press Ctrl+C to end the session\n")
        
        # Main chat loop
        while True:
            try:
                # Get user input
                question = console.input("[bold cyan]You:[/bold cyan] ")
                
                if question.lower().strip() in ['exit', 'quit', 'q']:
                    console.print("👋 [yellow]Goodbye![/yellow]")
                    break
                
                if not question.strip():
                    continue
                
                # Process question
                with console.status("[bold green]Thinking..."):
                    result = rag_ask(question)
                
                # Display response
                console.print("\n[bold green]Assistant:[/bold green]")
                if stream:
                    if char_mode:
                        typewriter_effect_char(result, delay=stream_delay, style=stream_style)
                    else:
                        typewriter_effect(result, delay=stream_delay, style=stream_style)
                elif pretty:
                    format_for_terminal(result)
                else:
                    # Even in plain mode, use enhanced formatting for better readability
                    format_for_terminal(result)
                console.print()
                
            except KeyboardInterrupt:
                console.print("\n👋 [yellow]Chat session ended[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error processing question: {e}[/red]")
                continue
                
    except ImportError as e:
        console.print(f"[red]❌ Failed to import required modules: {e}[/red]")
        raise Exit(1)

# ============================================================================
# BUILD COMMAND - Chemical Plant Simulation
# ============================================================================

@app.command("build")
@handle_errors
def build_plant(
    template_id: Optional[int] = Option(None, "--template", "-t", help="Plant template ID (1-22)"),
    feedstock: Optional[str] = Option(None, "--feedstock", "-f", help="Feedstock type"),
    capacity: Optional[int] = Option(None, "--capacity", "-c", help="Production capacity (tons/year)"),
    location: Optional[str] = Option(None, "--location", "-l", help="Plant location"),
    hours: Optional[int] = Option(None, "--hours", help="Operating hours per year"),
    output_file: Optional[Path] = Option(None, "--output", "-o", help="Save results to JSON file"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging"),
    interactive: bool = Option(True, "--interactive/--no-interactive", help="Interactive mode for missing parameters"),
    financial_analysis: bool = Option(True, "--financial/--no-financial", help="Perform comprehensive financial analysis")
):
    """🏗️  Build plant simulation using modular templates with comprehensive financial analysis"""
    
    logger = setup_logging(verbose)
    
    console.print("🏗️  [bold blue]PyNucleus Enhanced Plant Builder[/bold blue]")
    console.print("🌍 [dim]Focused on African markets with comprehensive feasibility analysis[/dim]")
    
    try:
        # Import required components
        from pynucleus.data.mock_data_manager import get_mock_data_manager
        from pynucleus.pipeline.financial_analyzer import FinancialAnalyzer
        
        # Initialize components
        mock_data_manager = get_mock_data_manager()
        financial_analyzer = FinancialAnalyzer()
        templates = mock_data_manager.get_all_plant_templates()
        
        # Interactive parameter collection if needed
        if interactive:
            # Show available templates with enhanced details
            console.print("\n📋 [bold]Available African Plant Templates (22 Total):[/bold]")
            console.print("─" * 100)
            
            for template in templates:
                # Calculate capacity in tons/day for display
                capacity_tpd = template['default_parameters']['production_capacity'] / 365
                
                console.print(f"[bold cyan]{template['id']:2d}[/bold cyan]. {template['name']}")
                console.print(f"    🔧 Technology: {template['technology']}")
                console.print(f"    📝 Description: {template['description']}")
                console.print(f"    📊 Default Capacity: {capacity_tpd:,.0f} tons/day ({template['default_parameters']['production_capacity']:,} tons/year)")
                console.print(f"    💰 Capital Cost: ${template['default_parameters']['capital_cost']:,.0f}")
                console.print(f"    💸 Operating Cost: ${template['default_parameters']['operating_cost']:,.0f}/year")
                console.print(f"    💵 Product Price: ${template['default_parameters']['product_price']}/ton")
                
                # Show sustainability impact
                sustainability = template.get('sustainability_impact', {})
                if sustainability:
                    employment = sustainability.get('community_employment', 0)
                    circularity = sustainability.get('resource_circularity', 'Unknown')
                    co2_reduction = sustainability.get('co2_reduction_potential_tpy', 0)
                    console.print(f"    🌱 Employment: {employment} jobs | Circularity: {circularity} | CO₂ Reduction: {co2_reduction:+,.0f} tons/year")
                
                console.print()
            
            # Template selection
            if template_id is None:
                while True:
                    template_input = console.input("🏭 Enter template ID (1-22): ")
                    try:
                        template_id = int(template_input)
                        if 1 <= template_id <= len(templates):
                            break
                        else:
                            console.print(f"[red]❌ Please enter a valid template ID between 1 and {len(templates)}[/red]")
                    except ValueError:
                        console.print("[red]❌ Invalid template ID[/red]")
                
                # Get selected template details
                selected_template = next((t for t in templates if t['id'] == template_id), None)
                if selected_template:
                    console.print(f"\n✅ Selected: {selected_template['name']}")
                    console.print(f"   Technology: {selected_template['technology']}")
                    console.print(f"   Description: {selected_template['description']}")
                    
                    # Show regional context
                    regional = selected_template.get('regional_context_africa', {})
                    if regional:
                        console.print(f"   🌍 Infrastructure: {regional.get('infrastructure_quality', 'Unknown')}")
                        console.print(f"   👥 Labor Availability: {regional.get('skilled_labor_availability', 'Unknown')}")
                        console.print(f"   📋 Regulatory Environment: {regional.get('regulatory_environment', 'Unknown')}")
                        console.print(f"   🏛️ Political Stability: {regional.get('political_stability_index', 0):.2f}")
            
            # Feedstock selection
            if feedstock is None and selected_template:
                console.print(f"\n🌿 Available feedstock options for {selected_template['name']}:")
                for i, option in enumerate(selected_template['feedstock_options'], 1):
                    console.print(f"   {i}. {option}")
                
                while True:
                    try:
                        choice = int(console.input(f"\nSelect feedstock (1-{len(selected_template['feedstock_options'])}): "))
                        if 1 <= choice <= len(selected_template['feedstock_options']):
                            feedstock = selected_template['feedstock_options'][choice - 1]
                            break
                        else:
                            console.print(f"[red]❌ Please enter a valid choice between 1 and {len(selected_template['feedstock_options'])}[/red]")
                    except ValueError:
                        console.print("[red]❌ Please enter a valid number[/red]")
                
                console.print(f"✅ Selected feedstock: {feedstock}")
            
            # Production capacity
            if capacity is None and selected_template:
                valid_range = selected_template['valid_ranges']['production_capacity']
                default_capacity = selected_template['default_parameters']['production_capacity']
                
                console.print(f"\n📊 Production Capacity:")
                console.print(f"   Valid range: {valid_range['min']:,} - {valid_range['max']:,} tons/year")
                console.print(f"   Default: {default_capacity:,} tons/year ({default_capacity/365:,.0f} tons/day)")
                
                while True:
                    try:
                        user_input = console.input(f"Enter production capacity in tons/year (or press Enter for default): ")
                        if user_input.strip() == "":
                            capacity = default_capacity
                            break
                        else:
                            capacity = int(user_input)
                            if valid_range['min'] <= capacity <= valid_range['max']:
                                break
                            else:
                                console.print(f"[red]❌ Please enter a value between {valid_range['min']:,} and {valid_range['max']:,}[/red]")
                    except ValueError:
                        console.print("[red]❌ Please enter a valid number[/red]")
                
                console.print(f"✅ Production capacity: {capacity:,} tons/year ({capacity/365:,.0f} tons/day)")
            
            # Plant location
            if location is None and selected_template:
                console.print(f"\n📍 Available African locations (with cost factors):")
                location_options = list(selected_template['location_factors'].items())
                for idx, (location_name, factor) in enumerate(location_options, 1):
                    factor_text = "Standard" if factor == 1.0 else f"{factor:.2f}x cost"
                    console.print(f"   {idx}. {location_name} ({factor_text})")
                
                while True:
                    location_input = console.input(f"\nEnter plant location (number 1-{len(location_options)} or name): ").strip()
                    
                    # Try number input first
                    try:
                        location_num = int(location_input)
                        if 1 <= location_num <= len(location_options):
                            location = location_options[location_num - 1][0]
                            break
                        else:
                            console.print(f"[red]❌ Please enter a valid number between 1 and {len(location_options)}[/red]")
                    except ValueError:
                        # Try name input
                        if location_input in selected_template['location_factors']:
                            location = location_input
                            break
                        else:
                            console.print("[red]❌ Please enter a valid location from the list above[/red]")
                
                console.print(f"✅ Plant location: {location}")
            
            # Operating hours
            if hours is None and selected_template:
                valid_range = selected_template['valid_ranges']['operating_hours']
                default_hours = selected_template['default_parameters']['operating_hours']
                
                console.print(f"\n⏰ Operating Hours:")
                console.print(f"   Valid range: {valid_range['min']:,} - {valid_range['max']:,} hours/year")
                console.print(f"   Default: {default_hours:,} hours/year")
                
                while True:
                    try:
                        user_input = console.input(f"Enter operating hours per year (or press Enter for default): ")
                        if user_input.strip() == "":
                            hours = default_hours
                            break
                        else:
                            hours = int(user_input)
                            if valid_range['min'] <= hours <= valid_range['max']:
                                break
                            else:
                                console.print(f"[red]❌ Please enter a value between {valid_range['min']:,} and {valid_range['max']:,}[/red]")
                    except ValueError:
                        console.print("[red]❌ Please enter a valid number[/red]")
                
                console.print(f"✅ Operating hours: {hours:,} hours/year")
        
        # Validate required parameters
        required_params = [template_id, feedstock, capacity, location, hours]
        if any(param is None for param in required_params):
            console.print("[red]❌ Missing required parameters. Use --interactive or provide all parameters[/red]")
            raise Exit(1)
        
        # Get selected template
        selected_template = next((t for t in templates if t['id'] == template_id), None)
        if not selected_template:
            console.print(f"[red]❌ Template ID {template_id} not found[/red]")
            raise Exit(1)
        
        # Calculate adjusted costs based on location factor
        location_factor = selected_template['location_factors'].get(location, 1.0)
        adjusted_capital_cost = int(selected_template['default_parameters']['capital_cost'] * location_factor)
        adjusted_operating_cost = int(selected_template['default_parameters']['operating_cost'] * location_factor)
        
        # Create plant configuration
        plant_config = {
            "parameters": {
                "feedstock": feedstock,
                "production_capacity": capacity,
                "plant_location": location,
                "operating_hours": hours
            },
            "financial_parameters": {
                "capital_cost": adjusted_capital_cost,
                "operating_cost": adjusted_operating_cost,
                "product_price": selected_template['default_parameters']['product_price']
            },
            "template_info": selected_template,
            "location_adjustment": {
                "factor": location_factor,
                "original_capital_cost": selected_template['default_parameters']['capital_cost'],
                "original_operating_cost": selected_template['default_parameters']['operating_cost']
            }
        }
        
        # Display configuration summary
        console.print("\n🎉 [bold green]Plant configuration completed![/bold green]")
        console.print("\n📊 [bold]Enhanced Plant Configuration Summary:[/bold]")
        console.print("─" * 80)
        
        # Standardized daily production (calendar-day basis)
        daily_production = capacity / 365
        
        console.print(f"🏭 Plant Type: {selected_template['name']}")
        console.print(f"🔧 Technology: {selected_template['technology']}")
        console.print(f"⛽ Feedstock: {feedstock}")
        console.print(f"📊 Production Capacity: {capacity:,.2f} tons/year ({daily_production:,.2f} tons/day)")
        console.print(f"📍 Location: {location} (Cost Factor: {location_factor:.2f}x)")
        console.print(f"⏰ Operating Hours: {hours:,.2f} hours/year")
        # CapEx label: show location-adjusted if actually adjusted
        if location_factor != 1.0:
            console.print(f"💰 Capital Cost: ${adjusted_capital_cost:,.2f} (location-adjusted by {location_factor:.2f}x)")
        else:
            console.print(f"💰 Capital Cost: ${adjusted_capital_cost:,.2f}")
        # OpEx label: always show location-adjusted if factor != 1.0
        if location_factor != 1.0:
            console.print(f"💸 Operating Cost: ${adjusted_operating_cost:,.2f}/year (location-adjusted by {location_factor:.2f}x)")
        else:
            console.print(f"💸 Operating Cost: ${adjusted_operating_cost:,.2f}/year")
        console.print(f"💵 Product Price: ${selected_template['default_parameters']['product_price']:.2f}/ton")
        
        # Show sustainability metrics
        sustainability = selected_template.get('sustainability_impact', {})
        if sustainability:
            console.print(f"🌱 Community Employment: {sustainability.get('community_employment', 0):,.2f} jobs")
            console.print(f"♻️ Resource Circularity: {sustainability.get('resource_circularity', 'Unknown')}")
            co2_reduction = sustainability.get('co2_reduction_potential_tpy', 0)
            if co2_reduction != 0:
                console.print(f"🌍 CO₂ Impact: {co2_reduction:+,.2f} tons/year")
        
        # Perform financial analysis if requested
        if financial_analysis:
            console.print("\n💰 [bold]Performing Comprehensive Financial Analysis...[/bold]")
            
            try:
                with console.status("[bold green]Analyzing financial feasibility..."):
                    financial_report = financial_analyzer.generate_financial_report(plant_config)
                
                console.print("\n" + financial_report)
                
                # Additional feasibility insights
                console.print("\n🔍 [bold]Feasibility Assessment:[/bold]")
                
                # Get financial metrics for assessment
                metrics = financial_analyzer.calculate_financial_metrics(plant_config)
                if "error" not in metrics:
                    profitability = metrics["profitability_metrics"]
                    risks = metrics["risk_assessment"]
                    
                    # Feasibility scoring with proper ranges
                    feasibility_score = 0
                    feasibility_factors = []
                    
                    # ROI assessment (0-25 points)
                    roi = profitability.get("roi_percent", 0)
                    if roi >= 20:
                        feasibility_score += 25
                        feasibility_factors.append("✅ Excellent ROI (>20%)")
                    elif roi >= 15:
                        feasibility_score += 20
                        feasibility_factors.append("✅ Good ROI (15-20%)")
                    elif roi >= 10:
                        feasibility_score += 15
                        feasibility_factors.append("⚠️ Moderate ROI (10-15%)")
                    elif roi >= 5:
                        feasibility_score += 10
                        feasibility_factors.append("⚠️ Low ROI (5-10%)")
                    elif roi >= 0:
                        feasibility_score += 5
                        feasibility_factors.append("❌ Very low ROI (0-5%)")
                    else:
                        feasibility_factors.append("❌ Negative ROI (<0%)")
                    
                    # Payback period assessment (0-25 points)
                    payback = profitability.get("payback_period_years", float('inf'))
                    if payback == float('inf'):
                        feasibility_factors.append("❌ No payback (negative profit)")
                    elif payback <= 5:
                        feasibility_score += 25
                        feasibility_factors.append("✅ Fast payback (≤5 years)")
                    elif payback <= 8:
                        feasibility_score += 20
                        feasibility_factors.append("✅ Reasonable payback (5-8 years)")
                    elif payback <= 12:
                        feasibility_score += 15
                        feasibility_factors.append("⚠️ Long payback (8-12 years)")
                    elif payback <= 20:
                        feasibility_score += 10
                        feasibility_factors.append("⚠️ Very long payback (12-20 years)")
                    else:
                        feasibility_score += 5
                        feasibility_factors.append("❌ Extremely long payback (>20 years)")
                    
                    # Profit margin assessment (0-25 points)
                    margin = profitability.get("profit_margin_percent", 0)
                    if margin >= 25:
                        feasibility_score += 25
                        feasibility_factors.append("✅ High profit margin (>25%)")
                    elif margin >= 15:
                        feasibility_score += 20
                        feasibility_factors.append("✅ Good profit margin (15-25%)")
                    elif margin >= 8:
                        feasibility_score += 15
                        feasibility_factors.append("⚠️ Moderate profit margin (8-15%)")
                    elif margin >= 0:
                        feasibility_score += 10
                        feasibility_factors.append("⚠️ Low profit margin (0-8%)")
                    else:
                        feasibility_factors.append("❌ Negative profit margin (<0%)")
                    
                    # Risk assessment (0-25 points)
                    risk_count = len(risks)
                    if risk_count == 0:
                        feasibility_score += 25
                        feasibility_factors.append("✅ Low risk profile")
                    elif risk_count <= 2:
                        feasibility_score += 20
                        feasibility_factors.append("✅ Moderate risk profile")
                    elif risk_count <= 4:
                        feasibility_score += 15
                        feasibility_factors.append("⚠️ Higher risk profile")
                    elif risk_count <= 6:
                        feasibility_score += 10
                        feasibility_factors.append("⚠️ High risk profile")
                    else:
                        feasibility_score += 5
                        feasibility_factors.append("❌ Very high risk profile")
                    
                    # Display feasibility assessment
                    console.print(f"\n📊 Feasibility Score: {feasibility_score}/100")
                    
                    if feasibility_score >= 80:
                        console.print("🎉 [bold green]EXCELLENT FEASIBILITY[/bold green]")
                    elif feasibility_score >= 60:
                        console.print("✅ [bold green]GOOD FEASIBILITY[/bold green]")
                    elif feasibility_score >= 40:
                        console.print("⚠️ [bold yellow]MODERATE FEASIBILITY[/bold yellow]")
                    else:
                        console.print("❌ [bold red]LOW FEASIBILITY[/bold red]")
                    
                    console.print("\n📋 Feasibility Factors:")
                    for factor in feasibility_factors:
                        console.print(f"   {factor}")
                    
                    # Strategic recommendations
                    console.print("\n💡 [bold]Strategic Recommendations:[/bold]")
                    if feasibility_score >= 80:
                        console.print("   • Proceed with project implementation")
                        console.print("   • Consider scaling up for better economies")
                        console.print("   • Explore financing options for capital requirements")
                    elif feasibility_score >= 60:
                        console.print("   • Proceed with caution and risk mitigation")
                        console.print("   • Optimize operating parameters for better margins")
                        console.print("   • Consider phased implementation approach")
                    elif feasibility_score >= 40:
                        console.print("   • Conduct detailed feasibility study")
                        console.print("   • Explore cost reduction opportunities")
                        console.print("   • Consider alternative locations or technologies")
                    else:
                        console.print("   • Reconsider project fundamentals")
                        console.print("   • Explore alternative business models")
                        console.print("   • Consider smaller scale pilot project")
                
            except Exception as e:
                console.print(f"[red]❌ Financial analysis failed: {e}[/red]")
                logger.error(f"Financial analysis error: {e}")
        
        # Save configuration if requested
        if output_file:
            import json
            from datetime import datetime
            
            # Prepare comprehensive output data
            output_data = {
                "plant_configuration": plant_config,
                "financial_analysis": financial_analyzer.calculate_financial_metrics(plant_config) if financial_analysis else None,
                "feasibility_assessment": {
                    "template_id": template_id,
                    "template_name": selected_template['name'],
                    "technology": selected_template['technology'],
                    "sustainability_impact": selected_template.get('sustainability_impact', {}),
                    "regional_context": selected_template.get('regional_context_africa', {}),
                    "location_factors": selected_template.get('location_factors', {}),
                    "build_timestamp": datetime.now().isoformat()
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            console.print(f"\n💾 Comprehensive analysis saved to {output_file}")
        
        console.print("\n✅ [bold green]Enhanced plant build and analysis completed successfully![/bold green]")
        console.print("🌍 [dim]Ready for African market deployment with comprehensive feasibility assessment[/dim]")
            
    except ImportError as e:
        console.print(f"[red]❌ Failed to import required modules: {e}[/red]")
        raise Exit(1)

# ============================================================================
# STATS COMMAND - System Statistics and Monitoring  
# ============================================================================

@app.command("system-status")
@handle_errors
def system_status(
    subcommand: Optional[str] = Argument(None, help="Sub-command: comprehensive, validator"),
    output_file: Optional[str] = Option(None, "--output", "-o", help="Save results to JSON file"),
    live: bool = Option(False, "--live", "-l", help="Live updating dashboard"),
    hours: int = Option(24, "--hours", "-h", help="Hours of historical data to analyze"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """📊 System status monitoring with comprehensive diagnostics and fast validation"""
    
    logger = setup_logging(verbose)
    
    console.print("📊 [bold blue]PyNucleus System Status[/bold blue]")
    
    # Handle sub-commands
    if subcommand == "comprehensive":
        console.print("🔍 [yellow]Running comprehensive system diagnostics...[/yellow]")
        try:
            from pynucleus.metrics.system_statistics import run_system_statistics_dashboard
            from pynucleus.diagnostics.runner import run_comprehensive_diagnostics
            
            # Run comprehensive resource, DB, vector-store, VENV check
            console.print("📈 System resource analysis...")
            run_system_statistics_dashboard(
                output_file=output_file,
                show_trends=True,
                hours=hours,
                live_mode=live
            )
            
            console.print("🔧 Comprehensive diagnostics...")
            diagnostics_result = run_comprehensive_diagnostics()
            
            if diagnostics_result.get('success', True):
                console.print("✅ [bold green]Comprehensive system check completed successfully[/bold green]")
            else:
                console.print("⚠️  [yellow]System check completed with warnings[/yellow]")
                
        except ImportError as e:
            console.print(f"[red]❌ Failed to import diagnostics modules: {e}[/red]")
            raise Exit(1)
            
    elif subcommand == "validator":
        console.print("⚡ [yellow]Running fast system validation...[/yellow]")
        try:
            # Fast self-test: config files present, models loadable, env vars set
            console.print("📋 Checking configuration files...")
            
            # Check essential config files
            config_files = [
                "configs/production_config.json",
                "configs/development_config.json",
                "configs/logging.yaml"
            ]
            
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    console.print(f"✅ {config_file}")
                else:
                    console.print(f"❌ {config_file} [red](missing)[/red]")
            
            console.print("🤖 Checking model availability...")
            try:
                from pynucleus.llm.answer_engine import DSPyAnswerEngine
                engine = DSPyAnswerEngine()
                console.print("✅ LLM models accessible")
            except Exception as e:
                console.print(f"❌ LLM models [red](error: {e})[/red]")
            
            console.print("🔧 Checking environment variables...")
            import os
            env_vars = ["PYTHONPATH"]
            for var in env_vars:
                if os.getenv(var):
                    console.print(f"✅ {var}")
                else:
                    console.print(f"⚠️  {var} [yellow](not set)[/yellow]")
            
            console.print("⚡ [bold green]Fast validation completed[/bold green]")
            
        except ImportError as e:
            console.print(f"[red]❌ Failed to import validation modules: {e}[/red]")
            raise Exit(1)
            
    else:
        # Show available sub-commands
        console.print("Available sub-commands:")
        console.print("  📊 [bold cyan]comprehensive[/bold cyan]  - Full resource, DB, vector-store, VENV check")
        console.print("  ⚡ [bold cyan]validator[/bold cyan]      - Fast self-test (configs, models, env vars)")
        console.print("\nUsage:")
        console.print("  pynucleus system-status comprehensive")
        console.print("  pynucleus system-status validator")
        
        if subcommand:
            console.print(f"[red]❌ Unknown sub-command: {subcommand}[/red]")
            raise Exit(1)

# ============================================================================
# INGEST COMMAND GROUP - Document Ingestion with Sub-commands
# ============================================================================

ingest_app = Typer(
    name="ingest",
    help="📥 Document ingestion with vectorDB management",
    add_completion=False,
    pretty_exceptions_enable=True,
    rich_markup_mode="rich",
    no_args_is_help=True
)

@ingest_app.command("auto")
@handle_errors
def ingest_auto(
    source: Optional[Path] = Argument(None, help="File or directory to ingest (auto-detects if not specified)"),
    file_types: List[str] = Option([".pdf", ".txt", ".md"], "--types", "-t", help="File extensions to process"),
    recursive: bool = Option(False, "--recursive", "-r", help="Process subdirectories recursively"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """🔍 Auto-detect and ingest documents from default locations"""
    
    logger = setup_logging(verbose)
    
    console.print("🔍 [bold blue]Auto-Detect Document Ingestion[/bold blue]")
    
    try:
        from pynucleus.rag.collector import ingest_single_file, ingest
        
        # Auto-detect source if not specified
        if source is None:
            default_sources = [
                Path("data/01_raw/source_documents"),
                Path("data/01_raw"),
                Path("docs")
            ]
            
            console.print("🔍 Auto-detecting source directory...")
            for default_source in default_sources:
                if default_source.exists():
                    source = default_source
                    console.print(f"📁 Found: {source}")
                    break
            
            if source is None:
                console.print("[red]❌ No source specified and no default directories found[/red]")
                console.print("Available options:")
                console.print("  • Specify a file: pynucleus ingest auto /path/to/file.pdf")
                console.print("  • Specify a directory: pynucleus ingest auto /path/to/directory")
                console.print("  • Create data/01_raw/source_documents/ and place files there")
                raise Exit(1)
        
        # Validate source
        if not source.exists():
            console.print(f"[red]❌ Source not found: {source}[/red]")
            raise Exit(2)
        
        # Single ingestion mode
        if source.is_file():
            console.print(f"📄 Ingesting file: {source}")
            result = ingest_single_file(str(source))
            console.print("✅ [green]File ingested successfully[/green]")
        else:
            console.print(f"📁 Ingesting directory: {source}")
            console.print(f"📄 File types: {', '.join(file_types)}")
            
            with console.status("[bold green]Processing documents..."):
                result = ingest(
                    str(source)
                )
            
            # Handle case where ingest function returns None
            if result is None:
                console.print("✅ [green]Directory ingestion completed[/green]")
            elif result.get('success', True):
                processed = result.get('processed_files', 0)
                console.print(f"✅ [green]Directory ingestion completed: {processed} files processed[/green]")
            else:
                console.print("⚠️  [yellow]Directory ingestion completed with warnings[/yellow]")
        
    except ImportError as e:
        console.print(f"[red]❌ Failed to import ingestion modules: {e}[/red]")
        raise Exit(1)

@ingest_app.command("watch")
@handle_errors
def ingest_watch(
    source: Path = Argument(..., help="Directory to watch for new files"),
    file_types: List[str] = Option([".pdf", ".txt", ".md"], "--types", "-t", help="File extensions to monitor"),
    recursive: bool = Option(False, "--recursive", "-r", help="Watch subdirectories recursively"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """👀 Watch directory for new files and auto-ingest them"""
    
    logger = setup_logging(verbose)
    
    console.print("👀 [bold blue]Document Watch Mode[/bold blue]")
    console.print(f"📁 Watching: {source}")
    console.print(f"📄 File types: {', '.join(file_types)}")
    console.print("💡 Press Ctrl+C to stop watching\n")
    
    # Validate watch directory
    if not source.exists():
        console.print(f"[red]❌ Watch directory does not exist: {source}[/red]")
        raise Exit(2)
    
    if source.is_file():
        console.print("[red]❌ Watch mode requires a directory, not a file[/red]")
        raise Exit(1)
    
    try:
        import time
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        from pynucleus.rag.collector import ingest_single_file
        
        class DocumentHandler(FileSystemEventHandler):
            def __init__(self, file_extensions: List[str], logger):
                self.file_extensions = set(ext.lower() for ext in file_extensions)
                self.logger = logger
            
            def on_created(self, event):
                if event.is_directory:
                    return
                
                file_path = Path(event.src_path)
                
                # Check file extension and ignore hidden/temp files
                if (file_path.suffix.lower() in self.file_extensions and 
                    not file_path.name.startswith('.')):
                    
                    console.print(f"📄 New file detected: {file_path.name}")
                    self.logger.info(f"Processing new file: {file_path}")
                    
                    # Wait briefly for file to be fully written
                    time.sleep(2)
                    
                    try:
                        result = ingest_single_file(str(file_path))
                        console.print(f"✅ Successfully ingested: {file_path.name}")
                        self.logger.info(f"Ingestion result: {result}")
                    except Exception as e:
                        console.print(f"[red]❌ Failed to ingest {file_path.name}: {e}[/red]")
                        self.logger.error(f"Ingestion failed for {file_path}: {e}")
        
        # Setup file watcher
        event_handler = DocumentHandler(file_types, logger)
        observer = Observer()
        observer.schedule(event_handler, str(source), recursive=recursive)
        
        # Start watching
        observer.start()
        console.print("✅ [green]Document watching started[/green]")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n⏹️  Stopping document watcher...")
            observer.stop()
        
        observer.join()
        console.print("✅ Document watcher stopped")
        
    except ImportError as e:
        console.print(f"[red]❌ Required dependencies not found: {e}[/red]")
        console.print("Install with: pip install watchdog")
        raise Exit(1)

@ingest_app.command("single")
@handle_errors
def ingest_single(
    file_path: Path = Argument(..., help="Single file to ingest"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """📄 Ingest a single document file"""
    
    logger = setup_logging(verbose)
    
    console.print("📄 [bold blue]Single File Ingestion[/bold blue]")
    console.print(f"📁 File: {file_path}")
    
    # Validate file
    if not file_path.exists():
        console.print(f"[red]❌ File not found: {file_path}[/red]")
        raise Exit(2)
    
    if file_path.is_dir():
        console.print("[red]❌ Expected a file, got a directory. Use 'ingest auto' for directories.[/red]")
        raise Exit(1)
    
    try:
        from pynucleus.rag.collector import ingest_single_file
        
        with console.status("[bold green]Processing document..."):
            result = ingest_single_file(str(file_path))
        
        console.print("✅ [green]File ingested successfully[/green]")
        
    except ImportError as e:
        console.print(f"[red]❌ Failed to import ingestion modules: {e}[/red]")
        raise Exit(1)

@ingest_app.command("info")
@handle_errors
def ingest_info(
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """📊 Show vector database information and statistics"""
    
    logger = setup_logging(verbose)
    
    console.print("📊 [bold blue]Vector Database Information[/bold blue]")
    
    try:
        from pynucleus.rag.vector_store import VectorStore
        
        # Initialize vector store
        vector_store = VectorStore()
        
        # Get collection info
        console.print("🔍 Retrieving database statistics...")
        
        # Show basic stats using vector store methods
        stats = vector_store.get_index_stats()
        
        if not stats.get('exists', False):
            console.print("📭 [yellow]No vector index found[/yellow]")
            console.print("💡 Use 'pynucleus ingest auto' to create an index")
            return
            
        count = stats.get('total_vectors', 0)
        
        console.print(f"📄 Total documents: {count}")
        
        # Show recent additions if any
        if count > 0:
            console.print("\n📈 [bold]Database Details:[/bold]")
            console.print(f"  • Index files: {', '.join(stats.get('index_files', []))}")
            console.print(f"  • Document count: {count:,}")
            console.print(f"  • Dimensions: {stats.get('dimensions', 'Unknown')}")
            console.print(f"  • Index size: {stats.get('index_size_mb', 'Unknown')} MB")
            console.print(f"  • Last updated: {stats.get('last_updated', 'Unknown')}")
        else:
            console.print("📭 [yellow]No documents found in vector database[/yellow]")
            console.print("💡 Use 'pynucleus ingest auto' to add documents")
        
    except ImportError as e:
        console.print(f"[red]❌ Failed to import vector store modules: {e}[/red]")
        raise Exit(1)

@ingest_app.command("clear")
@handle_errors
def ingest_clear(
    confirm: bool = Option(False, "--confirm", "-y", help="Skip confirmation prompt"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """🗑️  Clear the vector database (removes all ingested documents)"""
    
    logger = setup_logging(verbose)
    
    console.print("🗑️  [bold red]Clear Vector Database[/bold red]")
    
    if not confirm:
        console.print("⚠️  [yellow]This will permanently delete all ingested documents from the vector database.[/yellow]")
        confirm_input = console.input("Type 'yes' to confirm: ")
        if confirm_input.lower() != 'yes':
            console.print("❌ Operation cancelled")
            return
    
    try:
        from pynucleus.rag.vector_store import VectorStore
        
        console.print("🔍 Clearing vector database...")
        
        # Initialize vector store
        vector_store = VectorStore()
        
        # Get count before clearing
        stats = vector_store.get_index_stats()
        old_count = stats.get('total_vectors', 0)
        
        # Clear the index (remove index files)
        import shutil
        if vector_store.index_dir.exists():
            shutil.rmtree(vector_store.index_dir)
            vector_store.index_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"✅ [green]Successfully cleared {old_count} documents from vector database[/green]")
        logger.info(f"Vector database cleared: {old_count} documents removed")
        
    except ImportError as e:
        console.print(f"[red]❌ Failed to import vector store modules: {e}[/red]")
        raise Exit(1)

@ingest_app.command("validate")
@handle_errors
def ingest_validate(
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """✅ Validate existing ingested documents and vector database integrity"""
    
    logger = setup_logging(verbose)
    
    console.print("✅ [bold blue]Vector Database Validation[/bold blue]")
    
    try:
        from pynucleus.rag.vector_store import VectorStore
        
        console.print("🔍 Validating vector database...")
        
        # Initialize vector store
        vector_store = VectorStore()
        
        # Check basic connectivity
        try:
            stats = vector_store.get_index_stats()
            count = stats.get('total_vectors', 0)
            console.print(f"✅ Database connection: OK ({count} documents)")
        except Exception as e:
            console.print(f"❌ Database connection: FAILED ({e})")
            raise Exit(1)
        
        # Validate sample documents
        if count > 0:
            console.print("🔍 Validating document integrity...")
            
            try:
                # Check if index files exist
                if stats.get('exists', False):
                    console.print("✅ Index files: OK")
                    console.print(f"✅ Index structure: {stats.get('index_count', 0)} files found")
                else:
                    console.print("❌ Index files: MISSING")
                    
            except Exception as e:
                console.print(f"❌ Document validation: FAILED ({e})")
        
        # Test search functionality
        console.print("🔍 Testing search functionality...")
        try:
            test_results = vector_store.search("test query", top_k=1)
            if test_results:
                console.print("✅ Search functionality: OK")
            else:
                console.print("⚠️  Search functionality: OK (no results)")
        except Exception as e:
            console.print(f"❌ Search functionality: FAILED ({e})")
        
        console.print("\n✅ [bold green]Vector database validation completed[/bold green]")
        
    except ImportError as e:
        console.print(f"[red]❌ Failed to import validation modules: {e}[/red]")
        raise Exit(1)

# Register ingest app as a sub-app
app.add_typer(ingest_app, name="ingest")

# ============================================================================
# AUTO-INGEST COMMAND - DEPRECATED (Legacy Support)
# ============================================================================

@app.command("auto-ingest")
@handle_errors
def auto_ingest_documents(
    watch_dir: Path = Option("data/01_raw/source_documents", "--watch-dir", "-w", help="Directory to watch for new files"),
    file_types: List[str] = Option([".pdf", ".txt", ".md"], "--types", "-t", help="File extensions to monitor"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging"),
    daemon: bool = Option(False, "--daemon", "-d", help="Run as background daemon"),
    recursive: bool = Option(False, "--recursive", "-r", help="Watch subdirectories recursively")
):
    """🔄 DEPRECATED: Auto-ingest documents with file system watching (use 'ingest --watch' instead)"""
    
    logger = setup_logging(verbose)
    
    # Show deprecation warning
    console.print("⚠️  [bold yellow]DEPRECATION WARNING[/bold yellow]")
    console.print("[yellow]The 'auto-ingest' command is deprecated. Use 'ingest --watch' instead.[/yellow]")
    console.print("[dim]This command will be removed in a future version.[/dim]\n")
    
    logger.warning("DEPRECATION: 'auto-ingest' command used. Use 'ingest --watch' instead.")
    
    # Redirect to new ingest command with watch mode
    console.print("🔄 [blue]Redirecting to new ingest command...[/blue]")
    
    # Redirect to new ingest watch command
    ingest_watch(
        source=watch_dir,
        file_types=file_types,
        recursive=recursive,
        verbose=verbose
    )

# ============================================================================
# ASK COMMAND - Single Question to RAG  
# ============================================================================

@app.command("ask")
@handle_errors
def ask_question(
    question: str = Argument(..., help="Question to ask the RAG system"),
    pretty: bool = Option(True, "--pretty/--plain", help="Use enhanced formatting"),
    model: str = Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", "-m", help="LLM model ID"),
    top_k: int = Option(5, "--top-k", "-k", help="Number of results to retrieve"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """❓ DEPRECATED: Ask a single question to the PyNucleus RAG system (use 'chat --single' instead)"""
    
    logger = setup_logging(verbose)
    
    # Show deprecation warning
    console.print("⚠️  [bold yellow]DEPRECATION WARNING[/bold yellow]")
    console.print("[yellow]The 'ask' command is deprecated. Use 'chat --single \"your question\"' instead.[/yellow]")
    console.print("[dim]This command will be removed in a future version.[/dim]\n")
    
    logger.warning("DEPRECATION: 'ask' command used. Use 'chat --single' instead.")
    
    # Redirect to new chat command with single mode
    console.print("🔄 [blue]Redirecting to new chat command...[/blue]")
    
    # Call the new interactive_chat function directly
    interactive_chat(
        model_id=model,
        top_k=top_k,
        verbose=verbose,
        temperature=0.7,
        max_tokens=512,
        single=question,
        pretty=pretty,
        stream=True,  # Enable streaming by default for ask command
        stream_delay=0.03,
        stream_style="bold green",
        char_mode=False
    )

# ============================================================================
# EVAL COMMAND - Evaluation and Testing
# ============================================================================

eval_app = Typer(name="eval", help="🧪 Evaluation and testing commands")
app.add_typer(eval_app)

@eval_app.command("golden")
@handle_errors
def eval_golden_dataset(
    threshold: float = Option(0.8, "--threshold", "-t", help="Minimum passing threshold"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """🏆 Run golden dataset evaluation"""
    
    logger = setup_logging(verbose)
    
    console.print("🏆 [bold blue]Golden Dataset Evaluation[/bold blue]")
    
    try:
        from pynucleus.eval.golden_eval import run_eval
        
        console.print(f"🎯 Threshold: {threshold}")
        console.print("🔄 Running evaluation...")
        
        with console.status("[bold green]Evaluating..."):
            passed = run_eval()
        
        if passed:
            console.print("✅ [bold green]Golden dataset evaluation PASSED[/bold green]")
        else:
            console.print(f"❌ [bold red]Golden dataset evaluation FAILED (below {threshold})[/bold red]")
            raise Exit(1)
            
    except ImportError as e:
        console.print(f"[red]❌ Failed to import evaluation modules: {e}[/red]")
        raise Exit(1)

@eval_app.command("metrics")
@handle_errors
def compute_rag_metrics(
    retrieved_file: Path = Option(..., "--retrieved", "-r", help="File with retrieved document IDs"),
    relevant_file: Path = Option(..., "--relevant", "-g", help="File with ground truth relevant IDs"), 
    k: int = Option(5, "--k", help="Number of top documents to consider"),
    output_file: Optional[Path] = Option(None, "--output", "-o", help="Output file for metrics"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """📊 Compute RAG retrieval metrics (precision, recall, F1)"""
    
    logger = setup_logging(verbose)
    
    console.print("📊 [bold blue]RAG Metrics Computation[/bold blue]")
    
    # Validate input files
    for file_path, name in [(retrieved_file, "retrieved"), (relevant_file, "relevant")]:
        if not file_path.exists():
            console.print(f"[red]❌ {name.title()} file not found: {file_path}[/red]")
            raise Exit(2)
    
    try:
        from pynucleus.metrics.system_statistics import compute_retrieval_metrics
        
        # Read files
        with open(retrieved_file, 'r') as f:
            retrieved_ids = [line.strip() for line in f if line.strip()]
        
        with open(relevant_file, 'r') as f:
            relevant_ids = set(line.strip() for line in f if line.strip())
        
        # Compute metrics
        console.print("🔄 Computing metrics...")
        metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k)
        
        # Display results
        console.print(f"\n[bold green]RAG Retrieval Metrics (k={k})[/bold green]")
        console.print(f"📊 Precision: {metrics.precision:.3f}")
        console.print(f"📈 Recall: {metrics.recall:.3f}")
        console.print(f"🎯 F1 Score: {metrics.f1:.3f}")
        console.print(f"📋 Retrieved: {metrics.num_retrieved}")
        console.print(f"✅ Relevant: {metrics.num_relevant}")
        
        # Save results if requested
        if output_file:
            import json
            metrics_data = {
                'k': metrics.k,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1': metrics.f1,
                'num_relevant': metrics.num_relevant,
                'num_retrieved': metrics.num_retrieved,
                'retrieved_ids': retrieved_ids[:k],
                'relevant_ids': list(relevant_ids)
            }
            with open(output_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            console.print(f"💾 Metrics saved to {output_file}")
            
    except ImportError as e:
        console.print(f"[red]❌ Failed to import metrics modules: {e}[/red]")
        raise Exit(1)

# ============================================================================
# SERVE COMMAND - Web Server Management
# ============================================================================

serve_app = Typer(name="serve", help="🌐 Web server management commands")
app.add_typer(serve_app)

@serve_app.command("start")
@handle_errors
def start_server(
    port: int = Option(5001, "--port", "-p", help="Server port"),
    host: str = Option("0.0.0.0", "--host", help="Server host"),
    workers: int = Option(1, "--workers", "-w", help="Number of worker processes"),
    reload: bool = Option(False, "--reload", help="Enable auto-reload for development"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """🚀 Start the PyNucleus web server"""
    
    logger = setup_logging(verbose)
    
    console.print("🌐 [bold blue]PyNucleus Web Server[/bold blue]")
    console.print(f"🔗 Starting server at http://{host}:{port}")
    
    try:
        import subprocess
        import os
        
        # Check if server is already running
        from scripts.check_ports import check_port
        if not check_port(port):
            console.print(f"[yellow]⚠️  Port {port} is already in use[/yellow]")
            should_kill = console.input("Kill existing server? [y/N]: ").lower().strip()
            if should_kill == 'y':
                # Kill existing server
                os.system(f"pkill -f 'port {port}' || lsof -ti:{port} | xargs kill -9")
                console.print("🔪 Killed existing server")
            else:
                console.print("❌ Server start cancelled")
                raise Exit(1)
        
        # Start server using gunicorn for production
        cmd = [
            "gunicorn",
            "--bind", f"{host}:{port}",
            "--workers", str(workers),
            "--timeout", "300",
            "--worker-class", "sync",
            "src.pynucleus.api.wsgi:app"
        ]
        
        if reload:
            cmd.extend(["--reload", "--reload-extra-file", "src/"])
        
        console.print(f"🚀 Starting server with command: {' '.join(cmd)}")
        
        # Start server
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Failed to start server: {e}[/red]")
        raise Exit(1)
    except FileNotFoundError:
        console.print("[red]❌ Gunicorn not found. Install with: pip install gunicorn[/red]")
        raise Exit(1)

@serve_app.command("stop")
@handle_errors
def stop_server(
    port: int = Option(5001, "--port", "-p", help="Server port to stop"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """⏹️  Stop the PyNucleus web server"""
    
    logger = setup_logging(verbose)
    
    console.print("⏹️  [bold blue]Stopping PyNucleus Web Server[/bold blue]")
    
    try:
        import os
        import subprocess
        
        # Kill processes on the specified port
        result = subprocess.run([
            "lsof", "-ti", f":{port}"
        ], capture_output=True, text=True)
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    console.print(f"🔪 Killed process {pid}")
                except ProcessLookupError:
                    pass
            console.print(f"✅ Server on port {port} stopped")
        else:
            console.print(f"⚠️  No server found running on port {port}")
            
    except FileNotFoundError:
        console.print("[red]❌ lsof command not found[/red]")
        raise Exit(1)

@serve_app.command("restart")
@handle_errors  
def restart_server(
    port: int = Option(5001, "--port", "-p", help="Server port"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """🔄 Restart the PyNucleus web server"""
    
    console.print("🔄 [bold blue]Restarting PyNucleus Web Server[/bold blue]")
    
    # Stop then start
    stop_server(port=port, verbose=verbose)
    start_server(port=port, verbose=verbose)

# ============================================================================
# DIAGNOSTICS COMMAND - System Health and Diagnostics
# ============================================================================

# ============================================================================
# HEALTH CHECK COMMAND GROUP - System Diagnostics with Sub-commands
# ============================================================================

health_app = Typer(
    name="health",
    help="🩺 System health checks and diagnostics",
    add_completion=False,
    pretty_exceptions_enable=True,
    rich_markup_mode="rich",
    no_args_is_help=True
)

@health_app.command("quick")
@handle_errors
def health_quick(
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """⚡ Fast health check - essential systems only"""
    
    logger = setup_logging(verbose)
    
    console.print("⚡ [bold blue]Quick Health Check[/bold blue]")
    
    try:
        # Check Python environment
        console.print("🐍 Python environment...")
        console.print(f"  ✅ Python {sys.version.split()[0]}")
        
        # Check critical imports
        console.print("📦 Critical dependencies...")
        critical_deps = ["typer", "rich", "pathlib"]
        for dep in critical_deps:
            try:
                __import__(dep)
                console.print(f"  ✅ {dep}")
            except ImportError:
                console.print(f"  ❌ {dep} [red](missing)[/red]")
        
        # Check configuration files
        console.print("📋 Configuration files...")
        config_files = ["configs/production_config.json", "configs/development_config.json"]
        for config_file in config_files:
            if Path(config_file).exists():
                console.print(f"  ✅ {config_file}")
            else:
                console.print(f"  ⚠️  {config_file} [yellow](missing)[/yellow]")
        
        # Check data directories
        console.print("📁 Data directories...")
        data_dirs = ["data", "logs"]
        for data_dir in data_dirs:
            if Path(data_dir).exists():
                console.print(f"  ✅ {data_dir}/")
            else:
                console.print(f"  ⚠️  {data_dir}/ [yellow](missing)[/yellow]")
        
        console.print("\n⚡ [bold green]Quick health check completed[/bold green]")
        
    except Exception as e:
        console.print(f"[red]❌ Quick health check failed: {e}[/red]")
        raise Exit(1)

@health_app.command("full")
@handle_errors
def health_full(
    output_file: Optional[Path] = Option(None, "--output", "-o", help="Save diagnostics report"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """🔍 Comprehensive system diagnostics"""
    
    logger = setup_logging(verbose)
    
    console.print("🔍 [bold blue]Comprehensive System Diagnostics[/bold blue]")
    
    try:
        from pynucleus.diagnostics.runner import run_comprehensive_diagnostics
        
        console.print("🔄 Running comprehensive diagnostics...")
        
        with console.status("[bold green]Analyzing system..."):
            result = run_comprehensive_diagnostics()
        
        # Display results
        if result.get('success', True):
            console.print("✅ [bold green]System is healthy[/bold green]")
        else:
            console.print("⚠️  [yellow]System has issues[/yellow]")
        
        # Display key findings
        if 'issues' in result:
            for issue in result['issues']:
                console.print(f"❌ {issue}")
        
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            console.print(f"💾 Report saved to {output_file}")
            
    except ImportError as e:
        console.print(f"[red]❌ Failed to import diagnostics modules: {e}[/red]")
        raise Exit(1)

@health_app.command("network")
@handle_errors
def health_network(
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """🌐 Network connectivity and external service checks"""
    
    logger = setup_logging(verbose)
    
    console.print("🌐 [bold blue]Network Connectivity Check[/bold blue]")
    
    try:
        import socket
        import urllib.request
        
        # Check internet connectivity
        console.print("🌍 Testing internet connectivity...")
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            console.print("  ✅ Internet connection: OK")
        except OSError:
            console.print("  ❌ Internet connection: FAILED")
        
        # Check external services
        console.print("🔗 Testing external services...")
        test_urls = [
            ("Hugging Face", "https://huggingface.co"),
            ("PyPI", "https://pypi.org"),
            ("GitHub", "https://github.com")
        ]
        
        for name, url in test_urls:
            try:
                response = urllib.request.urlopen(url, timeout=5)
                if response.getcode() == 200:
                    console.print(f"  ✅ {name}: OK")
                else:
                    console.print(f"  ⚠️  {name}: HTTP {response.getcode()}")
            except Exception as e:
                console.print(f"  ❌ {name}: FAILED ({e})")
        
        # Check local ports
        console.print("🔌 Checking local ports...")
        ports_to_check = [5001, 8000, 3000]
        for port in ports_to_check:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                if result == 0:
                    console.print(f"  ⚠️  Port {port}: IN USE")
                else:
                    console.print(f"  ✅ Port {port}: Available")
                sock.close()
            except Exception:
                console.print(f"  ❓ Port {port}: Unknown")
        
        console.print("\n🌐 [bold green]Network check completed[/bold green]")
        
    except Exception as e:
        console.print(f"[red]❌ Network check failed: {e}[/red]")
        raise Exit(1)

@health_app.command("storage")
@handle_errors
def health_storage(
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """💾 Storage and file system checks"""
    
    logger = setup_logging(verbose)
    
    console.print("💾 [bold blue]Storage and File System Check[/bold blue]")
    
    try:
        import shutil
        import os
        
        # Check disk space
        console.print("📊 Disk space analysis...")
        total, used, free = shutil.disk_usage(".")
        total_gb = total // (1024**3)
        used_gb = used // (1024**3)
        free_gb = free // (1024**3)
        
        console.print(f"  📁 Total: {total_gb} GB")
        console.print(f"  📊 Used: {used_gb} GB")
        console.print(f"  💿 Free: {free_gb} GB")
        
        if free_gb < 1:
            console.print("  ⚠️  [yellow]Low disk space warning[/yellow]")
        else:
            console.print("  ✅ Disk space: OK")
        
        # Check critical directories
        console.print("📁 Critical directories...")
        critical_dirs = [
            "data/01_raw",
            "data/03_processed",
            "logs",
            "configs"
        ]
        
        for dir_path in critical_dirs:
            path = Path(dir_path)
            if path.exists():
                if path.is_dir():
                    file_count = len(list(path.rglob("*")))
                    console.print(f"  ✅ {dir_path}/ ({file_count} items)")
                else:
                    console.print(f"  ❌ {dir_path} [red](not a directory)[/red]")
            else:
                console.print(f"  ⚠️  {dir_path}/ [yellow](missing)[/yellow]")
        
        # Check file permissions
        console.print("🔒 File permissions...")
        test_file = "logs/test_permissions.tmp"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            console.print("  ✅ Write permissions: OK")
        except Exception as e:
            console.print(f"  ❌ Write permissions: FAILED ({e})")
        
        # Check database files
        console.print("🗄️  Database files...")
        db_files = [
            "data/03_processed/chromadb/chroma.sqlite3",
            "pynucleus.db"
        ]
        
        for db_file in db_files:
            if Path(db_file).exists():
                size = Path(db_file).stat().st_size
                size_mb = size / (1024**2)
                console.print(f"  ✅ {db_file} ({size_mb:.1f} MB)")
            else:
                console.print(f"  ⚠️  {db_file} [yellow](missing)[/yellow]")
        
        console.print("\n💾 [bold green]Storage check completed[/bold green]")
        
    except Exception as e:
        console.print(f"[red]❌ Storage check failed: {e}[/red]")
        raise Exit(1)

# Register health app as a sub-app
app.add_typer(health_app, name="health")

# ============================================================================
# LEGACY DIAGNOSTICS COMMAND - DEPRECATED (backward compatibility)
# ============================================================================

@app.command("diagnostics")
@handle_errors
def legacy_diagnostics(
    comprehensive: bool = Option(False, "--comprehensive", "-c", help="Run comprehensive system diagnostics"),
    output_file: Optional[Path] = Option(None, "--output", "-o", help="Save diagnostics report"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """🩺 DEPRECATED: System diagnostics (use 'health' command instead)"""
    
    logger = setup_logging(verbose)
    
    # Show deprecation warning
    console.print("⚠️  [bold yellow]DEPRECATION WARNING[/bold yellow]")
    console.print("[yellow]The 'diagnostics' command is deprecated. Use 'health' command instead.[/yellow]")
    console.print("[dim]This command will be removed in a future version.[/dim]\n")
    
    logger.warning("DEPRECATION: 'diagnostics' command used. Use 'health' command instead.")
    
    # Redirect to new health command
    console.print("🔄 [blue]Redirecting to new health command...[/blue]")
    
    if comprehensive:
        # Call health full
        health_full(output_file=output_file, verbose=verbose)
    else:
        # Call health quick
        health_quick(verbose=verbose)

# ============================================================================
# VERSION COMMAND
# ============================================================================

@app.command("version")
@handle_errors
def show_version():
    """📋 Show PyNucleus version information"""
    
    console.print("🧪 [bold blue]PyNucleus Chemical Process Simulation & RAG System[/bold blue]")
    
    try:
        # Try to get version from package
        import pkg_resources
        version = pkg_resources.get_distribution("pynucleus").version
    except:
        version = "development"
    
    console.print(f"📦 Version: {version}")
    console.print(f"🐍 Python: {sys.version.split()[0]}")
    console.print(f"📍 Location: {Path(__file__).parent.parent.parent}")
    
    # Show key dependencies
    deps = ["typer", "rich", "chromadb", "torch", "transformers"]
    console.print("\n📚 [bold]Key Dependencies:[/bold]")
    
    for dep in deps:
        try:
            import importlib
            importlib.import_module(dep)
            console.print(f"  ✅ {dep}")
        except ImportError:
            console.print(f"  ❌ {dep} [red](missing)[/red]")

# ============================================================================
# RAG COMMAND - Unified RAG System Operations
# ============================================================================

rag_app = Typer(
    name="rag",
    help="Unified RAG system: document vectorization, vectorDB, auto-ingest, evaluation, and more.",
    add_completion=False,
    pretty_exceptions_enable=True,
    rich_markup_mode="rich",
    no_args_is_help=True
)

@app.command("rag")
@handle_errors
def rag_entry(ctx: Context):
    """Unified RAG system: document vectorization, vectorDB, auto-ingest, evaluation, and more."""
    console.print("\n🧪 [bold blue]PyNucleus RAG System[/bold blue]")
    console.print("This is the main entry point for all RAG-related operations.")
    console.print("\nAvailable subcommands:")
    console.print("   • vectorize      Ingest/index documents, chunk analysis, clear DB, etc.")
    console.print("   • vectordb       VectorDB info, clear, backup, restore, etc.")
    console.print("   • auto-ingest    Watch directories, batch/single file, recursive, etc.")
    console.print("   • eval           Run golden dataset, metrics, etc.")
    console.print("\nRun 'rag <subcommand> --help' for more details.")
    ctx.exit()

# Register rag_app as a sub-app
app.add_typer(rag_app, name="rag")

# ============================================================================
# COMPLETION COMMAND
# ============================================================================

@app.command("completion")
@handle_errors
def install_completion(
    shell: str = Option(
        "bash", 
        "--shell", 
        "-s", 
        help="Shell type (bash, zsh, fish, powershell)",
        show_default=True
    ),
    install: bool = Option(
        False, 
        "--install", 
        "-i", 
        help="Install completion automatically"
    ),
    requirements: bool = Option(
        False,
        "--requirements",
        "-r", 
        help="Also install/update requirements.txt dependencies"
    ),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """🔧 Install shell completion and optionally requirements for pynucleus"""
    
    logger = setup_logging(verbose)
    
    console.print("🔧 [bold blue]PyNucleus Shell Completion & Setup[/bold blue]")
    console.print(f"🐚 Shell: {shell}")
    if requirements:
        console.print("📦 Requirements: Will be installed/updated")
    
    import subprocess
    import os
    from pathlib import Path
    
    # Install requirements if requested
    if requirements:
        console.print("\n📦 [yellow]Installing/updating requirements...[/yellow]")
        req_file = Path("requirements.txt")
        if req_file.exists():
            try:
                cmd = ["pip", "install", "-r", "requirements.txt", "--upgrade"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    console.print("✅ [green]Requirements installed successfully![/green]")
                else:
                    console.print(f"❌ [red]Requirements installation failed: {result.stderr}[/red]")
            except Exception as e:
                console.print(f"❌ [red]Error installing requirements: {e}[/red]")
        else:
            console.print("⚠️ [yellow]requirements.txt not found[/yellow]")
    
    # Get the completion script
    try:
        if install:
            console.print("\n📥 [yellow]Installing completion automatically...[/yellow]")
            
            # Run the completion installation command
            cmd = ["pynucleus", "--install-completion", shell]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print("✅ [green]Completion installed successfully![/green]")
                console.print("🔄 [yellow]Please restart your terminal or run:[/yellow]")
                console.print(f"[blue]source ~/.{shell}rc[/blue]")
            else:
                console.print(f"❌ [red]Installation failed: {result.stderr}[/red]")
                raise Exit(1)
        else:
            console.print("\n📋 [yellow]To install completion manually:[/yellow]")
            console.print(f"[blue]pynucleus --install-completion {shell}[/blue]")
            console.print("\n🔄 [yellow]Or use the automatic installation:[/yellow]")
            console.print("[blue]pynucleus completion --install[/blue]")
            console.print("[blue]pynucleus completion --install --requirements[/blue] (includes pip install)")
            
            # Show completion script
            console.print("\n📄 [yellow]Completion script preview:[/yellow]")
            cmd = ["pynucleus", "--show-completion", shell]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print(f"[dim]{result.stdout[:200]}...[/dim]")
            else:
                console.print("[red]❌ Failed to generate completion script[/red]")
                
    except Exception as e:
        console.print(f"❌ [red]Error setting up completion: {e}[/red]")
        raise Exit(1)
    
    console.print("\n💡 [yellow]Supported shells: bash, zsh, fish, powershell[/yellow]")
    console.print("🔗 [blue]After installation, you'll have tab completion for all pynucleus commands![/blue]")

# ============================================================================
# GIT INTEGRATION COMMANDS
# ============================================================================

@app.command("git-commit")
@handle_errors
def git_commit(
    message: str = Argument(..., help="Commit message"),
    add_all: bool = Option(False, "--add-all", "-a", help="Add all tracked files before commit"),
    push: bool = Option(False, "--push", "-p", help="Push to remote after commit"),
    branch: Optional[str] = Option(None, "--branch", "-b", help="Branch to push to (default: current)"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """📝 Easy git commit with optional add-all and push"""
    
    logger = setup_logging(verbose)
    
    console.print("📝 [bold blue]Git Commit Operation[/bold blue]")
    console.print(f"💬 Message: {message}")
    
    import subprocess
    import os
    
    try:
        # Check if in git repo
        result = subprocess.run(["git", "status"], capture_output=True, text=True)
        if result.returncode != 0:
            console.print("❌ [red]Not in a git repository[/red]")
            raise Exit(1)
        
        # Add all files if requested
        if add_all:
            console.print("📦 [yellow]Adding all tracked files...[/yellow]")
            result = subprocess.run(["git", "add", "-A"], capture_output=True, text=True)
            if result.returncode != 0:
                console.print(f"❌ [red]Git add failed: {result.stderr}[/red]")
                raise Exit(1)
            console.print("✅ [green]Files added[/green]")
        
        # Check for changes to commit
        result = subprocess.run(["git", "diff", "--cached", "--exit-code"], capture_output=True)
        if result.returncode == 0:
            console.print("⚠️ [yellow]No changes staged for commit[/yellow]")
            return
        
        # Commit
        console.print("💾 [yellow]Creating commit...[/yellow]")
        result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"❌ [red]Commit failed: {result.stderr}[/red]")
            raise Exit(1)
        
        console.print("✅ [green]Commit created successfully![/green]")
        
        # Push if requested
        if push:
            console.print("🚀 [yellow]Pushing to remote...[/yellow]")
            
            # Get current branch if not specified
            if not branch:
                result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
                if result.returncode == 0:
                    branch = result.stdout.strip()
                else:
                    branch = "main"
            
            push_cmd = ["git", "push", "origin", branch]
            result = subprocess.run(push_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print(f"✅ [green]Pushed to {branch} successfully![/green]")
            else:
                console.print(f"❌ [red]Push failed: {result.stderr}[/red]")
                console.print("💡 [yellow]Commit was successful, only push failed[/yellow]")
        
    except Exception as e:
        console.print(f"❌ [red]Error during git operation: {e}[/red]")
        raise Exit(1)

@app.command("git-clone")
@handle_errors
def git_clone(
    repo_url: str = Argument(..., help="Repository URL to clone"),
    directory: Optional[str] = Option(None, "--dir", "-d", help="Directory name (default: repo name)"),
    branch: Optional[str] = Option(None, "--branch", "-b", help="Specific branch to clone"),
    depth: Optional[int] = Option(None, "--depth", help="Shallow clone depth"),
    setup: bool = Option(False, "--setup", "-s", help="Run setup after clone (install requirements)"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """📥 Easy git clone with optional setup"""
    
    logger = setup_logging(verbose)
    
    console.print("📥 [bold blue]Git Clone Operation[/bold blue]")
    console.print(f"🔗 Repository: {repo_url}")
    
    import subprocess
    import os
    from pathlib import Path
    
    try:
        # Build clone command
        cmd = ["git", "clone"]
        
        if branch:
            cmd.extend(["--branch", branch])
            console.print(f"🌿 Branch: {branch}")
        
        if depth:
            cmd.extend(["--depth", str(depth)])
            console.print(f"📏 Depth: {depth}")
        
        cmd.append(repo_url)
        
        if directory:
            cmd.append(directory)
            target_dir = directory
        else:
            # Extract repo name from URL
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            target_dir = repo_name
        
        console.print(f"📁 Target directory: {target_dir}")
        
        # Check if directory exists
        if Path(target_dir).exists():
            console.print(f"⚠️ [yellow]Directory {target_dir} already exists[/yellow]")
            choice = console.input("Continue anyway? (y/N): ").lower().strip()
            if choice != 'y':
                console.print("🚫 [yellow]Clone cancelled[/yellow]")
                return
        
        # Clone repository
        console.print("📥 [yellow]Cloning repository...[/yellow]")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            console.print(f"❌ [red]Clone failed: {result.stderr}[/red]")
            raise Exit(1)
        
        console.print("✅ [green]Repository cloned successfully![/green]")
        
        # Setup if requested
        if setup:
            console.print("🔧 [yellow]Running setup...[/yellow]")
            
            # Change to target directory
            os.chdir(target_dir)
            
            # Look for requirements files
            req_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
            found_req = False
            
            for req_file in req_files:
                if Path(req_file).exists():
                    console.print(f"📦 [yellow]Installing {req_file}...[/yellow]")
                    if req_file == "pyproject.toml":
                        cmd = ["pip", "install", "-e", "."]
                    else:
                        cmd = ["pip", "install", "-r", req_file]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        console.print(f"✅ [green]{req_file} installed successfully![/green]")
                    else:
                        console.print(f"❌ [red]{req_file} installation failed: {result.stderr}[/red]")
                    found_req = True
            
            if not found_req:
                console.print("ℹ️ [blue]No requirements file found, skipping installation[/blue]")
            
            # Change back to original directory
            os.chdir("..")
        
        console.print(f"\n🎉 [bold green]Ready! Repository cloned to: {target_dir}[/bold green]")
        if not setup:
            console.print("💡 [yellow]Tip: Use --setup flag to automatically install requirements[/yellow]")
        
    except Exception as e:
        console.print(f"❌ [red]Error during clone operation: {e}[/red]")
        raise Exit(1)

@app.command("git-pull")
@handle_errors
def git_pull(
    branch: Optional[str] = Option(None, "--branch", "-b", help="Branch to pull from (default: current)"),
    remote: str = Option("origin", "--remote", "-r", help="Remote name"),
    rebase: bool = Option(False, "--rebase", help="Rebase instead of merge"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose logging")
):
    """🔄 Easy git pull with conflict handling"""
    
    logger = setup_logging(verbose)
    
    console.print("🔄 [bold blue]Git Pull Operation[/bold blue]")
    
    import subprocess
    import os
    
    try:
        # Check if in git repo
        result = subprocess.run(["git", "status"], capture_output=True, text=True)
        if result.returncode != 0:
            console.print("❌ [red]Not in a git repository[/red]")
            raise Exit(1)
        
        # Get current branch if not specified
        if not branch:
            result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
            if result.returncode == 0:
                branch = result.stdout.strip()
            else:
                branch = "main"
        
        console.print(f"🌿 Branch: {branch}")
        console.print(f"📡 Remote: {remote}")
        
        # Check for uncommitted changes
        result = subprocess.run(["git", "diff", "--exit-code"], capture_output=True)
        if result.returncode != 0:
            console.print("⚠️ [yellow]You have uncommitted changes[/yellow]")
            choice = console.input("Continue with pull? (y/N): ").lower().strip()
            if choice != 'y':
                console.print("🚫 [yellow]Pull cancelled[/yellow]")
                return
        
        # Build pull command
        cmd = ["git", "pull"]
        if rebase:
            cmd.append("--rebase")
        cmd.extend([remote, branch])
        
        console.print("📥 [yellow]Pulling from remote...[/yellow]")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("✅ [green]Pull completed successfully![/green]")
            
            # Show what was updated
            if result.stdout.strip():
                console.print("\n📋 [blue]Pull summary:[/blue]")
                console.print(f"[dim]{result.stdout}[/dim]")
        else:
            # Handle common error cases
            if "merge conflict" in result.stderr.lower() or "conflict" in result.stderr.lower():
                console.print("⚠️ [yellow]Merge conflicts detected![/yellow]")
                console.print("🔧 [blue]Please resolve conflicts manually, then run:[/blue]")
                console.print("   [cyan]git add .[/cyan]")
                console.print("   [cyan]git commit[/cyan]")
            elif "diverged" in result.stderr.lower():
                console.print("🔀 [yellow]Branch has diverged from remote[/yellow]")
                console.print("💡 [blue]Consider using --rebase flag or merge manually[/blue]")
            else:
                console.print(f"❌ [red]Pull failed: {result.stderr}[/red]")
                raise Exit(1)
        
        # Show current status
        console.print("\n📊 [blue]Current status:[/blue]")
        result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            console.print(f"[dim]{result.stdout}[/dim]")
        else:
            console.print("[green]Working directory clean[/green]")
        
    except Exception as e:
        console.print(f"❌ [red]Error during pull operation: {e}[/red]")
        raise Exit(1)

# ============================================================================
# POST-COMMAND OPTIONS SYSTEM
# ============================================================================

# Old show_post_command_options and show_context_menu functions removed
# They have been replaced by the enhanced menu system in menus.py

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def show_interactive_menu():
    """Display interactive menu for main PyNucleus commands"""
    while True:  # Main menu loop
        console.print("\n🧪 [bold blue]PyNucleus Chemical Process Simulation & RAG System[/bold blue]")
        console.print("Choose an option to get started:\n")
        
        # Main commands with descriptions (updated structure)
        commands = [
            ("1", "run", "Execute pipeline"),
            ("2", "chat", "Chat with LLM"),
            ("3", "build", "Build plant simulation"),
            ("4", "system-status", "System status"),
            ("5", "ingest", "Ingest documents"),
            ("6", "health", "Health check"),
            ("7", "version", "Show version"),
            ("8", "eval", "Run evaluations"),
            ("9", "serve", "Web server"),
            ("g", "git", "Git operations"),
            ("c", "completion", "Install shell completion & requirements"),
            ("0", "help", "Show help"),
            ("q", "quit", "Exit")
        ]
        
        # Display numbered menu
        for number, command, description in commands:
            console.print(f"[bold cyan]{number:>2}[/bold cyan]  {description}")
        
        console.print("\n" + "─" * 80)
        console.print("[dim]💡 Tip: You can also run commands directly, e.g., './pynucleus run --help'[/dim]")
        console.print("[dim]💡 Tip: Use '--verbose' flag for detailed output[/dim]")
        console.print("[dim]💡 Tip: Press 'c' to install shell completion for tab-completion[/dim]\n")
        
        # Get user choice
        choice = get_main_menu_choice(commands)
        if choice == 'quit':
            return
        elif choice == 'help':
            console.print("\n" + "=" * 80)
            console.print("[bold blue]PyNucleus CLI Help[/bold blue]")
            console.print("=" * 80)
            sys.argv = [sys.argv[0], '--help']
            app()
            # After showing help, return to menu
            continue
        
        # Handle command execution with context
        result = execute_command_with_context(choice, commands)
        if result == 'quit':
            return
        elif result == 'main':
            continue  # Stay in main menu loop

def get_main_menu_choice(commands):
    """Get and validate user choice from main menu"""
    max_attempts = 5
    attempts = 0
    
    while attempts < max_attempts:
        try:
            choice = console.input("[bold green]Enter your choice (0-9, g, c, q to quit): [/bold green]").strip().lower()
            
            if choice == 'q' or choice == 'quit':
                console.print("👋 [yellow]Goodbye![/yellow]")
                return 'quit'
            
            if choice == '0' or choice == 'help':
                return 'help'
            
            if choice == 'c' or choice == 'completion':
                return 'completion'
            
            if choice == 'g' or choice == 'git':
                return 'git'
            
            # Validate numeric choice
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= 9:
                    return commands[choice_num - 1][1]  # Return command name
                else:
                    console.print(f"[red]❌ Invalid choice: {choice_num}. Please enter 0-9, g, c, or 'q'[/red]")
                    attempts += 1
            except ValueError:
                console.print(f"[red]❌ Invalid input: '{choice}'. Please enter 0-9, g, c, or 'q'[/red]")
                attempts += 1
                
        except (KeyboardInterrupt, EOFError):
            console.print("\n👋 [yellow]Goodbye![/yellow]")
            return 'quit'
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            attempts += 1
            
    # If too many attempts, exit gracefully
    console.print("[yellow]⚠️  Too many invalid attempts. Exiting.[/yellow]")
    return 'quit'

def execute_command_with_context(command_name: str, main_commands: List[tuple]):
    """Execute a command and handle post-command options based on context"""
    from .menus import enhanced_context_menu, simple_command_wrapper
    
    # Context-specific command definitions
    contexts = {
        "ingest": [
            ("1", "Auto-ingest documents", "auto"),
            ("2", "Watch directory for files", "watch"),
            ("3", "Ingest single file", "single"),
            ("4", "Show ingest information", "info"),
            ("5", "Clear document database", "clear"),
            ("6", "Validate ingest system", "validate")
        ],
        "health": [
            ("1", "Quick health check", "quick"),
            ("2", "Full system diagnostics", "full"),
            ("3", "Network connectivity check", "network"),
            ("4", "Storage system check", "storage")
        ],
        "eval": [
            ("1", "Run golden dataset evaluation", "golden"),
            ("2", "Compute RAG metrics", "metrics")
        ],
        "serve": [
            ("1", "Start web server", "start"),
            ("2", "Stop web server", "stop"),
            ("3", "Restart web server", "restart")
        ],
        "system-status": [
            ("1", "Comprehensive system status", "comprehensive"),
            ("2", "System validator", "validator")
        ],
        "git": [
            ("1", "Git commit (with message)", "commit"),
            ("2", "Git clone repository", "clone"),
            ("3", "Git pull from remote", "pull")
        ]
    }
    
    # Commands that don't have sub-contexts
    simple_commands = ["run", "chat", "build", "version", "completion"]
    
    if command_name in simple_commands:
        # Execute simple command with enhanced menu wrapper
        def command_executor():
            execute_simple_command(command_name)
        
        simple_command_wrapper(command_name, command_executor)
    elif command_name in contexts:
        # Handle contextual commands with enhanced sub-menus
        menu_options = contexts[command_name]
        def command_executor(subcommand: str):
            execute_contextual_subcommand(command_name, subcommand)
        
        enhanced_context_menu(command_name.title(), menu_options, command_executor)
    else:
        console.print(f"[red]❌ Unknown command: {command_name}[/red]")
        return 'main'

def execute_simple_command(command_name: str):
    """Execute a simple command that doesn't have sub-commands"""
    try:
        if command_name == "run":
            # Handle the run command by prompting for config file
            console.print("🚀 [bold blue]Execute Pipeline[/bold blue]")
            console.print("Available configuration files:\n")
            
            # List available config files
            config_dir = Path("configs")
            config_files = []
            if config_dir.exists():
                for file in config_dir.iterdir():
                    if file.suffix == '.json':
                        config_files.append(file)
            
            if not config_files:
                console.print("[red]❌ No configuration files found in configs/ directory[/red]")
                return
            
            # Display config options
            for i, config_file in enumerate(config_files, 1):
                console.print(f"[bold cyan]{i:>2}[/bold cyan]  {config_file.name}")
            
            console.print(f"[bold cyan] 0[/bold cyan]  Enter custom config path")
            
            # Get user choice
            while True:
                try:
                    choice = console.input(f"\n[bold green]Select config file (1-{len(config_files)}, 0 for custom): [/bold green]").strip()
                    
                    if choice == "0":
                        custom_path = console.input("[bold green]Enter config file path: [/bold green]").strip()
                        if custom_path and Path(custom_path).exists():
                            config_path = custom_path
                            break
                        else:
                            console.print("[red]❌ File not found. Please try again.[/red]")
                            continue
                    
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(config_files):
                        config_path = str(config_files[choice_num - 1])
                        break
                    else:
                        console.print(f"[red]❌ Invalid choice. Please enter 1-{len(config_files)} or 0.[/red]")
                        
                except ValueError:
                    console.print(f"[red]❌ Invalid input. Please enter a number.[/red]")
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[yellow]Operation cancelled.[/yellow]")
                    return
            
            console.print(f"\n[yellow]🔄 Running pipeline with config: {config_path}[/yellow]")
            sys.argv = [sys.argv[0], 'run', '--config', config_path]
        elif command_name == "build":
            sys.argv = [sys.argv[0], 'build', '--interactive']
        elif command_name == "chat":
            console.print("💬 [bold blue]Starting Interactive Chat[/bold blue]")
            console.print("💡 Type 'exit', 'quit', or press Ctrl+C to end the session\n")
            sys.argv = [sys.argv[0], 'chat']
        elif command_name == "completion":
            console.print("🔧 [bold blue]Shell Completion Setup[/bold blue]")
            console.print("💡 This will help you install tab completion for pynucleus commands\n")
            sys.argv = [sys.argv[0], 'completion']
        else:
            sys.argv = [sys.argv[0], command_name]
        
        app()
    except SystemExit:
        # Handle normal command completion
        pass
    except Exception as e:
        console.print(f"[red]❌ Command failed: {e}[/red]")
        raise  # Re-raise so the wrapper can handle it

# Old handle_contextual_command function removed - replaced by enhanced_context_menu from menus.py

def execute_contextual_subcommand(context_name: str, subcommand: str):
    """Execute a sub-command within a specific context"""
    try:
        if context_name == "ingest":
            execute_ingest_command(subcommand)
        elif context_name == "health":
            sys.argv = [sys.argv[0], 'health', subcommand]
            app()
        elif context_name == "eval":
            execute_eval_command(subcommand)
        elif context_name == "serve":
            execute_serve_command(subcommand)
        elif context_name == "system-status":
            args = [sys.argv[0], 'system-status', subcommand]
            sys.argv = args
            app()
        elif context_name == "git":
            execute_git_command(subcommand)
    except SystemExit:
        # Handle normal command completion
        pass
    except Exception as e:
        console.print(f"[red]❌ Command failed: {e}[/red]")
        raise  # Re-raise so the wrapper can handle it

def execute_ingest_command(subcommand: str):
    """Execute ingest sub-commands with interactive prompts"""
    args = [sys.argv[0], 'ingest', subcommand]
    
    if subcommand in ['auto', 'single']:
        source = console.input("📁 Source path (optional, press Enter to skip): ").strip()
        if source:
            args.append(source)
    elif subcommand == 'watch':
        source = console.input("📁 Watch directory (default: data/01_raw/source_documents): ").strip()
        if not source:
            source = "data/01_raw/source_documents"
        args.append(source)
    
    if subcommand in ['auto', 'watch']:
        recursive = console.input("📂 Process subdirectories? (y/N): ").lower().startswith('y')
        if recursive:
            args.extend(['--recursive'])
    
    sys.argv = args
    app()

def execute_eval_command(subcommand: str):
    """Execute eval sub-commands with interactive prompts"""
    args = [sys.argv[0], 'eval', subcommand]
    
    if subcommand == 'metrics':
        retrieved_file = console.input("📁 Retrieved file path: ").strip()
        relevant_file = console.input("📁 Relevant file path: ").strip()
        if retrieved_file and relevant_file:
            args.extend(['--retrieved', retrieved_file, '--relevant', relevant_file])
    
    sys.argv = args
    app()

def execute_serve_command(subcommand: str):
    """Execute serve sub-commands with interactive prompts"""
    args = [sys.argv[0], 'serve', subcommand]
    
    if subcommand in ['start', 'restart']:
        try:
            port_input = console.input("🔌 Port (default: 5001): ").strip()
            if not port_input:
                port = "5001"
            else:
                # Validate that it's a number
                int(port_input)  # This will raise ValueError if invalid
                port = port_input
        except (ValueError, EOFError, KeyboardInterrupt):
            console.print("[yellow]ℹ️ Using default port 5001[/yellow]")
            port = "5001"
        args.extend(['--port', port])
    elif subcommand == 'stop':
        try:
            port_input = console.input("🔌 Port to stop (default: 5001): ").strip()
            if not port_input:
                port = "5001"
            else:
                # Validate that it's a number
                int(port_input)  # This will raise ValueError if invalid
                port = port_input
        except (ValueError, EOFError, KeyboardInterrupt):
            console.print("[yellow]ℹ️ Using default port 5001[/yellow]")
            port = "5001"
        args.extend(['--port', port])
    
    sys.argv = args
    app()

def execute_git_command(subcommand: str):
    """Execute git sub-commands with interactive prompts"""
    if subcommand == 'commit':
        # Interactive git commit
        message = console.input("💬 Commit message: ").strip()
        if not message:
            console.print("[red]❌ Commit message required[/red]")
            return
        
        add_all = console.input("📦 Add all files? (y/N): ").lower().strip() == 'y'
        push_after = console.input("🚀 Push after commit? (y/N): ").lower().strip() == 'y'
        
        args = [sys.argv[0], 'git-commit', message]
        if add_all:
            args.append('--add-all')
        if push_after:
            args.append('--push')
        
        sys.argv = args
        app()
        
    elif subcommand == 'clone':
        # Interactive git clone
        repo_url = console.input("🔗 Repository URL: ").strip()
        if not repo_url:
            console.print("[red]❌ Repository URL required[/red]")
            return
        
        directory = console.input("📁 Directory name (Enter for default): ").strip()
        branch = console.input("🌿 Branch (Enter for default): ").strip()
        setup_after = console.input("🔧 Run setup after clone? (y/N): ").lower().strip() == 'y'
        
        args = [sys.argv[0], 'git-clone', repo_url]
        if directory:
            args.extend(['--dir', directory])
        if branch:
            args.extend(['--branch', branch])
        if setup_after:
            args.append('--setup')
        
        sys.argv = args
        app()
        
    elif subcommand == 'pull':
        # Interactive git pull
        branch = console.input("🌿 Branch (Enter for current): ").strip()
        remote = console.input("📡 Remote (default: origin): ").strip()
        if not remote:
            remote = "origin"
        
        use_rebase = console.input("🔄 Use rebase instead of merge? (y/N): ").lower().strip() == 'y'
        
        args = [sys.argv[0], 'git-pull']
        if branch:
            args.extend(['--branch', branch])
        if remote != "origin":
            args.extend(['--remote', remote])
        if use_rebase:
            args.append('--rebase')
        
        sys.argv = args
        app()

def main():
    """Main CLI entry point with error handling"""
    try:
        # Check if no arguments provided (just 'pynucleus')
        if len(sys.argv) == 1:
            show_interactive_menu()
        else:
            app()
    except Exception as e:
        console.print(f"[red]❌ Fatal error: {e}[/red]")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            console.print_exception(show_locals=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 