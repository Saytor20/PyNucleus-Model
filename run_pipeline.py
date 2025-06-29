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
    model_id: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model-id", help="LLM model ID"),
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

@app.command("ingest")
def ingest_documents(
    source_dir: Path = typer.Option(..., "--source-dir", help="Source directory containing documents to ingest"),
    output_dir: Path = typer.Option("data/03_intermediate", "--output-dir", help="Output directory for processed documents"),
    backend: str = typer.Option("chroma", "--backend", help="Vector store backend (chroma|faiss|qdrant)"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Verbose logging"),
):
    """Ingest and process documents for RAG system."""
    try:
        # Setup logging
        logger = configure_logging(level="DEBUG" if verbose else "INFO")
        cli_logger = get_logger(__name__)
        
        cli_logger.info("🚀 Starting document ingestion")
        cli_logger.info(f"📁 Source directory: {source_dir}")
        cli_logger.info(f"📁 Output directory: {output_dir}")
        
        # Validate source directory
        if not source_dir.exists():
            cli_logger.error(f"❌ Source directory not found: {source_dir}")
            raise typer.Exit(1)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate backend selection
        if backend not in ['chroma', 'faiss', 'qdrant']:
            cli_logger.error(f"❌ Invalid backend: {backend}. Must be 'chroma', 'faiss', or 'qdrant'")
            raise typer.Exit(1)
        
        cli_logger.info(f"🔧 Using vector store backend: {backend}")
        
        if backend == 'qdrant':
            cli_logger.info("⚠️  Qdrant stub -- not yet enabled")
            print("Qdrant stub -- not yet enabled")
            return
        
        # Initialize RAG core
        from pynucleus.rag import RAGCore
        rag_core = RAGCore(data_dir=output_dir.parent)
        
        # Process documents using real ChromaDB ingestion
        cli_logger.info("🔄 Processing documents...")
        
        if backend == 'chroma':
            # Use the actual ChromaDB ingestion function
            from pynucleus.rag.collector import ingest
            ingest(source_dir=str(source_dir))
            cli_logger.info("✅ Successfully ingested documents into ChromaDB")
            print(f"📊 Documents ingested into ChromaDB collection: pynucleus_documents")
        else:
            # Fallback to RAG core for other backends
            result = rag_core.process_documents(source_dir=str(source_dir))
            if result["status"] == "success":
                cli_logger.info(f"✅ Successfully processed {result['processed_count']} documents")
                print(f"📊 Processed {result['processed_count']} out of {result['total_files']} files")
            else:
                cli_logger.error(f"❌ Document processing failed: {result['message']}")
                raise typer.Exit(1)
            
    except Exception as e:
        cli_logger.error(f"❌ Ingestion failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

@app.command("build-faiss")  
def build_faiss_index(
    chunk_dir: Path = typer.Option("data/03_intermediate/converted_chunked_data", "--chunk-dir", help="Directory containing chunk data"),
    index_dir: Path = typer.Option("data/04_models/chunk_reports", "--index-dir", help="Output directory for FAISS index"),
    force_rebuild: bool = typer.Option(False, "--force-rebuild", help="Force rebuild even if index exists"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Verbose logging"),
):
    """Build FAISS vector index from processed chunks."""
    try:
        # Setup logging
        logger = configure_logging(level="DEBUG" if verbose else "INFO")
        cli_logger = get_logger(__name__)
        
        cli_logger.info("🚀 Starting FAISS index building")
        cli_logger.info(f"📁 Chunk directory: {chunk_dir}")
        cli_logger.info(f"📁 Index directory: {index_dir}")
        cli_logger.info(f"🔄 Force rebuild: {force_rebuild}")
        
        # Validate chunk directory
        if not chunk_dir.exists():
            cli_logger.error(f"❌ Chunk directory not found: {chunk_dir}")
            raise typer.Exit(1)
        
        # Create index directory
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RAG core
        from pynucleus.rag import RAGCore
        rag_core = RAGCore(data_dir=index_dir.parent.parent)
        
        # Build index
        cli_logger.info("🔄 Building FAISS index...")
        result = rag_core.build_index(force_rebuild=force_rebuild)
        
        if result["status"] == "success":
            cli_logger.info(f"✅ Successfully built FAISS index")
            print(f"📊 Index size: {result.get('index_size', 'Unknown')}")
            print(f"📐 Dimensions: {result.get('dimensions', 'Unknown')}")
            print(f"📄 Chunks indexed: {result.get('chunks_indexed', 'Unknown')}")
            print(f"📁 Index saved to: {index_dir}")
        elif result["status"] == "exists":
            print(f"ℹ️  Index already exists: {result['message']}")
            print(f"📁 Existing indices: {result.get('existing_indices', [])}")
        else:
            cli_logger.error(f"❌ Index building failed: {result['message']}")
            raise typer.Exit(1)
            
    except Exception as e:
        cli_logger.error(f"❌ FAISS index building failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

@app.command("ask")
def ask_question(
    question: str = typer.Option(..., "--question", help="Question to ask the RAG system"),
    model_id: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model-id", help="LLM model ID for enhanced responses"),
    top_k: int = typer.Option(5, "--top-k", help="Number of top results to retrieve"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Verbose logging"),
):
    """Ask a question to the RAG system."""
    try:
        # Setup logging
        logger = configure_logging(level="DEBUG" if verbose else "INFO")
        cli_logger = get_logger(__name__)
        
        cli_logger.info("🚀 Starting RAG query")
        cli_logger.info(f"❓ Question: {question}")
        cli_logger.info(f"🤖 Model ID: {model_id}")
        cli_logger.info(f"📊 Top K: {top_k}")
        
        # DSPy components removed, using RAG system directly
        cli_logger.info("🔄 Using RAG system for query processing")
        
        # Initialize RAG pipeline
        from pynucleus.pipeline.pipeline_rag import RAGPipeline
        rag_pipeline = RAGPipeline(data_dir="data")
        
        # Query the RAG system
        cli_logger.info("🔄 Processing question...")
        result = rag_pipeline.query(question, top_k=top_k)
        
        # Display results
        print("=" * 60)
        print("📋 RAG SYSTEM RESPONSE")
        print("=" * 60)
        print(f"Question: {question}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print("-" * 60)
        print(f"Answer: {result.get('answer', 'No answer available')}")
        print("-" * 60)
        print("Sources:")
        for i, source in enumerate(result.get('sources', []), 1):
            print(f"  {i}. {source}")
        print("=" * 60)
        
        # Always use LLM for enhanced response
        try:
            from pynucleus.llm.llm_runner import LLMRunner
            llm_runner = LLMRunner(model_id=model_id)
            
            cli_logger.info(f"🤖 Generating enhanced response with {model_id}")
            
            # Create enhanced prompt with context using Guidance integration
            from pynucleus.llm.prompting import build_prompt
            context = result.get('answer', '')
            enhanced_prompt = build_prompt(context, question)
            
            llm_response = llm_runner.ask(
                question=enhanced_prompt,
                max_length=500,
                temperature=0.7
            )
            
            print("\n🤖 ENHANCED LLM RESPONSE")
            print("-" * 60)
            print(llm_response)
            print("=" * 60)
            
        except Exception as llm_error:
            cli_logger.error(f"❌ LLM processing failed: {llm_error}")
            print(f"\n❌ LLM processing failed: {llm_error}")
            print("🔄 Falling back to RAG-only response")
                
        cli_logger.info("✅ Question processed successfully")
        
    except Exception as e:
        cli_logger.error(f"❌ Question processing failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

# DSPy command removed - DSPy functionality disabled to avoid API dependencies
# Use the regular 'ask' or 'chat' commands instead for local LLM processing

@app.command("chat")
def interactive_chat(
    model_id: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model-id", help="LLM model ID for responses"),
    top_k: int = typer.Option(5, "--top-k", help="Number of top results to retrieve from RAG"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Verbose logging"),
    use_dspy: bool = typer.Option(False, "--use-dspy/--no-dspy", help="DSPy disabled - using local RAG+LLM instead"),
):
    """Start an interactive chat session with the LLM."""
    try:
        # Setup logging
        logger = configure_logging(level="DEBUG" if verbose else "INFO")
        cli_logger = get_logger(__name__)
        
        cli_logger.info("🚀 Starting interactive chat session")
        cli_logger.info(f"🤖 Model ID: {model_id}")
        cli_logger.info(f"📊 Top K: {top_k}")
        # DSPy functionality has been removed to avoid API dependencies
        # Using local RAG + LLM processing only
        
        # Initialize systems
        rag_pipeline = None
        llm_runner = None
        
        # Initialize RAG pipeline and LLM
        try:
            from pynucleus.pipeline.pipeline_rag import RAGPipeline
            from pynucleus.llm.llm_runner import LLMRunner
            
            cli_logger.info("🔧 Initializing RAG pipeline and LLM runner")
            rag_pipeline = RAGPipeline(data_dir="data")
            llm_runner = LLMRunner(model_id=model_id)
            print("✅ RAG + LLM mode enabled")
        except Exception as e:
            cli_logger.error(f"❌ Failed to initialize RAG/LLM systems: {e}")
            print(f"❌ Failed to initialize systems: {e}")
            raise typer.Exit(1)
        
        # Start interactive session
        print("\n" + "=" * 60)
        print("🤖 PYNUCLEUS INTERACTIVE CHAT")
        print("=" * 60)
        print("💡 Ask questions about chemical processes, simulations, or technical topics")
        print("📝 Type 'quit', 'exit', or press Ctrl+C to end the session")
        print("🔄 Type 'clear' to clear the screen")
        print("❓ Type 'help' for more commands")
        print("=" * 60)
        
        question_count = 0
        
        while True:
            try:
                # Get user input
                print(f"\n[Q{question_count + 1}]", end=" ")
                question = input("💭 Your question: ").strip()
                
                # Handle special commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Chat session ended.")
                    break
                elif question.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                elif question.lower() == 'help':
                    print("\n📋 Available Commands:")
                    print("  • quit/exit/q  - End chat session")
                    print("  • clear        - Clear screen")
                    print("  • help         - Show this help")
                    print("  • Any question - Ask the AI system")
                    continue
                elif not question:
                    print("❌ Please enter a question or command")
                    continue
                
                question_count += 1
                
                print(f"\n🔄 Processing question {question_count}...")
                start_time = datetime.now()
                
                # Process with RAG + LLM
                try:
                    # Get RAG response
                    rag_result = rag_pipeline.query(question, top_k=top_k)
                    
                    # Get enhanced LLM response using Guidance integration
                    from pynucleus.llm.prompting import build_prompt
                    context = rag_result.get('answer', '')
                    enhanced_prompt = build_prompt(context, question)
                    
                    llm_response = llm_runner.ask(
                        question=enhanced_prompt,
                        max_length=500,
                        temperature=0.7
                    )
                    
                    # Display results
                    print("\n" + "📋 " + "=" * 58)
                    print(f"Answer: {llm_response}")
                    print("=" * 60)
                    
                    # Show sources if available
                    sources = rag_result.get('sources', [])
                    if sources:
                        print("📚 Sources:")
                        for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
                            print(f"  {i}. {source}")
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    print(f"⏱️  Response time: {duration:.2f}s")
                    print(f"🤖 Model: {model_id}")
                    print(f"📊 Confidence: {rag_result.get('confidence', 0):.2f}")
                    
                except Exception as e:
                    cli_logger.error(f"❌ RAG/LLM processing failed: {e}")
                    print(f"❌ Processing failed: {e}")
                    continue
                
            except KeyboardInterrupt:
                print("\n\n⏹️ Chat session interrupted by user")
                break
            except EOFError:
                print("\n\n👋 Chat session ended")
                break
            except Exception as e:
                cli_logger.error(f"❌ Unexpected error: {e}")
                print(f"❌ Unexpected error: {e}")
                continue
        
        # Session summary
        print(f"\n📊 Chat session summary:")
        print(f"  • Questions asked: {question_count}")
        print(f"  • System used: RAG + LLM")
        print(f"  • Model: {model_id}")
        
        cli_logger.info(f"✅ Chat session ended. Questions processed: {question_count}")
        
    except Exception as e:
        cli_logger.error(f"❌ Chat session failed: {str(e)}")
        print(f"❌ Error starting chat: {str(e)}")
        raise typer.Exit(1)

@app.command("build")
def build_plant(
    template_id: Optional[int] = typer.Option(None, "--template-id", help="Plant template ID (1-22) - if not provided, will prompt interactively"),
    feedstock: Optional[str] = typer.Option(None, "--feedstock", help="Feedstock type - if not provided, will prompt interactively"),
    production_capacity: Optional[int] = typer.Option(None, "--production-capacity", help="Production capacity in tons/year - if not provided, will prompt interactively"),
    plant_location: Optional[str] = typer.Option(None, "--plant-location", help="Plant location - if not provided, will prompt interactively"),
    operating_hours: Optional[int] = typer.Option(None, "--operating-hours", help="Operating hours per year - if not provided, will prompt interactively"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Verbose logging"),
    output_file: Optional[Path] = typer.Option(None, "--output-file", help="Save results to JSON file"),
):
    """Build a modular chemical plant and perform financial analysis with interactive prompts."""
    try:
        # Setup logging
        logger = configure_logging(level="DEBUG" if verbose else "INFO")
        cli_logger = get_logger(__name__)
        
        cli_logger.info("🚀 Starting Interactive Plant Builder")
        
        # Import required components
        from pynucleus.pipeline.plant_builder import PlantBuilder
        from pynucleus.pipeline.financial_analyzer import FinancialAnalyzer
        from pynucleus.data.mock_data_manager import MockDataManager
        
        # Initialize components
        plant_builder = PlantBuilder()
        financial_analyzer = FinancialAnalyzer()
        mock_data_manager = MockDataManager()
        
        # Get available templates
        templates = mock_data_manager.get_all_plant_templates()
        
        print("=" * 60)
        print("🏭 INTERACTIVE MODULAR PLANT BUILDER")
        print("=" * 60)
        
        # Step 1: Template Selection
        if template_id is None:
            print("\n📋 Available Plant Templates:")
            print("-" * 40)
            for template in templates:
                print(f"{template['id']:2d}. {template['name']}")
                print(f"    Technology: {template['technology']}")
                print(f"    Description: {template['description']}")
                print()
            
            while True:
                try:
                    template_id = int(input("Enter template ID (1-22): ").strip())
                    if 1 <= template_id <= 22:
                        break
                    else:
                        print("❌ Please enter a valid template ID between 1 and 22.")
                except ValueError:
                    print("❌ Please enter a valid number.")
        
        # Get selected template details
        selected_template = next((t for t in templates if t['id'] == template_id), None)
        if not selected_template:
            raise ValueError(f"Template ID {template_id} not found")
        
        print(f"\n✅ Selected: {selected_template['name']}")
        print(f"   Technology: {selected_template['technology']}")
        print(f"   Description: {selected_template['description']}")
        
        # Step 2: Feedstock Selection
        if feedstock is None:
            print(f"\n🌿 Available feedstock options for {selected_template['name']}:")
            for i, option in enumerate(selected_template['feedstock_options'], 1):
                print(f"   {i}. {option}")
            
            while True:
                try:
                    choice = int(input(f"\nSelect feedstock (1-{len(selected_template['feedstock_options'])}): ").strip())
                    if 1 <= choice <= len(selected_template['feedstock_options']):
                        feedstock = selected_template['feedstock_options'][choice - 1]
                        break
                    else:
                        print(f"❌ Please enter a valid choice between 1 and {len(selected_template['feedstock_options'])}.")
                except ValueError:
                    print("❌ Please enter a valid number.")
        
        print(f"✅ Selected feedstock: {feedstock}")
        
        # Step 3: Production Capacity
        if production_capacity is None:
            valid_range = selected_template['valid_ranges']['production_capacity_tpd']
            default_capacity = selected_template['default_parameters']['production_capacity_tpd']
            
            print(f"\n📊 Production Capacity:")
            print(f"   Valid range: {valid_range['min']:,} - {valid_range['max']:,} tons/day")
            print(f"   Default: {default_capacity:,} tons/day")
            
            while True:
                try:
                    user_input = input(f"Enter production capacity in tons/day (or press Enter for default): ").strip()
                    if user_input == "":
                        production_capacity = default_capacity
                        break
                    else:
                        production_capacity = int(user_input)
                        if valid_range['min'] <= production_capacity <= valid_range['max']:
                            break
                        else:
                            print(f"❌ Please enter a value between {valid_range['min']:,} and {valid_range['max']:,}.")
                except ValueError:
                    print("❌ Please enter a valid number.")
        
        print(f"✅ Production capacity: {production_capacity:,} tons/day")
        
        # Step 4: Plant Location
        if plant_location is None:
            print(f"\n📍 Available locations (with cost factors):")
            for location, factor in selected_template['location_factors'].items():
                factor_text = "Standard" if factor == 1.0 else f"{factor:.1f}x cost"
                print(f"   • {location} ({factor_text})")
            
            while True:
                plant_location = input(f"\nEnter plant location (or press Enter for 'Texas, USA'): ").strip()
                if plant_location == "":
                    plant_location = "Texas, USA"
                    break
                elif plant_location in selected_template['location_factors']:
                    break
                else:
                    print("❌ Please enter a valid location from the list above.")
        
        print(f"✅ Plant location: {plant_location}")
        
        # Step 5: Operating Hours
        if operating_hours is None:
            valid_range = selected_template['valid_ranges']['operating_hours']
            default_hours = selected_template['default_parameters']['operating_hours']
            
            print(f"\n⏰ Operating Hours:")
            print(f"   Valid range: {valid_range['min']:,} - {valid_range['max']:,} hours/year")
            print(f"   Default: {default_hours:,} hours/year")
            
            while True:
                try:
                    user_input = input(f"Enter operating hours per year (or press Enter for default): ").strip()
                    if user_input == "":
                        operating_hours = default_hours
                        break
                    else:
                        operating_hours = int(user_input)
                        if valid_range['min'] <= operating_hours <= valid_range['max']:
                            break
                        else:
                            print(f"❌ Please enter a value between {valid_range['min']:,} and {valid_range['max']:,}.")
                except ValueError:
                    print("❌ Please enter a valid number.")
        
        print(f"✅ Operating hours: {operating_hours:,} hours/year")
        
        # Convert production capacity from tons/day to tons/year for the API
        production_capacity_tpy = production_capacity * 365
        
        # Prepare custom parameters
        custom_parameters = {
            "feedstock": feedstock,
            "production_capacity": production_capacity_tpy,  # API expects tons/year
            "plant_location": plant_location,
            "operating_hours": operating_hours
        }
        
        print("\n" + "=" * 60)
        print("📋 BUILD SUMMARY")
        print("=" * 60)
        print(f"Template: {selected_template['name']}")
        print(f"Technology: {selected_template['technology']}")
        print(f"Feedstock: {feedstock}")
        print(f"Production Capacity: {production_capacity:,} tons/day ({production_capacity_tpy:,} tons/year)")
        print(f"Location: {plant_location}")
        print(f"Operating Hours: {operating_hours:,} hours/year")
        print("=" * 60)
        
        # Confirm before proceeding
        proceed = input("\nProceed with plant build? (y/N): ").strip().lower()
        if proceed not in ['y', 'yes']:
            print("❌ Plant build cancelled.")
            return
        
        # Step 6: Build plant configuration
        print("\n🔄 Step 1: Building plant configuration...")
        plant_config = plant_builder.build_plant(template_id, custom_parameters)
        
        print("✅ Plant configuration built successfully!")
        print(f"   Template: {plant_config['template_info']['name']}")
        print(f"   Technology: {plant_config['template_info']['technology']}")
        print(f"   Capital Cost: ${plant_config['financial_parameters']['capital_cost']:,.0f}")
        print(f"   Operating Cost: ${plant_config['financial_parameters']['operating_cost']:,.0f}/year")
        print(f"   Product Price: ${plant_config['financial_parameters']['product_price']}/ton")
        
        # Step 7: Perform financial analysis
        print("\n🔄 Step 2: Performing financial analysis...")
        financial_analysis = financial_analyzer.analyze_financials(plant_config)
        
        print("✅ Financial analysis completed!")
        
        # Display results
        print("\n" + "=" * 60)
        print("💰 FINANCIAL ANALYSIS RESULTS")
        print("=" * 60)
        
        llm_analysis = financial_analysis.get("llm_analysis", {})
        basic_calc = financial_analysis.get("basic_calculations", {})
        
        print(f"Annual Revenue: ${llm_analysis.get('annual_revenue', 0):,.0f}")
        print(f"Profit Margin: {llm_analysis.get('profit_margin_percent', 0):.1f}%")
        print(f"ROI: {llm_analysis.get('roi_percent', 0):.1f}%")
        
        # Display risks
        risks = llm_analysis.get('financial_risks', [])
        if risks:
            print(f"\n⚠️  Financial Risks:")
            for i, risk in enumerate(risks, 1):
                print(f"   {i}. {risk}")
        
        # Display recommendations
        recommendations = llm_analysis.get('strategic_recommendations', '')
        if recommendations:
            print(f"\n💡 Strategic Recommendations:")
            print(f"   {recommendations}")
        
        # Display basic calculations for verification
        print(f"\n📊 Basic Calculations (Verification):")
        print(f"   Annual Revenue: ${basic_calc.get('annual_revenue', 0):,.0f}")
        print(f"   Profit Margin: {basic_calc.get('profit_margin_percent', 0):.1f}%")
        print(f"   ROI: {basic_calc.get('roi_percent', 0):.1f}%")
        
        # Save results if output file specified
        if output_file:
            import json
            results = {
                "plant_configuration": plant_config,
                "financial_analysis": financial_analysis,
                "build_metadata": {
                    "template_id": template_id,
                    "custom_parameters": custom_parameters,
                    "build_timestamp": datetime.now().isoformat()
                }
            }
            
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")
        
        print("=" * 60)
        print("🎉 Plant build and analysis completed successfully!")
        print("=" * 60)
        
        cli_logger.info("✅ Plant build and analysis completed successfully")
        
    except ValueError as e:
        cli_logger.error(f"❌ Validation error: {str(e)}")
        print(f"❌ Validation error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        cli_logger.error(f"❌ Plant build failed: {str(e)}")
        print(f"❌ Plant build failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 