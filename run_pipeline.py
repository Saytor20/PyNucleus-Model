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
    backend: str = typer.Option("faiss", "--backend", help="Vector store backend (faiss|qdrant)"),
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
        if backend not in ['faiss', 'qdrant']:
            cli_logger.error(f"❌ Invalid backend: {backend}. Must be 'faiss' or 'qdrant'")
            raise typer.Exit(1)
        
        cli_logger.info(f"🔧 Using vector store backend: {backend}")
        
        if backend == 'qdrant':
            cli_logger.info("⚠️  Qdrant stub -- not yet enabled")
            print("Qdrant stub -- not yet enabled")
            return
        
        # Initialize RAG core
        from pynucleus.rag import RAGCore
        rag_core = RAGCore(data_dir=output_dir.parent)
        
        # Process documents
        cli_logger.info("🔄 Processing documents...")
        result = rag_core.process_documents(source_dir=str(source_dir))
        
        if result["status"] == "success":
            cli_logger.info(f"✅ Successfully processed {result['processed_count']} documents")
            print(f"📊 Processed {result['processed_count']} out of {result['total_files']} files")
            print(f"📁 Results saved to: {output_dir}")
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
        
        # Check for compiled DSPy program first
        from src.pynucleus.llm.dspy_compile import DSPyCompiler
        
        compiler = DSPyCompiler()
        compiled_program = compiler.load_compiled_program()
        
        if compiled_program:
            cli_logger.info("🎯 Using compiled DSPy program")
            # Use DSPy answer engine
            from src.pynucleus.llm.answer_engine import DSPyAnswerEngine
            engine = DSPyAnswerEngine(model_id=model_id)
            result = engine.answer_general(question)
            
            # Display DSPy results
            print("=" * 60)
            print("🧠 DSPY ENHANCED RESPONSE")
            print("=" * 60)
            print(f"Question: {question}")
            print(f"Generation Time: {result.get('generation_time', 0):.2f}s")
            print("-" * 60)
            print(f"Answer: {result.get('answer', 'No answer available')}")
            print("-" * 60)
            print(f"Model: {result.get('model_id', model_id)}")
            print(f"DSPy Used: {result.get('dspy_used', False)}")
            print("=" * 60)
        else:
            cli_logger.warning("⚠️ No compiled DSPy program found, using fallback RAG system")
            
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
            
            # Create enhanced prompt with context
            context = result.get('answer', '')
            enhanced_prompt = f"Based on this scientific information: {context}\n\nQuestion: {question}\n\nProvide a clear, technical answer:"
            
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

@app.command("dspy-compile")
def dspy_compile(
    csv_path: Path = typer.Option("docs/devset/dspy_examples.csv", "--csv-path", help="Path to development examples CSV"),
    output_dir: Path = typer.Option("data/dspy_artifacts", "--output-dir", help="Output directory for compiled artifacts"),
    ci: bool = typer.Option(False, "--ci", help="Run in CI mode (create artifact but don't commit)"),
    create_sample: bool = typer.Option(False, "--create-sample", help="Create sample development dataset"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Verbose logging"),
):
    """Compile DSPy programs using development examples."""
    try:
        # Setup logging
        logger = configure_logging(level="DEBUG" if verbose else "INFO")
        cli_logger = get_logger(__name__)
        
        cli_logger.info("🧠 Starting DSPy compilation")
        cli_logger.info(f"📋 CSV Path: {csv_path}")
        cli_logger.info(f"📁 Output Directory: {output_dir}")
        cli_logger.info(f"🔧 CI Mode: {ci}")
        
        # Import the compilation utility
        from src.pynucleus.llm.dspy_compile import compile_dspy_main
        
        # Run compilation
        exit_code = compile_dspy_main(
            csv_path=str(csv_path),
            output_dir=str(output_dir),
            ci_mode=ci,
            create_sample=create_sample
        )
        
        if exit_code == 0:
            if create_sample:
                cli_logger.info("✅ Sample dataset created successfully")
                print("📋 Sample development dataset created!")
                print(f"📁 Location: {csv_path}")
                print("💡 Edit this file to add your own examples, then run compilation again")
            elif ci:
                cli_logger.info("✅ CI compilation completed successfully")
                print("🎯 DSPy compilation completed in CI mode!")
                print("📄 Artifact created but not committed")
            else:
                cli_logger.info("✅ DSPy compilation completed successfully")
                print("🎉 DSPy compilation completed!")
                print(f"📁 Compiled artifacts saved to: {output_dir}")
                print("💡 Now 'pynucleus ask' will use the compiled DSPy program")
        else:
            cli_logger.error("❌ DSPy compilation failed")
            print("❌ DSPy compilation failed. Check logs for details.")
            raise typer.Exit(1)
            
    except Exception as e:
        cli_logger.error(f"❌ DSPy compilation error: {e}")
        print(f"❌ Error: {str(e)}")
        raise typer.Exit(1)

@app.command("chat")
def interactive_chat(
    model_id: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model-id", help="LLM model ID for responses"),
    top_k: int = typer.Option(5, "--top-k", help="Number of top results to retrieve from RAG"),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Verbose logging"),
    use_dspy: bool = typer.Option(True, "--use-dspy/--no-dspy", help="Use DSPy if available"),
):
    """Start an interactive chat session with the LLM."""
    try:
        # Setup logging
        logger = configure_logging(level="DEBUG" if verbose else "INFO")
        cli_logger = get_logger(__name__)
        
        cli_logger.info("🚀 Starting interactive chat session")
        cli_logger.info(f"🤖 Model ID: {model_id}")
        cli_logger.info(f"📊 Top K: {top_k}")
        cli_logger.info(f"🧠 DSPy enabled: {use_dspy}")
        
        # Initialize systems
        dspy_engine = None
        rag_pipeline = None
        llm_runner = None
        
        # Check for compiled DSPy program first
        if use_dspy:
            try:
                from src.pynucleus.llm.dspy_compile import DSPyCompiler
                from src.pynucleus.llm.answer_engine import DSPyAnswerEngine
                
                compiler = DSPyCompiler()
                compiled_program = compiler.load_compiled_program()
                
                if compiled_program:
                    cli_logger.info("🎯 Initializing DSPy answer engine")
                    dspy_engine = DSPyAnswerEngine(model_id=model_id)
                    
                    # Check if DSPy, LocalDSPy, or SimpleLocal was actually configured successfully
                    if dspy_engine.local_dspy_configured:
                        print("✅ LocalDSPy enhanced mode enabled (LangChain + structured prompting)")
                        print("🎯 Using DSPy-like structured reasoning with local models")
                    elif dspy_engine.simple_local_configured:
                        print("✅ SimpleLocal enhanced mode enabled (Direct transformers + structured prompting)")
                        print("🎯 Using basic structured reasoning with local models")
                    elif dspy_engine.dspy_configured:
                        print("✅ DSPy enhanced mode enabled")
                    else:
                        print("⚠️ DSPy disabled - local models not supported, using standard RAG + LLM")
                        dspy_engine = None  # Clear the engine so we use fallback
                else:
                    cli_logger.warning("⚠️ No compiled DSPy program found")
                    print("⚠️ No compiled DSPy program found, using standard RAG + LLM")
            except Exception as e:
                cli_logger.warning(f"⚠️ DSPy initialization failed: {e}")
                print(f"⚠️ DSPy initialization failed: {e}")
        
        # Fallback to RAG pipeline
        if not dspy_engine:
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
                
                # Process with DSPy if available
                if dspy_engine:
                    try:
                        result = dspy_engine.answer_general(question)
                        
                        # Display DSPy results
                        print("\n" + "🧠 " + "=" * 58)
                        print(f"Answer: {result.get('answer', 'No answer available')}")
                        print("=" * 60)
                        print(f"⏱️  Generation time: {result.get('generation_time', 0):.2f}s")
                        print(f"🤖 Model: {result.get('model_id', model_id)}")
                        if dspy_engine.local_dspy_configured:
                            dspy_mode = "LocalDSPy"
                        elif dspy_engine.simple_local_configured:
                            dspy_mode = "SimpleLocal"
                        elif dspy_engine.dspy_configured:
                            dspy_mode = "DSPy"
                        else:
                            dspy_mode = "None"
                        print(f"🎯 Enhanced Mode: {dspy_mode}")
                        
                    except Exception as e:
                        cli_logger.error(f"❌ DSPy processing failed: {e}")
                        print(f"❌ DSPy processing failed: {e}")
                        continue
                
                # Process with RAG + LLM
                else:
                    try:
                        # Get RAG response
                        rag_result = rag_pipeline.query(question, top_k=top_k)
                        
                        # Get enhanced LLM response
                        context = rag_result.get('answer', '')
                        enhanced_prompt = f"Based on this scientific information: {context}\n\nQuestion: {question}\n\nProvide a clear, technical answer:"
                        
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
        if dspy_engine:
            if dspy_engine.local_dspy_configured:
                system_type = "LocalDSPy Enhanced"
            elif dspy_engine.simple_local_configured:
                system_type = "SimpleLocal Enhanced"
            elif dspy_engine.dspy_configured:
                system_type = "DSPy Enhanced"
            else:
                system_type = "DSPy (fallback mode)"
        else:
            system_type = "RAG + LLM"
        
        print(f"  • System used: {system_type}")
        print(f"  • Model: {model_id}")
        
        cli_logger.info(f"✅ Chat session ended. Questions processed: {question_count}")
        
    except Exception as e:
        cli_logger.error(f"❌ Chat session failed: {str(e)}")
        print(f"❌ Error starting chat: {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 