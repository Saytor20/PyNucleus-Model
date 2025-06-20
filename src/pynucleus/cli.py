import rich
from typer import Typer, Option
from .rag.engine import ask as rag_ask
from .utils.pretty_formatter import format_for_terminal

app = Typer()

@app.command()
def ingest_docs(
    source_dir: str = "data/01_raw", 
    extract_pdf_tables: bool = Option(True, "--extract-pdf-tables/--skip-pdf-tables", help="Run enhanced PDF table extraction and processing")
):
    from pathlib import Path
    from pynucleus.rag.document_processor import DocumentProcessor
    from pynucleus.rag.collector import ingest

    # Enhanced PDF table extraction using document processor
    if extract_pdf_tables:
        processor = DocumentProcessor()
        pdfs = list(Path(source_dir).rglob("*.pdf"))
        
        print(f"🔍 Found {len(pdfs)} PDF files for table extraction...")
        
        for pdf in pdfs:
            print(f"📊 Processing tables from: {pdf.name}")
            result = processor.process_document(pdf)
            
            if result["tables_extracted"] > 0:
                print(f"  ✅ Extracted {result['tables_extracted']} tables")
                print(f"  📁 Created {len(result['table_files'])} CSV files")
                for table_type in result.get('table_types', []):
                    print(f"    - {table_type} tables")
            else:
                print(f"  ⚠️  No tables found in {pdf.name}")

    # Ingest both original documents and extracted CSV tables
    print("\n📚 Ingesting documents into RAG system...")
    ingest(source_dir)
    
    # Also ingest the extracted CSV tables
    tables_dir = Path("data/02_processed/tables")
    if tables_dir.exists() and list(tables_dir.glob("*.csv")):
        print("📊 Ingesting extracted table data...")
        ingest(str(tables_dir))

@app.command()
def ask(question: str, pretty: bool = Option(True, "--pretty/--plain", help="Use enhanced formatting")):
    """Ask a question to the RAG system with enhanced formatting"""
    result = rag_ask(question)
    
    if pretty:
        # Use pretty formatter for enhanced display
        format_for_terminal(result)
    else:
        # Fallback to plain rich print
        rich.print(result["answer"])

@app.command()
def eval_golden():
    from pynucleus.eval.golden_eval import run_eval
    passed = run_eval()
    if not passed:
        raise SystemExit("Golden dataset evaluation below threshold!")

def main():
    app() 