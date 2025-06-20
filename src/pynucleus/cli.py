import rich
from typer import Typer, Option
from .rag.engine import ask as rag_ask
from .utils.pretty_formatter import format_for_terminal

app = Typer()

@app.command()
def ingest_docs(source_dir: str = "data/01_raw"):
    from pynucleus.rag.collector import ingest
    ingest(source_dir)

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