import rich
from typer import Typer
app = Typer()

@app.command()
def ingest_docs(source_dir: str = "data/01_raw"):
    from pynucleus.rag.collector import ingest
    ingest(source_dir)

@app.command()
def ask(question: str):
    from pynucleus.pipeline.pipeline_rag import rag_pipeline
    res = rag_pipeline.query(question)
    rich.print(res["answer"])

@app.command()
def eval_golden():
    from pynucleus.eval.golden_eval import run_eval
    passed = run_eval()
    if not passed:
        raise SystemExit("Golden dataset evaluation below threshold!")

def main():
    app() 