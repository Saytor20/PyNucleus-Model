from pynucleus.rag.engine import ask

def test_basic_query():
    out = ask("Define distillation.")
    assert out["sources"] and "distill" in out["answer"].lower() 