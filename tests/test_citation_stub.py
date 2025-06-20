"""
Test for basic citation functionality
"""
from pynucleus.rag.engine import ask

def test_refs():
    """Test that References section appears when sources are available"""
    out = ask("What is distillation?")
    assert "References:" in out["answer"] or not out["sources"] 