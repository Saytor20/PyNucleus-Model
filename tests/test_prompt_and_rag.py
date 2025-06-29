from pynucleus.rag.engine import ask
from pynucleus.llm.prompting import build_prompt

def test_prompt_paths():
    p1 = build_prompt("", "What is distillation?")
    p2 = build_prompt("A"*900, "What is distillation?")
    assert ("RETRIEVED CONTEXT" in p2) and ("RETRIEVED CONTEXT" not in p1)

def test_rag_basic():
    out = ask("Define distillation.")
    assert out["answer"] 