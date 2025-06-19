from pynucleus.llm.model_loader import generate

def test_generate():
    out = generate("Hello, how are you?", max_tokens=10)
    assert isinstance(out, str) and len(out) > 0 