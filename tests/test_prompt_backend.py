"""
Test prompt backend functionality including enhanced RAG prompting.
"""

from pynucleus.llm.prompting import build_prompt, build_enhanced_rag_prompt, build_simple_prompt, build_detailed_prompt


def test_prompt_switch():
    """Test that build_prompt works with enhanced RAG prompting."""
    context = "H2O boils at 100°C."
    question = "What is boiling point?"
    
    prompt = build_prompt(context, question)
    
    # Verify the prompt contains relevant content
    assert "boiling" in prompt.lower()
    assert "100°C" in prompt
    assert isinstance(prompt, str)
    assert len(prompt.strip()) > 0


def test_enhanced_rag_prompt():
    """Test that enhanced RAG prompt works correctly."""
    context = "Chemical engineering involves unit operations."
    question = "What is chemical engineering?"
    
    prompt = build_enhanced_rag_prompt(context, question)
    
    # Should contain the context and question in some form
    assert "chemical engineering" in prompt.lower()
    assert "unit operations" in prompt.lower()
    assert "expert chemical engineer assistant" in prompt.lower()


def test_prompt_content_structure():
    """Test that prompt has expected structure elements."""
    context = "Chemical engineering involves unit operations."
    question = "What is chemical engineering?"
    
    prompt = build_prompt(context, question)
    
    # Should contain the context and question in some form
    assert "chemical engineering" in prompt.lower()
    assert "unit operations" in prompt.lower()


def test_simple_prompt():
    """Test simple prompt functionality."""
    context = "Distillation separates mixtures based on boiling points."
    question = "How does distillation work?"
    
    prompt = build_simple_prompt(context, question)
    
    assert "distillation" in prompt.lower()
    assert "boiling points" in prompt.lower()
    assert "experienced chemical engineer" in prompt.lower()


def test_detailed_prompt():
    """Test detailed prompt functionality."""
    context = "Heat exchangers transfer thermal energy between fluids."
    question = "Explain heat exchanger design principles."
    
    prompt = build_detailed_prompt(context, question)
    
    assert "heat exchangers" in prompt.lower()
    assert "thermal energy" in prompt.lower()
    assert "senior chemical engineering consultant" in prompt.lower() 