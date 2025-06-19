"""
Test prompt backend functionality including Guidance integration.
"""

from pynucleus.llm.prompting import USE_GUIDANCE, build_prompt


def test_prompt_switch():
    """Test that build_prompt works regardless of Guidance availability."""
    context = "H2O boils at 100°C."
    question = "What is boiling point?"
    
    prompt = build_prompt(context, question)
    
    # Verify the prompt contains relevant content
    assert "boiling" in prompt.lower()
    assert "100°C" in prompt
    assert isinstance(prompt, str)
    assert len(prompt.strip()) > 0


def test_prompt_backend_detection():
    """Test that USE_GUIDANCE flag is properly set."""
    assert isinstance(USE_GUIDANCE, bool)


def test_prompt_content_structure():
    """Test that prompt has expected structure elements."""
    context = "Chemical engineering involves unit operations."
    question = "What is chemical engineering?"
    
    prompt = build_prompt(context, question)
    
    # Should contain the context and question in some form
    assert "chemical engineering" in prompt.lower()
    assert "unit operations" in prompt.lower() 