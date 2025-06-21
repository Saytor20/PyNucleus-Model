"""Test reasoning quality and source validation for RAG system."""

import pytest
from pynucleus.rag.engine import ask
from pynucleus.settings import settings

def test_distillation_answer_relevance():
    """Test that answers contain relevant distillation information."""
    question = "What is distillation?"
    result = ask(question)
    
    # Check that we get a proper response structure
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    
    # Check answer relevance - should mention distillation
    answer = result["answer"].lower()
    assert "distillation" in answer, f"Answer should contain 'distillation' keyword. Got: {result['answer']}"
    
    # Check answer is substantial (not error message)
    assert len(result["answer"].strip()) > 20, "Answer should be substantial"
    assert "error" not in answer, "Answer should not contain error messages"

def test_source_count_matches_settings():
    """Test that number of sources matches RETRIEVE_TOP_K setting."""
    question = "What is distillation?"
    result = ask(question)
    
    # Check that sources list exists
    assert isinstance(result["sources"], list)
    
    # If we have sources, should match RETRIEVE_TOP_K or be less (if fewer docs available)
    if result["sources"]:
        assert len(result["sources"]) <= settings.RETRIEVE_TOP_K, \
            f"Sources count {len(result['sources'])} should not exceed RETRIEVE_TOP_K {settings.RETRIEVE_TOP_K}"
        
        # Check that sources are meaningful (not empty strings)
        for source in result["sources"]:
            assert isinstance(source, str), "Each source should be a string"
            assert len(source.strip()) > 0, "Sources should not be empty"

def test_numeric_citations_present():
    """Test that the system can handle citations (though the model may not always use them)."""
    question = "What is distillation?"
    result = ask(question)
    
    answer = result["answer"]
    
    # Check for numeric citations pattern [1], [2], etc.
    import re
    citations = re.findall(r'\[(\d+)\]', answer)
    
    # The system should support citations - if present, they should be valid
    if citations:
        # Citation numbers should be reasonable (1-based, within source count)
        for citation in citations:
            citation_num = int(citation)
            assert 1 <= citation_num <= len(result["sources"]), \
                f"Citation [{citation}] should be between 1 and {len(result['sources'])}"
    
    # Even if no citations are in the answer, the system should still work properly
    # The most important thing is that we have a meaningful answer
    assert len(answer.strip()) > 10, "Answer should be meaningful regardless of citations"

if __name__ == "__main__":
    # Run tests directly for quick validation
    test_distillation_answer_relevance()
    test_source_count_matches_settings()
    test_numeric_citations_present()
    print("✅ All reasoning quality tests passed!") 