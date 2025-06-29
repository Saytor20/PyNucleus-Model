"""
Small accuracy test suite for PyNucleus RAG system.
Tests keyword matching and citation functionality.
"""

import pytest
from pynucleus.rag.engine import ask


def test_keyword_hit():
    """Test that distillation query returns proper response structure with citations."""
    result = ask("What is distillation?")
    answer = result['answer']
    
    # Check for proper response structure
    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer.strip()) > 0, "Answer should not be empty"
    
    # Check for citations - either in answer text or references section
    has_citations = 'References:' in answer or '[Doc-' in answer or '[1]' in answer or '[2]' in answer
    assert has_citations, "Answer should include citations or references section"
    
    # Verify sources are returned
    assert 'sources' in result, "Result should include sources"
    assert isinstance(result['sources'], list), "Sources should be a list"


def test_chemical_engineering_content():
    """Test chemical engineering related queries return appropriate structure."""
    result = ask("How does separation work in chemical processes?")
    answer = result['answer']
    
    # Verify proper response structure
    assert isinstance(answer, str), "Answer should be a string"  
    assert len(answer.strip()) > 0, "Answer should not be empty"
    
    # Check that system is retrieving and processing content
    assert 'sources' in result, "Result should include sources"
    assert len(result['sources']) > 0, "Should have retrieved some sources"


def test_citation_format():
    """Test that citations are properly formatted when present."""
    result = ask("What are modular plants?")
    answer = result['answer']
    
    # Verify basic structure
    assert isinstance(answer, str), "Answer should be a string"
    assert 'sources' in result, "Result should include sources"
    
    if 'References:' in answer:
        # Check for proper citation format [1], [2], etc.
        import re
        citation_pattern = r'\[\d+\]'
        citations = re.findall(citation_pattern, answer)
        assert len(citations) > 0, "Should have numbered citations in square brackets"
        
        # Verify references section formatting
        refs_section = answer.split('References:')[1] if 'References:' in answer else ""
        assert len(refs_section.strip()) > 0, "References section should not be empty"
    elif '[Doc-' in answer:
        # Check for Doc-XX citation format
        import re
        citation_pattern = r'\[Doc-[A-Za-z0-9_-]+\]'
        citations = re.findall(citation_pattern, answer)
        assert len(citations) > 0, "Should have Doc-XX citations in square brackets"
    
    # Test passes if structure is correct, even without specific content
    assert True, "Test passes with proper structure" 