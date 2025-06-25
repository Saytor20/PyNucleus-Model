"""
Answer processing module for RAG pipeline with deduplication and citation enforcement.
"""

import re
from typing import List, Dict, Any, Optional
from rapidfuzz import fuzz
from ..utils.logger import logger
from ..settings import settings

def is_answer_duplicate(answer: str, context_chunks: List[str], threshold: float = 90.0) -> bool:
    """
    Check if answer is a duplicate of context chunks using RapidFuzz.
    
    Args:
        answer: Generated answer to check
        context_chunks: List of context chunks to compare against
        threshold: Similarity threshold for considering as duplicate (default: 90%)
    
    Returns:
        True if answer is considered a duplicate, False otherwise
    """
    if not answer or not context_chunks:
        return False
    
    for chunk in context_chunks:
        if chunk and len(chunk.strip()) > 20:  # Only check substantial chunks
            similarity = fuzz.partial_ratio(answer, chunk)
            if similarity > threshold:
                logger.debug(f"Duplicate detected (similarity: {similarity:.1f}%): '{answer[:50]}...'")
                return True
    
    return False

def deduplicate_answer(sentences: List[str], threshold: float = 90.0) -> List[str]:
    """
    Remove duplicate sentences using RapidFuzz similarity matching.
    
    Args:
        sentences: List of sentences to deduplicate
        threshold: Similarity threshold for considering sentences as duplicates (default: 90%)
    
    Returns:
        List of unique sentences
    """
    if not sentences:
        return sentences
    
    unique_sentences = []
    
    for sentence in sentences:
        # Skip empty or very short sentences
        if not sentence or len(sentence.strip()) < 10:
            continue
            
        sentence = sentence.strip()
        
        # Check if this sentence is similar to any existing unique sentence
        is_duplicate = False
        for unique_sentence in unique_sentences:
            similarity = fuzz.ratio(sentence, unique_sentence)
            if similarity > threshold:
                is_duplicate = True
                logger.debug(f"Duplicate detected (similarity: {similarity:.1f}%): '{sentence[:50]}...'")
                break
        
        if not is_duplicate:
            unique_sentences.append(sentence)
    
    logger.info(f"Deduplication: {len(sentences)} -> {len(unique_sentences)} sentences")
    return unique_sentences

def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text using improved sentence boundary detection.
    
    Args:
        text: Input text to split into sentences
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Handle common abbreviations that shouldn't trigger sentence breaks
    text = re.sub(r'\b(?:Dr|Prof|Mr|Mrs|Ms|etc|vs|e\.g|i\.e|Fig|Table|Eq|Vol|No)\.\s*', 
                  lambda m: m.group(0).replace('.', '<!DOT!>'), text)
    
    # Split on sentence-ending punctuation followed by whitespace and capital letter
    sentences = re.split(r'[.!?]+\s+(?=[A-Z])', text)
    
    # Restore abbreviation dots
    sentences = [s.replace('<!DOT!>', '.').strip() for s in sentences]
    
    # Filter out empty sentences and normalize
    sentences = [s for s in sentences if s and len(s.strip()) > 5]
    
    return sentences

def has_valid_citations(text: str) -> bool:
    """
    Check if the text contains valid citations in the expected format.
    
    Args:
        text: Text to check for citations
        
    Returns:
        True if valid citations are found, False otherwise
    """
    if not text:
        return False
    
    # Look for various citation patterns
    citation_patterns = [
        r'\[Doc-\w+\]',  # [Doc-ID] format
        r'\[\d+\]',      # [1], [2], etc.
        r'\[doc_\d+\]',  # [doc_1], [doc_2], etc.
        r'\[[A-Za-z0-9_-]+\]'  # General document ID format
    ]
    
    for pattern in citation_patterns:
        if re.search(pattern, text):
            logger.debug(f"Found citation matching pattern: {pattern}")
            return True
    
    logger.debug("No valid citations found in text")
    return False

def extract_citations(text: str) -> List[str]:
    """
    Extract all citations from the text.
    
    Args:
        text: Text to extract citations from
        
    Returns:
        List of found citations
    """
    if not text:
        return []
    
    citations = []
    citation_patterns = [
        r'\[Doc-\w+\]',
        r'\[\d+\]',
        r'\[doc_\d+\]',
        r'\[[A-Za-z0-9_-]+\]'
    ]
    
    for pattern in citation_patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)
    
    return list(set(citations))  # Remove duplicates

def enforce_citation_format(text: str, available_sources: List[str]) -> str:
    """
    Ensure proper citation format and add missing citations where appropriate.
    
    Args:
        text: Text to process
        available_sources: List of available source IDs
        
    Returns:
        Text with enforced citation format
    """
    if not text or not available_sources:
        return text
    
    # If no citations exist, try to add them intelligently
    if not has_valid_citations(text):
        sentences = extract_sentences(text)
        if sentences:
            # Add a citation to the first substantive sentence
            first_sentence = sentences[0]
            if len(available_sources) > 0:
                enhanced_sentence = f"{first_sentence} [Doc-{available_sources[0]}]"
                text = text.replace(first_sentence, enhanced_sentence, 1)
                logger.info("Added citation to answer lacking proper citations")
    
    return text

def process_answer_quality(answer: str, sources: List[str], retry_count: int = 0) -> Dict[str, Any]:
    """
    Process answer for quality, deduplication, and citation enforcement.
    
    Args:
        answer: Raw answer text
        sources: List of source IDs used
        retry_count: Number of retries attempted
        
    Returns:
        Dictionary with processed answer and quality metrics
    """
    if not answer:
        return {
            "processed_answer": "I don't have enough information to provide a complete answer.",
            "has_citations": False,
            "sentence_count": 0,
            "citations_found": [],
            "deduplication_applied": False,
            "retry_count": retry_count,
            "quality_score": 0.0
        }
    
    # Extract sentences for deduplication
    sentences = extract_sentences(answer)
    original_sentence_count = len(sentences)
    
    # Apply deduplication
    threshold = getattr(settings, 'DEDUPLICATION_THRESHOLD', 90.0)
    deduplicated_sentences = deduplicate_answer(sentences, threshold)
    deduplication_applied = len(deduplicated_sentences) < len(sentences)
    
    # Reconstruct answer from deduplicated sentences
    processed_answer = ' '.join(deduplicated_sentences)
    
    # Enforce citation format
    processed_answer = enforce_citation_format(processed_answer, sources)
    
    # Check citation quality
    has_citations = has_valid_citations(processed_answer)
    citations_found = extract_citations(processed_answer)
    
    # Calculate quality score
    quality_score = calculate_answer_quality(processed_answer, has_citations, len(citations_found))
    
    return {
        "processed_answer": processed_answer,
        "has_citations": has_citations,
        "sentence_count": len(deduplicated_sentences),
        "original_sentence_count": original_sentence_count,
        "citations_found": citations_found,
        "deduplication_applied": deduplication_applied,
        "retry_count": retry_count,
        "quality_score": quality_score
    }

def calculate_answer_quality(answer: str, has_citations: bool, citation_count: int) -> float:
    """
    Calculate a quality score for the answer based on various factors.
    
    Args:
        answer: The answer text
        has_citations: Whether the answer has citations
        citation_count: Number of citations found
        
    Returns:
        Quality score between 0.0 and 1.0
    """
    if not answer:
        return 0.0
    
    score = 0.0
    
    # Length factor (reasonable length gets points)
    length = len(answer)
    if 50 <= length <= 1000:
        score += 0.3
    elif length > 1000:
        score += 0.2  # Very long answers might be verbose
    
    # Citation factor
    if has_citations:
        score += 0.4
        # Bonus for multiple citations
        if citation_count > 1:
            score += min(0.2, citation_count * 0.05)
    
    # Content quality factors
    if re.search(r'\b(process|system|method|technique|procedure)\b', answer.lower()):
        score += 0.1  # Technical content
    
    if re.search(r'\b(because|therefore|thus|due to|result)\b', answer.lower()):
        score += 0.1  # Explanatory content
    
    return min(1.0, score)

def should_retry_generation(quality_result: Dict[str, Any], max_retries: int = 2) -> bool:
    """
    Determine if answer generation should be retried based on quality metrics.
    
    Args:
        quality_result: Result from process_answer_quality
        max_retries: Maximum number of retries allowed
        
    Returns:
        True if generation should be retried, False otherwise
    """
    retry_count = quality_result.get('retry_count', 0)
    
    # Don't retry if max retries reached
    if retry_count >= max_retries:
        return False
    
    # Retry if no citations and citations are required
    require_citations = getattr(settings, 'REQUIRE_CITATIONS', True)
    if require_citations and not quality_result.get('has_citations', False):
        logger.info(f"Retrying generation due to missing citations (attempt {retry_count + 1})")
        return True
    
    # Retry if quality score is very low
    quality_score = quality_result.get('quality_score', 0.0)
    if quality_score < 0.3:
        logger.info(f"Retrying generation due to low quality score: {quality_score:.2f} (attempt {retry_count + 1})")
        return True
    
    return False

# Export main functions
__all__ = [
    "is_answer_duplicate",
    "deduplicate_answer", 
    "extract_sentences", 
    "has_valid_citations", 
    "extract_citations",
    "enforce_citation_format",
    "process_answer_quality",
    "calculate_answer_quality",
    "should_retry_generation"
] 