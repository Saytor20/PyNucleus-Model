"""
Answer processing module for RAG pipeline with deduplication and citation enforcement.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import fuzz
from collections import Counter
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

def truncate_answer(text: str, max_length: int = None) -> str:
    """
    Truncate answer text to a reasonable length while preserving sentence boundaries.
    
    Args:
        text: Text to truncate
        max_length: Maximum length in characters (defaults to settings.MAX_ANSWER_LENGTH)
        
    Returns:
        Truncated text that ends at a sentence boundary
    """
    if not text:
        return text
    
    if max_length is None:
        max_length = getattr(settings, 'MAX_ANSWER_LENGTH', 500)
    
    if len(text) <= max_length:
        return text
    
    # Find the last complete sentence within the limit
    truncated = text[:max_length]
    
    # Look for sentence endings in the truncated text
    sentence_endings = ['.', '!', '?']
    last_sentence_end = -1
    
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > last_sentence_end:
            last_sentence_end = pos
    
    # If we found a sentence ending, truncate there
    if last_sentence_end > max_length * 0.7:  # Only if it's not too early
        truncated = truncated[:last_sentence_end + 1]
    else:
        # Otherwise, find the last word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:
            truncated = truncated[:last_space] + '...'
        else:
            truncated = truncated + '...'
    
    return truncated.strip()

def remove_meta_commentary(text: str) -> str:
    """
    Remove meta-commentary and thinking sections from LLM responses.
    
    Args:
        text: Text that may contain meta-commentary
        
    Returns:
        Text with meta-commentary removed
    """
    if not text:
        return text
    
    # More specific patterns that indicate meta-commentary or thinking
    meta_patterns = [
        # Page references with commentary
        r'\[Page \d+\] --- In this context, I can see that.*?\.',
        r'\[Page \d+\] - In this context, I can see that.*?\.',
        
        # "However, there seems to be no direct mention..." type patterns
        r'However, there seems to be no direct mention.*?\.',
        r'However, there is no direct mention.*?\.',
        r'However, the text does not directly mention.*?\.',
        r'However, there appears to be no specific mention.*?\.',
        
        # "Could you please provide more details..." type patterns
        r'Could you please provide more details.*?\.',
        r'Could you please clarify.*?\.',
        r'Could you please explain.*?\.',
        r'Could you please specify.*?\.',
        
        # "Additionally, could you explain..." type patterns
        r'Additionally, could you explain.*?\.',
        r'Furthermore, could you explain.*?\.',
        r'Moreover, could you explain.*?\.',
        
        # "Here's a detailed explanation:" type patterns
        r"Here's a detailed explanation:",
        r"Here is a detailed explanation:",
        r"Let me provide a detailed explanation:",
        r"Let me explain in detail:",
    ]
    
    cleaned_text = text
    
    # Remove each pattern
    for pattern in meta_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up multiple spaces and line breaks
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
    
    # Remove leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    # If the text starts with a lowercase letter after cleaning, capitalize it
    if cleaned_text and cleaned_text[0].islower():
        cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]
    
    return cleaned_text

def process_answer_quality(answer: str, sources: List[str], retry_count: int = 0, 
                          question: str = None, expected_keywords: List[str] = None) -> Dict[str, Any]:
    """
    Enhanced answer processing with quality validation and improvement for chemical engineering.
    
    Args:
        answer: Generated answer text
        sources: List of source documents
        retry_count: Current retry attempt number
        question: Original question (for validation)
        expected_keywords: Expected keywords for domain validation
        
    Returns:
        Dictionary with processed answer and quality metrics
    """
    logger = logging.getLogger(__name__)
    
    if not answer:
        return {
            "processed_answer": "",
            "quality_score": 0.0,
            "deduplication_applied": False,
            "has_citations": False,
            "domain_relevance": 0.0,
            "technical_accuracy": 0.0,
            "improvement_applied": False
        }
    
    original_answer = answer
    
    # Step 1: Remove meta-commentary and thinking sections
    cleaned_answer = remove_meta_commentary(answer)
    
    # Step 2: Clean and format the raw answer
    cleaned_answer = clean_and_format_answer(cleaned_answer)
    
    # Step 3: Clean and deduplicate
    sentences = extract_sentences(cleaned_answer)
    unique_sentences = deduplicate_answer(sentences)
    logger.info(f"Deduplication: {len(sentences)} -> {len(unique_sentences)} sentences")
    
    # Step 4: Remove low-quality sentences
    filtered_sentences = _filter_low_quality_sentences(unique_sentences)
    
    # Step 5: Chemical engineering domain validation
    domain_score = _assess_domain_relevance(filtered_sentences, expected_keywords)
    
    # Step 6: Technical accuracy assessment
    technical_score = _assess_technical_accuracy(filtered_sentences, question)
    
    # Step 7: Reconstruct answer
    processed_answer = ' '.join(filtered_sentences)
    
    # Step 8: Truncate answer if too long
    processed_answer = truncate_answer(processed_answer)
    
    # Step 9: Add citations if missing
    has_citations = bool(re.search(r'\[Doc-\w+\]', processed_answer))
    if not has_citations and sources:
        processed_answer = enforce_citation_format(processed_answer, sources)
        has_citations = True
        logger.info("Added citation to answer lacking proper citations")
    
    # Step 10: Quality improvement for poor answers
    improvement_applied = False
    if (domain_score < 0.3 or technical_score < 0.3) and question and retry_count == 0:
        improved_answer = _attempt_answer_improvement(question, processed_answer, sources)
        if improved_answer and improved_answer != processed_answer:
            processed_answer = improved_answer
            improvement_applied = True
            logger.info("Applied answer improvement due to low quality scores")
    
    # Step 11: Final quality assessment
    quality_score = _calculate_quality_score(
        processed_answer, sources, domain_score, technical_score, 
        has_citations, len(filtered_sentences)
    )
    
    return {
        "processed_answer": processed_answer,
        "quality_score": quality_score,
        "deduplication_applied": len(sentences) != len(unique_sentences),
        "has_citations": has_citations,
        "domain_relevance": domain_score,
        "technical_accuracy": technical_score,
        "improvement_applied": improvement_applied,
        "sentence_count": len(filtered_sentences),
        "original_sentence_count": len(sentences)
    }

def _filter_low_quality_sentences(sentences: List[str]) -> List[str]:
    """Filter out low-quality sentences that don't contribute to answer quality."""
    filtered = []
    
    for sentence in sentences:
        # Clean the sentence first
        sentence = clean_and_format_answer(sentence)
        
        # Skip very short sentences (unless they're important statements)
        if len(sentence.strip()) < 20 and not _is_important_short_statement(sentence):
            continue
            
        # Skip repetitive or filler sentences
        if _is_filler_sentence(sentence):
            continue
            
        # Skip sentences that are mostly formatting or navigation
        if _is_formatting_sentence(sentence):
            continue
            
        # Skip sentences with too many repeated words
        if _has_excessive_repetition(sentence):
            continue
            
        # Skip sentences with terminal formatting artifacts
        if '│' in sentence or sentence.strip().startswith('│'):
            continue
            
        filtered.append(sentence)
    
    return filtered

def _is_important_short_statement(sentence: str) -> bool:
    """Check if a short sentence contains important information."""
    important_patterns = [
        r'\b(yes|no)\b',  # Direct answers
        r'\b\d+(\.\d+)?\s*(°C|°F|K|bar|psi|atm)\b',  # Temperature/pressure values
        r'\b\d+(\.\d+)?\s*(mol|kg|L|m³)\b',  # Quantities
    ]
    
    return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in important_patterns)

def _is_filler_sentence(sentence: str) -> bool:
    """Identify filler sentences that don't add value."""
    filler_patterns = [
        r'^(please|thank you|you\'re welcome)',
        r'(let me know|feel free to ask|any questions)',
        r'^(as mentioned|as stated|as discussed)',
        r'^(in conclusion|to summarize|in summary)',
        r'(hope this helps|hope that helps)',
    ]
    
    sentence_lower = sentence.lower().strip()
    return any(re.search(pattern, sentence_lower) for pattern in filler_patterns)

def _is_formatting_sentence(sentence: str) -> bool:
    """Identify sentences that are primarily formatting or navigation."""
    formatting_patterns = [
        r'^(answer:|question:|context:)',
        r'^(step \d+|stage \d+|phase \d+):\s*$',
        r'^(###|##|\*\*)',  # Markdown headers
        r'^\s*[-\*]\s*$',  # Empty list items
    ]
    
    return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in formatting_patterns)

def _has_excessive_repetition(sentence: str) -> bool:
    """Check if sentence has excessive word repetition."""
    words = sentence.lower().split()
    if len(words) < 5:
        return False
    
    word_counts = Counter(words)
    max_count = max(word_counts.values())
    
    # If any word appears more than 3 times in a sentence, it's likely repetitive
    return max_count > 3

def _assess_domain_relevance(sentences: List[str], expected_keywords: List[str] = None) -> float:
    """Assess how relevant the answer is to chemical engineering domain."""
    if not sentences:
        return 0.0
    
    text = ' '.join(sentences).lower()
    
    # Chemical engineering domain indicators
    domain_keywords = [
        'chemical', 'engineering', 'process', 'reactor', 'distillation', 'separation',
        'heat exchanger', 'mass transfer', 'heat transfer', 'fluid', 'pressure', 'temperature',
        'catalyst', 'reaction', 'equilibrium', 'thermodynamics', 'kinetics', 'phase',
        'absorption', 'adsorption', 'extraction', 'crystallization', 'evaporation',
        'filtration', 'membrane', 'pump', 'compressor', 'turbine', 'vessel', 'column',
        'efficiency', 'optimization', 'design', 'operation', 'control', 'safety',
        'plant', 'unit operation', 'flowsheet', 'piping', 'instrumentation'
    ]
    
    # Count domain keyword matches
    domain_matches = sum(1 for keyword in domain_keywords if keyword in text)
    domain_score = min(domain_matches / 10, 1.0)  # Normalize to 0-1
    
    # Boost score if expected keywords are present
    if expected_keywords:
        expected_matches = sum(1 for keyword in expected_keywords if keyword.lower() in text)
        expected_score = expected_matches / len(expected_keywords) if expected_keywords else 0
        domain_score = (domain_score + expected_score) / 2
    
    return domain_score

def _assess_technical_accuracy(sentences: List[str], question: str = None) -> float:
    """Assess the technical accuracy indicators in the answer."""
    if not sentences:
        return 0.0
    
    text = ' '.join(sentences).lower()
    accuracy_score = 0.5  # Start with neutral score
    
    # Positive indicators (increase score)
    positive_indicators = [
        r'\b\d+(\.\d+)?\s*(°C|°F|K|bar|psi|atm|kPa|MPa)\b',  # Specific values with units
        r'\b(equation|formula|principle|law|theory)\b',  # References to established knowledge
        r'\b(typically|usually|generally|commonly)\b',  # Appropriate qualifiers
        r'\b(according to|based on|studies show)\b',  # Evidence-based language
    ]
    
    for pattern in positive_indicators:
        if re.search(pattern, text):
            accuracy_score += 0.1
    
    # Negative indicators (decrease score)
    negative_indicators = [
        r'\b(always|never|all|every|none)\b(?!\s+(?:chemical|engineering))',  # Overgeneralizations
        r'\b(might be|could be|perhaps|maybe)\b.*\b(is|are)\b',  # Uncertain statements presented as facts
        r'\b(cats|dogs|animals|weather|food|sports)\b',  # Completely unrelated content
        r'\b(i think|i believe|in my opinion)\b',  # Personal opinions in technical context
    ]
    
    for pattern in negative_indicators:
        if re.search(pattern, text):
            accuracy_score -= 0.2
    
    # Specific chemical engineering accuracy checks
    if question:
        question_lower = question.lower()
        if 'distillation' in question_lower and 'distillation' in text:
            # Check for correct distillation concepts
            if any(term in text for term in ['boiling point', 'vapor', 'liquid', 'separation']):
                accuracy_score += 0.1
            if 'cat' in text or 'animal' in text:  # Wrong context
                accuracy_score -= 0.3
    
    return max(0.0, min(1.0, accuracy_score))

def _attempt_answer_improvement(question: str, poor_answer: str, sources: List[str]) -> Optional[str]:
    """Attempt to improve a poor quality answer using enhanced prompting."""
    try:
        from ..llm.prompting import build_answer_improvement_prompt
        from ..llm.model_loader import generate
        
        # Build context from sources for improvement
        context = f"Sources: {'; '.join(sources[:3])}" if sources else "No specific sources available"
        
        improvement_prompt = build_answer_improvement_prompt(question, poor_answer, context)
        
        # Generate improved answer with conservative settings
        improved_answer = generate(
            improvement_prompt,
            max_tokens=300,  # Shorter for focused improvement
            temperature=0.1  # Low temperature for accuracy
        )
        
        if improved_answer and len(improved_answer.strip()) > 50:
            # Clean the improved answer
            improved_answer = improved_answer.strip()
            
            # Basic validation - ensure it's actually better
            if (len(improved_answer) > len(poor_answer) * 0.8 and  # Not too short
                not _is_filler_sentence(improved_answer) and  # Not just filler
                _assess_domain_relevance([improved_answer]) > 0.3):  # Domain relevant
                return improved_answer
    
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Answer improvement failed: {e}")
    
    return None

def _calculate_quality_score(answer: str, sources: List[str], domain_score: float, 
                           technical_score: float, has_citations: bool, 
                           sentence_count: int) -> float:
    """Calculate overall quality score for the processed answer."""
    if not answer:
        return 0.0
    
    # Base score components
    domain_weight = 0.3
    technical_weight = 0.3
    citation_weight = 0.2
    completeness_weight = 0.2
    
    # Domain relevance score
    domain_component = domain_score * domain_weight
    
    # Technical accuracy score
    technical_component = technical_score * technical_weight
    
    # Citation score
    citation_component = (1.0 if has_citations else 0.0) * citation_weight
    
    # Completeness score (based on length and sentence count)
    answer_length = len(answer)
    completeness_score = min(1.0, answer_length / 200)  # Normalize to reasonable length
    sentence_factor = min(1.0, sentence_count / 3)  # Prefer multi-sentence answers
    completeness_component = (completeness_score * sentence_factor) * completeness_weight
    
    # Calculate final score
    quality_score = (domain_component + technical_component + 
                    citation_component + completeness_component)
    
    return round(min(1.0, max(0.0, quality_score)), 3)

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

def _remove_unwanted_prefixes(text: str) -> str:
    """Remove unwanted prefixes and formatting artifacts from text."""
    if not text:
        return text
    
    # Remove common unwanted prefixes
    unwanted_patterns = [
        r'^###\s*ANSWER:\s*',
        r'^ANSWER:\s*',
        r'^HUMAN:\s*',
        r'^ASSISTANT:\s*',
        r'^###\s*',
        r'^Question:\s*',
        r'^Context:\s*',
        r'^Response:\s*',
        r'^Explanation:\s*',
    ]
    
    for pattern in unwanted_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove standalone formatting markers
    text = re.sub(r'^\s*[-*]\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*###\s*$', '', text, flags=re.MULTILINE)
    
    return text.strip()

def clean_and_format_answer(text: str) -> str:
    """
    Clean and format answer text to remove formatting artifacts and improve readability.
    
    Args:
        text: Raw answer text that may contain formatting issues
        
    Returns:
        Cleaned and properly formatted text
    """
    if not text:
        return text
    
    # Step 1: Remove terminal-style formatting
    # Remove pipe characters and their surrounding whitespace
    text = re.sub(r'\s*│\s*', ' ', text)
    
    # Step 2: Remove unwanted prefixes first
    text = _remove_unwanted_prefixes(text)
    
    # Step 3: Fix spacing issues
    # Add spaces between words that are joined together
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # camelCase -> camel Case
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)     # word123 -> word 123
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # 123word -> 123 word
    
    # Fix common spacing issues
    text = re.sub(r'(\w)\s*\(\s*(\w)', r'\1 (\2', text)  # word( word -> word (word
    text = re.sub(r'(\w)\s*\)\s*(\w)', r'\1) \2', text)  # word) word -> word) word
    text = re.sub(r'(\w)\s*,\s*(\w)', r'\1, \2', text)   # word,word -> word, word
    
    # Step 4: Clean up citations
    # Fix malformed citation patterns
    text = re.sub(r'\[Doc-([^]]+)\]', r'[Doc-\1]', text)
    
    # Clean up source names in citations
    def clean_source_name(match):
        source = match.group(1)
        # Remove common formatting artifacts
        source = re.sub(r'[()]', '', source)  # Remove parentheses
        source = re.sub(r'\s+', ' ', source)  # Normalize whitespace
        source = source.strip()
        return f'[Doc-{source}]'
    
    text = re.sub(r'\[Doc-([^]]+)\]', clean_source_name, text)
    
    # Step 5: Fix mathematical notation
    # Improve LaTeX-style math formatting
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)', text)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)', text)
    
    # Fix broken LaTeX commands
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1/\2)', text)
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1/\2)', text)
    
    # Fix broken \frac commands
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1/\2)', text)
    text = re.sub(r'rac\{([^}]*)\}\{([^}]*)\}', r'(\1/\2)', text)
    
    # Clean up mathematical expressions
    text = re.sub(r'\(\s*([^)]+)\s*\)', r'(\1)', text)  # Remove extra spaces in parentheses
    text = re.sub(r'\[\s*([^\]]+)\s*\]', r'[\1]', text)  # Remove extra spaces in brackets
    
    # Step 6: Fix punctuation and sentence structure
    # Add missing periods at the end of sentences
    text = re.sub(r'([a-z])\s*$', r'\1.', text)
    
    # Fix double spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Step 7: Improve paragraph structure
    # Add line breaks for better readability
    text = re.sub(r'\.\s+([A-Z])', r'.\n\n\1', text)
    
    # Step 8: Additional cleaning for specific issues
    # Remove excessive underscores
    text = re.sub(r'_{3,}', '', text)
    
    # Clean up broken citations with parentheses
    text = re.sub(r'\[\(([^)]+)\)\s*([^]]+)\]', r'[\1 \2]', text)
    
    # Remove standalone punctuation marks
    text = re.sub(r'\s+[.,;:]\s+', ' ', text)
    
    # Step 9: Final cleanup
    text = text.strip()
    
    # Step 10: Ensure proper sentence endings
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    
    # Step 11: Final formatting improvements
    # Remove any remaining excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Remove excessive line breaks
    text = re.sub(r' +', ' ', text)  # Normalize spaces
    
    # Ensure proper paragraph breaks
    text = re.sub(r'\. ([A-Z][a-z])', r'.\n\n\1', text)
    
    # Clean up any remaining formatting artifacts
    text = re.sub(r'^\s*[-*]\s*', '', text, flags=re.MULTILINE)  # Remove list markers
    text = re.sub(r'^\s*###\s*', '', text, flags=re.MULTILINE)   # Remove markdown headers
    
    return text.strip()

# Export main functions
__all__ = [
    "is_answer_duplicate",
    "deduplicate_answer", 
    "extract_sentences", 
    "has_valid_citations", 
    "extract_citations",
    "enforce_citation_format",
    "process_answer_quality",
    "should_retry_generation",
    "clean_and_format_answer",
    "truncate_answer",
    "remove_meta_commentary"
] 