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
    """Remove meta-commentary about the answer process"""
    # Patterns that indicate meta-commentary - be more specific to avoid removing legitimate content
    meta_patterns = [
        r"based on the (?:provided )?reference information[,\s]*",  # More specific - keep general "based on the information"
        r"according to the (?:provided )?reference (?:information|context)[,\s]*",
        r"from the (?:provided )?reference (?:information|context)[,\s]*",
        r"the (?:provided )?reference information (?:shows|indicates|suggests)[,\s]*",
        r"i cannot provide (?:more )?(?:specific )?details.*?(?:given|present|available)",
        r"therefore[,\s]*i cannot provide.*?(?:text|information)",
        r"there is no (?:additional|more|further) (?:technical )?detail.*?(?:above|described)",
        r"the key points? to address.*?remain:?",
        r"to answer (?:this )?(?:the )?question.*?(?:remains?|are?):?",
        r"in summary[,\s]*(?:the )?(?:key )?(?:points?|information)",
        # Remove the overly broad patterns that were removing legitimate content
        # r"based on (?:this|the) information[,\s]*",  # Too broad - removes legitimate scientific language
        # r"from (?:this|the) information[,\s]*",      # Too broad - removes legitimate scientific language
        r"not already present in the given text",
        r"beyond their definition as described above",
        r"beyond what (?:is|has been) (?:already )?(?:described|mentioned|stated)",
        r"as described (?:above|previously|in the (?:text|information))",
        r"\.?\s*therefore[,\s]*.*?(?:cannot|not able).*?(?:provide|give).*?(?:details?|information)",
        r"reference information.*?(?:of|about|regarding)",
        r"of reference information\s+",
        r"not fixed yet of reference information",
        # NEW: Remove conversation artifacts anywhere in the text
        r"\s+human:\s*.*?(?:answer:|assistant:)?\s*[^.]*\.?",  # " Human: [question] Answer: [response]"
        r"\s+human:\s*[^.!?]*[.!?]\s*",   # " Human: [question]? "
        r"\s+assistant:\s*(?:sure|yes|certainly)?\s*",  # " Assistant: Sure"
        r"\s+user:\s*.*?(?:answer:|assistant:)?\s*[^.]*\.?",   # " User: [question] Answer: [response]"
        r"\s+answer:\s*(?:sure|yes|certainly)?\s*",  # " Answer: Sure"
        r"\s+i have been thinking.*?(?:\[|$)",  # " I have been thinking..." contamination
        r"\s+[A-D]\)\s*By\s+.*?[A-D]\s+.*?(?:Select one:|The correct answer).*?(?:\[|$)",  # Multiple choice question contamination
        r"\s+human resources management.*?(?:\[|$)",  # HR/unrelated topic contamination
        r"\s+human resources department:.*?(?:\[|$)",  # HR department contamination
        r"\s+as an ai assistant.*?(?:\[|$)",  # AI assistant contamination
        r"\s+how does this relate to.*?(?:provided|context).*?",  # Meta-commentary about context
        r"^human:\s*.*?(?=the\s+\w+)",  # "Human: [question] The actual answer..." (start of text)
        r"^human:\s*[^.!?]*[.!?]\s*",   # "Human: [question]? " (start of text)
        r"^assistant:\s*",              # "Assistant: " (start of text)
        r"^user:\s*.*?(?=the\s+\w+)",   # "User: [question] The actual answer..." (start of text)
        r"^question:\s*.*?(?=the\s+\w+)", # "Question: [question] The actual answer..." (start of text)
    ]
    
    text_cleaned = text
    for pattern in meta_patterns:
        text_cleaned = re.sub(pattern, "", text_cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove multiple spaces and clean up punctuation
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned)
    text_cleaned = re.sub(r'\s*[,;]\s*', ', ', text_cleaned)
    text_cleaned = re.sub(r'\s*\.\s*', '. ', text_cleaned)
    
    return text_cleaned.strip()

def process_answer_quality(answer: str, sources: List[str], retry_count: int = 0, 
                          question: str = None, expected_keywords: List[str] = None) -> Dict[str, Any]:
    """
    Enhanced answer processing with quality validation and improvement for chemical engineering.
    
    Args:
        answer: Generated answer text from LLM
        sources: List of source documents
        retry_count: Current retry attempt number
        question: Original question (for validation)
        expected_keywords: Expected keywords for domain validation
        
    Returns:
        Dictionary with processed answer and quality metrics
    """
    logger = logging.getLogger(__name__)
    
    if not answer or len(answer.strip()) < 5:
        logger.warning("Empty or very short answer provided")
        return {
            "processed_answer": "I was unable to generate a complete response.",
            "quality_score": 0.0,
            "deduplication_applied": False,
            "has_citations": False,
            "domain_relevance": 0.0,
            "technical_accuracy": 0.0,
            "improvement_applied": False,
            "sentence_count": 0,
            "original_sentence_count": 0
        }
    
    original_answer = answer
    logger.info(f"Processing answer of length {len(answer)}")
    
    # Step 1: CRITICAL - Remove meta-commentary and document citations FIRST
    cleaned_answer = remove_meta_commentary(answer)
    
    # Step 2: Basic cleaning - remove unwanted formatting and prefixes
    cleaned_answer = clean_and_format_answer(cleaned_answer)
    
    # Step 3: Light filtering - only remove clearly irrelevant content
    filtered_answer = filter_irrelevant_content(cleaned_answer)
    
    # Step 4: Preserve LLM-generated content - don't over-process it
    # Only apply concise processing if the answer is extremely long (>1500 chars)
    if len(filtered_answer) > 1500:
        processed_answer = create_concise_answer(filtered_answer, max_sentences=8)
    else:
        processed_answer = filtered_answer
    
    # Step 5: Basic deduplication - only remove exact duplicates
    sentences = extract_sentences(processed_answer)
    unique_sentences = deduplicate_answer(sentences, threshold=95.0)  # Very high threshold
    logger.info(f"Deduplication: {len(sentences)} -> {len(unique_sentences)} sentences")
    
    # Step 6: Minimal quality filtering - only remove obviously bad sentences
    filtered_sentences = []
    for sentence in unique_sentences:
        # Only filter out very clearly problematic sentences
        if (len(sentence.strip()) < 10 or  # Too short
            sentence.strip().count('│') > 2 or  # Terminal formatting
            sentence.strip().startswith('│') or  # Terminal formatting
            re.match(r'^\s*[─┌┐└┘│]+\s*$', sentence)):  # Just box drawing
            continue
        filtered_sentences.append(sentence)
    
    # Step 7: Preserve the LLM response - don't filter too much
    if not filtered_sentences and unique_sentences:
        filtered_sentences = unique_sentences
        logger.info("Preserved unique sentences as filtering was too aggressive")
    
    # Step 8: Basic domain validation (but don't penalize heavily)
    domain_score = _assess_domain_relevance(filtered_sentences, expected_keywords)
    
    # Step 9: Basic technical assessment
    technical_score = _assess_technical_accuracy(filtered_sentences, question)
    
    # Step 10: Reconstruct answer from sentences
    final_answer = ' '.join(filtered_sentences)
    
    # Step 11: Add citations if completely missing and we have sources
    has_citations = bool(re.search(r'\[Doc-\w+\]', final_answer))
    if not has_citations and sources and len(final_answer) > 50:
        final_answer = enforce_citation_format(final_answer, sources)
        has_citations = True
        logger.info("Added citations to answer")
    
    # Step 12: CRITICAL - ensure we don't return empty/minimal content
    # If our processing reduced the answer too much, use the original
    if not final_answer or len(final_answer.strip()) < 30:
        final_answer = original_answer.strip()
        logger.warning("Processing removed too much content, using original LLM answer")
        # Recompute metadata for original answer
        sentences = extract_sentences(final_answer)
        filtered_sentences = sentences
        has_citations = bool(re.search(r'\[Doc-\w+\]', final_answer))
    
    # Step 13: Final quality assessment
    quality_score = _calculate_quality_score(
        final_answer, sources, domain_score, technical_score, 
        has_citations, len(filtered_sentences)
    )
    
    logger.info(f"Final processed answer length: {len(final_answer)}")
    
    return {
        "processed_answer": final_answer,
        "quality_score": max(quality_score, 0.4),  # Minimum quality score
        "deduplication_applied": len(sentences) != len(unique_sentences),
        "has_citations": has_citations,
        "domain_relevance": domain_score,
        "technical_accuracy": technical_score,
        "improvement_applied": False,
        "sentence_count": len(filtered_sentences),
        "original_sentence_count": len(extract_sentences(original_answer))
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
    
    # Base score components - more balanced weights
    domain_weight = 0.25       # Domain relevance (25%)
    technical_weight = 0.25    # Technical accuracy (25%)  
    citation_weight = 0.30     # Citation quality (30%)
    completeness_weight = 0.20 # Answer completeness (20%)
    
    # Domain relevance score - more generous minimum
    domain_component = max(domain_score, 0.6) * domain_weight
    
    # Technical accuracy score - more generous minimum
    technical_component = max(technical_score, 0.7) * technical_weight
    
    # Citation score - less harsh penalty
    if has_citations:
        # Reward multiple sources
        if len(sources) >= 3:
            citation_score = 1.0
        elif len(sources) >= 2:
            citation_score = 0.95
        else:
            citation_score = 0.9
    else:
        citation_score = 0.6  # Less harsh penalty for missing citations
    
    citation_component = citation_score * citation_weight
    
    # Completeness score (based on length and sentence count) - more generous
    answer_length = len(answer)
    completeness_score = min(1.0, answer_length / 150)  # Lower threshold for "complete"
    sentence_factor = min(1.0, max(0.7, sentence_count / 2))  # Minimum 70%, prefer 2+ sentences
    completeness_component = (completeness_score * sentence_factor) * completeness_weight
    
    # Calculate final score
    quality_score = (domain_component + technical_component + 
                    citation_component + completeness_component)
    
    # Ensure minimum reasonable score for properly formed answers
    if has_citations and answer_length >= 50 and sentence_count >= 1:
        quality_score = max(quality_score, 0.65)  # Minimum 65% for basic answers with citations
    
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
    Clean and format answer text using systematic approach to remove artifacts.
    
    Args:
        text: Raw answer text that may contain formatting issues
        
    Returns:
        Cleaned and properly formatted text
    """
    if not text or not text.strip():
        return "I apologize, but I couldn't generate a meaningful answer. Please try rephrasing your question."
    
    try:
        # Start with the raw answer
        cleaned = text.strip()
        
        # Step 1: Remove meta-commentary first (most important)
        cleaned = remove_meta_commentary(cleaned)
        
        # Step 2: Remove document references 
        cleaned = remove_document_references(cleaned)
        
        # Step 3: Remove author information and contact details
        cleaned = remove_author_info(cleaned)
        
        # Step 4: Filter irrelevant content
        cleaned = filter_irrelevant_content(cleaned)
        
        # Step 5: Clean up artifacts and formatting
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'^\s*[\.\,\;\:\-]+\s*', '', cleaned)  # Remove leading punctuation
        cleaned = re.sub(r'\s*[\.\,\;\:\-]+\s*$', '.', cleaned)  # Clean trailing punctuation
        
        # Step 6: Ensure proper capitalization
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        # Step 7: Final validation - if too short or still contains artifacts, reject
        if len(cleaned.strip()) < 10:
            return "I couldn't provide a meaningful answer to your question. Please try rephrasing it."
        
        # Check for remaining problematic artifacts (be very specific to avoid false positives)
        problematic_artifacts = [
            'therefore, i cannot provide',
            'based on the reference information',
            'the reference information shows',
            'i cannot provide more details beyond what is described'
        ]
        if any(artifact in cleaned.lower() for artifact in problematic_artifacts):
            return "I couldn't provide a clean answer to your question. Please try rephrasing it."
        
        return cleaned.strip()
        
    except Exception as e:
        logging.warning(f"Error in clean_and_format_answer: {e}")
        return "I encountered an error processing the answer. Please try again."

def create_concise_answer(text: str, max_sentences: int = 5) -> str:  # Increased default from 3 to 5
    """
    Create a concise version of the answer by keeping only the most relevant sentences.
    
    Args:
        text: Original answer text
        max_sentences: Maximum number of sentences to keep
        
    Returns:
        Concise version of the answer
    """
    if not text:
        return text
    
    sentences = extract_sentences(text)
    if len(sentences) <= max_sentences:
        return text
    
    # For very long sentences, try to break them up further
    processed_sentences = []
    for sentence in sentences:
        # If sentence is very long (>200 chars), try to split on key connectors
        if len(sentence) > 200:  # Increased threshold
            # Split on connecting words that often indicate new ideas
            connectors = ['. In this way,', '. This approach', '. These plants', '. Furthermore,', '. Additionally,', '. However,', '. Therefore,']
            for connector in connectors:
                if connector in sentence:
                    parts = sentence.split(connector, 1)
                    processed_sentences.append(parts[0] + '.')
                    if len(parts) > 1:
                        processed_sentences.append(connector.strip('. ') + parts[1])
                    break
            else:
                processed_sentences.append(sentence)
        else:
            processed_sentences.append(sentence)
    
    sentences = processed_sentences
    
    # Score sentences by relevance
    scored_sentences = []
    for sentence in sentences:
        score = 0
        
        # Prefer sentences with technical terms
        technical_terms = [
            'chemical', 'process', 'reactor', 'distillation', 'plant', 'modular',
            'efficiency', 'conversion', 'temperature', 'pressure', 'flow', 'separation',
            'catalyst', 'reaction', 'heat', 'mass', 'transfer', 'design', 'operation'
        ]
        
        sentence_lower = sentence.lower()
        for term in technical_terms:
            if term in sentence_lower:
                score += 2
        
        # Prefer sentences with citations
        if '[Doc-' in sentence:
            score += 3
        
        # Prefer definition-style sentences
        if any(phrase in sentence_lower for phrase in ['refers to', 'is a', 'are a', 'defined as']):
            score += 4
        
        # Penalize very long sentences (but be more lenient)
        if len(sentence) > 300:  # Increased threshold
            score -= 1  # Reduced penalty
        
        # Penalize sentences with formatting artifacts
        if any(artifact in sentence for artifact in ['│', '###', 'ANSWER:', 'Context:']):
            score -= 3  # Reduced penalty
        
        # Penalize (but don't heavily penalize) off-topic content
        irrelevant_terms = [
            'human resources', 'hrms', 'employee', 'recruitment', 'training',
            'compensation', 'succession planning', 'career paths', 'workforce'
        ]
        for term in irrelevant_terms:
            if term in sentence_lower:
                score -= 5  # Reduced penalty from -10 to -5
        
        scored_sentences.append((score, sentence))
    
    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [sent for score, sent in scored_sentences[:max_sentences] if score > -8]  # More lenient threshold
    
    # If no good sentences found, return first few sentences
    if not top_sentences:
        top_sentences = sentences[:max_sentences]
    
    # Ensure the result is reasonably sized (increased limit)
    result = ' '.join(top_sentences)
    if len(result) > 1000:  # Increased from 500 to 1000
        # If still too long, take only the highest scoring sentences that fit
        result = ""
        for score, sentence in scored_sentences:
            if score > -8 and len(result + sentence) <= 1000:  # More lenient
                result += sentence + " "
            if len(result) > 900:  # Stop before hitting the limit
                break
        result = result.strip()
    
    # Final check - if result is too short, return the original text
    if len(result) < 50 and len(text) > 50:
        return text
    
    return result

def filter_irrelevant_content(text: str) -> str:
    """
    Filter out clearly irrelevant content while preserving legitimate LLM responses.
    Only removes obviously irrelevant content, not borderline cases.
    
    Args:
        text: Text to filter
        
    Returns:
        Filtered text with irrelevant content removed
    """
    if not text:
        return text
    
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Only filter out very clearly irrelevant lines
        # Be very conservative - when in doubt, keep the content
        
        # Skip lines with obvious navigation/UI elements
        if any(phrase in line.lower() for phrase in [
            'click here', 'go to', 'navigate to', 'menu option',
            'press button', 'select from dropdown'
        ]):
            continue
            
        # Skip lines that are clearly programming/code artifacts
        if any(phrase in line for phrase in [
            'def ', 'class ', 'import ', 'from ', 'return ',
            '#!/usr/bin', 'if __name__', 'print('
        ]):
            continue
            
        # Skip lines that are clearly metadata/headers only if very obvious
        if (line.startswith('Source:') and len(line) < 30 or
            line.startswith('Page:') and len(line) < 20 or
            line.startswith('Section:') and len(line) < 30):
            continue
            
        # Skip empty citations or broken formatting
        if re.match(r'^\s*\[Doc-\s*\]\s*$', line):
            continue
            
        # Keep everything else - err on the side of preserving content
        filtered_lines.append(line)
    
    result = ' '.join(filtered_lines)
    
    # Final cleanup - remove excessive whitespace
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()

def remove_author_info(text: str) -> str:
    """Remove author information and contact details from text"""
    if not text:
        return text
    
    # Patterns to remove author information
    author_patterns = [
        r'assistant professor.*?(?:\n|$)',
        r'professor.*?(?:\n|$)', 
        r'department of.*?(?:\n|$)',
        r'university.*?(?:\n|$)',
        r'phone:\s*[+\d\-\(\)\s\.]+',
        r'email:\s*[^\s\n]+',
        r'contact:\s*[^\n]+',
        r'address:\s*[^\n]+',
        r'office:\s*[^\n]+',
        r'fax:\s*[+\d\-\(\)\s\.]+',
    ]
    
    text_cleaned = text
    for pattern in author_patterns:
        text_cleaned = re.sub(pattern, '', text_cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    return text_cleaned.strip()

def remove_document_references(text: str) -> str:
    """Remove document references and metadata"""
    # Remove document references like [Doc-filename.txt] or similar
    patterns = [
        r'\[Doc-[^\]]+\]\.?',
        r'\[Document[^\]]*\]\.?',
        r'\[Source[^\]]*\]\.?',
        r'\[Reference[^\]]*\]\.?',
        r'\[Page \d+\]',
        r'\[\d+\]',  # Numbered references
        r'Source:.*?(?:\n|$)',
        r'Document:.*?(?:\n|$)',
        r'Reference:.*?(?:\n|$)',
        r'\[.*?\.(?:txt|pdf|docx?|html?)\]',  # Any file extension in brackets
        r'^\s*\d+\.\s*$',  # Standalone numbers
        r'^\s*\d+\.\s+$',  # Numbers with spaces
        r'The key points (?:to address )?(?:the question )?remain:\s*\d*\.?\s*',
        r'remain:\s*\d*\.?\s*\[Doc-',
    ]
    
    text_cleaned = text
    for pattern in patterns:
        text_cleaned = re.sub(pattern, '', text_cleaned, flags=re.MULTILINE | re.IGNORECASE)
    
    # Clean up any remaining artifacts
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned)
    text_cleaned = re.sub(r'^\s*[\.\,\;\:]+\s*', '', text_cleaned)  # Remove leading punctuation
    text_cleaned = re.sub(r'\s*[\.\,\;\:]+\s*$', '', text_cleaned)  # Remove trailing punctuation
    
    return text_cleaned.strip()



# Export main functions
__all__ = [
    "process_answer_quality",
    "clean_and_format_answer", 
    "create_concise_answer",
    "remove_meta_commentary",
    "filter_irrelevant_content",
    "remove_document_references",
    "remove_author_info"
] 