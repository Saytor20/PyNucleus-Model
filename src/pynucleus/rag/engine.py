"""
Enhanced RAG Pipeline for PyNucleus with improved retrieval, metadata indexing, and citation enforcement.
"""

# Apply telemetry patch before any ChromaDB imports
from ..utils.telemetry_patch import apply_telemetry_patch
apply_telemetry_patch()

import chromadb
from chromadb.config import Settings
from pathlib import Path
from ..settings import settings
from ..llm.model_loader import generate
from ..utils.logger import logger
from .answer_processing import process_answer_quality, should_retry_generation, is_answer_duplicate
from ..llm.prompting import build_enhanced_rag_prompt
import re

# Safe import of metrics with fallback
try:
    from ..metrics import Metrics, inc, start, stop
except ImportError:
    # Fallback no-op functions if metrics module is not available
    class NoOpMetrics:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    Metrics = NoOpMetrics()
    inc = lambda *args, **kwargs: None
    start = lambda *args, **kwargs: None
    stop = lambda *args, **kwargs: None
from typing import Optional, Dict

# Confidence calibration integration
_confidence_calibrator = None

# Simple in-memory cache for CLI usage
_in_memory_cache = {}
_cache_enabled = True

def _load_confidence_calibrator():
    """Load confidence calibration model on startup."""
    global _confidence_calibrator
    if _confidence_calibrator is None:
        try:
            from ..eval import load_latest_model
            _confidence_calibrator = load_latest_model()
            if _confidence_calibrator is None:
                logger.info("No trained confidence calibration model found, using identity function")
                _confidence_calibrator = lambda x: x  # Identity function
            else:
                logger.info("Confidence calibration model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load confidence calibration model: {e}")
            _confidence_calibrator = lambda x: x  # Fallback to identity
    return _confidence_calibrator

def _get_cached_response(question: str) -> Optional[Dict]:
    """Get cached response from in-memory cache"""
    if not _cache_enabled:
        return None
    
    # Normalize question for caching
    normalized_question = question.lower().strip()
    
    # Check if we have a cached response
    if normalized_question in _in_memory_cache:
        cached_data = _in_memory_cache[normalized_question]
        
        # Check if cache is still valid (1 hour TTL)
        import time
        if time.time() - cached_data.get('cached_at', 0) < 3600:  # 1 hour
            logger.info(f"Cache hit for question: {question[:50]}...")
            return cached_data['response']
        else:
            # Remove expired cache entry
            del _in_memory_cache[normalized_question]
    
    return None

def _cache_response(question: str, response: Dict) -> None:
    """Cache response in in-memory cache"""
    if not _cache_enabled:
        return
    
    # Normalize question for caching
    normalized_question = question.lower().strip()
    
    # Store response with timestamp
    import time
    _in_memory_cache[normalized_question] = {
        'response': response,
        'cached_at': time.time()
    }
    
    logger.info(f"Cached response for question: {question[:50]}...")
    
    # Limit cache size to prevent memory issues
    if len(_in_memory_cache) > 1000:
        # Remove oldest entries
        oldest_key = min(_in_memory_cache.keys(), 
                        key=lambda k: _in_memory_cache[k]['cached_at'])
        del _in_memory_cache[oldest_key]
        logger.debug("Removed oldest cache entry to prevent memory overflow")

# Global ChromaDB client for connection reuse
_client = None

def _get_chromadb_client():
    """Get or create ChromaDB client with robust error handling."""
    global _client
    if _client is None:
        try:
            # Disable telemetry at environment level
            import os
            os.environ["ANONYMIZED_TELEMETRY"] = "false"
            os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
            
            # Ensure directory exists
            Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
            
            # Use consistent settings across all modules (match vector_store & collector)
            client_settings = Settings(
                anonymized_telemetry=settings.CHROMA_TELEMETRY_ENABLED,
                allow_reset=True,
                chroma_client_auth_provider=None,
                chroma_server_host=None,
                chroma_server_http_port=None
            )
            _client = chromadb.PersistentClient(
                path=settings.CHROMA_PATH,
                settings=client_settings
            )
            
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            _client = None
    
    return _client

def _initialize_collection():
    """Initialize and return ChromaDB collection."""
    try:
        client = _get_chromadb_client()
        if client is None:
            logger.error("ChromaDB client not available")
            return None
        
        # Get or create collection
        collection = client.get_or_create_collection("pynucleus_documents")
        logger.info("ChromaDB collection initialized successfully")
        return collection
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB collection: {e}")
        return None

def retrieve_enhanced(question: str) -> tuple[list, list, list]:
    """
    Enhanced retrieval with metadata filtering and optimized performance.
    Returns top 3 most relevant documents from DIVERSE sources for better coverage.
    """
    try:
        client = _get_chromadb_client()
        if client is None:
            logger.warning("ChromaDB client not available")
            return [], [], []
        
        # Get or create collection
        collection = client.get_or_create_collection("pynucleus_documents")
        
        # Use larger initial retrieval to get diverse sources
        initial_top_k = getattr(settings, 'INITIAL_RETRIEVE_TOP_K', 15)  # Get more initially
        final_top_k = getattr(settings, 'ENHANCED_RETRIEVE_TOP_K', 3)    # Final diverse results
        similarity_threshold = getattr(settings, 'RAG_SIMILARITY_THRESHOLD', 0.3)
        
        # Enhanced query with metadata inclusion
        results = collection.query(
            query_texts=[question],
            n_results=initial_top_k,  # Get more results initially
            include=['documents', 'metadatas', 'distances']
        )
        
        # Extract and filter results by similarity
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        
        # Filter by similarity threshold and deduplicate by source
        source_seen = set()
        diverse_results = []
        
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            # Estimate similarity (this is approximate)
            similarity = max(0, 1 - (dist / 2))  # Rough conversion
            
            if similarity < similarity_threshold:
                continue
                
            # Extract source name
            source_name = "Unknown Source"
            if meta and 'source' in meta:
                source_name = meta['source']
            
            # Only keep if we haven't seen this source before
            if source_name not in source_seen:
                source_seen.add(source_name)
                diverse_results.append((doc, meta, similarity, source_name))
                
                # Stop if we have enough diverse sources
                if len(diverse_results) >= final_top_k:
                    break
        
        # Extract final results
        filtered_documents = [result[0] for result in diverse_results]
        filtered_metadatas = [result[1] for result in diverse_results]
        sources = [result[3] for result in diverse_results]
        
        logger.info(f"Enhanced retrieval: {len(filtered_documents)} documents from {len(source_seen)} unique sources (filtered from {len(documents)}) for query")
        return filtered_documents, sources, filtered_metadatas
        
    except Exception as e:
        logger.error(f"Enhanced retrieval failed: {e}")
        return [], [], []

def retrieve(question: str, k: int = None) -> tuple[list[str], list[str]]:
    """Legacy retrieve function for backward compatibility."""
    documents, sources, _ = retrieve_enhanced(question)
    return documents, sources

def build_enhanced_context(documents: list, sources: list, metadatas: list) -> str:
    """
    Build enhanced context with metadata from top documents.
    Optimized for quality by using better filtering and structuring.
    """
    if not documents:
        return ""
    
    # Use optimized settings for better quality
    max_docs = min(getattr(settings, 'MAX_CONTEXT_CHUNKS', 3), len(documents))
    documents = documents[:max_docs]
    sources = sources[:max_docs]
    metadatas = metadatas[:max_docs] if metadatas else []
    
    context_parts = []
    
    for i, (doc, source) in enumerate(zip(documents, sources)):
        if not doc or len(doc.strip()) < 30:  # Increased minimum length
            continue
            
        # Clean the document content aggressively to remove ALL formatting artifacts
        doc_content = doc.strip()
        
        # Remove ALL document structure artifacts and metadata
        doc_content = re.sub(r'^[\]\[\s]*Section:\s*[^\n]*\n?', '', doc_content, flags=re.MULTILINE)
        doc_content = re.sub(r'^[\]\[\s]*Page:\s*[^\n]*\n?', '', doc_content, flags=re.MULTILINE)
        doc_content = re.sub(r'^[\]\[\s]*\d{1,4}[\]\[\s]*\n?', '', doc_content, flags=re.MULTILINE)  # Remove numbers
        doc_content = re.sub(r'^[\]\[\s]*[a-z]{1,5}[\]\[\s]*\n?', '', doc_content, flags=re.MULTILINE)  # Remove fragments
        doc_content = re.sub(r'^\s*[\]\[\(\)]+\s*', '', doc_content, flags=re.MULTILINE)  # Remove brackets/parentheses
        
        # CRITICAL: Remove author information and contact details
        doc_content = re.sub(r'Assistant Professor.*?Phone:.*?\n?', '', doc_content, flags=re.MULTILINE | re.DOTALL)
        doc_content = re.sub(r'Professor.*?Phone:.*?\n?', '', doc_content, flags=re.MULTILINE | re.DOTALL)
        doc_content = re.sub(r'Dr\..*?Phone:.*?\n?', '', doc_content, flags=re.MULTILINE | re.DOTALL)
        doc_content = re.sub(r'Phone:\s*\+?\d+[-\s\d\(\)]*\n?', '', doc_content, flags=re.MULTILINE)
        doc_content = re.sub(r'Email:\s*[^\n]*\n?', '', doc_content, flags=re.MULTILINE)
        doc_content = re.sub(r'Department of.*?Engineering\s*\n?', '', doc_content, flags=re.MULTILINE)
        doc_content = re.sub(r'University.*?\n?', '', doc_content, flags=re.MULTILINE)
        doc_content = re.sub(r'[A-Z][a-z]+\s+University\s*\n?', '', doc_content, flags=re.MULTILINE)
        
        # Remove document citations and references
        doc_content = re.sub(r'\[Doc-[^\]]*\]', '', doc_content)
        doc_content = re.sub(r'\[Page \d+\]', '', doc_content)
        doc_content = re.sub(r'\[Section \d+\]', '', doc_content)
        
        # Remove markdown headers and formatting
        doc_content = re.sub(r'^#+\s*', '', doc_content, flags=re.MULTILINE)
        doc_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', doc_content)  # Remove bold
        doc_content = re.sub(r'\*([^*]+)\*', r'\1', doc_content)  # Remove italics
        
        # Remove box drawing and ALL terminal characters
        doc_content = re.sub(r'[─┌┐└┘│├┤┬┴┼]+', '', doc_content)
        doc_content = re.sub(r'[│\|]+', ' ', doc_content)
        
        # Remove standalone numbers/fragments at start of lines
        doc_content = re.sub(r'^\s*\d{1,3}\s+', '', doc_content, flags=re.MULTILINE)
        doc_content = re.sub(r'^\s*[a-z]{1,3}\s+', '', doc_content, flags=re.MULTILINE)
        
        # Remove common document continuation markers
        doc_content = re.sub(r'^\s*\(continues?\)\s*', '', doc_content, flags=re.MULTILINE)
        doc_content = re.sub(r'^\s*\.\.\.\s*', '', doc_content, flags=re.MULTILINE)
        
        # Remove lines that are just author/contact information
        lines = doc_content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that are clearly author/contact info
            if (re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+.*?(Professor|Dr\.|PhD)', line) or
                re.match(r'^Phone:.*?', line) or
                re.match(r'^Email:.*?', line) or
                re.match(r'^Department of.*?', line) or
                re.match(r'^[A-Z][a-z]+\s+University.*?', line) or
                len(line) < 5):  # Skip very short lines
                continue
            cleaned_lines.append(line)
        
        doc_content = '\n'.join(cleaned_lines)
        
        # Normalize whitespace and remove excessive line breaks
        doc_content = re.sub(r'\s+', ' ', doc_content)
        doc_content = re.sub(r'\n\s*\n\s*\n', '\n\n', doc_content)
        
        # Clean up sentence starts - ensure proper capitalization
        sentences = re.split(r'(?<=[.!?])\s+', doc_content)
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Only keep substantial sentences
                # Capitalize first letter
                if sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                cleaned_sentences.append(sentence)
        
        doc_content = ' '.join(cleaned_sentences)
        
        # Limit each document to ~600 characters for focused responses
        if len(doc_content) > 600:
            # Cut at sentence boundary
            sentences = re.split(r'[.!?]+', doc_content)
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence) <= 600:
                    truncated += sentence + ". "
                else:
                    break
            doc_content = truncated.strip()
        
        # Only include if content is substantial and meaningful
        if doc_content and len(doc_content.strip()) > 30:
            # Present as clean factual information
            context_parts.append(doc_content)
    
    # Join all context parts with clear separation
    if context_parts:
        return "\n\n".join(context_parts)
    else:
        return "No relevant technical information available."

def clean_model_response(response: str) -> str:
    """
    Clean model response while preserving the full answer quality.
    
    Args:
        response: Raw model response
        
    Returns:
        Cleaned response with full content preserved
    """
    if not response or not isinstance(response, str):
        return response
    
    # Remove any trailing metadata or system messages
    lines = response.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Stop at system metadata markers
        if any(marker in line.lower() for marker in [
            '── metadata ──', 'metadata:', 'response time:', 'model:', 
            'processing time:', 'timestamp:', 'model_id:', 'generation_time:',
            '── processing metadata ──', '── system info ──'
        ]):
            break
            
        clean_lines.append(line)
    
    # Rejoin and clean whitespace
    response = ' '.join(clean_lines)
    response = re.sub(r'\s+', ' ', response).strip()
    
    return response

def ask_enhanced(question: str, retry_count: int = 0) -> dict:
    """
    Enhanced ask function with improved retrieval, citation enforcement, retry logic, and LLM fallback.
    
    Args:
        question: The question to answer
        retry_count: Current retry attempt count
        
    Returns:
        Dictionary with enhanced answer and quality metrics
    """
    inc("queries_total")
    max_retries = getattr(settings, 'MAX_RETRY_ATTEMPTS', 2)
    
    try:
        # Enhanced retrieval with metadata
        start("enhanced_retrieval")
        documents, sources, metadatas = retrieve_enhanced(question)
        retrieval_time = stop("enhanced_retrieval")
        
        # Log retrieval results for debugging
        logger.info(f"Retrieved {len(documents)} documents for query: '{question}'")
        if documents:
            logger.info(f"Document lengths: {[len(doc) for doc in documents if doc]}")
        
        # Determine if we have sufficient RAG context
        has_rag_context = len(documents) > 0 and any(len(doc.strip()) > 50 for doc in documents)
        
        if has_rag_context:
            # Use RAG approach with retrieved documents
            context = build_enhanced_context(documents, sources, metadatas)
            prompt = build_enhanced_rag_prompt(context, question)
            answer_source = "rag"
            logger.info(f"Using RAG approach with {len([d for d in documents if d])} documents")
            
        else:
            # Fallback to LLM general knowledge with enhanced prompting
            logger.info(f"No relevant RAG context found for '{question}', using LLM general knowledge")
            
            # Create a general knowledge prompt that encourages comprehensive answers
            general_prompt = f"""You are an expert chemical engineer with deep knowledge in process simulation, equipment design, and industrial operations. 

### QUESTION
{question}

### INSTRUCTIONS
- Provide a comprehensive, detailed answer based on your general knowledge
- Use professional chemical engineering terminology
- Include relevant technical details, principles, and applications
- Structure your response clearly with explanations
- If applicable, mention common industrial practices and considerations
- Be thorough but concise

### ANSWER
"""
            prompt = general_prompt
            answer_source = "llm_general"
            context = ""
        
        # Generate model response with latency tracking
        start("enhanced_generation")
        answer = generate(
            prompt, 
            max_tokens=settings.MAX_TOKENS,
            temperature=getattr(settings, 'TEMPERATURE', 0.3)
        )
        generation_time = stop("enhanced_generation")
        
        # Process answer quality with enhanced validation and improvement
        # For LLM general knowledge, we're more lenient with citations
        if answer_source == "llm_general":
            # Don't require citations for general knowledge answers
            quality_result = process_answer_quality(
                answer, [], retry_count, 
                question=question, expected_keywords=[]
            )
            # Override citation requirement for general knowledge
            quality_result["has_citations"] = True  # General knowledge doesn't need citations
            quality_result["quality_score"] = max(quality_result["quality_score"], 0.6)  # Boost quality for general knowledge
        else:
            # Enhanced processing with question context for better validation
            quality_result = process_answer_quality(
                answer, sources, retry_count,
                question=question, expected_keywords=[]
            )
        
        # Check if retry is needed (only for RAG answers)
        if answer_source == "rag" and should_retry_generation(quality_result, max_retries):
            logger.info(f"Retrying RAG answer generation (attempt {retry_count + 1}/{max_retries})")
            return ask_enhanced(question, retry_count + 1)
        
        # Final answer processing
        final_answer = quality_result["processed_answer"]
        
        # Enhanced answer quality validation
        min_length = getattr(settings, 'MIN_ANSWER_LENGTH', 100)
        if not final_answer or len(final_answer.strip()) < min_length:
            if answer_source == "rag" and documents and context:
                final_answer = f"Based on the available information: {context[:200]}..."
            elif answer_source == "llm_general":
                # For general knowledge, provide a more helpful response
                final_answer = f"Based on general chemical engineering knowledge: {answer[:300]}..."
            else:
                final_answer = "I was unable to generate a complete response based on the available information."
        
        return {
            "answer": final_answer,
            "sources": sources[:len([d for d in documents if d])] if answer_source == "rag" else [],
            "answer_source": answer_source,
            "quality_metrics": {
                "has_citations": quality_result["has_citations"],
                "citations_found": quality_result["citations_found"],
                "quality_score": quality_result["quality_score"],
                "sentence_count": quality_result["sentence_count"],
                "deduplication_applied": quality_result["deduplication_applied"],
                "retry_count": retry_count
            },
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_chunks_retrieved": len(documents),
            "chunks_used": len([d for d in documents if d]) if answer_source == "rag" else 0,
            "rag_context_used": answer_source == "rag"
        }
        
    except Exception as e:
        logger.error(f"Enhanced ask function failed: {e}")
        return {
            "answer": f"An error occurred while processing your question: {str(e)}",
            "sources": [],
            "answer_source": "error",
            "quality_metrics": {
                "has_citations": False,
                "quality_score": 0.0,
                "retry_count": retry_count
            }
        }

def ask(question: str, max_retries: int = 2) -> dict:
    """
    Enhanced RAG pipeline with smart token management, duplication checking, confidence calibration, and caching.
    
    Args:
        question: User question
        max_retries: Maximum retry attempts for duplicate/incomplete answers
        
    Returns:
        Dictionary with answer, sources, metadata, and calibrated confidence scores:
        - answer: Generated response text
        - sources: List of source identifiers used
        - confidence_raw: Original confidence score from quality assessment [0.0-1.0]
        - confidence_cal: Calibrated confidence score using trained isotonic regression [0.0-1.0]
        - confidence: Alias for confidence_cal (backward compatibility)
        - Additional metadata: retrieval_count, retry_count, response_time, etc.
    """
    import time
    start_time = time.time()
    
    # Check cache first
    cached_response = _get_cached_response(question)
    if cached_response:
        # Add cache metadata
        cached_response['metadata'] = cached_response.get('metadata', {})
        cached_response['metadata']['cache_hit'] = True
        cached_response['metadata']['response_time'] = 0.01  # Very fast for cached responses
        logger.info(f"Returning cached response for: {question[:50]}...")
        return cached_response
    
    try:
        # Enhanced retrieval - get top 3 most relevant chunks
        documents, sources, metadatas = retrieve_enhanced(question)
        
        if not documents:
            # Apply confidence calibration even for no-document case
            raw_confidence = 0.0
            calibrator = _load_confidence_calibrator()
            
            try:
                calibrated_confidence = calibrator(raw_confidence)
                calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))
                
                # Record metrics
                try:
                    from ..metrics.prometheus import record_confidence_calibration
                    record_confidence_calibration(raw_confidence, calibrated_confidence, 'success')
                except ImportError:
                    pass  # Metrics module not available
                
            except Exception as e:
                logger.warning(f"Failed to apply confidence calibration for empty result: {e}")
                calibrated_confidence = raw_confidence
                
                # Record failure metrics
                try:
                    from ..metrics.prometheus import record_confidence_calibration
                    record_confidence_calibration(raw_confidence, raw_confidence, 'failure')
                except ImportError:
                    pass  # Metrics module not available
            
            return {
                "answer": "I don't have enough information to provide a complete answer.",
                "sources": [],
                "confidence_raw": raw_confidence,
                "confidence_cal": calibrated_confidence,
                "confidence": calibrated_confidence,
                "retrieval_count": 0
            }
        
        # Build enhanced context from top 3 chunks
        context = build_enhanced_context(documents, sources, metadatas)
        context_length = len(context)
        
        # Determine optimal token limit
        optimal_tokens = get_optimal_token_limit(question, context_length)
        is_complex = is_complex_question(question)
        
        logger.info(f"Question complexity: {'Complex' if is_complex else 'Simple'}, "
                   f"Token limit: {optimal_tokens}, Context length: {context_length}")
        
        # Generate answer with retry logic for duplicates and incomplete answers
        answer = None
        retry_count = 0
        final_tokens_used = optimal_tokens
        
        while retry_count <= max_retries:
            # Generate answer using enhanced prompt
            prompt = build_enhanced_rag_prompt(context, question)
            logger.info(f"DEBUG: Generated prompt length: {len(prompt)}")
            
            raw_answer = generate(prompt, max_tokens=final_tokens_used)
            logger.info(f"DEBUG: Raw answer from model: '{raw_answer[:200]}...' (length: {len(raw_answer)})")
            
            # Clean the response
            cleaned_answer = clean_model_response(raw_answer)
            logger.info(f"DEBUG: Cleaned answer: '{cleaned_answer[:200]}...' (length: {len(cleaned_answer)})")
            
            # Check for duplication
            if is_answer_duplicate(cleaned_answer, documents):
                logger.info(f"Duplicate answer detected, retry {retry_count + 1}/{max_retries + 1}")
                retry_count += 1
                if retry_count > max_retries:
                    cleaned_answer = "Information provided in context is insufficient to provide a complete answer."
                    break
                continue
            
            # Check for completeness
            if is_complex and not is_answer_complete(cleaned_answer, question):
                logger.info(f"Incomplete answer detected for complex question, retry {retry_count + 1}/{max_retries + 1}")
                retry_count += 1
                # Increase token limit for retry
                final_tokens_used = min(final_tokens_used + 100, getattr(settings, 'MAX_TOKENS_COMPLEX', 600))
                if retry_count > max_retries:
                    # Add completion note to existing answer
                    if cleaned_answer and not cleaned_answer.endswith('.'):
                        cleaned_answer += '.'
                    cleaned_answer += " Additional details would require more comprehensive analysis."
                    break
                continue
            
            # Exit retry loop - we have a good answer
            answer = cleaned_answer
            break
        
        # Ensure we have an answer
        if answer is None:
            answer = cleaned_answer or "Unable to generate a complete response."
            logger.warning(f"DEBUG: No valid answer after retries, using: '{answer}'")
        
        logger.info(f"DEBUG: Final answer before quality processing: '{answer[:200]}...' (length: {len(answer)})")
        
        # Process answer for quality and enhancement
        quality_result = process_answer_quality(
            answer, sources, retry_count,
            question=question, expected_keywords=[]
        )
        
        logger.info(f"DEBUG: Processed answer: '{quality_result['processed_answer'][:200]}...' (length: {len(quality_result['processed_answer'])})")
        
        # Apply confidence calibration
        raw_confidence = quality_result["quality_score"]
        
        try:
            calibrator = _load_confidence_calibrator()
            calibrated_confidence = calibrator(raw_confidence)
            calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))
            
            # Record metrics
            try:
                from ..metrics.prometheus import record_confidence_calibration
                record_confidence_calibration(raw_confidence, calibrated_confidence, 'success')
            except ImportError:
                pass  # Metrics module not available
            
        except Exception as e:
            logger.warning(f"Failed to apply confidence calibration: {e}")
            calibrated_confidence = raw_confidence
            
            # Record failure metrics
            try:
                from ..metrics.prometheus import record_confidence_calibration
                record_confidence_calibration(raw_confidence, raw_confidence, 'failure')
            except ImportError:
                pass  # Metrics module not available
        
        # Check if the answer was rejected during processing
        processed_answer = quality_result["processed_answer"]
        answer_was_rejected = (
            "couldn't provide" in processed_answer.lower() or
            "unable to generate" in processed_answer.lower() or
            "try rephrasing" in processed_answer.lower() or
            len(processed_answer) < 50
        )
        
        # If answer was rejected, reset sources and confidence to reflect the failure
        if answer_was_rejected:
            final_sources = []
            final_confidence_raw = 0.2  # Low confidence for rejected answers
            final_confidence_cal = 0.2
            final_has_citations = False
            logger.info("Answer was rejected during processing - clearing sources and reducing confidence")
        else:
            final_sources = sources
            final_confidence_raw = raw_confidence
            final_confidence_cal = calibrated_confidence
            final_has_citations = quality_result["has_citations"]
        
        # Create result dictionary with both confidence scores
        result = {
            "answer": processed_answer,
            "sources": final_sources,
            "confidence_raw": final_confidence_raw,
            "confidence_cal": final_confidence_cal,
            "confidence": final_confidence_cal,  # Backward compatibility - use calibrated
            "retrieval_count": len(documents),
            "deduplication_applied": quality_result["deduplication_applied"],
            "has_citations": final_has_citations,
            "retry_count": retry_count,
            "tokens_used": final_tokens_used,
            "is_complex_question": is_complex,
            "context_length": context_length,
            "response_time": time.time() - start_time,
            "retrieval_score": 0.5  # Placeholder - should come from actual retrieval scoring
        }
        
        logger.info(f"DEBUG: Final result answer: '{result['answer'][:200]}...' (length: {len(result['answer'])})")
        
        # Cache the successful response
        _cache_response(question, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        
        # Apply confidence calibration for error case
        raw_confidence = 0.0
        try:
            calibrator = _load_confidence_calibrator()
            calibrated_confidence = calibrator(raw_confidence)
            calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))
            
            # Record metrics
            try:
                from ..metrics.prometheus import record_confidence_calibration
                record_confidence_calibration(raw_confidence, calibrated_confidence, 'success')
            except ImportError:
                pass  # Metrics module not available
        except:
            calibrated_confidence = raw_confidence
            
            # Record failure metrics  
            try:
                from ..metrics.prometheus import record_confidence_calibration
                record_confidence_calibration(raw_confidence, raw_confidence, 'failure')
            except ImportError:
                pass  # Metrics module not available
        
        return {
            "answer": f"An error occurred while processing your question: {str(e)}",
            "sources": [],
            "confidence_raw": raw_confidence,
            "confidence_cal": calibrated_confidence,
            "confidence": calibrated_confidence,
            "retrieval_count": 0
        }

def ask_streaming(question: str):
    """Ask a question using RAG pipeline with streaming response."""
    try:
        # Use enhanced retrieval
        documents, sources, metadatas = retrieve_enhanced(question)
        
        # Handle empty results gracefully
        if not documents:
            def simple_fallback():
                yield {"sources": []}
                yield "I don't have enough information in my knowledge base to answer this question."
            return simple_fallback()
        
        # Build enhanced context
        context = build_enhanced_context(documents, sources, metadatas)
        
        # Import enhanced prompting
        from ..llm.prompting import build_enhanced_rag_prompt
        prompt = build_enhanced_rag_prompt(context, question)
        
        # Generate streaming answer
        stream_response = generate(prompt, max_tokens=settings.MAX_TOKENS, stream=True)
        
        # Create streaming generator
        def streaming_generator():
            # First, yield sources metadata
            yield {"sources": sources}
            
            # Handle different types of stream responses
            if hasattr(stream_response, '__iter__') and not isinstance(stream_response, str):
                # True streaming response - collect all chunks first to clean
                collected_response = ""
                for chunk in stream_response:
                    if chunk and isinstance(chunk, str):
                        collected_response += chunk
                
                # Process collected response for quality with enhanced validation
                quality_result = process_answer_quality(
                    collected_response, sources, 0,
                    question=question, expected_keywords=[]
                )
                cleaned_response = quality_result["processed_answer"]
                
                # Yield cleaned response word by word
                words = cleaned_response.split()
                for i, word in enumerate(words):
                    if i == 0:
                        yield word
                    else:
                        yield " " + word
            else:
                # Fallback: simulate streaming for non-streaming responses
                answer = str(stream_response) if stream_response else "Unable to generate response."
                quality_result = process_answer_quality(
                    answer, sources, 0,
                    question=question, expected_keywords=[]
                )
                cleaned_answer = quality_result["processed_answer"]
                
                # Split answer into words and yield word by word
                words = cleaned_answer.split()
                for i, word in enumerate(words):
                    if i == 0:
                        yield word
                    else:
                        yield " " + word
                    
                    # Small delay simulation for demonstration
                    import time
                    time.sleep(0.05)  # 50ms delay between words
        
        return streaming_generator()
        
    except Exception as e:
        logger.error(f"Enhanced streaming ask function failed: {e}")
        
        def error_generator():
            yield {"sources": []}
            yield f"An error occurred while processing your question: {str(e)}"
        
        return error_generator()

def is_complex_question(question: str) -> bool:
    """
    Detect if a question is complex and requires more detailed answers.
    
    Args:
        question: User question
        
    Returns:
        True if question is complex, False otherwise
    """
    question_lower = question.lower()
    complex_keywords = getattr(settings, 'COMPLEX_QUESTION_KEYWORDS', [
        "design", "how to", "process", "methodology", "steps", "procedure", 
        "implementation", "development", "construction", "analysis", "optimization"
    ])
    
    return any(keyword in question_lower for keyword in complex_keywords)

def is_answer_complete(answer: str, question: str) -> bool:
    """
    Check if an answer appears complete based on question type and answer structure.
    
    Args:
        answer: Generated answer
        question: Original question
        
    Returns:
        True if answer appears complete, False otherwise
    """
    if not answer or len(answer.strip()) < 50:
        return False
    
    answer_lower = answer.lower()
    question_lower = question.lower()
    
    # Check for incomplete indicators
    incomplete_indicators = [
        "1.", "2.", "3.", "4.", "5.",  # Numbered lists that might be cut off
        "step", "steps", "phase", "phases",
        "first", "second", "third", "next", "then", "finally",
        "additionally", "furthermore", "moreover", "also",
        "in conclusion", "to summarize", "in summary"
    ]
    
    # If it's a complex question, check for structured response
    if is_complex_question(question):
        # For design questions, look for specific design elements
        if "design" in question_lower:
            design_elements = [
                "principle", "consideration", "methodology", "approach",
                "component", "module", "system", "process", "step",
                "factor", "requirement", "specification", "standard"
            ]
            has_design_content = any(element in answer_lower for element in design_elements)
            
            # Check if answer has substantial content (not just repetitive text)
            word_count = len(answer.split())
            unique_words = len(set(answer.split()))
            diversity_ratio = unique_words / word_count if word_count > 0 else 0
            
            # Answer should have design content and reasonable diversity
            return has_design_content and diversity_ratio > 0.6 and word_count >= 80
        
        # For other complex questions, check for structured response
        has_structure = any(indicator in answer_lower for indicator in incomplete_indicators)
        
        # Check if answer ends abruptly (no proper conclusion)
        ends_abruptly = not any(ending in answer_lower[-100:] for ending in [
            ".", "!", "?", "conclusion", "summary", "therefore", "thus"
        ])
        
        # Check if answer is too short for a complex question
        too_short = len(answer.split()) < 50
        
        return has_structure and not ends_abruptly and not too_short
    
    # For simple questions, just check if it's substantial
    return len(answer.split()) >= 20

def get_optimal_token_limit(question: str, context_length: int) -> int:
    """
    Determine optimal token limit based on question complexity and context.
    
    Args:
        question: User question
        context_length: Length of retrieved context
        
    Returns:
        Optimal token limit for generation
    """
    base_tokens = getattr(settings, 'MAX_TOKENS', 400)
    min_tokens = getattr(settings, 'MIN_TOKENS', 100)
    max_complex_tokens = getattr(settings, 'MAX_TOKENS_COMPLEX', 600)
    
    if is_complex_question(question):
        # Complex questions get more tokens
        return int(min(max_complex_tokens, base_tokens * 1.5))
    else:
        # Simple questions get conservative tokens regardless of context length
        return int(max(min_tokens, min(base_tokens, 250))) 