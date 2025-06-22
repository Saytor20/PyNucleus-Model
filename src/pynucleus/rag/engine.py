"""
RAG Pipeline for PyNucleus with ChromaDB backend and enhanced response cleaning.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from ..settings import settings
from ..llm.model_loader import generate
from ..utils.logger import logger
import re

# Global ChromaDB client for connection reuse
_client = None

def _get_chromadb_client():
    """Get or create ChromaDB client with robust error handling."""
    global _client
    if _client is None:
        try:
            # Ensure directory exists
            Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
            
            # Use consistent settings across all modules (match vector_store & collector)
            client_settings = ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
                chroma_client_auth_provider=None,
                chroma_server_host=None,
                chroma_server_http_port=None
            )
            _client = chromadb.PersistentClient(
                path=settings.CHROMA_PATH,
                settings=client_settings
            )
            logger.info("ChromaDB client initialized successfully")
            
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            _client = None
    
    return _client

def retrieve(question: str, k: int = None) -> tuple[list[str], list[str]]:
    """Retrieve documents and sources with graceful failure handling."""
    k = k or settings.RETRIEVE_TOP_K
    
    try:
        client = _get_chromadb_client()
        if client is None:
            logger.warning("ChromaDB client not available")
            return [], []
        
        # Get or create collection
        collection = client.get_or_create_collection("pynucleus_documents")
        
        # Query documents
        results = collection.query(
            query_texts=[question],
            n_results=k,
            include=['documents', 'metadatas']
        )
        
        # Extract documents and sources
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        sources = [
            meta.get('source', f'doc_{i}') if meta else f'doc_{i}' 
            for i, meta in enumerate(metadatas)
        ]
        
        logger.info(f"Retrieved {len(documents)} documents for query")
        return documents, sources
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return [], []

def clean_model_response(response: str) -> str:
    """
    Clean model response by extracting only the first factual sentence.
    
    Args:
        response: Raw model response
        
    Returns:
        Cleaned response with only the first factual sentence
    """
    if not response or not isinstance(response, str):
        return response
    
    # First, remove any metadata or internal reasoning that might be at the end
    lines = response.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Stop at common metadata markers
        if any(marker in line.lower() for marker in [
            '── metadata ──', 'metadata:', 'response time:', 'model:', 
            'processing time:', 'timestamp:', 'model_id:', 'generation_time:',
            '── processing metadata ──', '── system info ──'
        ]):
            break
            
        # Stop at reasoning markers
        if any(marker in line.lower() for marker in [
            'reasoning:', 'thinking:', 'analysis:', 'step 1:', 'step 2:', 'step 3:',
            'first,', 'next,', 'then,', 'finally,', 'therefore,', 'thus,',
            'looking at', 'examining', 'considering', 'based on'
        ]):
            break
            
        clean_lines.append(line)
    
    # Rejoin and clean
    response = ' '.join(clean_lines)
    
    # Split into sentences and find the first substantive one
    sentences = response.replace('\n', ' ').replace('  ', ' ').split('.')
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Skip empty or very short sentences
        if len(sentence) < 15:
            continue
            
        # Skip meta-commentary sentences - enhanced patterns
        skip_patterns = [
            'the correct answer', 'the answer should', 'so the answer', 'wait', 
            'let me', 'according to', 'however', 'therefore', 'but since',
            'based on', 'thus', 'in fact', 'to summarize', 'given this',
            'looking at', 'first', 'next', 'now', 'then', 'since there',
            'note:', 'answer:', 'final answer', 'citation', 'reference',
            'source [', 'user expects', 'instruction', 'please', 'check',
            'okay', 'let me start', 'let me understand', 'let me analyze',
            'looking back', 'looking at the context', 'looking at the information',
            'the context mentions', 'the context states', 'the context shows',
            'from the context', 'based on the context', 'according to the context',
            'the user asked', 'the question is', 'the question asks',
            'to answer this', 'to respond to', 'to address this',
            'i need to', 'i should', 'i will', 'i can see',
            'this means', 'this indicates', 'this shows', 'this suggests',
            'therefore', 'thus', 'hence', 'consequently',
            'in conclusion', 'to summarize', 'in summary',
            'the answer is', 'the response is', 'the solution is',
            'wait', 'hold on', 'let me think', 'let me check',
            'i think', 'i believe', 'i would say', 'i would suggest',
            'it appears', 'it seems', 'it looks like', 'it appears that',
            'as mentioned', 'as stated', 'as shown', 'as indicated',
            'the text says', 'the document says', 'the source says',
            'user expects', 'user wants', 'user is asking',
            'instruction', 'direction', 'guidance', 'requirement',
            'please note', 'please check', 'please verify',
            'should be', 'must be', 'needs to be', 'has to be',
            'correct answer', 'right answer', 'proper answer',
            'final answer', 'definitive answer', 'complete answer'
        ]
        
        if any(pattern in sentence.lower() for pattern in skip_patterns):
            continue
            
        # Skip sentences that are too generic or non-informative
        generic_patterns = [
            'this is a', 'this refers to', 'this means',
            'it is a', 'it refers to', 'it means',
            'there is a', 'there are', 'there exists',
            'we can see', 'we can observe', 'we can notice',
            'you can see', 'you can observe', 'you can notice',
            'one can see', 'one can observe', 'one can notice'
        ]
        
        if any(pattern in sentence.lower() for pattern in generic_patterns):
            continue
            
        # This looks like a factual sentence - clean it and return
        sentence = re.sub(r'\$.*?\$', '', sentence)  # Remove math notation
        sentence = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', sentence)  # Remove LaTeX
        sentence = re.sub(r'\([0-9]+\)', '', sentence)  # Remove citation numbers in parentheses
        sentence = re.sub(r'\s+', ' ', sentence).strip()  # Clean whitespace
        
        # Remove any remaining citation patterns that might be malformed
        sentence = re.sub(r'\[\d+\]\s*\d+', '', sentence)  # Remove patterns like [2] 4
        sentence = re.sub(r'^\d+\s*', '', sentence)  # Remove leading numbers
        
        # Add citation if it looks like it should have one
        if '[' not in sentence and len(sentence) > 20:
            # Check if there's a [1] or similar pattern later in the response
            citation_match = re.search(r'\[(\d+)\]', response)
            if citation_match:
                sentence += f' [{citation_match.group(1)}]'
        
        if sentence:
            return sentence + '.' if not sentence.endswith('.') else sentence
    
    # Fallback: if no good sentence found, return a simple version
    response_clean = response.split('.')[0].strip()
    if len(response_clean) > 10:
        # Clean up the fallback response
        response_clean = re.sub(r'\[\d+\]\s*\d+', '', response_clean)
        response_clean = re.sub(r'^\d+\s*', '', response_clean)
        return response_clean + '.'
    
    return "I don't have enough information to provide a complete answer."

def ask(question: str) -> dict:
    """Ask a question using RAG pipeline with robust handling and numeric citations."""
    try:
        # Retrieve documents and sources
        documents, sources = retrieve(question, settings.RETRIEVE_TOP_K)
        
        # Handle empty results gracefully
        if not documents:
            return {
                "answer": "I don't have enough information in my knowledge base to answer this question.",
                "sources": []
            }
        
        # Context slicing by MAX_CONTEXT_CHARS
        context_parts = []
        current_chars = 0
        
        for i, doc in enumerate(documents):
            # Add citation number to each document
            doc_with_citation = f"[{i+1}] {doc}"
            
            if current_chars + len(doc_with_citation) <= settings.MAX_CONTEXT_CHARS:
                context_parts.append(doc_with_citation)
                current_chars += len(doc_with_citation)
            else:
                # Truncate the last document to fit within limit
                remaining_chars = settings.MAX_CONTEXT_CHARS - current_chars
                if remaining_chars > 50:  # Only add if meaningful content can fit
                    truncated_doc = doc[:remaining_chars-20] + "..."
                    context_parts.append(f"[{i+1}] {truncated_doc}")
                break
        
        # Build context from processed documents
        context = "\n\n".join(context_parts)
        
        # Use enhanced prompting for consistent formatting
        from ..llm.prompting import build_prompt
        
        # Build enhanced prompt with direct answer instructions
        prompt = build_prompt(context, question)
        
        # Generate answer with token limit
        answer = generate(prompt, max_tokens=settings.MAX_TOKENS)
        
        # Validate answer quality
        if not answer or len(answer.strip()) < 5:
            answer = "I was unable to generate a complete response based on the available information."
        
        return {
            "answer": clean_model_response(answer.strip()),
            "sources": sources[:len(context_parts)]  # Only return sources that were actually used
        }
        
    except Exception as e:
        logger.error(f"Ask function failed: {e}")
        return {
            "answer": f"An error occurred while processing your question: {str(e)}",
            "sources": []
        }

def ask_streaming(question: str):
    """Ask a question using RAG pipeline with streaming response."""
    try:
        # Retrieve documents and sources
        documents, sources = retrieve(question, settings.RETRIEVE_TOP_K)
        
        # Handle empty results gracefully
        if not documents:
            def simple_fallback():
                yield {"sources": []}
                yield "I don't have enough information in my knowledge base to answer this question."
            return simple_fallback()
        
        # Context slicing by MAX_CONTEXT_CHARS
        context_parts = []
        current_chars = 0
        
        for i, doc in enumerate(documents):
            # Add citation number to each document
            doc_with_citation = f"[{i+1}] {doc}"
            
            if current_chars + len(doc_with_citation) <= settings.MAX_CONTEXT_CHARS:
                context_parts.append(doc_with_citation)
                current_chars += len(doc_with_citation)
            else:
                # Truncate the last document to fit within limit
                remaining_chars = settings.MAX_CONTEXT_CHARS - current_chars
                if remaining_chars > 50:  # Only add if meaningful content can fit
                    truncated_doc = doc[:remaining_chars-20] + "..."
                    context_parts.append(f"[{i+1}] {truncated_doc}")
                break
        
        # Build context from processed documents
        context = "\n\n".join(context_parts)
        
        # Import prompting for enhanced reasoning
        from ..llm.prompting import build_prompt
        
        # Build enhanced prompt with step-by-step reasoning
        prompt = build_prompt(context, question)
        
        # Generate streaming answer
        stream_response = generate(prompt, max_tokens=settings.MAX_TOKENS, stream=True)
        
        # Create streaming generator
        def streaming_generator():
            # First, yield sources metadata
            yield {"sources": sources[:len(context_parts)]}
            
            # Handle different types of stream responses
            if hasattr(stream_response, '__iter__') and not isinstance(stream_response, str):
                # True streaming response - collect all chunks first to clean
                collected_response = ""
                for chunk in stream_response:
                    if chunk and isinstance(chunk, str):
                        collected_response += chunk
                
                # Clean the collected response
                cleaned_response = clean_model_response(collected_response)
                
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
                cleaned_answer = clean_model_response(answer)
                
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
        logger.error(f"Streaming ask function failed: {e}")
        
        def error_generator():
            yield {"sources": []}
            yield f"An error occurred while processing your question: {str(e)}"
        
        return error_generator() 