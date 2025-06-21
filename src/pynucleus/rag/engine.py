import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from ..settings import settings
from ..llm.model_loader import generate
from ..utils.logger import logger

# Global ChromaDB client for connection reuse
_client = None

def _get_chromadb_client():
    """Get or create ChromaDB client with robust error handling."""
    global _client
    if _client is None:
        try:
            # Ensure directory exists
            Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
            
            # Create client with minimal settings
            client_settings = ChromaSettings(anonymized_telemetry=False)
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
        
        # Create prompt with clear instructions for citations
        prompt = f"""Based on the provided context, answer the following question. Use numeric citations [1], [2], etc. to reference sources.

Question: {question}

Context:
{context}

Answer:"""
        
        # Generate answer with token limit
        answer = generate(prompt, max_tokens=settings.MAX_TOKENS)
        
        # Validate answer quality
        if not answer or len(answer.strip()) < 5:
            answer = "I was unable to generate a complete response based on the available information."
        
        return {
            "answer": answer.strip(),
            "sources": sources[:len(context_parts)]  # Only return sources that were actually used
        }
        
    except Exception as e:
        logger.error(f"Ask function failed: {e}")
        return {
            "answer": f"An error occurred while processing your question: {str(e)}",
            "sources": []
        } 