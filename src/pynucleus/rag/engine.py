import chromadb
from chromadb.config import Settings
from pathlib import Path
from ..settings import settings
from ..llm.model_loader import generate
from ..llm.prompting import build_prompt
from ..utils.logger import logger
from ..rag.vector_store import ChromaVectorStore

# Centralized ChromaDB client management
_client = None
_coll = None
_store = None

def _get_chromadb_client():
    """Get or create ChromaDB client with consistent settings."""
    global _client
    if _client is None:
        try:
            # Ensure directory exists
            Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
            
            # Create client with consistent settings
            client_settings = Settings(
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
            
            # Test client connectivity
            _client.list_collections()
            logger.info("ChromaDB client initialized successfully in engine")
            
        except Exception as e:
            if "already exists" in str(e).lower():
                # Handle existing instance conflict by archiving and recreating
                logger.warning(f"ChromaDB instance conflict detected, archiving old DB...")
                try:
                    _archive_and_recreate_chromadb()
                    
                    # Recreate client with consistent settings
                    _client = chromadb.PersistentClient(
                        path=settings.CHROMA_PATH,
                        settings=client_settings
                    )
                    logger.info("ChromaDB client recreated after archiving")
                    
                except Exception as reset_error:
                    logger.error(f"Failed to archive/recreate ChromaDB: {reset_error}")
                    _client = None
            else:
                logger.error(f"ChromaDB initialization failed: {e}")
                _client = None
    
    return _client

def _archive_and_recreate_chromadb():
    """Archive existing ChromaDB and create a fresh one."""
    from datetime import datetime
    import shutil
    
    chroma_path = Path(settings.CHROMA_PATH)
    if chroma_path.exists():
        # Create archive directory
        archive_dir = chroma_path.parent / "archived_vector_dbs"
        archive_dir.mkdir(exist_ok=True)
        
        # Archive with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"vector_db_archived_{timestamp}"
        
        try:
            shutil.move(str(chroma_path), str(archive_path))
            logger.info(f"Archived old ChromaDB to {archive_path}")
            
            # Recreate fresh directory
            chroma_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created fresh ChromaDB directory at {chroma_path}")
            
        except Exception as e:
            logger.error(f"Failed to archive ChromaDB: {e}")
            # Fallback: try to remove and recreate
            try:
                shutil.rmtree(str(chroma_path))
                chroma_path.mkdir(parents=True, exist_ok=True)
                logger.info("Removed and recreated ChromaDB directory")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup ChromaDB directory: {cleanup_error}")
                raise

def _initialize_collection():
    """Initialize ChromaDB collection using centralized client."""
    global _coll
    if _coll is None:
        try:
            client = _get_chromadb_client()
            if client is None:
                return None
            
            # Use consistent collection name with other modules
            _coll = client.get_or_create_collection("pynucleus_documents")
            logger.success("ChromaDB collection initialized successfully")
            
        except Exception as e:
            logger.warning(f"ChromaDB collection initialization failed: {e}")
            _coll = None
    
    return _coll

def _get_vector_store():
    """Get or create ChromaVectorStore instance."""
    global _store
    if _store is None:
        _store = ChromaVectorStore()
    return _store

def _retrieve(q: str, k: int | None = None):
    """Retrieve documents and sources using ChromaVectorStore."""
    k = k or settings.RETRIEVE_TOP_K
    
    try:
        # Use ChromaVectorStore for consistent retrieval
        store = _get_vector_store()
        results = store.search(q, top_k=k)
        
        # Extract documents and sources from search results
        documents = []
        sources = []
        
        for result in results:
            documents.append(result.get('text', ''))
            sources.append(result.get('source', 'unknown'))
            
        return documents, sources
        
    except Exception as e:
        logger.warning(f"ChromaVectorStore retrieval failed, falling back to direct collection query: {e}")
        
        # Fallback to direct collection query
        coll = _initialize_collection()
        if coll is None:
            logger.warning("ChromaDB not available, returning empty results")
            return [], []
        
        try:
            res = coll.query(query_texts=[q], n_results=k, include=['documents', 'metadatas'])
            documents = res["documents"][0]
            metadatas = res["metadatas"][0]
            sources = [meta.get('source', f"doc_{i}") if meta else f"doc_{i}" for i, meta in enumerate(metadatas)]
            return documents, sources
        except Exception as e:
            logger.warning(f"Direct collection retrieval failed: {e}")
            return [], []

def retrieve(q: str, k: int | None = None):
    """Retrieve documents using ChromaDB collection (legacy interface)."""
    documents, _ = _retrieve(q, k)
    return documents

def ask(question: str):
    """Ask a question using RAG pipeline with enhanced prompting and citations."""
    # Use ChromaVectorStore for optimized retrieval
    store = _get_vector_store()
    docs, sources = store.search(question, top_k=5)  # Balanced retrieval
    
    # Smart context management: estimate tokens and balance input/output
    target_input_tokens = 1000  # Reserve ~1000 tokens for response generation
    chars_per_token = 4  # Rough estimate: 4 chars per token
    max_context_chars = target_input_tokens * chars_per_token
    
    # Calculate per-document allocation
    if docs:
        chars_per_doc = min(400, max_context_chars // len(docs))  # Max 400 chars per doc
        chars_per_doc = max(200, chars_per_doc)  # But at least 200 chars
    else:
        chars_per_doc = 400
    
    # Process documents with intelligent truncation
    processed_docs = []
    for doc in docs:
        if len(doc) <= chars_per_doc:
            processed_docs.append(doc)
        else:
            # Smart truncation: try to preserve complete sentences
            truncated = doc[:chars_per_doc]
            
            # Find the best cut point (sentence ending)
            for punct in ['. ', '.\n', '! ', '? ']:
                last_punct = truncated.rfind(punct)
                if last_punct > chars_per_doc * 0.6:  # If we keep at least 60% of content
                    truncated = truncated[:last_punct + 1]
                    break
            else:
                # If no good sentence boundary found, add ellipsis
                truncated = truncated.rstrip() + "..."
            
            processed_docs.append(truncated)
    
    # Build optimized context
    ctx = "\n\n".join(processed_docs)
    
    # Efficient prompt design
    prompt = (
        f"You are a chemical process engineer. Using the context below, answer: {question}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Provide a detailed answer with citations [1], [2], etc:"
    )
    
    # Generate with appropriate token allocation
    max_response_tokens = min(400, settings.MAX_TOKENS)  # Cap at 400 tokens for response
    answer = generate(prompt, max_tokens=max_response_tokens)
    
    # Add reference section
    if sources:
        refs = "\n\nReferences:\n" + "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sources))
        answer += refs
    
    return {"answer": answer.strip(), "sources": sources} 