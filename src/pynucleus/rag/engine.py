import chromadb
from chromadb.config import Settings
from pathlib import Path
from ..settings import settings
from ..llm.model_loader import generate
from ..llm.prompting import build_prompt
from ..utils.logger import logger

# Centralized ChromaDB client management
_client = None
_coll = None

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

def retrieve(q: str, k: int | None = None):
    """Retrieve documents using ChromaDB collection."""
    k = k or settings.RETRIEVE_TOP_K
    
    # Lazy initialization of collection
    coll = _initialize_collection()
    if coll is None:
        logger.warning("ChromaDB not available, returning empty results")
        return []
    
    try:
        return coll.query(query_texts=[q], n_results=k)["documents"][0]
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}")
        return []

def ask(question: str):
    """Ask a question using RAG pipeline."""
    chunks = retrieve(question)
    # Tag chunks for inline citation markers [1] etc.
    tagged_ctx = [f"[{i+1}] {c}" for i,c in enumerate(chunks)]
    context = "\n\n".join(tagged_ctx) if chunks else ""
    prompt = build_prompt(context, question)
    answer = generate(prompt, max_tokens=50)
    
    if chunks:
        refs = "\n\nReferences:\n" + "\n".join(f"[{i+1}] {c[:60]}..." for i,c in enumerate(chunks))
        answer += refs
    
    return {"answer": answer.strip(), "sources": chunks or ["General Knowledge"]} 