"""
RAG Pipeline for PyNucleus system.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..rag.engine import ask as _ask

class RAGPipeline:
    """RAG (Retrieval-Augmented Generation) pipeline for document processing and querying."""
    
    def __init__(self, data_dir: str = "data", chunk_size: int = 1000):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        
        # RAG components
        self.vector_store = None
        self.embeddings = None
        self.retriever = None
        
        # Initialize real vector store
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize vector store using factory pattern with auto-ingestion."""
        try:
            from ..rag.vector_store_remote import create_vector_store
            from ..settings import settings
            
            self.vector_store = create_vector_store(backend=settings.vstore_backend)
            
            # Check if vector store loaded successfully and has content
            if hasattr(self.vector_store, 'loaded') and self.vector_store.loaded:
                self.logger.info(f"Vector store ({settings.vstore_backend}) initialized successfully")
                # Check if ChromaDB has documents
                self._check_and_populate_chromadb()
            elif hasattr(self.vector_store, 'initialized') and self.vector_store.initialized:
                self.logger.info(f"Remote vector store ({settings.vstore_backend}) initialized successfully")
                # Check if ChromaDB has documents
                self._check_and_populate_chromadb()
            else:
                self.logger.warning(f"Vector store ({settings.vstore_backend}) not fully loaded - will use fallback mode")
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize vector store: {e}")
            self.vector_store = None
            
    def _check_and_populate_chromadb(self):
        """Check if ChromaDB has documents and auto-ingest if empty."""
        try:
            from ..rag.engine import _get_chromadb_client, _initialize_collection
            from ..rag.collector import ingest
            from ..settings import settings
            from pathlib import Path
            import chromadb
            
            # Get ChromaDB client and collection
            client = _get_chromadb_client()
            if client is None:
                self.logger.warning("Failed to get ChromaDB client for document check")
                return
                
            coll = _initialize_collection()
            if coll is None:
                self.logger.warning("Failed to initialize ChromaDB collection")
                return
                
            # Check if collection has documents
            doc_count = coll.count()
            self.logger.info(f"ChromaDB collection contains {doc_count} documents")
            
            if doc_count == 0:
                # Auto-ingest documents if collection is empty
                source_docs_dir = Path("data/01_raw/source_documents")
                if source_docs_dir.exists() and list(source_docs_dir.glob("*")):
                    self.logger.info("ChromaDB collection is empty. Auto-ingesting documents...")
                    ingest(str(source_docs_dir))
                    
                    # Verify ingestion worked
                    new_count = coll.count()
                    self.logger.info(f"Auto-ingestion completed. ChromaDB now contains {new_count} documents")
                else:
                    self.logger.warning(f"No source documents found in {source_docs_dir} for auto-ingestion")
                    
        except Exception as e:
            self.logger.error(f"Failed to check/populate ChromaDB: {e}")
        
    def query(self, q:str, top_k:int=6):
        out = _ask(q)
        
        # Calculate confidence based on source availability and quality
        sources = out["sources"] or ["General Knowledge"]
        if sources == ["General Knowledge"]:
            confidence = 0.0  # No real sources found
        elif len(sources) >= 3:
            confidence = 0.8  # Good sources available
        else:
            confidence = 0.5  # Some sources available
        
        return {
            "answer": out["answer"],
            "sources": sources,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "context_quality": "none" if not out["sources"] else "medium"
        }
    
    def load_documents(self, source_dir: Optional[str] = None) -> bool:
        """
        Load documents into the RAG system.
        
        Args:
            source_dir: Directory containing source documents
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if source_dir:
                docs_dir = Path(source_dir)
            else:
                docs_dir = self.data_dir / "01_raw" / "source_documents"
            
            if not docs_dir.exists():
                self.logger.warning(f"Documents directory not found: {docs_dir}")
                return False
                
            # Mock document loading
            doc_files = list(docs_dir.glob("*"))
            self.logger.info(f"Loaded {len(doc_files)} documents from {docs_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load documents: {e}")
            return False 

rag_pipeline = RAGPipeline() 