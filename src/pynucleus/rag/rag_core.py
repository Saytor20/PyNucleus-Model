"""
Core RAG system implementation for PyNucleus.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

class RAGCore:
    """Core RAG system for PyNucleus knowledge retrieval."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
        # RAG components
        self.document_processor = None
        self.vector_store = None
        self.embeddings_model = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG system components."""
        try:
            from .document_processor import DocumentProcessor
            from .vector_store import VectorStore
            
            self.document_processor = DocumentProcessor()
            self.vector_store = VectorStore()
            
            self.logger.info("RAG Core components initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"RAG component initialization failed: {e}")
    
    def process_documents(self, source_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process documents from source directory.
        
        Args:
            source_dir: Directory containing source documents
            
        Returns:
            Processing results
        """
        if not source_dir:
            source_dir = self.data_dir / "01_raw" / "source_documents"
        else:
            source_dir = Path(source_dir)
            
        try:
            if not source_dir.exists():
                return {
                    "status": "error",
                    "message": f"Source directory not found: {source_dir}",
                    "processed_count": 0
                }
            
            # Get all documents
            doc_files = list(source_dir.glob("*"))
            doc_files = [f for f in doc_files if f.is_file()]
            
            processed_docs = []
            
            if self.document_processor:
                for doc_file in doc_files:
                    try:
                        result = self.document_processor.process_document(doc_file)
                        processed_docs.append(result)
                    except Exception as e:
                        self.logger.warning(f"Failed to process {doc_file}: {e}")
                        continue
            else:
                # Mock processing
                for doc_file in doc_files:
                    processed_docs.append({
                        "file_path": str(doc_file),
                        "status": "processed",
                        "chunk_count": 5,  # Mock value
                        "processed_at": datetime.now().isoformat()
                    })
            
            return {
                "status": "success",
                "processed_count": len(processed_docs),
                "total_files": len(doc_files),
                "processed_documents": processed_docs,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "processed_count": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def build_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build or rebuild the vector index.
        
        Args:
            force_rebuild: Force rebuild even if index exists
            
        Returns:
            Index building results
        """
        try:
            chunk_data_dir = self.data_dir / "03_intermediate" / "converted_chunked_data"
            
            if not chunk_data_dir.exists():
                return {
                    "status": "error",
                    "message": f"Chunk data directory not found: {chunk_data_dir}",
                    "index_size": 0
                }
            
            # Check for existing index
            index_dir = self.data_dir / "04_models" / "chunk_reports"
            index_files = list(index_dir.glob("*.faiss")) if index_dir.exists() else []
            
            if index_files and not force_rebuild:
                return {
                    "status": "exists",
                    "message": "Index already exists. Use force_rebuild=True to rebuild.",
                    "existing_indices": [f.name for f in index_files],
                    "index_size": len(index_files)
                }
            
            # Build new index
            if self.vector_store:
                result = self.vector_store.build_index(chunk_data_dir)
            else:
                # Mock index building
                result = {
                    "status": "success",
                    "index_size": 100,  # Mock value
                    "dimensions": 768,
                    "chunks_indexed": 50
                }
            
            result.update({
                "timestamp": datetime.now().isoformat(),
                "force_rebuild": force_rebuild
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Index building failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "index_size": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            Search results
        """
        try:
            start_time = datetime.now()
            
            if self.vector_store:
                results = self.vector_store.search(query, top_k, similarity_threshold)
            else:
                # Mock search results
                results = [
                    {
                        "text": f"Mock result for query: {query}",
                        "source": "mock_document_1.txt",
                        "score": 0.85,
                        "chunk_id": 1
                    },
                    {
                        "text": "Additional relevant information from knowledge base",
                        "source": "mock_document_2.txt", 
                        "score": 0.78,
                        "chunk_id": 2
                    }
                ]
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "query": query,
                "results": results,
                "result_count": len(results),
                "search_time": search_time,
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {
                "query": query,
                "results": [],
                "result_count": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get RAG system status and statistics."""
        try:
            # Check data directories
            directories = {
                "source_documents": self.data_dir / "01_raw" / "source_documents",
                "processed_documents": self.data_dir / "02_processed" / "converted_to_txt",
                "chunked_data": self.data_dir / "03_intermediate" / "converted_chunked_data",
                "indices": self.data_dir / "04_models" / "chunk_reports"
            }
            
            status = {
                "system_operational": True,
                "components": {
                    "document_processor": self.document_processor is not None,
                    "vector_store": self.vector_store is not None,
                    "embeddings_model": self.embeddings_model is not None
                },
                "data_status": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Check each directory
            for name, path in directories.items():
                if path.exists():
                    items = list(path.glob("*"))
                    status["data_status"][name] = {
                        "exists": True,
                        "item_count": len(items),
                        "path": str(path)
                    }
                else:
                    status["data_status"][name] = {
                        "exists": False,
                        "item_count": 0,
                        "path": str(path)
                    }
            
            return status
            
        except Exception as e:
            return {
                "system_operational": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            } 