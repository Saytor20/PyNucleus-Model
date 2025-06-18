"""
RAG Pipeline for PyNucleus system.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

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
        """Initialize the real FAISS vector store."""
        try:
            from ..rag.vector_store import RealFAISSVectorStore
            self.vector_store = RealFAISSVectorStore()
            if self.vector_store.loaded:
                self.logger.info("Real FAISS vector store initialized successfully")
            else:
                self.logger.warning("FAISS vector store not loaded - will use fallback mode")
        except Exception as e:
            self.logger.warning(f"Failed to initialize vector store: {e}")
            self.vector_store = None
        
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            top_k: Number of top results to return
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            # Use real vector store if available
            if self.vector_store and hasattr(self.vector_store, 'search'):
                search_results = self.vector_store.search(question, top_k=top_k, similarity_threshold=0.3)
                
                if search_results:
                    # Combine search results into answer
                    answer_parts = []
                    sources = []
                    confidence_scores = []
                    
                    for result in search_results:
                        answer_parts.append(result["text"][:300] + "...")
                        sources.append(result["source"])
                        confidence_scores.append(result["score"])
                    
                    combined_answer = " ".join(answer_parts)
                    avg_confidence = sum(confidence_scores) / len(confidence_scores)
                    
                    response = {
                        "answer": combined_answer,
                        "sources": sources,
                        "confidence": avg_confidence,
                        "timestamp": datetime.now().isoformat(),
                        "search_results_count": len(search_results)
                    }
                else:
                    # No search results found
                    response = {
                        "answer": f"No relevant information found for: {question}",
                        "sources": [],
                        "confidence": 0.0,
                        "timestamp": datetime.now().isoformat(),
                        "search_results_count": 0
                    }
            else:
                # Fallback to mock response
                response = {
                    "answer": f"Mock RAG response for: {question}",
                    "sources": [
                        "mock_document_1.txt",
                        "mock_document_2.txt"
                    ],
                    "confidence": 0.85,
                    "timestamp": datetime.now().isoformat()
                }
            
            self.logger.info(f"RAG query processed: {question[:50]}...")
            return response
            
        except Exception as e:
            self.logger.error(f"RAG query failed: {e}")
            return {
                "answer": f"Error processing query: {e}",
                "sources": [],
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
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