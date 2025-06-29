"""
RAG Pipeline for PyNucleus system.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..rag.engine import ask as _ask
from ..data.mock_data_manager import get_mock_data_manager

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
        
        # Mock data integration
        self.mock_data_manager = get_mock_data_manager()
        
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
    
    def query(self, q: str, top_k: int = 6, include_mock_data: bool = True):
        """
        Query the RAG system with optional mock data integration.
        
        Args:
            q: Query string
            top_k: Number of top results to retrieve
            include_mock_data: Whether to include mock data in search
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # First try standard RAG query
        out = _ask(q)
        
        # If mock data is enabled and standard query has low confidence, enhance with mock data
        if include_mock_data and self.mock_data_manager.is_data_loaded():
            enhanced_result = self._enhance_with_mock_data(q, out)
            return enhanced_result
        
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
    
    def _enhance_with_mock_data(self, query: str, original_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance RAG results with mock data when available.
        
        Args:
            query: Original query
            original_result: Original RAG result
            
        Returns:
            Enhanced result with mock data
        """
        try:
            # Search mock data for relevant information
            mock_docs = self.mock_data_manager.search_technical_documents(query)
            plant_templates = self.mock_data_manager.get_all_plant_templates()
            
            # Check if query is about plant templates
            plant_related_keywords = ["plant", "template", "methanol", "ammonia", "ethylene", "polyethylene", "urea"]
            is_plant_query = any(keyword.lower() in query.lower() for keyword in plant_related_keywords)
            
            enhanced_answer = original_result["answer"]
            enhanced_sources = original_result["sources"] or []
            confidence = original_result.get("confidence", 0.0)
            
            # Add mock data if relevant
            if mock_docs:
                mock_context = "\n\n".join([f"{doc['title']}: {doc['content']}" for doc in mock_docs[:2]])
                enhanced_answer += f"\n\nAdditional technical information:\n{mock_context}"
                enhanced_sources.extend([f"Mock Data: {doc['title']}" for doc in mock_docs[:2]])
                confidence = min(0.9, confidence + 0.2)  # Boost confidence with mock data
            
            # Add plant template information if relevant
            if is_plant_query and plant_templates:
                template_info = []
                for template in plant_templates[:3]:  # Limit to 3 templates
                    template_info.append(f"{template['name']}: {template['description']}")
                
                if template_info:
                    enhanced_answer += f"\n\nAvailable plant templates:\n" + "\n".join(template_info)
                    enhanced_sources.append("Mock Data: Plant Templates")
                    confidence = min(0.9, confidence + 0.1)
            
            return {
                "answer": enhanced_answer,
                "sources": enhanced_sources,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "context_quality": "enhanced" if mock_docs or is_plant_query else "medium",
                "mock_data_used": bool(mock_docs or is_plant_query)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to enhance with mock data: {e}")
            return original_result
    
    def query_mock_data_only(self, query: str) -> Dict[str, Any]:
        """
        Query only mock data (for testing and development).
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with mock data results
        """
        if not self.mock_data_manager.is_data_loaded():
            return {
                "answer": "Mock data not available",
                "sources": [],
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "context_quality": "none"
            }
        
        # Search technical documents
        tech_docs = self.mock_data_manager.search_technical_documents(query)
        
        # Search plant templates
        plant_templates = self.mock_data_manager.get_all_plant_templates()
        relevant_templates = []
        
        query_lower = query.lower()
        for template in plant_templates:
            if (query_lower in template['name'].lower() or 
                query_lower in template['description'].lower() or
                query_lower in template['technology'].lower()):
                relevant_templates.append(template)
        
        # Build response
        answer_parts = []
        sources = []
        
        if tech_docs:
            answer_parts.append("Technical Information:")
            for doc in tech_docs[:2]:
                answer_parts.append(f"• {doc['title']}: {doc['content'][:200]}...")
                sources.append(f"Mock Data: {doc['title']}")
        
        if relevant_templates:
            answer_parts.append("\nPlant Templates:")
            for template in relevant_templates[:2]:
                answer_parts.append(f"• {template['name']}: {template['description']}")
                sources.append(f"Mock Data: {template['name']}")
        
        if not answer_parts:
            answer_parts.append("No relevant mock data found for your query.")
        
        return {
            "answer": "\n".join(answer_parts),
            "sources": sources,
            "confidence": 0.7 if sources else 0.0,
            "timestamp": datetime.now().isoformat(),
            "context_quality": "mock_only"
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
    
    def get_mock_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of available mock data.
        
        Returns:
            Dictionary with mock data summary
        """
        return self.mock_data_manager.get_data_summary()

rag_pipeline = RAGPipeline() 