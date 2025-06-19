"""
DSPy retriever adapter for PyNucleus RAG integration.
"""

from typing import List, Dict, Any, Optional
import logging

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    
    # Mock DSPy components for graceful fallback
    class MockDSPy:
        class Retrieve:
            def __init__(self, k=5):
                self.k = k
    
    dspy = MockDSPy()


class PyNucleusRetriever(dspy.Retrieve):
    """DSPy retriever adapter for PyNucleus RAG pipeline."""
    
    def __init__(self, k: int = 5, rag_pipeline=None):
        """
        Initialize retriever adapter.
        
        Args:
            k: Number of passages to retrieve
            rag_pipeline: Existing PyNucleus RAG pipeline instance
        """
        super().__init__(k=k)
        self.rag_pipeline = rag_pipeline
        self.logger = logging.getLogger(__name__)
        
    def forward(self, query_or_queries: str, k: Optional[int] = None) -> List[str]:
        """
        Retrieve relevant passages for given query.
        
        Args:
            query_or_queries: Search query or list of queries
            k: Number of passages to retrieve (overrides default)
            
        Returns:
            List of retrieved passage strings
        """
        k = k or self.k
        
        try:
            if self.rag_pipeline is None:
                # Fallback to mock retrieval
                return self._mock_retrieve(query_or_queries, k)
            
            # Use existing RAG pipeline for retrieval
            if isinstance(query_or_queries, str):
                queries = [query_or_queries]
            else:
                queries = query_or_queries
            
            all_passages = []
            for query in queries:
                passages = self._retrieve_from_pipeline(query, k)
                all_passages.extend(passages)
            
            # Deduplicate and limit to k total passages
            seen = set()
            unique_passages = []
            for passage in all_passages:
                if passage not in seen and len(unique_passages) < k:
                    seen.add(passage)
                    unique_passages.append(passage)
            
            return unique_passages
            
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return self._mock_retrieve(query_or_queries, k)
    
    def _retrieve_from_pipeline(self, query: str, k: int) -> List[str]:
        """Retrieve passages using the RAG pipeline."""
        try:
            # Call the RAG pipeline's search method
            if hasattr(self.rag_pipeline, 'search_documents'):
                results = self.rag_pipeline.search_documents(query, top_k=k)
                
                # Extract text content from results
                passages = []
                for result in results:
                    if isinstance(result, dict):
                        # Extract text from different possible keys
                        text = (result.get('content') or 
                               result.get('text') or 
                               result.get('passage') or 
                               str(result))
                    else:
                        text = str(result)
                    
                    passages.append(text)
                
                return passages
                
            elif hasattr(self.rag_pipeline, 'query'):
                # Alternative interface
                results = self.rag_pipeline.query(query, num_results=k)
                return [str(result) for result in results]
            
            else:
                self.logger.warning("RAG pipeline doesn't have expected search methods")
                return self._mock_retrieve(query, k)
                
        except Exception as e:
            self.logger.error(f"Pipeline retrieval failed: {e}")
            return self._mock_retrieve(query, k)
    
    def _mock_retrieve(self, query: str, k: int) -> List[str]:
        """Fallback mock retrieval when pipeline is unavailable."""
        mock_passages = [
            f"Mock passage 1 related to: {query}",
            f"Mock passage 2 with chemical engineering context for: {query}",
            f"Mock passage 3 about PyNucleus simulation for: {query}",
            f"Mock passage 4 containing technical details on: {query}",
            f"Mock passage 5 with process analysis of: {query}"
        ]
        
        return mock_passages[:k]


class DocumentRetriever:
    """Wrapper for document-specific retrieval operations."""
    
    def __init__(self, rag_pipeline=None):
        self.rag_pipeline = rag_pipeline
        self.logger = logging.getLogger(__name__)
    
    def retrieve_for_report_analysis(self, question: str, report_type: str = None) -> List[str]:
        """Retrieve context specifically for report analysis."""
        try:
            if self.rag_pipeline:
                # Enhance query with report context
                enhanced_query = f"report analysis {report_type or ''} {question}".strip()
                retriever = PyNucleusRetriever(k=3, rag_pipeline=self.rag_pipeline)
                return retriever.forward(enhanced_query)
            else:
                return [f"No additional context available for report analysis: {question}"]
        except Exception as e:
            self.logger.error(f"Report retrieval failed: {e}")
            return [f"Retrieval error for report analysis: {question}"]
    
    def retrieve_for_simulation(self, question: str, case_name: str = None) -> List[str]:
        """Retrieve context for simulation queries."""
        try:
            if self.rag_pipeline:
                # Enhance query with simulation context
                enhanced_query = f"simulation {case_name or ''} {question}".strip()
                retriever = PyNucleusRetriever(k=3, rag_pipeline=self.rag_pipeline)
                return retriever.forward(enhanced_query)
            else:
                return [f"No additional context available for simulation: {question}"]
        except Exception as e:
            self.logger.error(f"Simulation retrieval failed: {e}")
            return [f"Retrieval error for simulation: {question}"] 