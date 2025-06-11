"""
RAG Pipeline Module

Handles RAG (Retrieval-Augmented Generation) pipeline operations including:
- Document processing and chunking
- Wikipedia article scraping
- FAISS vector store building and evaluation
- Query testing and results collection
"""

import sys
import os
import importlib
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath('.'))

class RAGPipeline:
    """Main RAG Pipeline class for managing document processing and retrieval operations."""
    
    def __init__(self, results_dir="data/05_output/results"):
        """Initialize RAG Pipeline with results directory."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results_data = []
        self.manager = None
        self.documents = None
        
        # Import and reload modules
        self._setup_imports()
    
    def _setup_imports(self):
        """Setup and reload RAG modules."""
        print("🔧 Setting up RAG imports...")
        
        # Clear cached imports
        modules_to_reload = [
            'pynucleus.rag.config',
            'pynucleus.rag.document_processor',
            'pynucleus.rag.wiki_scraper',
            'pynucleus.rag.data_chunking',
            'pynucleus.rag.vector_store'
        ]
        
        # Remove old modules from sys.modules
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('core_modules'):
                del sys.modules[module_name]
        
        # Reload modules
        for module_name in modules_to_reload:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
        
        # Import modules
        from pynucleus.rag import config
        from pynucleus.rag.document_processor import process_documents
        from pynucleus.rag.wiki_scraper import scrape_wikipedia_articles
        from pynucleus.rag.data_chunking import load_and_chunk_files, save_chunked_data
        from pynucleus.rag.vector_store import FAISSDBManager, _load_docs
        
        # Store imports as instance variables
        self.config = config
        self.process_documents = process_documents
        self.scrape_wikipedia_articles = scrape_wikipedia_articles
        self.load_and_chunk_files = load_and_chunk_files
        self.save_chunked_data = save_chunked_data
        self.FAISSDBManager = FAISSDBManager
        self._load_docs = _load_docs
        
        print("✅ RAG imports ready!")
    
    def run_pipeline(self):
        """Execute the complete RAG pipeline."""
        print("📚 Starting RAG Pipeline...")
        
        # Step 1: Process source documents
        print("Step 1: Processing source documents...")
        self.process_documents()

        # Step 2: Scrape Wikipedia articles
        print("\nStep 2: Scraping Wikipedia articles...")
        self.scrape_wikipedia_articles()

        # Step 3: Process and chunk documents
        print("\nStep 3: Processing and chunking documents...")
        chunked_docs = self.load_and_chunk_files()
        self.save_chunked_data(chunked_docs)

        # Step 4: Build FAISS vector store
        print("\nStep 4: Building FAISS vector store...")   
        self.manager = self.FAISSDBManager()
        self.documents = self._load_docs(str(self.config.FULL_JSON_PATH), self.manager.log)
        self.manager.build(self.documents)
        self.manager.evaluate(self.config.GROUND_TRUTH_DATA)
        
        print(f"✅ RAG Pipeline completed! FAISS log → {self.manager.log_path}")
        return self.manager, self.documents
    
    def test_queries(self, test_queries=None):
        """Test RAG queries and collect results."""
        if not self.manager or not self.documents:
            print("❌ Pipeline not initialized. Run run_pipeline() first.")
            return
        
        print("🔍 Testing RAG queries...")
        
        if test_queries is None:
            test_queries = [
                "What are the key challenges in implementing modular chemical plants?",
                "How does supply chain management affect modular design?",
                "What are the economic benefits of modular construction?",
                "How does software architecture relate to modular design?",
                "What are the environmental impacts of modular manufacturing?"
            ]
        
        self.results_data.clear()
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            results = self.manager.search(query, k=3)
            
            for i, (doc, score) in enumerate(results, 1):
                print(f"   {i}. Score: {score:.4f} | Source: {doc.metadata.get('source', 'Unknown')}")
                
                self.results_data.append({
                    'query': query,
                    'result_rank': i,
                    'score': score,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'content_preview': doc.page_content[:200],
                    'full_content': doc.page_content,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        print(f"✅ Query testing completed! {len(self.results_data)} results collected.")
        return self.results_data
    
    def get_statistics(self):
        """Get RAG pipeline statistics."""
        if not self.documents:
            return {}
        
        stats = {
            'total_chunks': len(self.documents),
            'avg_chunk_size': sum(len(doc.page_content) for doc in self.documents) / len(self.documents),
            'num_sources': len(set(doc.metadata.get('source') for doc in self.documents)),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"📊 RAG Statistics:")
        print(f"   • Total Chunks: {stats['total_chunks']}")
        print(f"   • Average Chunk Size: {stats['avg_chunk_size']:.1f} characters")
        print(f"   • Number of Sources: {stats['num_sources']}")
        
        return stats
    
    def get_results(self):
        """Get collected results data."""
        return self.results_data
    
    def clear_results(self):
        """Clear collected results."""
        self.results_data.clear()
        print("🗑️ RAG results cleared.") 