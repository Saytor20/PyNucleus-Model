"""
RAG Pipeline Module

Handles RAG (Retrieval-Augmented Generation) pipeline operations including:
- Document processing and chunking
- Wikipedia article scraping
- FAISS vector store building and evaluation
- Query testing and results collection
- DWSIM simulation data integration
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
    
    def __init__(self, results_dir="data/05_output/results", enable_dwsim_integration=True):
        """Initialize RAG Pipeline with results directory."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results_data = []
        self.manager = None
        self.documents = None
        self.enable_dwsim_integration = enable_dwsim_integration
        
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
        
        # Import DWSIM integration if enabled
        if self.enable_dwsim_integration:
            try:
                from pynucleus.integration.dwsim_data_integrator import DWSIMDataIntegrator
                self.DWSIMDataIntegrator = DWSIMDataIntegrator
            except ImportError:
                print("⚠️ DWSIM integration not available - continuing without it")
                self.enable_dwsim_integration = False
                self.DWSIMDataIntegrator = None
        
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

        # Step 4: DWSIM Data Integration (if enabled and available)
        if self.enable_dwsim_integration and self.DWSIMDataIntegrator:
            print("\nStep 4: Integrating DWSIM simulation data...")
            self._integrate_dwsim_data()
        else:
            print("\nStep 4: Skipping DWSIM integration (not enabled or not available)")

        # Step 5: Build FAISS vector store
        print(f"\nStep {'5' if self.enable_dwsim_integration else '4'}: Building FAISS vector store...")   
        self.manager = self.FAISSDBManager()
        self.documents = self._load_docs(str(self.config.FULL_JSON_PATH), self.manager.log)
        self.manager.build(self.documents)
        self.manager.evaluate(self.config.GROUND_TRUTH_DATA)
        
        print(f"✅ RAG Pipeline completed! FAISS log → {self.manager.log_path}")
        return self.manager, self.documents
    
    def _integrate_dwsim_data(self):
        """Integrate DWSIM simulation data into the knowledge base."""
        try:
            # Initialize DWSIM data integrator
            integrator = self.DWSIMDataIntegrator()
            
            # Check if DWSIM results are available
            dwsim_results_file = Path("data/05_output/dwsim_simulation_results.csv")
            if not dwsim_results_file.exists():
                print("   ⚠️ No DWSIM simulation results found - skipping integration")
                return
            
            # Perform integration
            result = integrator.integrate_simulation_data()
            
            if result["success"]:
                print(f"   ✅ Integration successful: {result['simulation_chunks']} simulation chunks added")
                print(f"   📊 Total knowledge base: {result['total_chunks']} chunks")
                print(f"      ├── Documents: {result['document_chunks']}")
                print(f"      └── Simulations: {result['simulation_chunks']}")
                
                # Show integration summary
                summary = integrator.create_integration_summary()
                print(f"\n{summary}")
                
                # Show example queries
                print("\n🔍 Enhanced Query Capabilities:")
                examples = integrator.get_simulation_query_examples()[:3]
                for i, query in enumerate(examples, 1):
                    print(f"   {i}. {query}")
                    
            else:
                print(f"   ❌ Integration failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ DWSIM integration error: {str(e)}")
            # Continue without integration
    
    def test_queries(self):
        """Test RAG queries with enhanced support for simulation data."""
        print("🔍 Testing RAG queries...")
        
        # Standard queries
        standard_queries = [
            "What are the key challenges in implementing modular chemical plants?",
            "How does supply chain management affect modular design?",
            "What are the economic benefits of modular construction?",
            "How does software architecture relate to modular design?",
            "What are the environmental impacts of modular manufacturing?"
        ]
        
        # Enhanced queries that leverage simulation data (if available)
        simulation_queries = [
            "What are the performance metrics for the distillation simulation?",
            "How do the reactor conversion rates compare across different simulations?",
            "What operating conditions were used in the heat exchanger simulation?",
            "Which simulation showed the highest selectivity and why?",
            "How do the simulation results relate to modular plant design principles?"
        ]
        
        # Determine which queries to use based on integration status
        if self.enable_dwsim_integration and self._has_simulation_data():
            queries = standard_queries + simulation_queries[:2]  # Add 2 simulation queries
        else:
            queries = standard_queries
        
        all_results = []
        
        for query in queries:
            print(f"\n📝 Query: {query}")
            try:
                results = self.manager.search(query, k=3)
                for i, (doc, score) in enumerate(results, 1):
                    source_path = doc.metadata.get('source', 'Unknown source')
                    print(f"   {i}. Score: {score:.4f} | Source: {source_path}")
                    all_results.append({
                        'query': query,
                        'rank': i,
                        'score': score,
                        'source': source_path,
                        'content_preview': doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    })
            except Exception as e:
                print(f"   ❌ Query failed: {str(e)}")
        
        print(f"✅ Query testing completed! {len(all_results)} results collected.")
        return all_results
    
    def _has_simulation_data(self) -> bool:
        """Check if simulation data is available in the knowledge base."""
        try:
            stats_file = Path("data/03_intermediate/converted_chunked_data/chunked_data_stats.json")
            if not stats_file.exists():
                return False
            
            import json
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            return stats.get("integration_enabled", False) and stats.get("simulation_chunks", 0) > 0
        except:
            return False
    
    def get_statistics(self):
        """Get comprehensive statistics including simulation data integration."""
        try:
            stats_file = Path("data/03_intermediate/converted_chunked_data/chunked_data_stats.json")
            
            # Create default statistics file if it doesn't exist
            if not stats_file.exists():
                print("📄 Statistics file not found - initializing RAG data...")
                return self._initialize_rag_data()
            
            # Load existing statistics
            import json
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            # Check if data is empty and needs initialization
            if stats.get("total_chunks", 0) == 0 and stats.get("status") == "initialized":
                print("📄 No processed data found - running automatic RAG initialization...")
                return self._initialize_rag_data()
            
            # Add integration status
            if stats.get("integration_enabled", False):
                stats["integration_status"] = "DWSIM data integrated"
                stats["has_simulation_data"] = True
            else:
                stats["integration_status"] = "Documents only"
                stats["has_simulation_data"] = False
            
            return stats
            
        except Exception as e:
            print(f"⚠️ Error loading statistics: {str(e)}")
            print("🔧 Attempting to initialize RAG data...")
            return self._initialize_rag_data()

    def _initialize_rag_data(self):
        """Initialize RAG data by running basic pipeline steps."""
        try:
            print("🚀 Auto-initializing RAG pipeline data...")
            
            # Ensure directory exists
            stats_file = Path("data/03_intermediate/converted_chunked_data/chunked_data_stats.json")
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Check for source documents and process them
            source_docs_dir = Path("data/01_raw/source_documents")
            converted_docs_dir = Path("data/02_processed/converted_to_txt")
            
            total_chunks = 0
            sources = []
            
            # Process any available source documents
            if source_docs_dir.exists() and any(source_docs_dir.iterdir()):
                try:
                    print("📄 Processing source documents...")
                    self.process_documents()
                    sources.append("source_documents")
                except Exception as e:
                    print(f"⚠️ Source document processing failed: {e}")
            
            # Try to scrape Wikipedia articles as fallback
            try:
                print("🌐 Scraping Wikipedia articles for base knowledge...")
                self.scrape_wikipedia_articles()
                
                # Check if articles were scraped
                web_sources_dir = Path("data/01_raw/web_sources")
                if web_sources_dir.exists() and any(web_sources_dir.iterdir()):
                    sources.append("wikipedia_articles")
            except Exception as e:
                print(f"⚠️ Wikipedia scraping failed: {e}")
            
            # Process and chunk available documents
            try:
                print("🔪 Processing and chunking documents...")
                chunked_docs = self.load_and_chunk_files()
                if chunked_docs:
                    self.save_chunked_data(chunked_docs)
                    total_chunks = len(chunked_docs)
                    print(f"✅ Created {total_chunks} document chunks")
            except Exception as e:
                print(f"⚠️ Document chunking failed: {e}")
            
            # Create comprehensive statistics
            stats = {
                "total_chunks": total_chunks,
                "document_chunks": total_chunks,
                "simulation_chunks": 0,
                "sources": sources,
                "avg_chunk_size": 500,  # Reasonable default
                "integration_enabled": self.enable_dwsim_integration,
                "has_simulation_data": False,
                "integration_status": "Documents only" if total_chunks > 0 else "No data available",
                "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "status": "auto_initialized" if total_chunks > 0 else "empty"
            }
            
            # Save statistics file
            import json
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            
            if total_chunks > 0:
                print(f"✅ RAG data initialized successfully with {total_chunks} chunks")
                
                # Try to build FAISS vector store
                try:
                    print("🔍 Building FAISS vector store...")
                    self.manager = self.FAISSDBManager()
                    self.documents = self._load_docs(str(self.config.FULL_JSON_PATH), self.manager.log)
                    if self.documents:
                        self.manager.build(self.documents)
                        print("✅ FAISS vector store built successfully")
                except Exception as e:
                    print(f"⚠️ FAISS vector store creation failed: {e}")
            else:
                print("⚠️ No data sources available for RAG initialization")
                print("💡 Please add documents to data/01_raw/source_documents/ for processing")
            
            return stats
            
        except Exception as e:
            print(f"❌ RAG initialization failed: {str(e)}")
            # Return minimal fallback statistics
            fallback_stats = {
                "total_chunks": 0,
                "document_chunks": 0,
                "simulation_chunks": 0,
                "sources": [],
                "avg_chunk_size": 0,
                "integration_enabled": False,
                "has_simulation_data": False,
                "integration_status": "Initialization failed",
                "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "status": "error",
                "error": str(e)
            }
            
            # Still try to save the fallback file
            try:
                stats_file = Path("data/03_intermediate/converted_chunked_data/chunked_data_stats.json")
                stats_file.parent.mkdir(parents=True, exist_ok=True)
                import json
                with open(stats_file, 'w') as f:
                    json.dump(fallback_stats, f, indent=4)
            except:
                pass
                
            return fallback_stats
    
    def get_results(self):
        """Get collected results data."""
        return self.results_data
    
    def clear_results(self):
        """Clear previous RAG results."""
        self.results_data.clear()
        print("🗑️ RAG results cleared.")
    
    def print_status(self):
        """Print detailed pipeline status including integration information."""
        print("\n📊 RAG Pipeline Status:")
        print("=" * 50)
        
        try:
            stats = self.get_statistics()
            
            if "error" in stats and stats.get('total_chunks', 0) == 0:
                print(f"⚠️ Warning: {stats['error']}")
                print("📊 Using fallback statistics...")
            
            print(f"📚 Total Chunks: {stats.get('total_chunks', 0):,}")
            
            if stats.get("has_simulation_data", False):
                print(f"📄 Document Chunks: {stats.get('document_chunks', 0):,}")
                print(f"🔬 Simulation Chunks: {stats.get('simulation_chunks', 0):,}")
                print(f"🔗 Integration Status: {stats.get('integration_status', 'Unknown')}")
            else:
                print(f"📄 Document Sources: {len(stats.get('sources', []))}")
                print(f"🔗 Integration Status: {stats.get('integration_status', 'Documents only')}")
            
            print(f"📏 Average Chunk Size: {stats.get('avg_chunk_size', 0):.1f} characters")
            
            if hasattr(self, 'manager') and self.manager:
                print(f"🔍 Vector Store: Ready")
                print(f"📂 FAISS Index: Available")
            else:
                print(f"🔍 Vector Store: Not built")
                
        except Exception as e:
            print(f"❌ Error getting pipeline status: {str(e)}")
            print("📊 Pipeline status unavailable")
        
        print("=" * 50) 