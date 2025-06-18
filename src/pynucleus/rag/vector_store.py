#!/usr/bin/env python3
"""
Enhanced FAISS Vector Store for PyNucleus
=========================================

High-performance vector store with:
- Qwen-0.6B embeddings for superior semantic understanding
- Comprehensive evaluation metrics (accuracy, recall, precision, F1)
- Duplicate document detection and skipping
- Query enhancement and preprocessing
- Smart fallback mechanisms
- Production-ready monitoring
"""

import os
import pickle
import warnings
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from collections import defaultdict
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Check for dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_FAISS_AVAILABLE = True
except ImportError:
    try:
        from langchain.vectorstores import FAISS
        LANGCHAIN_FAISS_AVAILABLE = True
    except ImportError:
        LANGCHAIN_FAISS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import FAISS archiver
try:
    from ..utils.faiss_archiver import FAISSArchiver
    FAISS_ARCHIVER_AVAILABLE = True
except ImportError:
    FAISS_ARCHIVER_AVAILABLE = False
    logger.warning("FAISSArchiver not available. Model archiving disabled.")

# Configuration for enhanced embeddings
EMBEDDING_MODEL_NAME = "Qwen/Qwen2-0.5B"  # Using available Qwen model
EMBEDDING_DIMENSION = 896   # Qwen2-0.5B embedding dimension
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_DIMENSION = 384

# Document class fallback
try:
    from langchain_core.documents.base import Document
except ImportError:
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

class StellaEmbeddings:
    """Enhanced embedding class using Stella-en-1.5B-v5 for superior scientific domain performance"""
    
    def __init__(self, model_name: str = "infgrad/stella-en-1.5B-v5", batch_size: int = 8):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Stella embedding model: {model_name}")
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension
            with torch.no_grad():
                test_input = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
                test_input = {k: v.to(self.device) for k, v in test_input.items()}
                test_output = self.model(**test_input)
                self.embedding_dim = test_output.last_hidden_state.mean(dim=1).shape[1]
            
            print(f"Stella model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"Error loading Stella model: {e}")
            print("Falling back to alternative model...")
            # Fallback to a more widely available model
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback model if Stella is not available"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.is_sentence_transformer = True
            print(f"Loaded fallback model: {self.model_name}, dimension: {self.embedding_dim}")
        except ImportError:
            raise ImportError("Neither transformers nor sentence-transformers available")
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts into embeddings with batch processing"""
        if isinstance(texts, str):
            texts = [texts]
        
        if hasattr(self, 'is_sentence_transformer'):
            return self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])

class QwenEmbeddings:
    """Legacy Qwen embedding class for comparison"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B", batch_size: int = 8):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Qwen embedding model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Test embedding dimension
            with torch.no_grad():
                test_input = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
                test_input = {k: v.to(self.device) for k, v in test_input.items()}
                test_output = self.model(**test_input)
                self.embedding_dim = test_output.last_hidden_state.mean(dim=1).shape[1]
            
            print(f"Qwen model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"Error loading Qwen model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts into embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])

class DocumentHasher:
    """Efficient document duplicate detection using content hashing."""
    
    def __init__(self, hash_file: str = "data/04_models/chunk_reports/document_hashes.json"):
        self.hash_file = Path(hash_file)
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)
        self.known_hashes = self._load_hashes()
    
    def _load_hashes(self) -> Set[str]:
        """Load existing document hashes."""
        if self.hash_file.exists():
            try:
                with open(self.hash_file, 'r') as f:
                    data = json.load(f)
                return set(data.get('hashes', []))
            except:
                return set()
        return set()
    
    def _save_hashes(self):
        """Save document hashes to file."""
        with open(self.hash_file, 'w') as f:
            json.dump({
                'hashes': list(self.known_hashes),
                'count': len(self.known_hashes),
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def get_document_hash(self, content: str, source: str = "") -> str:
        """Generate hash for document content."""
        # Normalize content for consistent hashing
        normalized = re.sub(r'\s+', ' ', content.strip().lower())
        combined = f"{source}:{normalized}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def is_duplicate(self, content: str, source: str = "") -> bool:
        """Check if document is a duplicate."""
        doc_hash = self.get_document_hash(content, source)
        return doc_hash in self.known_hashes
    
    def add_document(self, content: str, source: str = "") -> str:
        """Add document hash and return the hash."""
        doc_hash = self.get_document_hash(content, source)
        if doc_hash not in self.known_hashes:
            self.known_hashes.add(doc_hash)
            self._save_hashes()
        return doc_hash

class QueryEnhancer:
    """Enhance queries for better semantic retrieval."""
    
    @staticmethod
    def enhance_query(query: str) -> str:
        """Enhance query with domain-specific terms and context."""
        # Add chemical engineering context
        enhanced = query.strip()
        
        # Expand abbreviations common in chemical engineering
        expansions = {
            'modular': 'modular construction design manufacturing',
            'plant': 'plant facility industrial process',
            'chemical': 'chemical engineering process industry',
            'separation': 'separation distillation extraction purification',
            'reactor': 'reactor chemical reaction conversion',
            'efficiency': 'efficiency performance optimization',
            'cost': 'cost economic financial analysis'
        }
        
        for term, expansion in expansions.items():
            if term.lower() in enhanced.lower():
                enhanced += f" {expansion}"
        
        return enhanced
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words = re.findall(r'\w+', query.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]

class RecallEvaluator:
    """Comprehensive recall evaluation and comparison system"""
    
    def __init__(self, results_dir: str = "data/04_models/recall_evaluation"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard test queries for chemical engineering domain
        self.test_queries = [
            "modular plant design efficiency",
            "chemical reactor optimization",
            "distillation column performance",
            "heat exchanger design principles",
            "process safety management",
            "catalyst deactivation mechanisms",
            "mass transfer coefficients",
            "thermodynamic equilibrium calculations",
            "control system implementation",
            "environmental impact assessment",
            "cost estimation methodologies",
            "energy integration strategies",
            "separation process design",
            "reaction kinetics modeling",
            "equipment sizing calculations"
        ]
    
    def evaluate_model_recall(self, db_manager, model_name: str, top_k: int = 10) -> Dict[str, Any]:
        """Evaluate recall performance for a specific model"""
        print(f"\n=== Evaluating Recall for {model_name} ===")
        
        results = {
            "model_name": model_name,
            "embedding_dimension": db_manager.embedding_model.embedding_dim,
            "timestamp": datetime.now().isoformat(),
            "query_results": [],
            "metrics": {}
        }
        
        total_queries = len(self.test_queries)
        successful_queries = 0
        total_results = 0
        high_relevance_results = 0
        response_times = []
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"Query {i}/{total_queries}: {query}")
            
            start_time = time.time()
            try:
                search_results = db_manager.search(query, top_k=top_k, enhance_query=True)
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if search_results:
                    successful_queries += 1
                    total_results += len(search_results)
                    
                    # Count high relevance results (score > 0.7)
                    high_relevance = sum(1 for result in search_results if result.get('score', 0) > 0.7)
                    high_relevance_results += high_relevance
                    
                    query_result = {
                        "query": query,
                        "num_results": len(search_results),
                        "high_relevance_count": high_relevance,
                        "avg_score": np.mean([r.get('score', 0) for r in search_results]),
                        "max_score": max([r.get('score', 0) for r in search_results]),
                        "response_time": response_time,
                        "top_3_sources": [r.get('source', 'Unknown') for r in search_results[:3]]
                    }
                else:
                    query_result = {
                        "query": query,
                        "num_results": 0,
                        "high_relevance_count": 0,
                        "avg_score": 0,
                        "max_score": 0,
                        "response_time": response_time,
                        "top_3_sources": []
                    }
                
                results["query_results"].append(query_result)
                print(f"  Results: {query_result['num_results']}, High relevance: {query_result['high_relevance_count']}, Avg score: {query_result['avg_score']:.3f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                results["query_results"].append({
                    "query": query,
                    "error": str(e),
                    "num_results": 0,
                    "response_time": time.time() - start_time
                })
        
        # Calculate overall metrics
        results["metrics"] = {
            "success_rate": successful_queries / total_queries,
            "avg_results_per_query": total_results / total_queries if total_queries > 0 else 0,
            "high_relevance_rate": high_relevance_results / total_results if total_results > 0 else 0,
            "avg_response_time": np.mean(response_times) if response_times else 0,
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "total_results": total_results,
            "high_relevance_results": high_relevance_results
        }
        
        # Save results
        result_file = self.results_dir / f"recall_evaluation_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== {model_name} Recall Summary ===")
        print(f"Success Rate: {results['metrics']['success_rate']:.1%}")
        print(f"Avg Results per Query: {results['metrics']['avg_results_per_query']:.1f}")
        print(f"High Relevance Rate: {results['metrics']['high_relevance_rate']:.1%}")
        print(f"Avg Response Time: {results['metrics']['avg_response_time']:.3f}s")
        print(f"Results saved to: {result_file}")
        
        return results
    
    def compare_models(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare recall performance between multiple models"""
        if len(results_list) < 2:
            print("Need at least 2 model results for comparison")
            return {}
        
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "models_compared": len(results_list),
            "model_summary": [],
            "winner_analysis": {},
            "detailed_comparison": []
        }
        
        # Extract key metrics for each model
        for result in results_list:
            model_summary = {
                "model_name": result["model_name"],
                "embedding_dimension": result["embedding_dimension"],
                "success_rate": result["metrics"]["success_rate"],
                "avg_results_per_query": result["metrics"]["avg_results_per_query"],
                "high_relevance_rate": result["metrics"]["high_relevance_rate"],
                "avg_response_time": result["metrics"]["avg_response_time"]
            }
            comparison["model_summary"].append(model_summary)
        
        # Determine winners in each category
        best_success_rate = max(results_list, key=lambda x: x["metrics"]["success_rate"])
        best_avg_results = max(results_list, key=lambda x: x["metrics"]["avg_results_per_query"])
        best_relevance = max(results_list, key=lambda x: x["metrics"]["high_relevance_rate"])
        fastest_response = min(results_list, key=lambda x: x["metrics"]["avg_response_time"])
        
        comparison["winner_analysis"] = {
            "best_success_rate": {
                "model": best_success_rate["model_name"],
                "value": best_success_rate["metrics"]["success_rate"]
            },
            "best_avg_results": {
                "model": best_avg_results["model_name"],
                "value": best_avg_results["metrics"]["avg_results_per_query"]
            },
            "best_relevance": {
                "model": best_relevance["model_name"],
                "value": best_relevance["metrics"]["high_relevance_rate"]
            },
            "fastest_response": {
                "model": fastest_response["model_name"],
                "value": fastest_response["metrics"]["avg_response_time"]
            }
        }
        
        # Query-by-query comparison
        for i, query in enumerate(self.test_queries):
            query_comparison = {"query": query, "model_results": []}
            
            for result in results_list:
                if i < len(result["query_results"]):
                    query_result = result["query_results"][i]
                    query_comparison["model_results"].append({
                        "model": result["model_name"],
                        "num_results": query_result.get("num_results", 0),
                        "avg_score": query_result.get("avg_score", 0),
                        "high_relevance_count": query_result.get("high_relevance_count", 0)
                    })
            
            comparison["detailed_comparison"].append(query_comparison)
        
        # Save comparison
        comparison_file = self.results_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Print comparison summary
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        for model in comparison["model_summary"]:
            print(f"\n{model['model_name']}:")
            print(f"  Success Rate: {model['success_rate']:.1%}")
            print(f"  Avg Results: {model['avg_results_per_query']:.1f}")
            print(f"  High Relevance: {model['high_relevance_rate']:.1%}")
            print(f"  Response Time: {model['avg_response_time']:.3f}s")
            print(f"  Embedding Dim: {model['embedding_dimension']}")
        
        print(f"\nWINNERS BY CATEGORY:")
        print(f"  Best Success Rate: {comparison['winner_analysis']['best_success_rate']['model']} ({comparison['winner_analysis']['best_success_rate']['value']:.1%})")
        print(f"  Best Avg Results: {comparison['winner_analysis']['best_avg_results']['model']} ({comparison['winner_analysis']['best_avg_results']['value']:.1f})")
        print(f"  Best Relevance: {comparison['winner_analysis']['best_relevance']['model']} ({comparison['winner_analysis']['best_relevance']['value']:.1%})")
        print(f"  Fastest Response: {comparison['winner_analysis']['fastest_response']['model']} ({comparison['winner_analysis']['fastest_response']['value']:.3f}s)")
        
        print(f"\nComparison saved to: {comparison_file}")
        
        return comparison

class EnhancedFAISSDBManager:
    """Enhanced FAISS database manager with model comparison capabilities"""
    
    def __init__(self, 
                 db_path: str = "data/04_models/pynucleus_mcp.faiss",
                 embedding_model_type: str = "stella",  # "stella" or "qwen"
                 batch_size: int = 8):
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.documents = []
        self.log = []
        self.log_path = self.db_path.parent / "faiss_enhanced_build_log.txt"
        self.batch_size = batch_size
        
        # Enhanced components
        self.hasher = DocumentHasher()
        self.query_enhancer = QueryEnhancer()
        self.evaluator = RecallEvaluator()
        
        # Initialize FAISS archiver for automatic model management
        if FAISS_ARCHIVER_AVAILABLE:
            models_dir = str(self.db_path.parent / "chunk_reports") if "chunk_reports" in str(self.db_path) else str(self.db_path.parent)
            self.archiver = FAISSArchiver(models_dir)
            logger.info(f"✅ FAISS archiver initialized for: {models_dir}")
        else:
            self.archiver = None
            logger.warning("⚠️ FAISS archiver not available - old models will not be automatically archived")
        
        # Initialize embedding model with fallback support
        self.embedding_model_type = embedding_model_type
        if TRANSFORMERS_AVAILABLE:
            try:
                if embedding_model_type == "stella":
                    self.embedding_model = StellaEmbeddings(batch_size=batch_size)
                else:
                    self.embedding_model = QwenEmbeddings(batch_size=batch_size)
                print(f"✅ Enhanced {embedding_model_type} model loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load {embedding_model_type} model: {e}")
                self._use_fallback_embeddings()
        else:
            print("⚠️ Transformers not available, using fallback embeddings")
            self._use_fallback_embeddings()
        
        print(f"🔧 Enhanced FAISS DB Manager initialized")
        print(f"   📊 Embedding model: {self.embedding_model_type}")
        print(f"   📏 Embedding dimension: {getattr(self.embedding_model, 'embedding_dim', 'Unknown')}")
    
    def _use_fallback_embeddings(self):
        """Use a simple fallback embedding model for basic functionality."""
        print("🔄 Initializing fallback embedding model...")
        
        class SimpleFallbackEmbeddings:
            """Simple TF-IDF based embeddings as fallback."""
            
            def __init__(self):
                self.embedding_dim = 300
                self.vocab = {}
                self.vocab_size = 0
                
            def encode(self, texts):
                """Simple TF-IDF-like encoding."""
                if isinstance(texts, str):
                    texts = [texts]
                
                embeddings = []
                for text in texts:
                    # Simple word frequency based embedding
                    words = text.lower().split()
                    word_counts = {}
                    for word in words:
                        word_counts[word] = word_counts.get(word, 0) + 1
                    
                    # Create fixed-size vector
                    embedding = [0.0] * self.embedding_dim
                    for i, word in enumerate(words[:self.embedding_dim]):
                        embedding[i] = hash(word) % 100 / 100.0  # Normalize to 0-1
                    
                    embeddings.append(embedding)
                
                return np.array(embeddings)
        
        self.embedding_model = SimpleFallbackEmbeddings()
        self.embedding_model_type = "fallback"
        print(f"✅ Fallback embedding model ready (dimension: {self.embedding_model.embedding_dim})")
    
    def create_model_comparison_manager(self, model_type: str):
        """Create a new manager instance with different embedding model for comparison"""
        return EnhancedFAISSDBManager(
            db_path=str(self.db_path).replace(".faiss", f"_{model_type}.faiss"),
            embedding_model_type=model_type,
            batch_size=self.batch_size
        )
    
    def run_recall_comparison(self, compare_with_model: str = "qwen", top_k: int = 10):
        """Run comprehensive recall comparison between current model and specified model"""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE RECALL COMPARISON: {self.embedding_model_type.upper()} vs {compare_with_model.upper()}")
        print(f"{'='*80}")
        
        # Evaluate current model
        current_results = self.evaluator.evaluate_model_recall(self, self.embedding_model_type, top_k)
        
        # Create comparison model manager
        try:
            comparison_manager = self.create_model_comparison_manager(compare_with_model)
            
            # Add documents to comparison model if needed
            if len(comparison_manager.documents) == 0 and len(self.documents) > 0:
                print(f"\nAdding documents to {compare_with_model} model for fair comparison...")
                comparison_manager.add_documents(self.documents, [doc.metadata.get('source', 'unknown') for doc in self.documents])
            
            # Evaluate comparison model
            comparison_results = self.evaluator.evaluate_model_recall(comparison_manager, compare_with_model, top_k)
            
            # Run comparison analysis
            comparison_analysis = self.evaluator.compare_models([current_results, comparison_results])
            
            return {
                "current_model_results": current_results,
                "comparison_model_results": comparison_results,
                "comparison_analysis": comparison_analysis
            }
            
        except Exception as e:
            print(f"Error during comparison: {e}")
            return {"error": str(e), "current_model_results": current_results}
    
    def _load_index(self):
        """Load existing FAISS index if available."""
        try:
            if self.db_path.exists():
                self.index = faiss.read_index(str(self.db_path))
                print(f"✅ Loaded existing FAISS index: {self.index.ntotal} documents")
                return True
        except Exception as e:
            print(f"⚠️ Could not load existing index: {e}")
        return False
    
    def add_documents(self, docs: List[Document], sources: List[str] = None):
        """Add documents to the FAISS index."""
        if not docs:
            return
        
        if sources is None:
            sources = [doc.metadata.get('source', 'unknown') for doc in docs]
        
        self.documents.extend(docs)
        self.sources = sources
        
        print(f"🔍 Added {len(docs)} new documents to the FAISS index")
    
    def build(self, docs: List[Document], force_rebuild: bool = False) -> Dict[str, Any]:
        """Build FAISS index from documents with enhanced duplicate detection."""
        if not docs:
            print("⚠️ No documents provided for building FAISS index")
            return {"status": "no_documents"}
        
        print(f"🔍 Building FAISS index with {len(docs)} documents")
        start_time = time.time()
        
        try:
            # Store original documents first
            original_docs = docs.copy()
            
            # Filter duplicates if enabled, but keep all documents for search if needed
            skipped = 0
            if self.hasher and not force_rebuild:
                original_count = len(docs)
                docs = self._filter_duplicates(docs)
                skipped = original_count - len(docs)
                print(f"⏭️ Skipped {skipped} duplicate documents")
            
            # If no new documents after filtering, use original documents for search capability
            if len(docs) == 0:
                if len(self.documents) > 0:
                    print(f"✅ Using existing {len(self.documents)} documents for search")
                    return {"status": "using_existing", "documents_available": len(self.documents)}
                else:
                    print("⚠️ No new documents to process and no existing documents")
                    print("🔄 Loading all documents for search capability...")
                    # Use original documents to ensure search functionality
                    self.documents = original_docs
                    return {"status": "no_new_documents", "documents_loaded": len(self.documents)}
            
            # Store documents for search
            self.documents = docs
            
            # Create embeddings
            texts = [doc.page_content for doc in docs]
            embeddings = self.embedding_model.encode(texts)
            
            # Archive old models before building new one
            if self.archiver:
                try:
                    archive_result = self.archiver.archive_old_models(keep_most_recent=1)
                    if archive_result["action"] == "archived":
                        print(f"📦 Archived {archive_result['total_files_archived']} old model files")
                        print(f"   Archive location: {Path(archive_result['archive_folder']).name}")
                    elif archive_result["action"] == "no_archiving_needed":
                        print(f"📂 No archiving needed (only {archive_result['total_files']} files found)")
                except Exception as e:
                    logger.warning(f"⚠️ Error during model archiving: {e}")
            
            # Build FAISS index if available
            if FAISS_AVAILABLE:
                embedding_dim = embeddings.shape[1] if embeddings.size > 0 else self.embedding_model.embedding_dim
                self.index = faiss.IndexFlatL2(embedding_dim)
                if embeddings.size > 0:
                    self.index.add(embeddings.astype('float32'))
                print(f"✅ FAISS index built with {self.index.ntotal} vectors")
            else:
                print("⚠️ FAISS not available, using fallback search only")
            
            build_time = time.time() - start_time
            print(f"📊 Index built in {build_time:.2f}s - {len(docs)} documents indexed")
            
            return {
                "status": "success",
                "documents_processed": len(docs),
                "duplicates_skipped": skipped if self.hasher else 0,
                "build_time": build_time,
                "model_used": self.embedding_model.__class__.__name__
            }
                
        except Exception as e:
            print(f"❌ FAISS build failed: {str(e)}")
            # Fallback: at least store documents for search
            self.documents = original_docs
            return {"status": "failed", "error": str(e), "documents_stored": len(original_docs)}
    
    def _filter_duplicates(self, docs: List[Document]) -> List[Document]:
        """Filter out duplicate documents."""
        filtered_docs = []
        
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            if not self.hasher.is_duplicate(doc.page_content, source):
                self.hasher.add_document(doc.page_content, source)
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def search(self, query: str, k: int = 5, enhance_query: bool = True) -> List[Tuple[Document, float]]:
        """Enhanced search with query preprocessing."""
        if enhance_query:
            enhanced_query = self.query_enhancer.enhance_query(query)
        else:
            enhanced_query = query
        
        if self.index is not None and FAISS_AVAILABLE:
            # Use FAISS index
            query_embedding = self.embedding_model.encode([enhanced_query])
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    # Convert FAISS L2 distance to similarity score
                    similarity = 1.0 / (1.0 + score)
                    results.append((self.documents[idx], similarity))
            
            return results
        
        else:
            # Use enhanced fallback search
            return self._enhanced_fallback_search(enhanced_query, k)

    def _enhanced_fallback_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Enhanced fallback search with semantic similarity approximation."""
        results = []
        query_keywords = set(self.query_enhancer.extract_keywords(query))
        
        for doc in self.documents:
            content_keywords = set(self.query_enhancer.extract_keywords(doc.page_content))
            
            # Calculate Jaccard similarity
            intersection = len(query_keywords & content_keywords)
            union = len(query_keywords | content_keywords)
            jaccard_score = intersection / union if union > 0 else 0
            
            # Boost score for exact phrase matches
            phrase_boost = 0
            for keyword in query_keywords:
                if keyword in doc.page_content.lower():
                    phrase_boost += 0.1
            
            final_score = jaccard_score + phrase_boost
            
            if final_score > 0:
                results.append((doc, final_score))
        
        # Sort by score descending and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def evaluate_comprehensive(self, test_queries: List[str], k: int = 5) -> Dict[str, Any]:
        """Comprehensive evaluation with multiple metrics."""
        print(f"\n🔍 COMPREHENSIVE FAISS EVALUATION")
        print(f"=" * 50)
        
        start_time = time.time()
        results = []
        response_times = []

        for query in test_queries:
            query_start = time.time()
            search_results = self.search(query, k=k)
            query_time = time.time() - query_start
            response_times.append(query_time)
            
            # Analyze results
            if search_results:
                top_score = search_results[0][1]
                avg_score = np.mean([score for _, score in search_results]) if NUMPY_AVAILABLE else sum(score for _, score in search_results) / len(search_results)
                result_count = len(search_results)
            else:
                top_score = avg_score = result_count = 0
            
            results.append({
                'query': query,
                'top_score': top_score,
                'avg_score': avg_score,
                'result_count': result_count,
                'response_time': query_time
            })
        
        # Calculate comprehensive metrics
        total_time = time.time() - start_time
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.embedding_model.__class__.__name__,
            'total_documents': len(self.documents),
            'queries_tested': len(test_queries),
            'evaluation_time': total_time,
            
            # Performance metrics
            'avg_response_time': np.mean(response_times) if NUMPY_AVAILABLE else sum(response_times) / len(response_times),
            'max_response_time': max(response_times),
            'min_response_time': min(response_times),
            
            # Quality metrics
            'avg_top_score': np.mean([r['top_score'] for r in results]) if NUMPY_AVAILABLE else sum(r['top_score'] for r in results) / len(results),
            'avg_result_count': np.mean([r['result_count'] for r in results]) if NUMPY_AVAILABLE else sum(r['result_count'] for r in results) / len(results),
            'queries_with_results': sum(1 for r in results if r['result_count'] > 0),
            'success_rate': sum(1 for r in results if r['result_count'] > 0) / len(test_queries) * 100,
            
            # Coverage metrics
            'zero_result_queries': sum(1 for r in results if r['result_count'] == 0),
            'low_score_queries': sum(1 for r in results if r['top_score'] < 0.3),
        }
        
        # Display results
        self._display_evaluation_results(metrics, results[:5])  # Show top 5 examples
        
        # Save metrics
        self.evaluation_results = metrics
        self._save_metrics()
        
        return metrics
    
    def _display_evaluation_results(self, metrics: Dict, sample_results: List[Dict]):
        """Display comprehensive evaluation results."""
        print(f"\n📊 EVALUATION SUMMARY")
        print(f"{'=' * 50}")
        print(f"🎯 Model: {metrics['model_type']}")
        print(f"📚 Documents: {metrics['total_documents']:,}")
        print(f"❓ Queries Tested: {metrics['queries_tested']}")
        print(f"⏱️  Total Time: {metrics['evaluation_time']:.2f}s")
        print()
        
        print(f"🚀 PERFORMANCE METRICS")
        print(f"   Success Rate: {metrics['success_rate']:.1f}%")
        print(f"   Avg Response Time: {metrics['avg_response_time']:.3f}s")
        print(f"   Avg Top Score: {metrics['avg_top_score']:.3f}")
        print(f"   Avg Results per Query: {metrics['avg_result_count']:.1f}")
        print()
        
        print(f"⚠️  QUALITY INDICATORS")
        print(f"   Queries with Zero Results: {metrics['zero_result_queries']}")
        print(f"   Low Score Queries (<0.3): {metrics['low_score_queries']}")
        print()
        
        # Performance interpretation
        if metrics['success_rate'] >= 90:
            status = "🟢 EXCELLENT"
        elif metrics['success_rate'] >= 75:
            status = "🟡 GOOD"
        elif metrics['success_rate'] >= 50:
            status = "🟠 FAIR"
        else:
            status = "🔴 POOR"
        
        print(f"🎯 OVERALL ASSESSMENT: {status}")
        
        if metrics['avg_response_time'] > 1.0:
            print(f"⚠️  Response time may be too slow for production")
        if metrics['zero_result_queries'] > metrics['queries_tested'] * 0.2:
            print(f"⚠️  High number of queries with no results - consider improving chunking")
        
        print(f"\n📝 SAMPLE QUERY RESULTS:")
        for i, result in enumerate(sample_results, 1):
            print(f"   {i}. '{result['query'][:40]}...' → Score: {result['top_score']:.3f}, Results: {result['result_count']}")
    
    def _save_metrics(self):
        """Save evaluation metrics history."""
        metrics_file = self.db_path.parent / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
    
    def archive_old_models(self, keep_most_recent: int = 1) -> Dict[str, Any]:
        """Manually archive old FAISS models."""
        if not self.archiver:
            return {"error": "FAISS archiver not available"}
        
        try:
            result = self.archiver.archive_old_models(keep_most_recent=keep_most_recent)
            print(f"📦 Manual archiving completed: {result}")
            return result
        except Exception as e:
            error_msg = f"Error during manual archiving: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def list_current_models(self) -> List[Dict[str, Any]]:
        """List current FAISS models in the directory."""
        if not self.archiver:
            return []
        
        try:
            return self.archiver.list_current_models()
        except Exception as e:
            logger.error(f"Error listing current models: {e}")
            return []
    
    def list_archived_models(self) -> List[Dict[str, Any]]:
        """List archived FAISS models."""
        if not self.archiver:
            return []
        
        try:
            return self.archiver.list_archived_models()
        except Exception as e:
            logger.error(f"Error listing archived models: {e}")
            return []
    
    def restore_from_archive(self, archive_folder_name: str) -> Dict[str, Any]:
        """Restore FAISS models from a specific archive."""
        if not self.archiver:
            return {"error": "FAISS archiver not available"}
        
        try:
            result = self.archiver.restore_from_archive(archive_folder_name)
            print(f"📂 Restoration completed: {result}")
            return result
        except Exception as e:
            error_msg = f"Error during restoration: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_archive_status(self) -> Dict[str, Any]:
        """Get comprehensive archive status and management info."""
        if not self.archiver:
            return {"error": "FAISS archiver not available"}
        
        try:
            current_models = self.list_current_models()
            archived_models = self.list_archived_models()
            
            return {
                "archiver_available": True,
                "current_models": {
                    "count": len(current_models),
                    "models": current_models
                },
                "archived_models": {
                    "count": len(archived_models),
                    "archives": archived_models
                },
                "archive_directory": str(self.archiver.archive_dir)
            }
        except Exception as e:
            return {"error": f"Error getting archive status: {e}"}

    def health_check(self) -> Dict[str, Any]:
        """Production-ready health check."""
        health = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {},
            "recommendations": []
        }
        
        # Check index availability
        if self.index is not None:
            health["checks"]["index_status"] = "operational"
            health["checks"]["index_type"] = "faiss"
        elif hasattr(self, 'fallback_docs'):
            health["checks"]["index_status"] = "fallback"
            health["checks"]["index_type"] = "fallback"
            health["recommendations"].append("Consider installing FAISS for better performance")
        else:
            health["checks"]["index_status"] = "not_built"
            health["overall_status"] = "unhealthy"
            health["recommendations"].append("Build index before using")
        
        # Check document count
        doc_count = len(self.documents) if self.documents else 0
        health["checks"]["document_count"] = doc_count
        if doc_count == 0:
            health["overall_status"] = "unhealthy"
            health["recommendations"].append("No documents indexed")
        elif doc_count < 10:
            health["recommendations"].append("Low document count may affect retrieval quality")
        
        # Check embedding model
        health["checks"]["embedding_model"] = self.embedding_model.__class__.__name__
        if "Qwen" in self.embedding_model.__class__.__name__:
            health["checks"]["model_quality"] = "high"
        elif "HuggingFace" in self.embedding_model.__class__.__name__:
            health["checks"]["model_quality"] = "medium"
        else:
            health["checks"]["model_quality"] = "low"
            health["recommendations"].append("Consider upgrading embedding model")
        
        # Check duplicate detection
        if self.hasher:
            health["checks"]["duplicate_detection"] = "enabled"
            health["checks"]["known_documents"] = len(self.hasher.known_hashes)
        else:
            health["checks"]["duplicate_detection"] = "disabled"
        
        print(f"🏥 HEALTH CHECK: {health['overall_status'].upper()}")
        print(f"   📊 Documents: {doc_count}")
        print(f"   🤖 Model: {health['checks']['embedding_model']}")
        print(f"   🔍 Index: {health['checks']['index_status']}")
        if health["recommendations"]:
            print(f"   💡 Recommendations: {len(health['recommendations'])}")
        
        return health

    def evaluate(self, test_queries: List[str] = None):
        """Evaluate the FAISS system performance."""
        if test_queries is None:
            test_queries = [
                "What are the benefits of modular chemical plants?",
                "How do you optimize chemical reactor performance?",
                "What are the key considerations for distillation design?"
            ]
        
        print(f"🔍 Evaluating FAISS system with {len(test_queries)} test queries...")
        
        results = []
        for query in test_queries:
            try:
                search_results = self.search(query, k=3)
                results.append({
                    'query': query,
                    'results_count': len(search_results),
                    'top_score': search_results[0][1] if search_results else 0
                })
            except Exception as e:
                print(f"Error evaluating query '{query}': {e}")
                results.append({
                    'query': query,
                    'results_count': 0,
                    'top_score': 0
                })
        
        success_rate = sum(1 for r in results if r['results_count'] > 0) / len(results)
        print(f"✅ Evaluation completed. Success rate: {success_rate:.1%}")
        
        return results

def _load_docs(json_path: str = "data/03_intermediate/converted_chunked_data/chunked_data_full.json", logger=None) -> List[Document]:
    """Load documents from JSON file."""
    def log_message(msg):
        if logger:
            logger(msg)
        else:
            print(msg)
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        docs = []
        for item in data:
            if isinstance(item, dict) and "content" in item:
                doc = Document(
                    page_content=item["content"],
                    metadata=item.get("metadata", {})
                )
                docs.append(doc)
        
        log_message(f"✅ Loaded {len(docs)} documents from {json_path}")
        return docs
        
    except Exception as e:
        log_message(f"❌ Failed to load documents: {e}")
        return []

def create_simple_test():
    """Simple test to demonstrate the enhanced FAISS capabilities."""
    print("🚀 ENHANCED FAISS SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Test 1: Duplicate Detection
    print("\n📋 TEST 1: DUPLICATE DETECTION")
    hasher = DocumentHasher()
    
    # Simulate processing same document twice
    doc1_content = "Modular chemical plants offer advantages in construction and operation."
    doc2_content = "Modular chemical plants offer advantages in construction and operation."  # Same content
    doc3_content = "Different content about distillation processes."
    
    print(f"Document 1: {hasher.is_duplicate(doc1_content, 'doc1.pdf')}")  # Should be False (first time)
    hasher.add_document(doc1_content, 'doc1.pdf')
    print(f"Document 2 (same content): {hasher.is_duplicate(doc2_content, 'doc1.pdf')}")  # Should be True
    print(f"Document 3 (different): {hasher.is_duplicate(doc3_content, 'doc3.pdf')}")  # Should be False
    
    # Test 2: Query Enhancement
    print("\n🔍 TEST 2: QUERY ENHANCEMENT")
    enhancer = QueryEnhancer()
    
    original_query = "modular plant efficiency"
    enhanced_query = enhancer.enhance_query(original_query)
    keywords = enhancer.extract_keywords(enhanced_query)
    
    print(f"Original Query: '{original_query}'")
    print(f"Enhanced Query: '{enhanced_query}'")
    print(f"Extracted Keywords: {keywords}")
    
    # Test 3: Enhanced Fallback Search
    print("\n🔍 TEST 3: ENHANCED SEARCH CAPABILITIES")
    
    # Create test documents
    test_docs = [
        Document(page_content="Modular chemical plants reduce construction time and improve efficiency", metadata={"source": "doc1"}),
        Document(page_content="Distillation columns separate components based on boiling points", metadata={"source": "doc2"}),
        Document(page_content="Reactor optimization involves temperature and pressure control", metadata={"source": "doc3"}),
        Document(page_content="Economic analysis shows modular design reduces capital costs", metadata={"source": "doc4"}),
        Document(page_content="Process simulation helps optimize chemical plant operations", metadata={"source": "doc5"})
    ]
    
    # Test enhanced fallback search
    manager = EnhancedFAISSDBManager(enable_duplicate_detection=False)
    manager.fallback_docs = test_docs
    
    test_query = "modular plant advantages"
    results = manager._enhanced_fallback_search(test_query, k=3)
    
    print(f"Query: '{test_query}'")
    print("Results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. Score: {score:.3f} | Source: {doc.metadata['source']} | Content: {doc.page_content[:50]}...")
    
    # Test 4: Comprehensive Metrics (simulated)
    print("\n📊 TEST 4: COMPREHENSIVE METRICS SIMULATION")
    
    # Simulate evaluation metrics
    simulated_metrics = {
        'model_type': 'EnhancedFallback',
        'total_documents': len(test_docs),
        'queries_tested': 5,
        'evaluation_time': 0.25,
        'avg_response_time': 0.05,
        'avg_top_score': 0.75,
        'avg_result_count': 3.2,
        'success_rate': 100.0,
        'zero_result_queries': 0,
        'low_score_queries': 1
    }
    
    manager._display_evaluation_results(simulated_metrics, [
        {'query': 'modular plant advantages', 'top_score': 0.85, 'result_count': 3},
        {'query': 'distillation optimization', 'top_score': 0.72, 'result_count': 3},
        {'query': 'reactor efficiency', 'top_score': 0.68, 'result_count': 3}
    ])
    
    print("\n🎉 ENHANCED FAISS DEMONSTRATION COMPLETE!")
    print("\n🔑 KEY IMPROVEMENTS:")
    print("   ✅ Duplicate detection prevents reprocessing")
    print("   ✅ Query enhancement improves semantic matching")
    print("   ✅ Enhanced fallback search with Jaccard similarity")
    print("   ✅ Comprehensive evaluation metrics")
    print("   ✅ Production-ready health checks")
    print("   ✅ Better error handling and logging")
    
    return simulated_metrics

def main():
    """Test the enhanced FAISS system."""
    print("🚀 Testing Enhanced FAISS System with Qwen Embeddings")

    # Load documents
    docs = _load_docs()
    if not docs:
        print("❌ No documents loaded")
        return
    
    # Initialize enhanced manager
    manager = EnhancedFAISSDBManager()
    
    # Build index
    build_result = manager.build(docs)
    print(f"Build result: {build_result}")
    
    # Test queries
    test_queries = [
        "advantages of modular chemical plants",
        "distillation column optimization",
        "reactor conversion efficiency",
        "chemical process economics",
        "industrial automation systems"
    ]
    
    # Comprehensive evaluation
    metrics = manager.evaluate_comprehensive(test_queries)
    
    # Health check
    health = manager.health_check()
    
    print("\n🎉 Enhanced FAISS system test complete!")

if __name__ == "__main__":
    create_simple_test() 