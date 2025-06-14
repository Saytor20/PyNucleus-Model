#!/usr/bin/env python3
"""
Vector Store Module for RAG Pipeline

Handles FAISS vector database operations for the PyNucleus RAG system.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Try to import required components
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    np = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import langchain components
try:
    from langchain_core.documents.base import Document
    DOCUMENT_AVAILABLE = True
except ImportError:
    try:
        from langchain.docstore.document import Document
        DOCUMENT_AVAILABLE = True
    except ImportError:
        DOCUMENT_AVAILABLE = False
        # Fallback for Document class
        class Document:
            def __init__(self, page_content: str, metadata: dict = None):
                self.page_content = page_content
                self.metadata = metadata or {}

# Try to import embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        EMBEDDINGS_AVAILABLE = True
    except ImportError:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
            EMBEDDINGS_AVAILABLE = True
        except ImportError:
            EMBEDDINGS_AVAILABLE = False

# Try to import FAISS from langchain
try:
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_FAISS_AVAILABLE = True
except ImportError:
    try:
        from langchain.vectorstores import FAISS
        LANGCHAIN_FAISS_AVAILABLE = True
    except ImportError:
        LANGCHAIN_FAISS_AVAILABLE = False

# Import from absolute paths instead of relative
try:
    from pynucleus.rag.config import RAGConfig
except ImportError:
    # Fallback config
    class RAGConfig:
        def __init__(self):
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.vector_dim = 384

logging.basicConfig(level=logging.INFO)

def _load_docs(json_path: str = "data/03_intermediate/converted_chunked_data/chunked_data_full.json", logger=None) -> List[Document]:
    """Load documents from JSON file."""
    def log_message(msg):
        if logger:
            if hasattr(logger, 'info'):
                logger.info(msg)
            else:
                logger(msg)
        else:
            print(msg)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log_message(f"Loaded {len(data)} documents from {json_path}")

        docs = []
        for i, d in enumerate(data):
            try:
                # Create metadata from the available fields
                metadata = {
                    "chunk_id": d.get("chunk_id"),
                    "source": d.get("source"),
                    "length": d.get("length"),
                }
                
                # Ensure content exists and is a string
                content = d.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)
            except Exception as e:
                log_message(f"⚠️ Error processing document {i}: {e}")
                log_message(f"   Document data: {d}")
                # Skip this document and continue
                continue
        return docs

    except Exception as e:
        log_message(f"⚠️  {e} – falling back to 3 dummy docs.")
        dummy = [
            (
                "Modular chemical plants reduce construction time.",
                {"source": "dummy_1"},
            ),
            (
                "Scalability is a key advantage of modular design.",
                {"source": "dummy_2"},
            ),
            ("Challenges include supply-chain coordination.", {"source": "dummy_3"}),
        ]
        return [Document(page_content=t, metadata=m) for t, m in dummy]

class FAISSDBManager:
    def __init__(
        self,
        index_path: str = "data/04_models/chunk_reports/pynucleus_mcp.faiss",
        embeddings_pkl_path: str = "data/04_models/chunk_reports/embeddings.pkl",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        log_dir: str = "data/04_models/chunk_reports",
    ):
        """Initialize FAISS vector store manager."""
        self.store_dir = Path(log_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"faiss_analysis_{timestamp}.txt"
        self.log = self._setup_logging()

        # Initialize embeddings with fallback
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception as e:
                self.log(f"⚠️ Failed to initialize HuggingFaceEmbeddings: {e}")
                self.embeddings = self._create_fallback_embeddings(model_name)
        else:
            self.log("⚠️ HuggingFaceEmbeddings not available, using fallback")
            self.embeddings = self._create_fallback_embeddings(model_name)

        # Initialize FAISS index
        self.index = None
        self.documents = []

    def _setup_logging(self):
        """Set up logging for FAISS operations."""
        def log_func(message: str):
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{message}\n")
            print(message)

        # Write header
        log_func("=== FAISS VectorDB Analysis ===")
        log_func(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return log_func

    def _create_fallback_embeddings(self, model_name: str):
        """Create fallback embeddings when HuggingFaceEmbeddings is not available."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use SentenceTransformer directly
            class FallbackEmbeddings:
                def __init__(self, model_name):
                    self.model = SentenceTransformer(model_name)
                
                def embed_documents(self, texts):
                    return self.model.encode(texts).tolist()
                
                def embed_query(self, text):
                    return self.model.encode([text])[0].tolist()
            
            return FallbackEmbeddings(model_name)
        else:
            # Simple dummy embeddings for testing
            class DummyEmbeddings:
                def embed_documents(self, texts):
                    return [[0.1] * 384 for _ in texts]
                
                def embed_query(self, text):
                    return [0.1] * 384
            
            return DummyEmbeddings()

    def build(self, docs: List[Document]):
        """Build FAISS index from documents."""
        try:
            self.documents = docs
            
            # Debug: Check what type of objects we're receiving
            self.log(f"🔍 Building FAISS index with {len(docs)} documents")
            if docs:
                first_doc = docs[0]
                self.log(f"   First document type: {type(first_doc)}")
                if hasattr(first_doc, 'page_content'):
                    self.log(f"   Has page_content: ✅")
                else:
                    self.log(f"   Has page_content: ❌ - This is the problem!")
                    self.log(f"   Document attributes: {dir(first_doc)}")
            
            if LANGCHAIN_FAISS_AVAILABLE and EMBEDDINGS_AVAILABLE:
                # Use full langchain FAISS
                self.index = FAISS.from_documents(docs, self.embeddings)
                
                # Save the index
                faiss_path = self.store_dir / "pynucleus_mcp.faiss"
                self.index.save_local(str(faiss_path))
                
                # Save embeddings
                embeddings_path = self.store_dir / "embeddings.pkl"
                with open(embeddings_path, "wb") as f:
                    pickle.dump(self.embeddings, f)
                
                self.log(f"✅ FAISS index built successfully with langchain")
                
            else:
                # Fallback: create simple index
                self.log("⚠️ Using fallback FAISS implementation")
                self._build_fallback_index(docs)
            
            # Log information
            self.log(f"Embedding device → cpu   | dim=384")
            self.log(f"Docs indexed : {len(docs)}")
            
            # List files
            self.log(f"\n-- Files in {self.store_dir.name}/ --")
            for f in os.listdir(self.store_dir):
                self.log(f"  · {f}")
                
        except Exception as e:
            self.log(f"❌ FAISS build failed: {str(e)}")
            # Continue without FAISS - system can still function
            self.log("⚠️ Continuing without vector store...")

    def _build_fallback_index(self, docs: List[Document]):
        """Build a simple fallback index when full FAISS is not available."""
        # Store documents in a simple format for basic search
        self.fallback_docs = docs
        self.log("✅ Fallback document storage created")

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if self.index is not None:
            # Use FAISS index if available
            return self.index.similarity_search_with_score(query, k)
        
        elif hasattr(self, 'fallback_docs'):
            # Use simple fallback search
            return self._fallback_search(query, k)
        
        else:
            raise ValueError("No search index available. Call build() first.")

    def _fallback_search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Simple fallback search when FAISS is not available."""
        results = []
        query_lower = query.lower()
        
        for doc in self.fallback_docs:
            # Simple keyword matching score
            content_lower = doc.page_content.lower()
            keywords = query_lower.split()
            score = sum(1 for keyword in keywords if keyword in content_lower)
            
            if score > 0:
                # Convert to similarity score (higher is better)
                results.append((doc, float(score)))
        
        # Sort by score descending and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def evaluate(self, ground_truth: Dict[str, str], k: int = 3) -> Dict[str, float]:
        """Evaluate vector store performance with comprehensive metrics."""
        self.log(f"\n=== Evaluation (Recall@{k}) ===")
        correct = 0
        total = len(ground_truth)
        scores = []
        response_times = []

        for query, expected_id in ground_truth.items():
            start_time = datetime.now()
            results = self.search(query, k=k)
            response_time = (datetime.now() - start_time).total_seconds()
            response_times.append(response_time)
            
            top_score = results[0][1] if results else 0
            scores.append(top_score)

            # Check if any result matches the expected document
            found = any(doc.metadata.get("source") == expected_id for doc, _ in results)
            correct += 1 if found else 0

            self.log(
                f"Q: {query[:40]:<40} {'✓' if found else '✗'}   top-score={top_score:.4f}   time={response_time:.3f}s"
            )

        recall = (correct / total) * 100 if total > 0 else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Calculate additional metrics for production monitoring
        metrics = {
            'recall_at_k': recall,
            'average_similarity_score': avg_score,
            'average_response_time': avg_response_time,
            'total_queries': total,
            'successful_queries': correct,
            'timestamp': datetime.now().isoformat()
        }
        
        self.log(f"\nRecall@{k}: {correct}/{total}  →  {recall:.1f}%")
        self.log(f"Avg Similarity Score: {avg_score:.4f}")
        self.log(f"Avg Response Time: {avg_response_time:.3f}s")
        
        # Save metrics for trend analysis
        self._save_performance_metrics(metrics)
        
        return metrics

    def _save_performance_metrics(self, metrics: Dict[str, float]):
        """Save performance metrics for trend analysis and monitoring."""
        metrics_file = self.store_dir / "performance_metrics.jsonl"
        
        # Append metrics to JSONL file for time series analysis
        with open(metrics_file, "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(metrics) + "\n")

    def benchmark_embedding_quality(self, sample_queries: List[str] = None, k: int = 5) -> Dict[str, Any]:
        """Comprehensive embedding quality benchmark for production systems."""
        if not sample_queries:
            sample_queries = [
                "modular chemical plant design",
                "process intensification benefits",
                "distillation column optimization",
                "reactor conversion efficiency",
                "heat exchanger performance",
                "separation process economics",
                "industrial automation systems",
                "chemical process safety",
                "sustainable manufacturing",
                "supply chain optimization"
            ]
        
        self.log(f"\n=== Embedding Quality Benchmark ===")
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'total_documents': len(self.documents),
            'embedding_dimension': 384,  # all-MiniLM-L6-v2 dimension
            'queries_tested': len(sample_queries),
            'retrieval_metrics': {},
            'diversity_metrics': {},
            'coverage_metrics': {}
        }
        
        all_scores = []
        all_response_times = []
        retrieved_docs = set()
        
        for i, query in enumerate(sample_queries):
            start_time = datetime.now()
            results = self.search(query, k=k)
            response_time = (datetime.now() - start_time).total_seconds()
            all_response_times.append(response_time)
            
            if results:
                query_scores = [score for _, score in results]
                all_scores.extend(query_scores)
                
                # Track document coverage
                for doc, _ in results:
                    doc_id = doc.metadata.get('source', f'doc_{hash(doc.page_content[:100])}')
                    retrieved_docs.add(doc_id)
                
                self.log(f"Query {i+1:2d}: {query[:30]:<30} | Top score: {query_scores[0]:.4f} | Time: {response_time:.3f}s")
        
        # Calculate comprehensive metrics
        if all_scores:
            benchmark_results['retrieval_metrics'] = {
                'avg_similarity_score': sum(all_scores) / len(all_scores),
                'min_similarity_score': min(all_scores),
                'max_similarity_score': max(all_scores),
                'score_std_dev': np.std(all_scores) if np else 0,
                'avg_response_time': sum(all_response_times) / len(all_response_times),
                'max_response_time': max(all_response_times),
                'queries_with_results': len([s for s in all_scores if s > 0])
            }
        
        # Document coverage analysis
        total_docs = len(self.documents)
        coverage_ratio = len(retrieved_docs) / total_docs if total_docs > 0 else 0
        benchmark_results['coverage_metrics'] = {
            'documents_retrieved': len(retrieved_docs),
            'total_documents': total_docs,
            'coverage_ratio': coverage_ratio,
            'coverage_percentage': coverage_ratio * 100
        }
        
        # Diversity analysis (check if we're getting diverse results)
        unique_sources = len(set(doc.metadata.get('source', 'unknown') for doc in self.documents))
        benchmark_results['diversity_metrics'] = {
            'unique_sources': unique_sources,
            'source_diversity_ratio': unique_sources / total_docs if total_docs > 0 else 0
        }
        
        self.log(f"\n📊 Benchmark Summary:")
        self.log(f"   Avg Similarity: {benchmark_results['retrieval_metrics'].get('avg_similarity_score', 0):.4f}")
        self.log(f"   Avg Response Time: {benchmark_results['retrieval_metrics'].get('avg_response_time', 0):.3f}s")
        self.log(f"   Document Coverage: {benchmark_results['coverage_metrics']['coverage_percentage']:.1f}%")
        
        # Save benchmark results
        benchmark_file = self.store_dir / f"embedding_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(benchmark_file, "w", encoding="utf-8") as f:
            import json
            f.write(json.dumps(benchmark_results, indent=2))
        
        return benchmark_results

    def monitor_embedding_drift(self, reference_queries: List[str] = None) -> Dict[str, Any]:
        """Monitor embedding drift over time for production systems."""
        if not reference_queries:
            reference_queries = [
                "chemical process optimization",
                "modular plant design",
                "industrial efficiency"
            ]
        
        drift_metrics = {
            'timestamp': datetime.now().isoformat(),
            'reference_queries': len(reference_queries),
            'drift_indicators': {}
        }
        
        # Load historical performance if available
        metrics_file = self.store_dir / "performance_metrics.jsonl"
        historical_scores = []
        
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        import json
                        data = json.loads(line.strip())
                        if 'average_similarity_score' in data:
                            historical_scores.append(data['average_similarity_score'])
                    except:
                        continue
        
        # Current performance
        current_scores = []
        for query in reference_queries:
            results = self.search(query, k=3)
            if results:
                current_scores.append(results[0][1])
        
        current_avg = sum(current_scores) / len(current_scores) if current_scores else 0
        
        if historical_scores and len(historical_scores) >= 5:
            historical_avg = sum(historical_scores[-5:]) / 5  # Last 5 measurements
            drift_percentage = ((current_avg - historical_avg) / historical_avg) * 100
            
            drift_metrics['drift_indicators'] = {
                'current_avg_score': current_avg,
                'historical_avg_score': historical_avg,
                'drift_percentage': drift_percentage,
                'drift_status': 'SIGNIFICANT' if abs(drift_percentage) > 10 else 'NORMAL',
                'measurements_count': len(historical_scores)
            }
            
            self.log(f"\n🔍 Embedding Drift Analysis:")
            self.log(f"   Current Avg Score: {current_avg:.4f}")
            self.log(f"   Historical Avg: {historical_avg:.4f}")
            self.log(f"   Drift: {drift_percentage:+.2f}%")
            self.log(f"   Status: {drift_metrics['drift_indicators']['drift_status']}")
        else:
            drift_metrics['drift_indicators'] = {
                'current_avg_score': current_avg,
                'status': 'BASELINE_ESTABLISHING',
                'note': 'Need more historical data for drift analysis'
            }
            self.log(f"\n🔍 Establishing baseline - Current avg score: {current_avg:.4f}")
        
        return drift_metrics

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for production vector store."""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'HEALTHY',
            'checks': {},
            'recommendations': []
        }
        
        # Check 1: Index availability
        if self.index is not None:
            health_status['checks']['index_available'] = True
        else:
            health_status['checks']['index_available'] = False
            health_status['overall_status'] = 'DEGRADED'
            health_status['recommendations'].append("Rebuild FAISS index")
        
        # Check 2: Document count
        doc_count = len(self.documents)
        health_status['checks']['document_count'] = doc_count
        if doc_count == 0:
            health_status['overall_status'] = 'CRITICAL'
            health_status['recommendations'].append("Add documents to vector store")
        elif doc_count < 10:
            health_status['recommendations'].append("Consider adding more documents for better retrieval")
        
        # Check 3: Embedding model availability
        try:
            test_embedding = self.embeddings.embed_query("test")
            health_status['checks']['embedding_model'] = True
            health_status['checks']['embedding_dimension'] = len(test_embedding)
        except Exception as e:
            health_status['checks']['embedding_model'] = False
            health_status['overall_status'] = 'CRITICAL'
            health_status['recommendations'].append(f"Fix embedding model: {str(e)}")
        
        # Check 4: Search functionality
        try:
            if doc_count > 0:
                test_results = self.search("test query", k=1)
                health_status['checks']['search_functional'] = len(test_results) > 0
            else:
                health_status['checks']['search_functional'] = None
        except Exception as e:
            health_status['checks']['search_functional'] = False
            health_status['overall_status'] = 'DEGRADED'
            health_status['recommendations'].append(f"Fix search functionality: {str(e)}")
        
        # Check 5: Performance metrics availability
        metrics_file = self.store_dir / "performance_metrics.jsonl"
        health_status['checks']['metrics_tracking'] = metrics_file.exists()
        
        self.log(f"\n🏥 Health Check Results:")
        self.log(f"   Overall Status: {health_status['overall_status']}")
        self.log(f"   Documents: {doc_count}")
        self.log(f"   Index Available: {health_status['checks']['index_available']}")
        self.log(f"   Search Functional: {health_status['checks']['search_functional']}")
        
        if health_status['recommendations']:
            self.log(f"   Recommendations: {len(health_status['recommendations'])}")
            for rec in health_status['recommendations']:
                self.log(f"     • {rec}")
        
        return health_status

def main():
    """Main function for testing."""
    if not FAISS_AVAILABLE:
        print("⚠️ FAISS not available. Please install faiss-cpu or faiss-gpu.")
        return

    # Load documents
    docs = _load_docs()
    
    # Initialize and build index
    manager = FAISSDBManager()
    manager.build(docs)
    
    # Test search
    query = "What are the advantages of modular plants?"
    results = manager.search(query, k=3)
    
    print("\nSearch Results:")
    for doc, score in results:
        print(f"\nScore: {score:.4f}")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Content: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main() 