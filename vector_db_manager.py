import os
import json
import pickle
import shutil
import torch
import warnings
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from chromadb.config import Settings
import chromadb

# Silence PostHog telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("posthog").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="posthog")

# Helper functions
def _mkdir_clean(path: str) -> None:
    """Clean and create a directory with full permissions."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    os.chmod(path, 0o777)  # Ensure directory is writable

def _mkdir_if_missing(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _timestamp() -> str:
    """Get current timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _now() -> str:
    """Get current time in YYYY-MM-DD HH:MM:SS format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _logfile(out_dir: str, prefix: str) -> Tuple[callable, str]:
    """Create a log file and return a write function and the log path."""
    _mkdir_if_missing(out_dir)
    path = os.path.join(out_dir, f"{prefix}_{_timestamp()}.txt")
    
    def write(msg: str, echo: bool = True) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        if echo:
            print(msg)
    
    return write, path

def _load_docs(json_path: str, log: callable) -> List[Document]:
    """Load documents from a JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log(f"Loaded {len(data)} documents from {json_path}")
        return [Document(page_content=d["content"], metadata={"source": d["source"]}) for d in data]
    except Exception as e:
        log(f"⚠️  {e} – fallback to dummy docs.")
        dummy = [
            ("Modular chemical plants reduce construction time.", "dummy_1"),
            ("Scalability is a key advantage of modular design.", "dummy_2"),
            ("Challenges include supply-chain coordination.", "dummy_3"),
        ]
        return [Document(page_content=t, metadata={"source": s}) for t, s in dummy]

class ChromaDBManager:
    """Manager class for ChromaDB operations."""
    
    def __init__(self,
                 vec_dir_name: str = "chroma_store",
                 out_dir_name: str = "vectordb_outputs",
                 collection: str = "pynucleus_mcp"):
        """Initialize the ChromaDB manager."""
        # Convert to absolute paths
        self.vec_dir = os.path.abspath(vec_dir_name)
        self.out_dir = os.path.abspath(out_dir_name)
        
        _mkdir_clean(self.vec_dir)
        
        self.collection = collection
        self.embed_path = os.path.join(self.vec_dir, "embeddings.pkl")
        
        self.log, self.log_path = _logfile(self.out_dir, "chroma_analysis")
        self.log("=== Chroma VectorDB Analysis (Absolute Paths) ===")
        self.log(f"Vector DB Directory (Absolute): {self.vec_dir}")
        self.log(f"Output Directory (Absolute): {self.out_dir}")
        self.log(f"Started: {_now()}")
        
        self.embeddings = None
        self.db = None
        self.client_settings = Settings(
            anonymized_telemetry=False,
            persist_directory=self.vec_dir,
            is_persistent=True
        )
        self.log(f"Chroma client settings configured for persistence at: {self.vec_dir}")

    def _emb(self) -> HuggingFaceEmbeddings:
        """Initialize or return existing embeddings."""
        if self.embeddings is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": dev},
                encode_kwargs={"normalize_embeddings": True}
            )
            self.log(f"Embedding device → {dev}   | dim=384")
        return self.embeddings

    def build(self, docs: List[Document]) -> None:
        """Build the vector database from documents."""
        emb = self._emb()
        
        client = chromadb.PersistentClient(
            path=self.vec_dir,
            settings=self.client_settings
        )
        self.log(f"ChromaDB PersistentClient initialized. Attempting to use/create collection: '{self.collection}' in '{self.vec_dir}'")
        
        self.db = Chroma.from_documents(
            documents=docs,
            embedding=emb,
            client=client,
            collection_name=self.collection,
            persist_directory=self.vec_dir  # Add persist_directory here
        )
        self.log("Chroma.from_documents completed. Persisting database changes.")
        
        # Save embeddings
        with open(self.embed_path, "wb") as f:
            pickle.dump(emb, f)
        
        self.log(f"Docs indexed : {len(docs)}")
        self.log(f"Embeds .pkl  : {self.embed_path}")
        
        # List files in vector store directory
        self.log("\n-- Files in vector store directory --")
        if os.path.exists(self.vec_dir):
            for root, _, files in os.walk(self.vec_dir):
                for f_name in files:
                    rel = os.path.relpath(os.path.join(root, f_name), self.vec_dir)
                    self.log(f"  · {rel} (Full: {os.path.join(root, f_name)})")
        else:
            self.log(f"Vector store directory {self.vec_dir} not found for listing files.")

    def load(self) -> None:
        """Load the database from disk."""
        if self.db is None:
            self.log("Database not in memory. Attempting to load from disk...")
            if self.embeddings is None and os.path.isfile(self.embed_path):
                self.log(f"Loading embeddings from {self.embed_path}")
                with open(self.embed_path, "rb") as f:
                    self.embeddings = pickle.load(f)
            
            if self.embeddings is None:
                self.log("Embeddings not found in memory or pickle, re-initializing for load.")
                self._emb()
            
            client = chromadb.PersistentClient(
                path=self.vec_dir,
                settings=self.client_settings
            )
            self.log(f"ChromaDB PersistentClient initialized for loading. Accessing collection: '{self.collection}' from '{self.vec_dir}'")
            
            self.db = Chroma(
                client=client,
                embedding_function=self.embeddings,
                collection_name=self.collection,
                persist_directory=self.vec_dir  # Add persist_directory here
            )
            self.log("Database loaded successfully.")

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Search the database for similar documents."""
        self.load()
        if self.db is None:
            self.log("⚠️ Error: Database not loaded, cannot perform search.")
            return []
        self.log(f"Performing similarity search for: '{query[:50]}...' (k={k})")
        return self.db.similarity_search_with_score(query, k)

    def evaluate(self, ground_truth: Dict[str, str], k: int = 3) -> None:
        """Evaluate the database performance against ground truth queries."""
        self.log("\n=== Evaluation (Recall@k) ===")
        self.log(f"Evaluating with k={k}")
        hits = 0
        total = len(ground_truth)
        
        if total == 0:
            self.log("No ground truth queries provided for evaluation.")
            return
        
        for query, expected in ground_truth.items():
            results = self.search(query, k)
            distance = results[0][1] if results and len(results) > 0 else float("inf")
            is_match = any(
                expected in doc.page_content or expected == doc.metadata["source"]
                for doc, _ in results
            )
            if is_match:
                hits += 1
            self.log(f"Q: {query[:45]:<45}  {'✓' if is_match else '✗'}   top-dist={distance:.4f}")
        
        recall = (hits / total) if total > 0 else 0
        self.log(f"\nRecall@{k}: {hits}/{total}  →   {recall:.1%}")

def main():
    """Main function to demonstrate usage."""
    # Example ground truth queries
    GROUND_TRUTH = {
        "advantages of modular chemical plants": "dummy_1",
        "scalability of modular design": "dummy_2",
    }
    JSON_PATH = "Chuncked_Data/chunked_data_full.json"
    
    # Initialize manager
    manager = ChromaDBManager(
        vec_dir_name="chroma_store_abs",
        out_dir_name="vectordb_outputs_abs"
    )
    
    # Load documents
    docs = _load_docs(JSON_PATH, manager.log)
    
    if not docs:
        manager.log("⚠️ No documents loaded. Halting build and evaluation.")
    else:
        manager.log(f"Proceeding to build database with {len(docs)} documents.")
        try:
            manager.build(docs)
            manager.log("Build process completed.")
            manager.evaluate(GROUND_TRUTH)
            manager.log("Evaluation completed.")
        except Exception as e:
            manager.log(f"💥 An error occurred during build or evaluation: {type(e).__name__} - {e}")
            import traceback
            manager.log(traceback.format_exc())
    
    print(f"\nChroma log → {manager.log_path}")

if __name__ == "__main__":
    main() 