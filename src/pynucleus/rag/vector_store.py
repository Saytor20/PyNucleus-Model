import json
import os
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from . import config

logging.basicConfig(level=logging.INFO)


def _load_docs(json_path: str = config.FULL_JSON_PATH, logger=None) -> List[Document]:
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
        for d in data:
            # Create metadata from the available fields
            metadata = {
                "chunk_id": d.get("chunk_id"),
                "source": d.get("source"),
                "length": d.get("length"),
            }
            doc = Document(page_content=d["content"], metadata=metadata)
            docs.append(doc)
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
        index_path: str = config.FAISS_INDEX_PATH,
        embeddings_pkl_path: str = config.EMBEDDINGS_PKL_PATH,
        model_name: str = config.EMBEDDING_MODEL,
        log_dir: str = config.REPORTS_DIR,
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

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

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

    def build(self, docs: List[Document]):
        """Build FAISS index from documents."""
        self.documents = docs
        self.index = FAISS.from_documents(docs, self.embeddings)

        # Save the index
        faiss_path = self.store_dir / "pynucleus_mcp.faiss"
        self.index.save_local(str(faiss_path))

        # Save embeddings
        embeddings_path = self.store_dir / "embeddings.pkl"
        with open(embeddings_path, "wb") as f:
            pickle.dump(self.embeddings, f)

        # Log information
        self.log(f"Embedding device → cpu   | dim=384")
        self.log(f"Docs indexed : {len(docs)}")
        self.log(f"Index file   : {faiss_path}")
        self.log(f"Embeds .pkl  : {embeddings_path}")

        # List files
        self.log(f"\n-- Files in {self.store_dir.name}/ --")
        for f in os.listdir(self.store_dir):
            self.log(f"  · {f}")

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        return self.index.similarity_search_with_score(query, k)

    def evaluate(self, ground_truth: Dict[str, str], k: int = 3) -> None:
        """Evaluate vector store performance."""
        self.log(f"\n=== Evaluation (Recall@{k}) ===")
        correct = 0
        total = len(ground_truth)

        for query, expected_id in ground_truth.items():
            results = self.search(query, k=k)
            top_score = results[0][1] if results else 0

            # Check if any result matches the expected document
            found = any(doc.metadata.get("source") == expected_id for doc, _ in results)
            correct += 1 if found else 0

            self.log(
                f"Q: {query[:40]:<40} {'✓' if found else '✗'}   top-score={top_score:.4f}"
            )

        recall = (correct / total) * 100 if total > 0 else 0
        self.log(f"\nRecall@{k}: {correct}/{total}  →  {recall:.1f}%")


def main():
    """Example usage."""
    GROUND_TRUTH = {
        "what are the benefits of modular design": "web_sources/wikipedia_modular_design.txt",
        "how does modular design work in vehicles": "web_sources/wikipedia_modular_design.txt",
    }
    JSON_PATH = "converted_chunked_data/chunked_data_full.json"

    f_mgr = FAISSDBManager()
    f_docs = _load_docs(JSON_PATH, f_mgr.log)
    f_mgr.build(f_docs)
    f_mgr.evaluate(GROUND_TRUTH)
    print(f"\nFAISS log → {f_mgr.log_path}")


if __name__ == "__main__":
    main()
