import os
import json
import pickle
import shutil
import torch
from datetime import datetime
from typing import Dict, List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Helper functions
def _mkdir_clean(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def _mkdir_if_missing(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _timestamp(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def _now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _logfile(out_dir: str, prefix: str):
    _mkdir_if_missing(out_dir)
    path = os.path.join(out_dir, f"{prefix}_{_timestamp()}.txt")
    def write(msg: str, echo=True):
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        if echo: print(msg)
    return write, path

def _load_docs(json_path: str, log) -> List[Document]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log(f"Loaded {len(data)} documents from {json_path}")
        return [Document(page_content=d["content"],
                        metadata={"source": d["source"]}) for d in data]
    except Exception as e:
        log(f"⚠️  {e} – falling back to 3 dummy docs.")
        dummy = [
            ("Modular chemical plants reduce construction time.", "dummy_1"),
            ("Scalability is a key advantage of modular design.", "dummy_2"),
            ("Challenges include supply-chain coordination.", "dummy_3"),
        ]
        return [Document(page_content=t, metadata={"source": s}) for t, s in dummy]

class FAISSDBManager:
    def __init__(self, vec_dir="faiss_store", out_dir="vectordb_outputs"):
        _mkdir_clean(vec_dir)      # re-create store each run
        self.vec_dir, self.out_dir = vec_dir, out_dir
        self.index_path = os.path.join(vec_dir, "pynucleus_mcp.faiss")
        self.embed_path = os.path.join(vec_dir, "embeddings.pkl")
        self.log, self.log_path = _logfile(out_dir, "faiss_analysis")
        self.log("=== FAISS VectorDB Analysis ===")
        self.log(f"Started: {_now()}")
        self.db, self.embeddings = None, None

    def _emb(self):
        if self.embeddings is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": dev},
                encode_kwargs={"normalize_embeddings": True})
            self.log(f"Embedding device → {dev}   | dim={len(self.embeddings.embed_query('hi'))}")
        return self.embeddings

    def build(self, docs):
        emb = self._emb()
        self.db = FAISS.from_documents(docs, emb)
        self.db.save_local(self.index_path)
        pickle.dump(emb, open(self.embed_path, "wb"))
        self.log(f"Docs indexed : {len(docs)}")
        self.log(f"Index file   : {self.index_path}")
        self.log(f"Embeds .pkl  : {self.embed_path}")

        # List everything in vec_dir so you can verify
        self.log("\n-- Files in faiss_store/ --")
        for f in os.listdir(self.vec_dir):
            self.log(f"  · {f}")

    def load(self):
        if self.db is None:
            if self.embeddings is None and os.path.isfile(self.embed_path):
                self.embeddings = pickle.load(open(self.embed_path, "rb"))
            self.db = FAISS.load_local(self.index_path,
                                       self.embeddings,
                                       allow_dangerous_deserialization=True)

    def search(self, q: str, k=3):
        self.load()
        return self.db.similarity_search_with_score(q, k)

    def evaluate(self, gt: Dict[str, str], k=3):
        self.log("\n=== Evaluation (Recall@3) ===")
        hits = 0
        for q, expect in gt.items():
            res = self.search(q, k)
            best = res[0][1] if res else float("inf")
            good = any(expect in d.page_content or expect == d.metadata["source"] for d, _ in res)
            hits += good
            self.log(f"Q: {q[:45]:<45}  {'✓' if good else '✗'}   top-score={best:.4f}")
        self.log(f"\nRecall@{k}: {hits}/{len(gt)}  →  {hits/len(gt):.1%}")

def main():
    """Example usage of the FAISS manager."""
    GROUND_TRUTH = {
        "advantages of modular chemical plants": "dummy_1",
        "scalability of modular design": "dummy_2",
    }
    JSON_PATH = "Chuncked_Data/chunked_data_full.json"

    f_mgr = FAISSDBManager()
    f_docs = _load_docs(JSON_PATH, f_mgr.log)
    f_mgr.build(f_docs)
    f_mgr.evaluate(GROUND_TRUTH)
    print(f"\nFAISS log → {f_mgr.log_path}")

if __name__ == "__main__":
    main() 