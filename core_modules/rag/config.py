# -*- coding: utf-8 -*-
"""
Configuration for the RAG module.
"""
from pathlib import Path

# --- Project Directories ---
ROOT_DIR = Path(__file__).parent.parent.parent
SOURCE_DOCS_DIR = ROOT_DIR / "source_documents"
CONVERTED_DIR = ROOT_DIR / "converted_to_txt"
WEB_SOURCES_DIR = ROOT_DIR / "web_sources"
CHUNKED_DATA_DIR = ROOT_DIR / "converted_chunked_data"
VECTOR_DB_DIR = ROOT_DIR / "vector_db"
REPORTS_DIR = ROOT_DIR / "chunk_reports"

# --- Data Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Vector Store ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = VECTOR_DB_DIR / "pynucleus_mcp.faiss"
EMBEDDINGS_PKL_PATH = VECTOR_DB_DIR / "embeddings.pkl"
FULL_JSON_PATH = CHUNKED_DATA_DIR / "chunked_data_full.json"

# --- Wikipedia Scraper ---
WIKI_SEARCH_KEYWORDS = [
    "modular design",
    "software architecture",
    "system design",
    "industrial design",
    "supply chain",
]

# --- Evaluation ---
GROUND_TRUTH_DATA = {
    "what are the benefits of modular design": str(
        WEB_SOURCES_DIR / "wikipedia_modular_design.txt"
    ),
    "how does modular design work in vehicles": str(
        WEB_SOURCES_DIR / "wikipedia_modular_design.txt"
    ),
} 