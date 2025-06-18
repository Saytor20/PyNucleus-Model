#!/usr/bin/env python3
"""
Data Chunking Module for RAG Pipeline

Handles text chunking and document segmentation for the PyNucleus RAG system.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import json
import time
import logging
import logging.config
import yaml
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Configure logging from YAML
def setup_logging():
    """Setup logging configuration from YAML file."""
    config_path = project_root / "configs" / "logging.yaml"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
        except Exception as e:
            # Fallback to basic logging if YAML config fails
            logging.basicConfig(level=logging.INFO)
            logging.error(f"Failed to load logging config: {e}")
    else:
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO)
        
setup_logging()
logger = logging.getLogger(__name__)

# Try to import langchain components
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents.base import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents.base import Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.docstore.document import Document
            LANGCHAIN_AVAILABLE = True
        except ImportError:
            LANGCHAIN_AVAILABLE = False
            # Fallback for text splitting
            class Document:
                def __init__(self, page_content: str, metadata: dict = None):
                    self.page_content = page_content
                    self.metadata = metadata or {}

# Import from absolute paths instead of relative
try:
    from pynucleus.rag.config import RAGConfig
except ImportError:
    # Fallback config
    class RAGConfig:
        def __init__(self):
            self.chunk_size = 1000
            self.chunk_overlap = 200

# Always import the proper metadata stripper
from pynucleus.rag.document_processor import strip_document_metadata

import warnings
from datetime import datetime
from typing import List

warnings.filterwarnings("ignore")

def load_and_chunk_files(
    chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """Load and chunk documents from both web_sources and converted_to_txt directories.

    Args:
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of Document objects with metadata
    """
    docs = []

    # Load Wikipedia articles from web_sources
    rag_config = RAGConfig()
    web_sources_dir = Path(getattr(rag_config, 'web_sources_dir', 'data/01_raw/web_sources'))
    if web_sources_dir.exists():
        wiki_files = list(web_sources_dir.glob("wikipedia_*.txt"))
        print(f"📰 Found {len(wiki_files)} Wikipedia articles")

        for file_path in wiki_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            doc = Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "type": "wikipedia",
                    "title": file_path.stem.replace("wikipedia_", "")
                    .replace("_", " ")
                    .title(),
                },
            )
            docs.append(doc)
    else:
        print("📰 Found 0 Wikipedia articles (web_sources directory not found)")

    # Load processed text files from converted_to_txt directory
    converted_dir = Path(getattr(rag_config, 'converted_dir', 'data/02_processed/converted_to_txt'))
    if converted_dir.exists():
        converted_files = list(converted_dir.glob("*.txt"))
        print(f"📄 Found {len(converted_files)} converted documents")

        for file_path in converted_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            doc = Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "type": "document",
                    "title": file_path.stem.replace("_", " ").title(),
                },
            )
            docs.append(doc)
    else:
        print("📄 Found 0 converted documents (converted_to_txt directory not found)")

    if not docs:
        print("⚠️ No documents found to process")
        return docs

    print(f"📋 Total documents loaded: {len(docs)}")

    # Split into chunks
    if LANGCHAIN_AVAILABLE:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(docs)
    else:
        # Fallback simple text splitting
        chunks = []
        for doc in docs:
            text = doc.page_content
            words = text.split()
            chunk_words = chunk_size // 4  # Rough approximation
            
            for i in range(0, len(words), chunk_words - (chunk_overlap // 4)):
                chunk_text = " ".join(words[i:i + chunk_words])
                chunk = Document(page_content=chunk_text, metadata=doc.metadata.copy())
                chunks.append(chunk)
    print(f"✂️ Split into {len(chunks)} chunks\n")

    return chunks

def save_chunked_data(
    chunks: List[Document], output_dir: str = "data/03_intermediate/converted_chunked_data"
) -> None:
    """Save chunked documents with metadata.

    Args:
        chunks: List of Document objects
        output_dir: Directory to save processed data
    """
    if not chunks:
        print("⚠️ No chunks to save")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save full data with metadata in the format expected by vector store
    full_data = []
    for i, chunk in enumerate(chunks):
        # Extract source file information
        source_path = chunk.metadata.get("source", "unknown")
        source_filename = Path(source_path).name if source_path != "unknown" else "unknown"
        
        # Determine file format based on extension
        file_format = "TXT"  # default
        if source_filename.lower().endswith(".pdf"):
            file_format = "PDF"
        elif source_filename.lower().endswith(".docx"):
            file_format = "DOCX"
        elif source_filename.lower().endswith(".txt"):
            file_format = "TXT"
        
        full_data.append(
            {
                "chunk_id": i,
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "length": len(chunk.page_content),
                # Required metadata per requirements
                "source_filename": source_filename,
                "file_format": file_format,
                "ingestion_timestamp": datetime.now().isoformat()
            }
        )

    with open(output_path / "chunked_data_full.json", "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)

    # Save statistics (avoid division by zero)
    stats = {
        "total_chunks": len(chunks),
        "sources": list(
            set(chunk.metadata.get("source", "unknown") for chunk in chunks)
        ),
        "avg_chunk_size": (
            sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
            if chunks
            else 0
        ),
        "generated_at": datetime.now().isoformat(),
    }

    with open(output_path / "chunked_data_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Save human-readable content
    with open(output_path / "chunked_data_content.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"\n--- Chunk {i} ---\n")
            f.write(f"Source: {chunk.metadata.get('source', 'unknown')}\n")
            f.write(f"Type: {chunk.metadata.get('type', 'unknown')}\n")
            f.write(f"Title: {chunk.metadata.get('title', 'unknown')}\n")
            f.write(f"Content:\n{chunk.page_content}\n")

def chunk_data(
    chunk_size: int = None, 
    chunk_overlap: int = None
) -> Dict[str, Any]:
    """Main function to chunk data and save results.

    Args:
        chunk_size: Optional override for chunk size
        chunk_overlap: Optional override for chunk overlap

    Returns:
        Dictionary with processing results
    """
    # Use provided values or defaults
    if chunk_size is None:
        chunk_size = 1000
    if chunk_overlap is None:
        chunk_overlap = 200

    # Load and chunk files
    chunks = load_and_chunk_files(chunk_size, chunk_overlap)
    
    # Save results
    save_chunked_data(chunks)
    
    return {
        "total_chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "timestamp": datetime.now().isoformat()
    }

def main():
    """Main function for testing."""
    results = chunk_data()
    print("\nChunking Results:")
    print(f"Total chunks: {results['total_chunks']}")
    print(f"Chunk size: {results['chunk_size']}")
    print(f"Chunk overlap: {results['chunk_overlap']}")
    print(f"Timestamp: {results['timestamp']}")

def chunk_text(
    text: str, 
    source_id: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """Chunk a single text document into overlapping segments.
    
    Args:
        text: Raw text to chunk
        source_id: Unique identifier for the source document
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of chunk dictionaries with standardized format
    """
    if not text or not text.strip():
        return []
    
    # Strip metadata before chunking
    cleaned_text = strip_document_metadata(text)
    
    if not cleaned_text:
        return []
    
    chunks = []
    
    if LANGCHAIN_AVAILABLE:
        # Use langchain text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        
        # Create a temporary document for splitting
        doc = Document(page_content=cleaned_text, metadata={})
        chunk_docs = text_splitter.split_documents([doc])
        
        for idx, chunk_doc in enumerate(chunk_docs):
            # Calculate positions in original cleaned text
            chunk_text = chunk_doc.page_content
            start_pos = cleaned_text.find(chunk_text)
            end_pos = start_pos + len(chunk_text) if start_pos != -1 else len(chunk_text)
            
            chunks.append({
                "id": f"{source_id}_{idx}",
                "text": chunk_text,
                "chunk_idx": idx,
                "start_pos": max(0, start_pos),
                "end_pos": end_pos
            })
    else:
        # Fallback simple text splitting
        words = cleaned_text.split()
        chunk_words = chunk_size // 4  # Rough approximation
        overlap_words = chunk_overlap // 4
        
        for i in range(0, len(words), chunk_words - overlap_words):
            chunk_words_slice = words[i:i + chunk_words]
            chunk_text = " ".join(chunk_words_slice)
            
            # Calculate approximate positions
            start_pos = len(" ".join(words[:i])) + (1 if i > 0 else 0)
            end_pos = start_pos + len(chunk_text)
            
            chunks.append({
                "id": f"{source_id}_{len(chunks)}",
                "text": chunk_text,
                "chunk_idx": len(chunks),
                "start_pos": start_pos,
                "end_pos": end_pos
            })
    
    return chunks

def save_chunked_document_json(
    chunks: List[Dict[str, Any]], 
    source_id: str, 
    output_dir: str = "data/03_intermediate"
) -> str:
    """Save chunks for a single document as JSON.
    
    Args:
        chunks: List of chunk dictionaries
        source_id: Source document identifier
        output_dir: Directory to save the JSON file
        
    Returns:
        Path to the saved JSON file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add metadata to each chunk
    enriched_chunks = []
    for chunk in chunks:
        enriched_chunk = chunk.copy()
        # Extract file information from source_id or chunk data
        source_filename = f"{source_id}.txt"  # default assumption
        file_format = "TXT"  # default
        
        enriched_chunk.update({
            "source_filename": source_filename,
            "file_format": file_format,
            "ingestion_timestamp": datetime.now().isoformat()
        })
        enriched_chunks.append(enriched_chunk)
    
    # Create standardized JSON structure
    json_data = {
        "source_id": source_id,
        "total_chunks": len(chunks),
        "generated_at": datetime.now().isoformat(),
        "chunks": enriched_chunks
    }
    
    # Save to individual JSON file
    json_file = output_path / f"{source_id}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    return str(json_file)

if __name__ == "__main__":
    main() 