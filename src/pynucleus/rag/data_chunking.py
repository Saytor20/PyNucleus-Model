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
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Try to import langchain components
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
        full_data.append(
            {
                "chunk_id": i,
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "length": len(chunk.page_content),
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

    print(f"✅ Successfully saved chunked data to {output_dir}/:")
    print(f"  • chunked_data_full.json - Complete data with metadata")
    print(f"  • chunked_data_stats.json - Statistical analysis")
    print(f"  • chunked_data_content.txt - Human-readable content\n")


def chunk_data(
    chunk_size: int = None, 
    chunk_overlap: int = None
) -> Dict[str, Any]:
    """
    Main data chunking pipeline.
    
    Args:
        chunk_size: Size of text chunks (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
        
    Returns:
        Dictionary with chunking results and statistics
    """
    start_time = datetime.now()
    print("Step 3: Processing and chunking documents...")
    
    # Initialize config
    rag_config = RAGConfig()
    
    # Use provided values or defaults from config
    if chunk_size is None:
        chunk_size = getattr(rag_config, 'chunk_size', 1000)
    if chunk_overlap is None:
        chunk_overlap = getattr(rag_config, 'chunk_overlap', 200)
    
    # Step 1: Load Wikipedia articles
    web_sources_dir = Path(getattr(rag_config, 'web_sources_dir', 'data/01_raw/web_sources'))
    wiki_docs = []
    try:
        if web_sources_dir.exists():
            wiki_files = list(web_sources_dir.glob("*.txt"))
            print(f"📰 Found {len(wiki_files)} Wikipedia articles")
            
            for file in wiki_files:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if LANGCHAIN_AVAILABLE:
                        doc = Document(page_content=content, metadata={"source": str(file)})
                    else:
                        doc = Document(page_content=content, metadata={"source": str(file)})
                    wiki_docs.append(doc)
        else:
            print(f"⚠️ Web sources directory not found: {web_sources_dir}")
    except Exception as e:
        print(f"⚠️ Error loading Wikipedia articles: {e}")
    
    # Step 2: Load converted documents 
    converted_dir = Path(getattr(rag_config, 'converted_dir', 'data/02_processed/converted_to_txt'))
    converted_docs = []
    try:
        if converted_dir.exists():
            converted_files = list(converted_dir.glob("*.txt"))
            print(f"📄 Found {len(converted_files)} converted documents")
            
            for file in converted_files:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if LANGCHAIN_AVAILABLE:
                        doc = Document(page_content=content, metadata={"source": str(file)})
                    else:
                        doc = Document(page_content=content, metadata={"source": str(file)})
                    converted_docs.append(doc)
        else:
            print(f"⚠️ Converted documents directory not found: {converted_dir}")
    except Exception as e:
        print(f"⚠️ Error loading converted documents: {e}")
    
    # Combine all documents
    all_docs = wiki_docs + converted_docs
    print(f"📋 Total documents loaded: {len(all_docs)}")
    
    # Step 3: Split documents into chunks
    if LANGCHAIN_AVAILABLE:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(all_docs)
    else:
        # Fallback simple text splitting
        chunks = []
        for doc in all_docs:
            text = doc.page_content
            words = text.split()
            chunk_words = chunk_size // 4  # Rough approximation
            
            for i in range(0, len(words), chunk_words - (chunk_overlap // 4)):
                chunk_text = " ".join(words[i:i + chunk_words])
                chunk = Document(page_content=chunk_text, metadata=doc.metadata.copy())
                chunks.append(chunk)
    
    print(f"✂️ Split into {len(chunks)} chunks")
    
    # Save chunked data
    output_dir = Path(getattr(rag_config, 'chunked_data_dir', 'data/03_intermediate/converted_chunked_data'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return save_chunked_data(chunks, str(output_dir))


def main():
    """Example usage of the data processor."""
    # Load and chunk the documents
    chunked_docs = load_and_chunk_files()

    # Save the chunked data
    save_chunked_data(chunked_docs)


if __name__ == "__main__":
    main()
