#!/usr/bin/env python3
"""
Document Processor for RAG Pipeline

Handles document loading, conversion, and preprocessing for the PyNucleus RAG system.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import json
import logging
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional

# Try to import unstructured
try:
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("Warning: unstructured not available. Document processing limited.")

# Try to import docx
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("Warning: python-docx not available. DOCX processing limited.")

# Try to import other document processors
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        import PyPDF2
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False
        print("Warning: PDF processing not available.")

# Import from absolute paths instead of relative
try:
    from pynucleus.rag.config import RAGConfig
except ImportError:
    # Fallback config
    class RAGConfig:
        def __init__(self):
            self.input_dir = "data/01_raw/source_documents"
            self.output_dir = "data/02_processed/converted_to_txt"

warnings.filterwarnings("ignore")

from langchain_unstructured import UnstructuredLoader
from PyPDF2 import PdfReader
from tqdm import tqdm

# Handle config import with fallback
try:
    from .config import SOURCE_DOCS_DIR, CONVERTED_DIR
except ImportError:
    # Fallback configuration
    SOURCE_DOCS_DIR = "data/01_raw/source_docs"
    CONVERTED_DIR = "data/02_intermediate/converted_docs"

from docx import Document

def process_documents(
    input_dir: str = SOURCE_DOCS_DIR,
    output_dir: str = CONVERTED_DIR,
    use_progress_bar: bool = True,
) -> None:
    """
    Process all documents in the input directory and convert them to text files.
    Handles PDF, DOCX, TXT, and other file types.
    """
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"📂 Creating directory: '{input_dir}'")
        os.makedirs(input_dir, exist_ok=True)
        print(
            f"ℹ Please place your files (PDF, DOCX, TXT, etc.) in the '{input_dir}' directory and run the script again."
        )
        return

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    files_to_process = [
        f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))
    ]

    if not files_to_process:
        print(f"ℹ The '{input_dir}' directory is empty. Nothing to process.")
        return

    print(
        f"--- 📄 Starting processing for {len(files_to_process)} file(s) in '{input_dir}' ---"
    )

    for filename in tqdm(
        files_to_process, desc="Processing files", disable=not use_progress_bar
    ):
        # Skip hidden files like .DS_Store
        if filename.startswith("."):
            continue

        input_path = os.path.join(input_dir, filename)
        # Use the original filename without "processed_" prefix for cleaner names
        output_filename = f"{os.path.splitext(os.path.basename(filename))[0]}.txt"
        output_path = os.path.join(output_dir, output_filename)

        print(f" ▶ Processing: {filename}")

        try:
            # Handle PDF files differently
            if filename.lower().endswith(".pdf"):
                # Use PyPDF2 for PDF files
                reader = PdfReader(input_path)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() + "\n\n"
            elif filename.lower().endswith(".docx"):
                # Use docx for DOCX files
                doc = Document(input_path)
                full_text = "\n\n".join([para.text for para in doc.paragraphs])
            else:
                # Use UnstructuredLoader for other file types
                loader = UnstructuredLoader(input_path)
                documents = loader.load()
                full_text = "\n\n".join([doc.page_content for doc in documents])

            # Save the extracted text to a new .txt file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            print(f"   • Success! Saved to: {output_path}")

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

    print(f"\n All files processed.")


def main():
    """Example usage of the document processor."""
    process_documents()


if __name__ == "__main__":
    main()
