"""Data processor for document chunking and management."""
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_files(directory="processed_txt_files"):
    """Load and chunk documents from the processed files directory."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as f:
                text = f.read()
                chunks.extend(text_splitter.split_text(text))
    
    return chunks

def save_chunked_data(chunks, output_dir="Chuncked_Data"):
    """Save chunked data to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full data
    with open(os.path.join(output_dir, "chunked_data_full.json"), 'w') as f:
        json.dump({"chunks": chunks}, f)
    
    # Save stats
    stats = {
        "total_chunks": len(chunks),
        "chunk_lengths": [len(chunk) for chunk in chunks],
        "sources": ["processed_txt_files"] * len(chunks)
    }
    with open(os.path.join(output_dir, "chunked_data_stats.json"), 'w') as f:
        json.dump(stats, f) 