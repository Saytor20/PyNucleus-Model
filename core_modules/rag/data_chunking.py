import os
import json
import warnings
from pathlib import Path
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List

warnings.filterwarnings("ignore")

def load_and_chunk_files(chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """Load and chunk documents from both web_sources and converted_to_txt directories.
    
    Args:
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects with metadata
    """
    docs = []
    
    # Load Wikipedia articles from web_sources
    web_sources_dir = Path("web_sources")
    if web_sources_dir.exists():
        wiki_files = list(web_sources_dir.glob("wikipedia_*.txt"))
        print(f"📰 Found {len(wiki_files)} Wikipedia articles")
        
        for file_path in wiki_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "type": "wikipedia",
                    "title": file_path.stem.replace("wikipedia_", "").replace("_", " ").title()
                }
            )
            docs.append(doc)
    else:
        print("📰 Found 0 Wikipedia articles (web_sources directory not found)")
    
    # Load processed text files from converted_to_txt directory
    converted_dir = Path("converted_to_txt")
    if converted_dir.exists():
        converted_files = list(converted_dir.glob("*.txt"))
        print(f"📄 Found {len(converted_files)} converted documents")
        
        for file_path in converted_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "type": "document",
                    "title": file_path.stem.replace("_", " ").title()
                }
            )
            docs.append(doc)
    else:
        print("📄 Found 0 converted documents (converted_to_txt directory not found)")
    
    if not docs:
        print("⚠️ No documents found to process")
        return docs
    
    print(f"📋 Total documents loaded: {len(docs)}")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"✂️ Split into {len(chunks)} chunks\n")
    
    return chunks

def save_chunked_data(chunks: List[Document], output_dir: str = "converted_chunked_data") -> None:
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
        full_data.append({
            "chunk_id": i,
            "content": chunk.page_content,
            "source": chunk.metadata.get("source", "unknown"),
            "length": len(chunk.page_content)
        })
    
    with open(output_path / "chunked_data_full.json", 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    
    # Save statistics (avoid division by zero)
    stats = {
        "total_chunks": len(chunks),
        "sources": list(set(chunk.metadata.get("source", "unknown") for chunk in chunks)),
        "avg_chunk_size": sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0,
        "generated_at": datetime.now().isoformat()
    }
    
    with open(output_path / "chunked_data_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Save human-readable content
    with open(output_path / "chunked_data_content.txt", 'w', encoding='utf-8') as f:
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

def main():
    """Example usage of the data processor."""
    # Load and chunk the documents
    chunked_docs = load_and_chunk_files()
    
    # Save the chunked data
    save_chunked_data(chunked_docs)

if __name__ == "__main__":
    main() 