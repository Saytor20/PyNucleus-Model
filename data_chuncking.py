import os
import json
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_and_chunk_files():
    """
    Load and chunk files from both data_sources and processed_txt_files directories
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_documents = []
    
    # Process files from both directories
    directories = ['data_sources', 'processed_txt_files']
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"⚠️ Directory {directory} not found")
            continue
            
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        # Create Document object with metadata
                        doc = Document(
                            page_content=text,
                            metadata={"source": file_path}
                        )
                        all_documents.append(doc)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    print(f"\nLoaded {len(all_documents)} documents for chunking")
    
    # Split documents into chunks
    chunked_documents = text_splitter.split_documents(all_documents)
    print(f"Split into {len(chunked_documents)} chunks")
    
    return chunked_documents

def save_chunked_data(chunked_documents, output_dir="Chuncked_Data"):
    """
    Save chunked documents into three separate files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save full content with metadata
    full_content = []
    for i, chunk in enumerate(chunked_documents):
        full_content.append({
            "chunk_id": i,
            "content": chunk.page_content,
            "source": chunk.metadata.get('source', 'N/A'),
            "length": len(chunk.page_content)
        })
    
    with open(os.path.join(output_dir, "chunked_data_full.json"), 'w', encoding='utf-8') as f:
        json.dump(full_content, f, indent=2, ensure_ascii=False)
    
    # 2. Save statistical analysis
    stats = {
        "total_chunks": len(chunked_documents),
        "chunk_lengths": [len(chunk.page_content) for chunk in chunked_documents],
        "sources": list(set(chunk.metadata.get('source', 'N/A') for chunk in chunked_documents)),
        "generated_at": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "chunked_data_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # 3. Save content-only version (for easy reading)
    with open(os.path.join(output_dir, "chunked_data_content.txt"), 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunked_documents):
            f.write(f"=== Chunk {i+1} ===\n")
            f.write(f"Source: {chunk.metadata.get('source', 'N/A')}\n")
            f.write(f"Length: {len(chunk.page_content)} characters\n")
            f.write("\nContent:\n")
            f.write(chunk.page_content)
            f.write("\n\n" + "="*50 + "\n\n")
    
    print(f"\n✅ Successfully saved chunked data to {output_dir}/:")
    print(f"  • chunked_data_full.json - Complete data with metadata")
    print(f"  • chunked_data_stats.json - Statistical analysis")
    print(f"  • chunked_data_content.txt - Human-readable content")

def main():
    """Example usage of the data processor."""
    # Load and chunk the documents
    chunked_docs = load_and_chunk_files()
    
    # Save the chunked data
    save_chunked_data(chunked_docs)

if __name__ == "__main__":
    main() 