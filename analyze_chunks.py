import numpy as np
import os
from collections import Counter
import json
from langchain.schema import Document

def load_chunked_data(json_file='Chuncked_Data/chunked_data_full.json'):
    """
    Load chunked data from JSON file and convert to Document objects.
    
    Args:
        json_file (str): Path to the JSON file containing chunked data
        
    Returns:
        list: List of Document objects
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            chunked_data = json.load(f)
        
        # Convert JSON data to Document objects
        chunked_documents = [
            Document(
                page_content=chunk['content'],
                metadata={'source': chunk['source']}
            ) for chunk in chunked_data
        ]
        return chunked_documents
    except Exception as e:
        print(f"Error loading chunked data: {str(e)}")
        return None

def analyze_chunked_data(chunked_documents):
    """
    Analyze the chunked documents and provide statistical insights.
    
    Args:
        chunked_documents (list): List of chunked Document objects
    """
    if not chunked_documents:
        print("⚠️ No chunked documents provided for analysis.")
        return
    
    print("\n--- Statistical Analysis & Quality Check ---")
    
    # Calculate the lengths of all chunks
    chunk_lengths = [len(chunk.page_content) for chunk in chunked_documents]
    
    # Calculate and print key statistics
    total_chunks = len(chunk_lengths)
    min_size = np.min(chunk_lengths)
    max_size = np.max(chunk_lengths)
    avg_size = np.mean(chunk_lengths)
    std_dev = np.std(chunk_lengths)
    median_size = np.median(chunk_lengths)
    
    print(f"Total Chunks: {total_chunks}")
    print(f"Minimum Chunk Size: {min_size} characters")
    print(f"Maximum Chunk Size: {max_size} characters")
    print(f"Average Chunk Size: {avg_size:.2f} characters")
    print(f"Median Chunk Size: {median_size:.2f} characters")
    print(f"Standard Deviation of Chunk Size: {std_dev:.2f}")
    
    # --- Source Distribution Analysis ---
    source_counts = Counter([chunk.metadata.get('source', 'N/A') for chunk in chunked_documents])
    print("\n--- Source Distribution ---")
    for source, count in source_counts.most_common():
        print(f"{os.path.basename(source)}: {count} chunks ({count/total_chunks*100:.1f}%)")
    
    # --- Automated Quality Feedback ---
    CHUNK_SIZE = 500  # Target chunk size
    
    # 1. Check for high variation in chunk size
    if std_dev > 150:
        print(f"\n[WARNING] High chunk size variation detected (Std Dev: {std_dev:.2f}).")
        print("  > This suggests documents may have irregular structures (e.g., many short lines or lists).")
        print("  > Resulting chunks may have inconsistent levels of context.")
    
    # 2. Check for and count potentially "orphaned" or very small chunks
    small_chunk_threshold = CHUNK_SIZE * 0.20  # Chunks smaller than 20% of the target size
    small_chunk_count = sum(1 for length in chunk_lengths if length < small_chunk_threshold)
    
    if small_chunk_count > 0:
        print(f"\n[ADVISORY] Found {small_chunk_count} chunks smaller than {small_chunk_threshold} characters.")
        print(f"  > The smallest chunk is {min_size} characters.")
        print("  > These small chunks might lack sufficient context and could clutter search results.")
        print("  > Consider cleaning the source documents or adjusting the chunking separators.")
    
    # 3. Check for very large chunks
    large_chunk_threshold = CHUNK_SIZE * 1.5  # Chunks larger than 150% of the target size
    large_chunk_count = sum(1 for length in chunk_lengths if length > large_chunk_threshold)
    
    if large_chunk_count > 0:
        print(f"\n[ADVISORY] Found {large_chunk_count} chunks larger than {large_chunk_threshold} characters.")
        print(f"  > The largest chunk is {max_size} characters.")
        print("  > These large chunks might contain too much information for effective processing.")
    
    # Add a success message if no issues are flagged
    if std_dev <= 150 and small_chunk_count == 0 and large_chunk_count == 0:
        print("\n[INFO] Chunking statistics appear healthy. Sizes are consistent.")
    
    # --- Sample Chunks Preview ---
    print("\n--- Sample Chunks Preview ---")
    for i, chunk in enumerate(chunked_documents[:3]):  # Print first 3 chunks
        chunk_source = os.path.basename(chunk.metadata.get('source', 'N/A'))
        print(f"\n--- Chunk {i+1} (Source: {chunk_source}, Length: {len(chunk.page_content)} chars) ---")
        print(chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content)

def main():
    """
    Main function to run the analysis.
    """
    # Load the chunked data
    chunked_documents = load_chunked_data()
    
    if chunked_documents:
        # Run the analysis
        analyze_chunked_data(chunked_documents)
    else:
        print("⚠️ Failed to load chunked data. Please check the JSON file path.")

if __name__ == "__main__":
    main() 