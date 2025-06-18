#!/usr/bin/env python3
"""
Chunk Demo Script

Demonstrates the refined chunking functionality with metadata stripping and JSON output.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pynucleus.rag.data_chunking import chunk_text, save_chunked_document_json
from pynucleus.rag.document_processor import strip_document_metadata


def create_sample_json():
    """Create a sample JSON record for demonstration."""
    sample_text = """Research on Modular Chemical Plants
    
    Dr. Sarah Wilson
    DEPARTMENT OF CHEMICAL ENGINEERING  
    MIT Institute of Technology
    sarah.wilson@mit.edu
    
    Page 1 of 15
    
    Abstract
    
    This study investigates the potential of modular chemical plant designs to reduce
    barriers to industrialization in developing regions. Modular plants offer several
    advantages including reduced capital costs, faster deployment, and improved
    operational flexibility compared to traditional large-scale facilities.
    
    Introduction
    
    The chemical industry has traditionally relied on large-scale, centralized
    manufacturing facilities to achieve economies of scale. However, this approach
    presents significant challenges for developing regions, including high capital
    requirements, complex infrastructure needs, and lengthy project timelines.
    
    Modular design represents a paradigm shift toward smaller, standardized units
    that can be manufactured off-site and rapidly deployed. This approach offers
    the potential to democratize access to chemical manufacturing capabilities.
    
    Page 2 of 15
    
    Literature Review
    
    Previous studies have examined various aspects of modular plant design,
    including economic considerations, technical feasibility, and operational
    challenges. Smith et al. (2019) demonstrated that modular plants can achieve
    cost parity with traditional facilities at smaller production scales.
    
    Methodology
    
    Our analysis employs a multi-criteria decision framework that considers
    technical, economic, and social factors. We evaluate modular plant designs
    across three key dimensions: capital efficiency, operational flexibility,
    and deployment speed.
    
    The study examines five different chemical processes: ethanol production,
    fertilizer synthesis, pharmaceutical intermediates, polymer manufacturing,
    and specialty chemicals production.
    """
    
    # Create sample directory structure
    sample_dir = Path("data/03_intermediate")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as sample record
    sample_file = sample_dir / "sample_record.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump({"content": sample_text, "title": "Modular Plants Research"}, f, indent=2)
    
    print(f"✅ Created sample record at: {sample_file}")
    return sample_file


def demo_chunking(json_file_path: str):
    """Demonstrate chunking functionality with a JSON record."""
    print(f"🔄 Processing: {json_file_path}")
    
    # Load the JSON record
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            record = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON file: {json_file_path}")
        return
    
    # Extract text content
    text = record.get("content", "")
    if not text:
        print("❌ No 'content' field found in JSON record")
        return
    
    title = record.get("title", "unknown_document")
    source_id = Path(json_file_path).stem
    
    print(f"📄 Document: {title}")
    print(f"🆔 Source ID: {source_id}")
    print(f"📏 Original length: {len(text)} characters")
    
    # Show original text sample (first 300 chars)
    print("\n" + "="*60)
    print("📖 ORIGINAL TEXT (first 300 chars):")
    print("="*60)
    print(text[:300] + "..." if len(text) > 300 else text)
    
    # Strip metadata
    cleaned_text = strip_document_metadata(text)
    print(f"\n🧹 After metadata stripping: {len(cleaned_text)} characters")
    
    # Show cleaned text sample
    print("\n" + "="*60)
    print("✨ CLEANED TEXT (first 300 chars):")
    print("="*60)
    print(cleaned_text[:300] + "..." if len(cleaned_text) > 300 else cleaned_text)
    
    # Chunk the text
    chunks = chunk_text(text, source_id, chunk_size=400, chunk_overlap=100)
    print(f"\n✂️ Created {len(chunks)} chunks")
    
    # Save as JSON
    json_output = save_chunked_document_json(chunks, source_id)
    print(f"💾 Saved chunks to: {json_output}")
    
    # Display first two chunks
    print("\n" + "="*60)
    print("📋 FIRST TWO CHUNKS:")
    print("="*60)
    
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {chunk['id']}")
        print(f"Position: {chunk['start_pos']}-{chunk['end_pos']}")
        print(f"Length: {len(chunk['text'])} characters")
        print(f"Text: {chunk['text'][:200]}{'...' if len(chunk['text']) > 200 else ''}")
    
    if len(chunks) > 2:
        print(f"\n... and {len(chunks) - 2} more chunks")
    
    return chunks


def show_recent_logs():
    """Display recent ingestion log entries."""
    log_file = Path("logs/ingestion.log")
    
    if not log_file.exists():
        print("❌ No ingestion log file found at logs/ingestion.log")
        return
    
    print("\n" + "="*60)
    print("📋 RECENT INGESTION LOG ENTRIES (last 20 lines):")
    print("="*60)
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Get last 20 lines
        recent_lines = lines[-20:] if len(lines) > 20 else lines
        
        for line in recent_lines:
            line = line.strip()
            if line:  # Skip empty lines
                print(line)
                
    except Exception as e:
        print(f"❌ Error reading log file: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Demo chunking functionality")
    parser.add_argument(
        "json_file", 
        nargs="?", 
        help="Path to JSON record file (optional - will create sample if not provided)"
    )
    parser.add_argument(
        "--create-sample", 
        action="store_true",
        help="Create a sample JSON record and exit"
    )
    parser.add_argument(
        "--log", 
        action="store_true",
        help="Display recent ingestion log entries after demo"
    )
    
    args = parser.parse_args()
    
    print("🎯 PyNucleus Chunking Demo")
    print("=" * 50)
    
    if args.create_sample:
        create_sample_json()
        return
    
    json_file = args.json_file
    if not json_file:
        # Create sample and use it
        print("No JSON file provided. Creating sample...")
        json_file = create_sample_json()
        print()
    
    # Run the demo
    chunks = demo_chunking(str(json_file))
    
    if chunks:
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"✂️ Generated {len(chunks)} chunks")
        print(f"📁 Check data/03_intermediate/ for output files")
        print("\n💡 Tips:")
        print("  - Metadata (titles, authors, page numbers) were automatically stripped")
        print("  - Each chunk has unique ID, position info, and standardized JSON format")
        print("  - Use this output for vector database ingestion or LLM processing")
    else:
        print("\n❌ Demo failed - no chunks generated")
    
    # Show recent logs if requested
    if args.log:
        show_recent_logs()


if __name__ == "__main__":
    main() 