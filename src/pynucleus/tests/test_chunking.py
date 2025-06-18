#!/usr/bin/env python3
"""
Tests for data chunking functionality.
"""

import sys
import os
from pathlib import Path
import pytest
import json
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from pynucleus.rag.data_chunking import chunk_text, save_chunked_document_json
from pynucleus.rag.document_processor import strip_document_metadata


class TestMetadataStripping:
    """Test metadata stripping functionality."""
    
    def test_strip_metadata(self):
        """Test that metadata artifacts are properly removed."""
        # Sample text with various metadata artifacts
        sample_text = """Research Paper Title
        
Author Name
DEPARTMENT OF ENGINEERING
University of Example
author@university.edu

Page 1 of 10

Introduction

This is the main content of the document. It contains valuable information
that should be preserved during the chunking process.

The content continues here with more substantial text that represents
the actual document content rather than metadata.

12

More content here that should be kept.

Page 2 of 10

Additional content that forms the body of the document.

UNIVERSITY OF EXAMPLE

Final paragraph with important information.
"""
        
        cleaned_text = strip_document_metadata(sample_text)
        
        # Check that metadata artifacts are removed
        assert "Research Paper Title" not in cleaned_text
        assert "Author Name" not in cleaned_text
        assert "DEPARTMENT OF ENGINEERING" not in cleaned_text
        assert "University of Example" not in cleaned_text
        assert "author@university.edu" not in cleaned_text
        assert "Page 1 of 10" not in cleaned_text
        assert "Page 2 of 10" not in cleaned_text
        assert "UNIVERSITY OF EXAMPLE" not in cleaned_text
        
        # Check that main content is preserved
        assert "Introduction" in cleaned_text
        assert "This is the main content" in cleaned_text
        assert "valuable information" in cleaned_text
        assert "Additional content" in cleaned_text
        assert "Final paragraph" in cleaned_text
    
    def test_strip_metadata_empty_input(self):
        """Test handling of empty or None input."""
        assert strip_document_metadata("") == ""
        assert strip_document_metadata(None) == None
        assert strip_document_metadata("   ") == ""
    
    def test_strip_metadata_no_artifacts(self):
        """Test that clean text is preserved."""
        clean_text = """This is a clean document with no metadata artifacts.

It contains multiple paragraphs of useful content.

All of this should be preserved in the output."""
        
        result = strip_document_metadata(clean_text)
        assert result.strip() == clean_text.strip()


class TestChunking:
    """Test text chunking functionality."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking functionality."""
        text = "This is a sample text. " * 100  # Create text that will need chunking
        source_id = "test_doc"
        
        chunks = chunk_text(text, source_id, chunk_size=200, chunk_overlap=50)
        
        # Basic assertions
        assert len(chunks) > 1  # Should create multiple chunks
        assert all(chunk["id"].startswith(source_id) for chunk in chunks)
        assert all("text" in chunk for chunk in chunks)
        assert all("chunk_idx" in chunk for chunk in chunks)
        assert all("start_pos" in chunk for chunk in chunks)
        assert all("end_pos" in chunk for chunk in chunks)
        
        # Check chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_idx"] == i
    
    def test_chunk_text_with_metadata(self):
        """Test that chunking strips metadata first."""
        text_with_metadata = """Document Title
        
        Author Name
        Department of Computer Science
        
        Page 1 of 5
        
        This is the actual content that should be chunked.
        It contains valuable information spread across multiple sentences.
        """ + "More content here. " * 50
        
        source_id = "test_metadata_doc"
        chunks = chunk_text(text_with_metadata, source_id, chunk_size=100, chunk_overlap=20)
        
        # Check that metadata was stripped from chunks
        all_chunk_text = " ".join(chunk["text"] for chunk in chunks)
        assert "Document Title" not in all_chunk_text
        assert "Author Name" not in all_chunk_text
        assert "Department of Computer Science" not in all_chunk_text
        assert "Page 1 of 5" not in all_chunk_text
        
        # Check that content was preserved
        assert "actual content" in all_chunk_text
        assert "valuable information" in all_chunk_text
    
    def test_chunk_text_empty_input(self):
        """Test handling of empty input."""
        assert chunk_text("", "test") == []
        assert chunk_text("   ", "test") == []
        assert chunk_text(None, "test") == []
    
    def test_chunk_text_positions(self):
        """Test that start_pos and end_pos are calculated correctly."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, "test", chunk_size=20, chunk_overlap=5)
        
        for chunk in chunks:
            assert chunk["start_pos"] >= 0
            assert chunk["end_pos"] > chunk["start_pos"]
            assert chunk["end_pos"] <= len(text)


class TestJSONOutput:
    """Test JSON output functionality."""
    
    def test_save_chunked_document_json(self):
        """Test saving chunks as JSON."""
        chunks = [
            {
                "id": "test_0",
                "text": "First chunk text",
                "chunk_idx": 0,
                "start_pos": 0,
                "end_pos": 16
            },
            {
                "id": "test_1", 
                "text": "Second chunk text",
                "chunk_idx": 1,
                "start_pos": 10,
                "end_pos": 27
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = save_chunked_document_json(chunks, "test_doc", temp_dir)
            
            assert os.path.exists(json_file)
            assert json_file.endswith("test_doc.json")
            
            # Load and verify JSON structure
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            assert data["source_id"] == "test_doc"
            assert data["total_chunks"] == 2
            assert "generated_at" in data
            assert "chunks" in data
            assert len(data["chunks"]) == 2
            
            # Verify chunk structure
            for i, chunk in enumerate(data["chunks"]):
                assert chunk["id"] == f"test_{i}"
                assert "text" in chunk
                assert "chunk_idx" in chunk
                assert "start_pos" in chunk
                assert "end_pos" in chunk


class TestIntegration:
    """Integration tests for the complete chunking pipeline."""
    
    def test_full_pipeline(self):
        """Test the complete pipeline from text to JSON."""
        # Sample document with metadata
        document_text = """Scientific Paper Title
        
        Dr. Jane Smith
        MASSACHUSETTS INSTITUTE OF TECHNOLOGY
        jane.smith@mit.edu
        
        Page 1 of 8
        
        Abstract
        
        This research investigates the effects of various parameters on system performance.
        The study utilizes advanced computational methods to analyze complex datasets.
        Results demonstrate significant improvements in efficiency when applying the proposed methodology.
        
        Introduction
        
        In recent years, there has been growing interest in optimizing system performance through
        innovative approaches. This paper presents a comprehensive analysis of current methodologies
        and proposes a novel framework for addressing existing limitations.
        
        The main contributions of this work include: detailed analysis of existing systems,
        development of improved algorithms, and experimental validation of the proposed approach.
        
        Page 2 of 8
        
        Methodology
        
        Our approach consists of three main phases: data collection, algorithm development,
        and performance evaluation. Each phase is designed to build upon the previous one,
        ensuring a systematic progression toward the final solution.
        """
        
        # Chunk the document
        source_id = "scientific_paper"
        chunks = chunk_text(document_text, source_id, chunk_size=300, chunk_overlap=50)
        
        # Save as JSON
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = save_chunked_document_json(chunks, source_id, temp_dir)
            
            # Verify the results
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check that chunks don't contain metadata
            all_text = " ".join(chunk["text"] for chunk in data["chunks"])
            assert "Scientific Paper Title" not in all_text
            assert "Dr. Jane Smith" not in all_text
            assert "MASSACHUSETTS INSTITUTE OF TECHNOLOGY" not in all_text
            assert "jane.smith@mit.edu" not in all_text
            assert "Page 1 of 8" not in all_text
            assert "Page 2 of 8" not in all_text
            
            # Check that content is preserved
            assert "Abstract" in all_text
            assert "research investigates" in all_text
            assert "Introduction" in all_text
            assert "Methodology" in all_text
            assert "three main phases" in all_text
            
            # Check JSON structure
            assert data["source_id"] == source_id
            assert data["total_chunks"] > 0
            assert len(data["chunks"]) == data["total_chunks"]


if __name__ == "__main__":
    pytest.main([__file__]) 