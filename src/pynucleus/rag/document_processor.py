"""
Document processor for PyNucleus RAG system.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

class DocumentProcessor:
    """Process documents for RAG system indexing."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processing results
        """
        try:
            # Read document content
            content = self._read_document(file_path)
            
            # Create chunks
            chunks = self._create_chunks(content)
            
            return {
                "file_path": str(file_path),
                "status": "success",
                "chunk_count": len(chunks),
                "content_length": len(content),
                "chunks": chunks,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process document {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "status": "error",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }
    
    def _read_document(self, file_path: Path) -> str:
        """Read document content based on file type."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif suffix in ['.md', '.markdown']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # For other file types, return a placeholder
            return f"Document: {file_path.name}\n[Content processing not implemented for {suffix} files]"
    
    def _create_chunks(self, content: str) -> List[Dict[str, Any]]:
        """Split content into chunks."""
        if not content:
            return []
        
        chunks = []
        words = content.split()
        
        # Simple word-based chunking
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                "chunk_id": len(chunks),
                "text": chunk_text,
                "word_count": len(chunk_words),
                "start_position": i,
                "end_position": min(i + self.chunk_size, len(words))
            })
        
        return chunks 