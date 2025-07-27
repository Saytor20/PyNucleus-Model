"""
Document processor for PyNucleus RAG system with PDF table extraction.
"""

import logging
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

class DocumentProcessor:
    """Process documents for RAG system indexing with automatic PDF table extraction."""
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 150, tables_output_dir: str = "data/02_processed/tables"):
        # Updated chunk size to match Step 4 requirements
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tables_output_dir = Path(tables_output_dir)
        self.tables_output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Table detection keywords
        self.table_keywords = [
            'table', 'data', 'figure', 'chart', 'matrix', 'schedule', 
            'list', 'inventory', 'summary', 'comparison', 'analysis',
            'results', 'specifications', 'parameters', 'properties'
        ]
        
        # Section header patterns for structure detection
        self.section_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Z\s]{3,}:?\s*)$',  # ALL CAPS headers
            r'^(\d+\.?\s+[A-Z].+)$',  # Numbered sections
            r'^([A-Z][a-z\s]{5,}:?\s*)$',  # Title case headers
            r'^(\*+\s+.+)$',  # Bullet point headers
            r'^(Chapter\s+\d+.*)$',  # Chapter headers
            r'^(Section\s+\d+.*)$',  # Section headers
            r'^(Part\s+[IVX]+.*)$',  # Part headers (Roman numerals)
        ]
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single document with automatic PDF table extraction and enhanced chunking.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processing results including extracted tables and enriched chunks
        """
        try:
            result = {
                "file_path": str(file_path),
                "status": "success",
                "processed_at": datetime.now().isoformat(),
                "tables_extracted": 0,
                "table_files": []
            }
            
            # Handle PDF files with text extraction and table extraction
            if file_path.suffix.lower() == '.pdf':
                # Extract full text content from PDF
                content = self._extract_pdf_text(file_path)
                
                # Also extract tables separately
                table_result = self._extract_pdf_tables(file_path)
                result.update(table_result)
                
                # If no text content was extracted, fall back to table summary
                if not content.strip():
                    content = self._create_table_summary(result["table_files"])
            else:
                # Read document content for non-PDF files
                content = self._read_document(file_path)
            
            # Save cleaned text to disk before chunking
            self._save_cleaned_text(content, file_path)
            
            # Enhanced chunking with metadata enrichment
            chunks = self._create_enhanced_chunks(content, file_path)
            result.update({
                "chunk_count": len(chunks),
                "content_length": len(content),
                "chunks": chunks
            })
            
            return result
            
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
            # Unsupported file type
            return f"Document: {file_path.name}\n[Content processing not implemented for {suffix} files]"
    
    def _save_cleaned_text(self, content: str, file_path: Path) -> None:
        """Save cleaned text content to the processed directory."""
        try:
            # Create output directory structure
            cleaned_txt_dir = Path("data/02_processed/cleaned_txt")
            cleaned_txt_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            output_filename = f"{file_path.stem}.txt"
            output_path = cleaned_txt_dir / output_filename
            
            # Clean and format the content for saving
            cleaned_content = self._clean_text_for_saving(content)
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            self.logger.info(f"Saved cleaned text to: {output_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save cleaned text for {file_path.name}: {e}")
    
    def _clean_text_for_saving(self, content: str) -> str:
        """Clean and format text content for saving to disk."""
        if not content:
            return ""
        
        # Basic text cleaning
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                # Remove excessive whitespace
                line = re.sub(r'\s+', ' ', line)
                # Remove special characters that might cause issues
                line = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/\&\%\$\#\@\+\=\<\>]', '', line)
                cleaned_lines.append(line)
        
        # Join with single newlines and add metadata header
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Add processing metadata header
        header = f"# Processed Document\n"
        header += f"# Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"# Content Length: {len(cleaned_text)} characters\n"
        header += f"# Word Count: {len(cleaned_text.split())} words\n"
        header += "# " + "="*50 + "\n\n"
        
        return header + cleaned_text
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text content from PDF using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():  # Only add non-empty pages
                    full_text += f"\n--- Page {page_num + 1} ---\n"
                    full_text += text + "\n"
            
            doc.close()
            
            if full_text.strip():
                self.logger.info(f"Extracted {len(full_text)} characters from PDF: {pdf_path.name}")
                return full_text
            else:
                self.logger.warning(f"No text content extracted from PDF: {pdf_path.name}")
                return ""
                
        except ImportError:
            self.logger.warning("PyMuPDF not available for PDF text extraction")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
            return ""
    
    def _create_enhanced_chunks(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """
        Enhanced chunking with structure awareness and metadata enrichment.
        
        Args:
            content: Document content
            file_path: Path to source file
            
        Returns:
            List of chunks with enriched metadata
        """
        if not content:
            return []
        
        try:
            # First, extract document structure
            sections = self._extract_document_structure(content)
            
            chunks = []
            current_section = None
            current_page = 1
            
            for section in sections:
                section_text = section['content']
                section_header = section['header']
                
                # Update current section context
                current_section = section_header
                
                # Split section into chunks using improved chunking strategy
                section_chunks = self._chunk_text_smartly(section_text)
                
                for chunk_idx, chunk_text in enumerate(section_chunks):
                    # Estimate page number (rough approximation: 500 words per page)
                    total_words_so_far = sum(chunk.get('word_count', 0) for chunk in chunks) if chunks else 0
                    estimated_page = max(1, (total_words_so_far // 500) + 1)
                    
                    # Create enriched chunk metadata
                    chunk_metadata = {
                        "chunk_id": len(chunks),
                        "text": chunk_text,
                        "word_count": len(chunk_text.split()),
                        "source_file": file_path.name,
                        "source_path": str(file_path),
                        "section_header": current_section,
                        "section_index": section['index'],
                        "chunk_index_in_section": chunk_idx,
                        "estimated_page": estimated_page,
                        "document_type": self._detect_document_type(file_path, content),
                        "chunk_type": self._classify_chunk_content(chunk_text),
                        "character_count": len(chunk_text),
                        "contains_technical_terms": self._contains_technical_terms(chunk_text),
                        "readability_score": self._calculate_readability_score(chunk_text)
                    }
                    
                    chunks.append(chunk_metadata)
            
            return chunks
            
        except Exception as e:
            self.logger.warning(f"Enhanced chunking failed for {file_path}: {e}. Using fallback chunking.")
            # Fallback to legacy chunking method
            return self._create_chunks(content)
    
    def _extract_document_structure(self, content: str) -> List[Dict[str, Any]]:
        """Extract document structure by identifying sections and headers."""
        lines = content.split('\n')
        sections = []
        current_section = {"header": "Introduction", "content": "", "index": 0}
        
        for line in lines:
            line = line.strip()
            if not line:
                current_section["content"] += "\n"
                continue
                
            # Check if line is a section header
            header = self._detect_section_header(line)
            if header:
                # Save current section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "header": header,
                    "content": "",
                    "index": len(sections)
                }
            else:
                current_section["content"] += line + "\n"
        
        # Add the last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        # If no sections were detected, treat entire document as one section
        if not sections:
            sections = [{
                "header": "Main Content",
                "content": content,
                "index": 0
            }]
        
        return sections
    
    def _detect_section_header(self, line: str) -> Optional[str]:
        """Detect if a line is a section header using various patterns."""
        for pattern in self.section_patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                # Extract the header text (remove markdown symbols, numbers, etc.)
                header = match.group(1).strip()
                # Clean up common header artifacts
                header = re.sub(r'^#+\s*', '', header)  # Remove markdown #
                header = re.sub(r'^[\d\.]+\s*', '', header)  # Remove leading numbers
                header = re.sub(r'[:\*]+$', '', header)  # Remove trailing : or *
                return header.strip()
        return None
    
    def _chunk_text_smartly(self, text: str) -> List[str]:
        """
        Improved chunking strategy that respects sentence boundaries and semantic units.
        """
        if not text.strip():
            return []
        
        # Split into sentences while preserving structure
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If adding this sentence would exceed chunk size
            if current_word_count + sentence_word_count > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_words = self._get_overlap_words(current_chunk)
                current_chunk = overlap_words + [sentence]
                current_word_count = len(' '.join(current_chunk).split())
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks if chunks else [text]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using improved regex patterns."""
        # Handle common abbreviations and edge cases
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|etc|vs|e\.g|i\.e)\.\s+', r'\1<DOT> ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore abbreviation dots and clean up
        sentences = [re.sub(r'<DOT>', '.', s.strip()) for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_words(self, current_chunk: List[str]) -> List[str]:
        """Get words for overlap between chunks."""
        if not current_chunk:
            return []
        
        # Get last few sentences for overlap
        chunk_text = ' '.join(current_chunk)
        words = chunk_text.split()
        
        # Take last N words for overlap, but don't exceed overlap limit
        overlap_word_count = min(self.chunk_overlap, len(words) // 2)
        return [' '.join(words[-overlap_word_count:])] if overlap_word_count > 0 else []
    
    def _detect_document_type(self, file_path: Path, content: str) -> str:
        """Detect document type based on filename and content."""
        filename = file_path.name.lower()
        content_lower = content.lower()
        
        # Check filename patterns
        if any(term in filename for term in ['readme', 'guide', 'manual']):
            return 'documentation'
        elif any(term in filename for term in ['spec', 'requirement']):
            return 'specification'
        elif any(term in filename for term in ['report', 'analysis']):
            return 'report'
        elif file_path.suffix.lower() == '.md':
            return 'markdown'
        elif file_path.suffix.lower() == '.pdf':
            return 'pdf'
        
        # Check content patterns
        if re.search(r'abstract|introduction|methodology|results|conclusion', content_lower):
            return 'academic_paper'
        elif re.search(r'function|class|import|def |return ', content_lower):
            return 'code_documentation'
        else:
            return 'general'
    
    def _classify_chunk_content(self, chunk_text: str) -> str:
        """Classify the type of content in a chunk."""
        text_lower = chunk_text.lower()
        
        if re.search(r'table|figure|chart|graph|diagram', text_lower):
            return 'table_or_figure'
        elif re.search(r'equation|formula|calculation', text_lower):
            return 'mathematical'
        elif re.search(r'procedure|step|method|process', text_lower):
            return 'procedural'
        elif re.search(r'definition|term|concept|meaning', text_lower):
            return 'definitional'
        elif re.search(r'example|case|instance|illustration', text_lower):
            return 'example'
        elif re.search(r'conclusion|summary|result|finding', text_lower):
            return 'conclusion'
        else:
            return 'narrative'
    
    def _contains_technical_terms(self, text: str) -> bool:
        """Check if chunk contains technical or domain-specific terms."""
        technical_indicators = [
            r'\b\w+ing\b',  # -ing words (often technical processes)
            r'\b\w+tion\b',  # -tion words (often technical concepts)
            r'\b\d+\.?\d*\s*(kg|g|lb|m|cm|mm|ft|in|°C|°F|K|Pa|bar|psi|mol|L|mL)\b',  # Units
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w*[Cc]hemical\w*\b',  # Chemical terms
            r'\b\w*[Ee]ngineering\w*\b',  # Engineering terms
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in technical_indicators)
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate a simple readability score (Flesch-like)."""
        if not text.strip():
            return 0.0
        
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease formula
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0.0, min(100.0, score))  # Clamp between 0-100
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simple approximation)."""
        word = word.lower().strip()
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_char_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    syllable_count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)

    def _create_chunks(self, content: str) -> List[Dict[str, Any]]:
        """
        Legacy chunking method - kept for backward compatibility.
        Use _create_enhanced_chunks for new functionality.
        """
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
    
    def _extract_pdf_tables(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract and process tables from PDF with robust error handling."""
        try:
            import camelot
            self.logger.info(f"Attempting table extraction from: {pdf_path.name}")
            
            # Extract tables using camelot
            tables = camelot.read_pdf(str(pdf_path), pages="all")
            
            extracted_files = []
            tables_by_type = {}
            
            for idx, table in enumerate(tables):
                # Clean the table
                df = self._clean_table(table.df)
                if df.empty:
                    continue
                
                # Detect table type and content
                table_type = self._detect_table_type(df)
                
                # Group tables by type
                if table_type not in tables_by_type:
                    tables_by_type[table_type] = []
                tables_by_type[table_type].append(df)
            
            # Save segregated tables
            for table_type, dfs in tables_by_type.items():
                if dfs:
                    # Combine tables of same type
                    combined_df = pd.concat(dfs, ignore_index=True)
                    
                    # Create meaningful filename
                    csv_filename = f"{pdf_path.stem}_{table_type}_tables.csv"
                    csv_path = self.tables_output_dir / csv_filename
                    
                    # Save with enhanced formatting
                    combined_df.to_csv(csv_path, index=False)
                    extracted_files.append(str(csv_path))
                    
                    self.logger.info(f"Extracted {table_type} tables → {csv_path}")
            
            return {
                "tables_extracted": len(tables),
                "table_files": extracted_files,
                "table_types": list(tables_by_type.keys())
            }
            
        except ImportError:
            self.logger.info("Table extraction disabled: Camelot library not available")
            return {"tables_extracted": 0, "table_files": [], "table_types": []}
        except Exception as e:
            error_msg = str(e)
            if "Ghostscript" in error_msg:
                self.logger.info(f"Table extraction skipped for {pdf_path.name}: Ghostscript not installed (this is optional)")
            elif "poppler" in error_msg.lower():
                self.logger.info(f"Table extraction skipped for {pdf_path.name}: Poppler utilities not installed (this is optional)")
            else:
                self.logger.warning(f"Table extraction failed for {pdf_path.name}: {error_msg}")
            
            return {"tables_extracted": 0, "table_files": [], "table_types": []}
    
    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format extracted table."""
        if df.empty:
            return df
        
        # First rename columns, then clean content
        df.columns = [col if col else f"col_{i}" for i, col in enumerate(df.columns)]
        
        # Clean cell content
        df = df.map(lambda x: str(x).strip() if pd.notna(x) else x)
        
        # Remove completely empty rows and columns
        df.dropna(axis=0, how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        
        # Remove rows that are mostly empty (less than 30% filled)
        min_filled = max(1, int(len(df.columns) * 0.3))
        df = df.dropna(thresh=min_filled)
        
        return df.reset_index(drop=True)
    
    def _detect_table_type(self, df: pd.DataFrame) -> str:
        """Detect table type based on content and structure."""
        if df.empty:
            return "unknown"
        
        # Convert all content to lowercase string for analysis
        content_text = " ".join([
            str(val).lower() for val in df.values.flatten() 
            if pd.notna(val) and str(val).strip()
        ])
        
        # Add column names to analysis
        column_text = " ".join([str(col).lower() for col in df.columns])
        full_text = content_text + " " + column_text
        
        # Detection patterns
        type_patterns = {
            "financial": ["cost", "price", "budget", "revenue", "profit", "expense", "financial"],
            "specifications": ["spec", "parameter", "property", "characteristic", "requirement"],
            "performance": ["efficiency", "rate", "speed", "capacity", "performance", "output"],
            "schedule": ["date", "time", "deadline", "schedule", "timeline", "duration"],
            "inventory": ["quantity", "stock", "item", "inventory", "catalog", "product"],
            "analysis": ["result", "analysis", "conclusion", "finding", "summary"],
            "comparison": ["vs", "versus", "compare", "comparison", "difference"],
            "data": ["measurement", "value", "reading", "observation", "data"]
        }
        
        # Score each type
        best_type = "general"
        best_score = 0
        
        for table_type, keywords in type_patterns.items():
            score = sum(1 for keyword in keywords if keyword in full_text)
            if score > best_score:
                best_score = score
                best_type = table_type
        
        return best_type
    
    def _create_table_summary(self, table_files: List[str]) -> str:
        """Create a text summary of extracted tables for RAG indexing."""
        if not table_files:
            return "No tables extracted from this PDF."
        
        summary_parts = [f"This document contains {len(table_files)} extracted table(s):"]
        
        for table_file in table_files:
            try:
                # Read the CSV to create a summary
                df = pd.read_csv(table_file)
                file_name = Path(table_file).name
                
                summary_parts.append(f"\nTable: {file_name}")
                summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
                summary_parts.append(f"Rows: {len(df)}")
                
                # Add sample data if available
                if not df.empty:
                    sample_row = df.iloc[0].to_dict()
                    sample_text = ", ".join([f"{k}: {v}" for k, v in sample_row.items() if pd.notna(v)])
                    summary_parts.append(f"Sample data: {sample_text}")
                
            except Exception as e:
                summary_parts.append(f"Table: {Path(table_file).name} (could not read: {e})")
        
        return "\n".join(summary_parts) 