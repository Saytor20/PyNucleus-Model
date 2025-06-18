#!/usr/bin/env python3
"""
Enhanced Text Cleaner for PyNucleus
====================================

A comprehensive text cleaning script that removes common artifacts from PDF-to-text conversion
and improves text readability across all document types.

Features:
- Removes single/double letter lines (common PDF artifacts)
- Eliminates excessive whitespace and blank lines
- Filters metadata and publisher information
- Removes page numbers and headers/footers
- Joins broken words across lines
- Handles table of contents artifacts
- Preserves important content structure

Usage:
    python scripts/enhanced_text_cleaner.py [input_dir] [output_dir]
    python scripts/enhanced_text_cleaner.py --file input.txt output.txt
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTextCleaner:
    """Comprehensive text cleaner for removing PDF conversion artifacts."""
    
    def __init__(self):
        self.stats = {
            'files_processed': 0,
            'lines_removed': 0,
            'blank_lines_removed': 0,
            'single_letters_removed': 0,
            'metadata_lines_removed': 0,
            'words_rejoined': 0
        }
        
        # Patterns for common artifacts
        self.artifact_patterns = {
            # Single letters or numbers on their own line
            'single_char': re.compile(r'^[A-Za-z0-9]$'),
            
            # Two characters that are likely artifacts
            'double_char': re.compile(r'^[A-Za-z]{1,2}$'),
            
            # Page numbers (various formats)
            'page_numbers': re.compile(r'^\s*(?:page\s+)?\d+\s*(?:of\s+\d+)?\s*$', re.IGNORECASE),
            
            # Roman numerals (often page numbers)
            'roman_numerals': re.compile(r'^\s*[ivxlcdm]+\s*$', re.IGNORECASE),
            
            # Lines with only punctuation and spaces
            'punctuation_only': re.compile(r'^[\s\.\-\(\)\[\]{},:;!?\'"_\|\\\/\*\+\=\~\`\#\@\$\%\^&]*$'),
            
            # Headers/footers with dots/dashes
            'header_footer_dots': re.compile(r'^\s*[.\-_]{3,}\s*$'),
            
            # Table of contents dots
            'toc_dots': re.compile(r'^.*\.{3,}.*\d+\s*$'),
            
            # URL fragments
            'url_fragments': re.compile(r'^(?:https?://|www\.|\.com|\.org|\.edu).*$', re.IGNORECASE),
            
            # Copyright and legal text patterns
            'copyright': re.compile(r'copyright|©|\(c\)|all rights reserved|printed in|isbn|doi:|preprint', re.IGNORECASE),
            
            # Publisher information
            'publisher': re.compile(r'press|publishing|publisher|taylor|francis|elsevier|springer|wiley|mcgraw|hill|pearson', re.IGNORECASE),
            
            # Author affiliations
            'affiliations': re.compile(r'university|college|department|institute|laboratory|@.*\.(edu|com|org)', re.IGNORECASE),
            
            # Date patterns (standalone)
            'standalone_dates': re.compile(r'^\s*\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\s*$'),
            
            # Common artifacts from OCR
            'ocr_artifacts': re.compile(r'^[il|]{1,3}$|^[0O]{1,2}$', re.IGNORECASE),
            
            # Lines that are mostly numbers and spaces
            'number_lines': re.compile(r'^\s*[\d\s\.\-\(\)]{3,}\s*$'),
            
            # Excessive spacing (lines with large gaps)
            'excessive_spacing': re.compile(r'^.{1,5}\s{10,}.*$'),
            
            # Broken word patterns (common at line breaks)
            'broken_word_end': re.compile(r'^.*[a-z]-\s*$'),
            'broken_word_start': re.compile(r'^\s*[a-z]+.*'),
        }
        
        # Metadata keywords to filter out
        self.metadata_keywords = [
            'copyright', 'all rights reserved', 'permission', 'reprinted', 'published',
            'taylor & francis', 'elsevier', 'springer', 'wiley', 'mcgraw', 'pearson',
            'isbn', 'doi:', 'printed in', 'no part of this', 'without written permission',
            'trademark', 'registered', 'proprietary', 'authorized', 'license',
            'reproduction', 'retrieval system', 'electronic', 'mechanical',
            'committee roster', 'foreword', 'acknowledgments', 'about the author'
        ]
        
        # Section headers that indicate real content
        self.content_indicators = [
            'introduction', 'abstract', 'summary', 'overview', 'background',
            'methodology', 'methods', 'results', 'discussion', 'conclusion',
            'chapter', 'section', 'part', 'appendix', 'references', 'bibliography',
            'table of contents', 'contents', 'index'
        ]

    def is_metadata_line(self, line: str) -> bool:
        """Check if a line contains metadata that should be removed."""
        line_lower = line.lower().strip()
        
        # Check for metadata keywords
        for keyword in self.metadata_keywords:
            if keyword in line_lower:
                return True
        
        # Check for email addresses
        if '@' in line and any(domain in line_lower for domain in ['.com', '.org', '.edu', '.gov']):
            return True
            
        # Check for URLs
        if any(url_start in line_lower for url_start in ['http://', 'https://', 'www.']):
            return True
            
        return False

    def is_likely_artifact(self, line: str, prev_line: str = "", next_line: str = "") -> bool:
        """Determine if a line is likely a conversion artifact."""
        stripped = line.strip()
        
        # Empty lines
        if not stripped:
            return False  # Handle separately
            
        # Check against artifact patterns
        for pattern_name, pattern in self.artifact_patterns.items():
            if pattern.match(stripped):
                # Special handling for some patterns
                if pattern_name == 'double_char':
                    # Don't remove common words or abbreviations
                    if stripped.lower() in ['is', 'to', 'of', 'or', 'in', 'on', 'at', 'by', 'an', 'as', 'be', 'do', 'go', 'he', 'me', 'we', 'my', 'no', 'so', 'up', 'us']:
                        continue
                    # Don't remove if it's part of a list or has context
                    if prev_line.strip() and (prev_line.strip().endswith(':') or '•' in prev_line or '1.' in prev_line):
                        continue
                return True
        
        # Check for metadata
        if self.is_metadata_line(line):
            return True
            
        # Lines that are too short and don't look like real content
        if len(stripped) <= 3 and not any(indicator in stripped.lower() for indicator in ['a', 'i', 'is', 'to', 'of']):
            return True
            
        return False

    def join_broken_words(self, lines: List[str]) -> List[str]:
        """Join words that were broken across lines due to PDF conversion."""
        if not lines:
            return lines
            
        result = []
        i = 0
        words_joined = 0
        
        while i < len(lines):
            current_line = lines[i]
            
            # Check if current line ends with a hyphen (broken word)
            if (i < len(lines) - 1 and 
                self.artifact_patterns['broken_word_end'].match(current_line)):
                
                next_line = lines[i + 1].strip()
                
                # Check if next line starts with lowercase (continuation)
                if (next_line and 
                    self.artifact_patterns['broken_word_start'].match(next_line)):
                    
                    # Join the lines, removing the hyphen
                    joined_line = current_line.rstrip().rstrip('-') + next_line
                    result.append(joined_line)
                    words_joined += 1
                    i += 2  # Skip next line since we merged it
                    continue
            
            result.append(current_line)
            i += 1
        
        self.stats['words_rejoined'] += words_joined
        return result

    def remove_excessive_blank_lines(self, lines: List[str]) -> List[str]:
        """Remove excessive blank lines, keeping at most 2 consecutive blank lines."""
        if not lines:
            return lines
            
        result = []
        blank_count = 0
        blank_lines_removed = 0
        
        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:  # Keep at most 2 consecutive blank lines
                    result.append(line)
                else:
                    blank_lines_removed += 1
            else:
                blank_count = 0
                result.append(line)
        
        self.stats['blank_lines_removed'] += blank_lines_removed
        return result

    def detect_content_start(self, lines: List[str]) -> int:
        """Find where real content starts (after title/metadata section)."""
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Look for content indicators
            for indicator in self.content_indicators:
                if indicator in line_lower and len(line.strip()) < 100:  # Likely a header
                    return i
                    
            # Look for paragraph-like content (longer lines)
            if (len(line.strip()) > 100 and 
                not self.is_metadata_line(line) and 
                not self.is_likely_artifact(line)):
                return max(0, i - 2)  # Start a bit before to keep context
        
        return 0  # If no clear start found, start from beginning

    def clean_text(self, text: str) -> str:
        """Main text cleaning function."""
        lines = text.splitlines()
        original_line_count = len(lines)
        
        # Step 1: Join broken words
        lines = self.join_broken_words(lines)
        
        # Step 2: Find content start
        content_start = self.detect_content_start(lines)
        
        # Step 3: Filter out artifacts and metadata
        cleaned_lines = []
        lines_removed = 0
        single_letters_removed = 0
        metadata_removed = 0
        
        for i, line in enumerate(lines):
            # Skip lines before content starts (but keep document header if present)
            if i < content_start and i > 10:  # Keep first 10 lines for document info
                continue
                
            prev_line = lines[i-1] if i > 0 else ""
            next_line = lines[i+1] if i < len(lines)-1 else ""
            
            if self.is_likely_artifact(line, prev_line, next_line):
                if self.artifact_patterns['single_char'].match(line.strip()):
                    single_letters_removed += 1
                elif self.is_metadata_line(line):
                    metadata_removed += 1
                lines_removed += 1
                continue
                
            cleaned_lines.append(line)
        
        # Step 4: Remove excessive blank lines
        cleaned_lines = self.remove_excessive_blank_lines(cleaned_lines)
        
        # Update statistics
        self.stats['lines_removed'] += lines_removed
        self.stats['single_letters_removed'] += single_letters_removed
        self.stats['metadata_lines_removed'] += metadata_removed
        
        return '\n'.join(cleaned_lines)

    def clean_file(self, input_path: Path, output_path: Path) -> bool:
        """Clean a single text file."""
        try:
            logger.info(f"Processing: {input_path.name}")
            
            # Read input file
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Skipping empty file: {input_path.name}")
                return False
            
            # Clean the content
            cleaned_content = self.clean_text(content)
            
            # Write output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            self.stats['files_processed'] += 1
            logger.info(f"✅ Cleaned: {input_path.name} -> {output_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {input_path.name}: {e}")
            return False

    def clean_directory(self, input_dir: Path, output_dir: Path) -> None:
        """Clean all text files in a directory."""
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return
        
        # Find all text files
        text_files = list(input_dir.glob("*.txt"))
        
        if not text_files:
            logger.warning(f"No .txt files found in {input_dir}")
            return
        
        logger.info(f"Found {len(text_files)} text files to process")
        
        # Process each file
        for txt_file in text_files:
            output_file = output_dir / txt_file.name
            self.clean_file(txt_file, output_file)

    def print_statistics(self) -> None:
        """Print cleaning statistics."""
        print("\n" + "="*60)
        print("ENHANCED TEXT CLEANING STATISTICS")
        print("="*60)
        print(f"📄 Files processed: {self.stats['files_processed']}")
        print(f"🗑️  Total lines removed: {self.stats['lines_removed']}")
        print(f"   • Single letters removed: {self.stats['single_letters_removed']}")
        print(f"   • Metadata lines removed: {self.stats['metadata_lines_removed']}")
        print(f"   • Blank lines removed: {self.stats['blank_lines_removed']}")
        print(f"🔗 Words rejoined: {self.stats['words_rejoined']}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Enhanced text cleaner for PyNucleus documents")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("output", help="Output file or directory")
    parser.add_argument("--file", action="store_true", help="Process single file instead of directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize cleaner
    cleaner = EnhancedTextCleaner()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"🧹 Enhanced Text Cleaner - Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.file:
        # Single file mode
        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return 1
        
        success = cleaner.clean_file(input_path, output_path)
        if not success:
            return 1
    else:
        # Directory mode
        cleaner.clean_directory(input_path, output_path)
    
    # Print statistics
    cleaner.print_statistics()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())