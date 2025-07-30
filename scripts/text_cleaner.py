"""
Text cleaning utility for PyNucleus wiki scraper.
"""

import re
from typing import Dict, Any


class TextCleaner:
    """Text cleaning utility with statistics tracking."""
    
    def __init__(self):
        self.stats = {
            'cleaned_texts': 0,
            'total_characters_before': 0,
            'total_characters_after': 0,
            'patterns_removed': 0,
            'whitespace_normalized': 0
        }
    
    def clean_text(self, text: str) -> str:
        """Clean text and update statistics."""
        if not text:
            return text
            
        original_length = len(text)
        self.stats['cleaned_texts'] += 1
        self.stats['total_characters_before'] += original_length
        
        cleaned = text
        
        # Remove excessive whitespace
        before_whitespace = len(cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Multiple newlines to double
        if len(cleaned) != before_whitespace:
            self.stats['whitespace_normalized'] += 1
        
        # Remove common noise patterns
        patterns = [
            r'\{\{.*?\}\}',  # Template markup
            r'\<ref[^>]*\>.*?\</ref\>',  # References
            r'\<ref[^>]*\/\>',  # Self-closing refs
            r'\[\[Category:.*?\]\]',  # Categories
            r'\[\[File:.*?\]\]',  # File links
            r'\[\[Image:.*?\]\]',  # Image links
            r'\<\!--.*?--\>',  # Comments
            r'\&[a-zA-Z]+;',  # HTML entities
        ]
        
        patterns_removed = 0
        for pattern in patterns:
            before_pattern = len(cleaned)
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            if len(cleaned) != before_pattern:
                patterns_removed += 1
        
        self.stats['patterns_removed'] += patterns_removed
        
        # Final cleanup
        cleaned = cleaned.strip()
        
        self.stats['total_characters_after'] += len(cleaned)
        
        return cleaned
    
    def get_cleaning_efficiency(self) -> float:
        """Calculate cleaning efficiency as percentage reduction."""
        if self.stats['total_characters_before'] == 0:
            return 0.0
        
        reduction = self.stats['total_characters_before'] - self.stats['total_characters_after']
        return (reduction / self.stats['total_characters_before']) * 100
    
    def reset_stats(self):
        """Reset cleaning statistics."""
        self.stats = {
            'cleaned_texts': 0,
            'total_characters_before': 0,
            'total_characters_after': 0,
            'patterns_removed': 0,
            'whitespace_normalized': 0
        } 