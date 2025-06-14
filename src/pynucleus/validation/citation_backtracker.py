# -*- coding: utf-8 -*-
"""
Citation Backtracking System for PyNucleus
==========================================

This module provides user-friendly citation backtracking from generated responses,
allowing users to trace back to source documents and verify information.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class SourceCitation:
    """Structure for source citations."""
    source_file: str
    chunk_id: Optional[int] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    confidence_score: float = 0.0
    relevant_text: str = ""
    citation_id: str = ""

@dataclass 
class CitedResponse:
    """Structure for responses with citations."""
    query: str
    answer: str
    citations: List[SourceCitation] = field(default_factory=list)
    confidence_score: float = 0.0
    response_time: float = 0.0
    timestamp: str = ""
    
    def get_formatted_response(self, format_type: str = "academic") -> str:
        """Get formatted response with citations."""
        if format_type == "academic":
            return self._format_academic_style()
        elif format_type == "footnote":
            return self._format_footnote_style()
        elif format_type == "inline":
            return self._format_inline_style()
        else:
            return self._format_simple_style()
    
    def _format_academic_style(self) -> str:
        """Format response in academic style with numbered references."""
        formatted_answer = self.answer
        references = []
        
        for i, citation in enumerate(self.citations, 1):
            ref_marker = f"[{i}]"
            # Insert reference markers in the text
            if citation.relevant_text:
                # Try to find where to insert the citation
                text_snippet = citation.relevant_text[:50]
                if text_snippet.lower() in formatted_answer.lower():
                    formatted_answer = formatted_answer.replace(
                        text_snippet, f"{text_snippet} {ref_marker}"
                    )
            
            # Create reference entry
            ref_entry = f"[{i}] {citation.source_file}"
            if citation.section:
                ref_entry += f", Section: {citation.section}"
            if citation.page_number:
                ref_entry += f", Page: {citation.page_number}"
            references.append(ref_entry)
        
        if references:
            formatted_answer += "\n\n**References:**\n" + "\n".join(references)
        
        return formatted_answer
    
    def _format_footnote_style(self) -> str:
        """Format response with footnote-style citations."""
        formatted_answer = self.answer
        footnotes = []
        
        for i, citation in enumerate(self.citations, 1):
            footnote_marker = f"^{i}"
            footnotes.append(f"{footnote_marker} Source: {citation.source_file}")
            
            # Add footnote markers to relevant parts of the text
            if citation.relevant_text:
                text_snippet = citation.relevant_text[:30]
                if text_snippet.lower() in formatted_answer.lower():
                    formatted_answer = formatted_answer.replace(
                        text_snippet, f"{text_snippet}{footnote_marker}"
                    )
        
        if footnotes:
            formatted_answer += "\n\n**Sources:**\n" + "\n".join(footnotes)
        
        return formatted_answer
    
    def _format_inline_style(self) -> str:
        """Format response with inline citations."""
        formatted_answer = self.answer
        
        for citation in self.citations:
            if citation.relevant_text:
                text_snippet = citation.relevant_text[:40]
                if text_snippet.lower() in formatted_answer.lower():
                    inline_citation = f" (Source: {Path(citation.source_file).stem})"
                    formatted_answer = formatted_answer.replace(
                        text_snippet, f"{text_snippet}{inline_citation}"
                    )
        
        return formatted_answer
    
    def _format_simple_style(self) -> str:
        """Format response with simple source list."""
        formatted_answer = self.answer
        
        if self.citations:
            sources = list(set(Path(c.source_file).stem for c in self.citations))
            formatted_answer += f"\n\n**Sources:** {', '.join(sources)}"
        
        return formatted_answer

class CitationBacktracker:
    """User-friendly citation backtracking system."""
    
    def __init__(self, 
                 vector_store_manager=None,
                 citation_cache_dir: str = "data/citations",
                 enable_cache: bool = True):
        """Initialize the citation backtracker.
        
        Args:
            vector_store_manager: Vector store manager for retrieving documents
            citation_cache_dir: Directory for caching citation data
            enable_cache: Whether to enable citation caching
        """
        self.vector_store_manager = vector_store_manager
        self.citation_cache_dir = Path(citation_cache_dir)
        self.enable_cache = enable_cache
        
        if self.enable_cache:
            self.citation_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Load document metadata for better citation formatting
        self.document_metadata = self._load_document_metadata()
        
        # Citation patterns for different document types
        self.citation_patterns = {
            'pdf': r'(?i)\.pdf$',
            'wiki': r'(?i)wikipedia_',
            'manuscript': r'(?i)manuscript|draft',
            'simulation': r'(?i)dwsim|simulation'
        }
    
    def _load_document_metadata(self) -> Dict[str, Dict]:
        """Load metadata about documents for better citation formatting."""
        metadata = {}
        
        # Try to load from vector store if available
        if self.vector_store_manager and hasattr(self.vector_store_manager, 'documents'):
            for doc in self.vector_store_manager.documents:
                source = doc.metadata.get('source', '')
                if source:
                    metadata[source] = {
                        'title': doc.metadata.get('title', Path(source).stem),
                        'type': doc.metadata.get('type', self._detect_document_type(source)),
                        'author': doc.metadata.get('author', 'Unknown'),
                        'date': doc.metadata.get('date', 'Unknown'),
                        'page_count': doc.metadata.get('page_count', 'Unknown')
                    }
        
        return metadata
    
    def _detect_document_type(self, source_path: str) -> str:
        """Detect document type from source path."""
        source_lower = source_path.lower()
        
        for doc_type, pattern in self.citation_patterns.items():
            if re.search(pattern, source_lower):
                return doc_type
        
        return 'document'
    
    def generate_cited_response(self, 
                              query: str, 
                              answer: str, 
                              retrieved_docs: List[Tuple[Any, float]],
                              response_time: float = 0.0) -> CitedResponse:
        """Generate a response with comprehensive citations.
        
        Args:
            query: Original query
            answer: Generated answer
            retrieved_docs: List of (document, score) tuples from retrieval
            response_time: Response generation time
            
        Returns:
            CitedResponse with comprehensive citation information
        """
        citations = []
        
        for i, (doc, score) in enumerate(retrieved_docs):
            # Extract citation information
            source_file = doc.metadata.get('source', f'Unknown_Source_{i}')
            
            # Get relevant text snippet
            relevant_text = self._extract_relevant_text(doc.page_content, answer)
            
            # Determine section/page information
            section = self._extract_section_info(doc)
            page_number = self._extract_page_number(doc)
            
            citation = SourceCitation(
                source_file=source_file,
                chunk_id=doc.metadata.get('chunk_id'),
                page_number=page_number,
                section=section,
                confidence_score=score,
                relevant_text=relevant_text,
                citation_id=f"cite_{i+1}"
            )
            
            citations.append(citation)
        
        # Calculate overall confidence
        confidence_score = sum(c.confidence_score for c in citations) / len(citations) if citations else 0.0
        
        cited_response = CitedResponse(
            query=query,
            answer=answer,
            citations=citations,
            confidence_score=confidence_score,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )
        
        # Cache the citation if enabled
        if self.enable_cache:
            self._cache_citation(cited_response)
        
        return cited_response
    
    def _extract_relevant_text(self, doc_content: str, answer: str, max_length: int = 200) -> str:
        """Extract most relevant text snippet from document for citation."""
        if not doc_content or not answer:
            return ""
        
        # Find overlapping words between document and answer
        doc_words = set(doc_content.lower().split())
        answer_words = set(answer.lower().split())
        common_words = doc_words.intersection(answer_words)
        
        if not common_words:
            return doc_content[:max_length]
        
        # Find the best sentence containing common words
        sentences = doc_content.split('.')
        best_sentence = ""
        max_overlap = 0
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(sentence_words.intersection(common_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sentence.strip()
        
        return best_sentence[:max_length] if best_sentence else doc_content[:max_length]
    
    def _extract_section_info(self, doc) -> Optional[str]:
        """Extract section information from document metadata."""
        # Look for section information in various metadata fields
        metadata = doc.metadata
        
        section_candidates = [
            metadata.get('section'),
            metadata.get('chapter'),
            metadata.get('heading'),
            metadata.get('title')
        ]
        
        for candidate in section_candidates:
            if candidate and isinstance(candidate, str):
                return candidate
        
        return None
    
    def _extract_page_number(self, doc) -> Optional[int]:
        """Extract page number from document metadata."""
        metadata = doc.metadata
        
        page_candidates = [
            metadata.get('page'),
            metadata.get('page_number'),
            metadata.get('page_num')
        ]
        
        for candidate in page_candidates:
            if candidate is not None:
                try:
                    return int(candidate)
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _cache_citation(self, cited_response: CitedResponse):
        """Cache citation data for future reference."""
        if not self.enable_cache:
            return
        
        cache_file = self.citation_cache_dir / f"citation_{hash(cited_response.query)}.json"
        
        try:
            cache_data = {
                'query': cited_response.query,
                'answer': cited_response.answer,
                'citations': [
                    {
                        'source_file': c.source_file,
                        'chunk_id': c.chunk_id,
                        'page_number': c.page_number,
                        'section': c.section,
                        'confidence_score': c.confidence_score,
                        'relevant_text': c.relevant_text,
                        'citation_id': c.citation_id
                    }
                    for c in cited_response.citations
                ],
                'confidence_score': cited_response.confidence_score,
                'response_time': cited_response.response_time,
                'timestamp': cited_response.timestamp
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.warning(f"Failed to cache citation: {e}")
    
    def get_cached_citation(self, query: str) -> Optional[CitedResponse]:
        """Retrieve cached citation for a query."""
        if not self.enable_cache:
            return None
        
        cache_file = self.citation_cache_dir / f"citation_{hash(query)}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            citations = []
            for c_data in cache_data['citations']:
                citation = SourceCitation(
                    source_file=c_data['source_file'],
                    chunk_id=c_data['chunk_id'],
                    page_number=c_data['page_number'],
                    section=c_data['section'],
                    confidence_score=c_data['confidence_score'],
                    relevant_text=c_data['relevant_text'],
                    citation_id=c_data['citation_id']
                )
                citations.append(citation)
            
            return CitedResponse(
                query=cache_data['query'],
                answer=cache_data['answer'],
                citations=citations,
                confidence_score=cache_data['confidence_score'],
                response_time=cache_data['response_time'],
                timestamp=cache_data['timestamp']
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to load cached citation: {e}")
            return None
    
    def verify_citation_accuracy(self, cited_response: CitedResponse) -> Dict[str, Any]:
        """Verify the accuracy of citations in a response."""
        verification_results = {
            'total_citations': len(cited_response.citations),
            'verified_citations': 0,
            'accuracy_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        for citation in cited_response.citations:
            # Check if source file exists and is accessible
            if not self._verify_source_exists(citation.source_file):
                verification_results['issues'].append(
                    f"Source file not found: {citation.source_file}"
                )
                continue
            
            # Check if relevant text is actually from the source
            if citation.relevant_text:
                if self._verify_text_in_source(citation.relevant_text, citation.source_file):
                    verification_results['verified_citations'] += 1
                else:
                    verification_results['issues'].append(
                        f"Relevant text not found in source: {citation.source_file}"
                    )
            else:
                verification_results['issues'].append(
                    f"No relevant text provided for citation: {citation.source_file}"
                )
        
        # Calculate accuracy score
        if verification_results['total_citations'] > 0:
            verification_results['accuracy_score'] = (
                verification_results['verified_citations'] / verification_results['total_citations']
            )
        
        # Generate recommendations
        if verification_results['accuracy_score'] < 0.8:
            verification_results['recommendations'].append(
                "Citation accuracy is below 80%. Review document chunking and retrieval settings."
            )
        
        if len(verification_results['issues']) > 0:
            verification_results['recommendations'].append(
                "Address citation issues to improve response reliability."
            )
        
        return verification_results
    
    def _verify_source_exists(self, source_file: str) -> bool:
        """Verify that a source file exists."""
        # Check if it's a file path
        if Path(source_file).exists():
            return True
        
        # Check in common source directories
        common_dirs = [
            "data/01_raw/source_documents",
            "data/01_raw/web_sources", 
            "data/02_processed/converted_to_txt"
        ]
        
        for dir_path in common_dirs:
            if (Path(dir_path) / source_file).exists():
                return True
        
        return False
    
    def _verify_text_in_source(self, text: str, source_file: str) -> bool:
        """Verify that specific text appears in the source file."""
        try:
            # Try to find and read the source file
            source_path = None
            
            if Path(source_file).exists():
                source_path = Path(source_file)
            else:
                # Check common directories
                common_dirs = [
                    "data/01_raw/source_documents",
                    "data/01_raw/web_sources",
                    "data/02_processed/converted_to_txt"
                ]
                
                for dir_path in common_dirs:
                    potential_path = Path(dir_path) / source_file
                    if potential_path.exists():
                        source_path = potential_path
                        break
            
            if not source_path:
                return False
            
            # Read file content
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if text appears in content (case-insensitive)
            return text.lower() in content.lower()
            
        except Exception as e:
            self.logger.warning(f"Error verifying text in source {source_file}: {e}")
            return False
    
    def generate_citation_report(self, 
                                cited_responses: List[CitedResponse],
                                output_file: Optional[str] = None) -> str:
        """Generate a comprehensive citation report."""
        if not cited_responses:
            return "No cited responses provided for report generation."
        
        # Analyze citation patterns
        source_frequency = defaultdict(int)
        citation_accuracy_scores = []
        
        for response in cited_responses:
            for citation in response.citations:
                source_frequency[citation.source_file] += 1
            
            # Verify citation accuracy
            verification = self.verify_citation_accuracy(response)
            citation_accuracy_scores.append(verification['accuracy_score'])
        
        # Generate report
        report = f"""
# Citation Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- Total Responses: {len(cited_responses)}
- Total Citations: {sum(len(r.citations) for r in cited_responses)}
- Average Citations per Response: {sum(len(r.citations) for r in cited_responses) / len(cited_responses):.2f}
- Average Citation Accuracy: {sum(citation_accuracy_scores) / len(citation_accuracy_scores):.2%}

## Most Frequently Cited Sources
"""
        
        # Sort sources by frequency
        sorted_sources = sorted(source_frequency.items(), key=lambda x: x[1], reverse=True)
        
        for source, count in sorted_sources[:10]:  # Top 10
            report += f"- {Path(source).name}: {count} citations\n"
        
        report += f"""

## Citation Quality Analysis
- High Quality Citations (>80% accuracy): {len([s for s in citation_accuracy_scores if s > 0.8])}
- Medium Quality Citations (50-80% accuracy): {len([s for s in citation_accuracy_scores if 0.5 <= s <= 0.8])}
- Low Quality Citations (<50% accuracy): {len([s for s in citation_accuracy_scores if s < 0.5])}

## Recommendations
"""
        
        # Generate recommendations based on analysis
        avg_accuracy = sum(citation_accuracy_scores) / len(citation_accuracy_scores)
        
        if avg_accuracy < 0.7:
            report += "- Improve citation accuracy by enhancing document preprocessing and chunking\n"
        
        if len(sorted_sources) < 5:
            report += "- Consider expanding the document collection for better source diversity\n"
        
        avg_citations = sum(len(r.citations) for r in cited_responses) / len(cited_responses)
        if avg_citations < 2:
            report += "- Increase the number of sources retrieved per query for better coverage\n"
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Citation report saved to: {output_file}")
        
        return report
    
    def create_interactive_citation_viewer(self, cited_response: CitedResponse) -> str:
        """Create an interactive HTML viewer for citations."""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Citation Viewer - PyNucleus</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .query {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .answer {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .citations {{ margin-top: 20px; }}
        .citation {{ 
            border: 1px solid #ddd; 
            margin: 10px 0; 
            padding: 10px; 
            border-radius: 5px;
            background: #f9f9f9;
        }}
        .citation-header {{ font-weight: bold; color: #0066cc; }}
        .citation-text {{ margin-top: 10px; font-style: italic; }}
        .confidence {{ color: #666; font-size: 0.9em; }}
        .highlight {{ background-color: yellow; }}
    </style>
</head>
<body>
    <h1>PyNucleus Citation Viewer</h1>
    
    <div class="query">
        <h2>Query:</h2>
        <p>{cited_response.query}</p>
    </div>
    
    <div class="answer">
        <h2>Answer:</h2>
        <p>{cited_response.answer}</p>
        <div class="confidence">
            Overall Confidence: {cited_response.confidence_score:.2%} | 
            Response Time: {cited_response.response_time:.2f}s
        </div>
    </div>
    
    <div class="citations">
        <h2>Citations ({len(cited_response.citations)}):</h2>
"""
        
        for i, citation in enumerate(cited_response.citations, 1):
            html_template += f"""
        <div class="citation">
            <div class="citation-header">
                [{i}] {Path(citation.source_file).name}
            </div>
            <div class="confidence">
                Confidence: {citation.confidence_score:.2%}
            </div>
            {f'<div><strong>Section:</strong> {citation.section}</div>' if citation.section else ''}
            {f'<div><strong>Page:</strong> {citation.page_number}</div>' if citation.page_number else ''}
            {f'<div class="citation-text">"{citation.relevant_text}"</div>' if citation.relevant_text else ''}
        </div>
"""
        
        html_template += """
    </div>
</body>
</html>
"""
        
        return html_template 