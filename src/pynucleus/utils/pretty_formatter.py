"""
Enhanced formatting utilities for PyNucleus CLI output.

Provides beautiful terminal formatting with boxes, citations, and confidence indicators.
"""

import re
from typing import Dict, Any, List

# Try to import rich for enhanced formatting
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.box import ROUNDED
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def clean_source_name(source: str) -> str:
    """Clean up source names for display"""
    if not source:
        return "Unknown Source"
    
    # Remove common file extensions and clean up
    cleaned = source.replace('.txt', '').replace('.pdf', '').replace('.md', '')
    cleaned = re.sub(r'[_-]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Capitalize words
    words = cleaned.split()
    if words:
        cleaned = ' '.join(word.capitalize() for word in words)
    
    return cleaned if cleaned else "Document"

def extract_sources_from_answer(answer: str) -> List[str]:
    """Extract source references from within the answer text"""
    # Find all [Doc-...] citations
    citations = re.findall(r'\[Doc-([^\]]+)\]', answer)
    return list(set(citations))  # Remove duplicates

def format_equations(text: str) -> str:
    """Format mathematical equations for better readability"""
    if not text:
        return text
    
    # Replace common LaTeX patterns with readable equivalents
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)', text)
    text = re.sub(r'\\Delta', 'Δ', text)
    text = re.sub(r'\\alpha', 'α', text)
    text = re.sub(r'\\beta', 'β', text)
    text = re.sub(r'\\gamma', 'γ', text)
    
    return text

def format_citations(text: str) -> str:
    """Format citation references for better readability"""
    if not text:
        return text
    
    # Improve citation formatting
    text = re.sub(r'\[Doc-([^\]]+)\]', r'[Source: \1]', text)
    
    return text

def create_cli_answer_box(answer: str, confidence: float = 0.0) -> str:
    """Create a beautiful CLI box for the answer"""
    if not RICH_AVAILABLE:
        # Fallback for plain text
        box_width = 80
        confidence_text = f"Confidence: {confidence:.1%}" if confidence > 0 else ""
        
        result = "\n" + "┌" + "─" * (box_width - 2) + "┐\n"
        result += f"│ 💡 ANSWER {confidence_text:>{box_width-12}}│\n"
        result += "├" + "─" * (box_width - 2) + "┤\n"
        
        # Wrap text to fit box
        words = answer.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= box_width - 4:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        for line in lines:
            result += f"│ {line:<{box_width-4}} │\n"
        
        result += "└" + "─" * (box_width - 2) + "┘\n"
        return result
    
    # Rich formatting
    console = Console()
    
    # Create confidence indicator
    confidence_text = ""
    if confidence > 0:
        if confidence >= 0.8:
            confidence_text = f"[bold green]🎯 {confidence:.1%}[/bold green]"
        elif confidence >= 0.6:
            confidence_text = f"[bold yellow]🎯 {confidence:.1%}[/bold yellow]"
        else:
            confidence_text = f"[bold red]🎯 {confidence:.1%}[/bold red]"
    
    # Format answer content
    formatted_answer = format_equations(answer)
    formatted_answer = format_citations(formatted_answer)
    
    # Create title with confidence
    title = "💡 Answer"
    if confidence_text:
        title += f" {confidence_text}"
    
    # Create panel
    panel = Panel(
        Markdown(formatted_answer),
        title=title,
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2)
    )
    
    # Print and return
    console.print(panel)
    return formatted_answer

def create_cli_citation_box(sources: List[str], citations_from_answer: List[str]) -> str:
    """Create a beautiful CLI box for citations and sources"""
    all_sources = list(set(sources + citations_from_answer))
    
    if not all_sources:
        return ""
    
    if not RICH_AVAILABLE:
        # Fallback for plain text
        box_width = 80
        result = "\n" + "┌" + "─" * (box_width - 2) + "┐\n"
        result += f"│ 📚 SOURCES & CITATIONS{' ' * (box_width - 25)}│\n"
        result += "├" + "─" * (box_width - 2) + "┤\n"
        
        for i, source in enumerate(all_sources[:5], 1):
            cleaned = clean_source_name(source)
            source_line = f"[{i}] {cleaned}"
            if len(source_line) > box_width - 6:
                source_line = source_line[:box_width-9] + "..."
            result += f"│ {source_line:<{box_width-4}} │\n"
        
        if len(all_sources) > 5:
            result += f"│ ... and {len(all_sources) - 5} more sources{' ' * (box_width - 25 - len(str(len(all_sources) - 5)))}│\n"
        
        result += "└" + "─" * (box_width - 2) + "┘\n"
        return result
    
    # Rich formatting
    console = Console()
    
    # Create table for sources
    table = Table(show_header=False, show_edge=False, padding=(0, 1))
    table.add_column("Number", style="bold cyan", width=5)
    table.add_column("Source", style="white")
    
    for i, source in enumerate(all_sources[:5], 1):
        cleaned = clean_source_name(source)
        table.add_row(f"[{i}]", cleaned)
    
    if len(all_sources) > 5:
        table.add_row("...", f"[dim]and {len(all_sources) - 5} more sources[/dim]")
    
    panel = Panel(
        table,
        title="📚 Sources & Citations",
        border_style="magenta",
        box=ROUNDED,
        padding=(0, 1)
    )
    
    console.print(panel)
    return "\n".join(all_sources)

def create_cli_metadata_box(result: Dict[str, Any]) -> str:
    """Create a metadata display box"""
    metadata = result.get("metadata", {})
    if not metadata:
        return ""
    
    if not RICH_AVAILABLE:
        # Simple text fallback
        lines = []
        if "response_time" in metadata:
            lines.append(f"⏱️  Response Time: {metadata['response_time']:.2f}s")
        if "retrieval_count" in metadata:
            lines.append(f"🔍 Documents Retrieved: {metadata['retrieval_count']}")
        if "tokens_used" in metadata:
            lines.append(f"💬 Tokens Used: {metadata['tokens_used']}")
        if "has_citations" in metadata:
            lines.append(f"🏷️  Citations: {'Yes' if metadata['has_citations'] else 'No'}")
        
        if lines:
            return "\n📊 " + " | ".join(lines) + "\n"
        return ""
    
    # Rich formatting
    console = Console()
    
    info_text = []
    if "response_time" in metadata:
        info_text.append(f"⏱️  {metadata['response_time']:.2f}s")
    if "retrieval_count" in metadata:
        info_text.append(f"🔍 {metadata['retrieval_count']} docs")
    if "tokens_used" in metadata:
        info_text.append(f"💬 {metadata['tokens_used']} tokens")
    if "has_citations" in metadata:
        info_text.append(f"🏷️  {'✓' if metadata['has_citations'] else '✗'} citations")
    
    if info_text:
        console.print(f"[dim]📊 {' | '.join(info_text)}[/dim]")
    
    return ""

def format_for_terminal(result: Dict[str, Any]) -> str:
    """
    Main function to format RAG results for terminal display with beautiful boxes.
    
    Args:
        result: Dictionary containing answer, sources, confidence, and metadata
        
    Returns:
        Formatted string for terminal display
    """
    if not result:
        return "No result to display"
    
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    confidence = result.get("confidence", 0.0)
    
    # Import and apply answer cleaning if available
    try:
        from ..rag.answer_processing import clean_and_format_answer, remove_meta_commentary, filter_irrelevant_content
        # Apply enhanced processing for clean output
        answer = remove_meta_commentary(answer)
        answer = filter_irrelevant_content(answer)
        answer = clean_and_format_answer(answer)
    except ImportError:
        pass  # Fallback if cleaning functions not available
    
    # Extract citations from answer text
    citations_from_answer = extract_sources_from_answer(answer)
    
    # Display answer box
    formatted_answer = create_cli_answer_box(answer, confidence)
    
    # Display citations box if we have sources
    if sources or citations_from_answer:
        create_cli_citation_box(sources, citations_from_answer)
    
    # Display metadata
    create_cli_metadata_box(result)
    
    return formatted_answer

# Legacy function for backwards compatibility
def pretty_print_answer_enhanced(result: Dict[str, Any], use_rich: bool = True) -> str:
    """Legacy function - redirects to format_for_terminal"""
    return format_for_terminal(result) 