"""
Pretty formatting utilities for enhanced output display
"""
import re
from typing import Dict, Any, List

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.columns import Columns
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def format_equations(text: str) -> str:
    """Detect and format equations with LaTeX-style wrapping"""
    # Simple equation detection patterns
    equation_patterns = [
        (r'\b([A-Za-z]+\s*=\s*[^,\n]+)', r'\\(\1\\)'),  # Basic equations like "x = 5"
        (r'(\d+\s*[+\-*/]\s*\d+)', r'\\(\1\\)'),  # Simple arithmetic
        (r'([A-Za-z]+\s*[+\-*/]\s*[A-Za-z]+)', r'\\(\1\\)'),  # Variable operations
    ]
    
    formatted_text = text
    for pattern, replacement in equation_patterns:
        formatted_text = re.sub(pattern, replacement, formatted_text)
    
    return formatted_text

def format_citations(text: str) -> str:
    """Enhance citation formatting"""
    # Make reference numbers bold
    text = re.sub(r'\[(\d+)\]', r'**[\1]**', text)
    return text

def extract_sources_from_answer(answer: str) -> List[str]:
    """Extract source references from within the answer text"""
    # Find all [Doc-...] citations
    citations = re.findall(r'\[Doc-([^\]]+)\]', answer)
    return list(set(citations))  # Remove duplicates

def clean_source_name(source_name: str) -> str:
    """Clean up source names for better display"""
    if not source_name:
        return "Unknown Source"
    
    # Remove common prefixes/suffixes
    cleaned = re.sub(r'^Doc-', '', source_name)
    cleaned = re.sub(r'\.txt$', '', cleaned)
    cleaned = re.sub(r'\.pdf$', '', cleaned)
    
    # Replace underscores and dashes with spaces
    cleaned = re.sub(r'[_-]', ' ', cleaned)
    
    # Capitalize words
    cleaned = ' '.join(word.capitalize() for word in cleaned.split())
    
    # Truncate if too long
    if len(cleaned) > 60:
        cleaned = cleaned[:57] + "..."
    
    return cleaned

def create_resource_box(sources: List[str], citations_from_answer: List[str]) -> str:
    """Create a formatted resource box"""
    if not RICH_AVAILABLE:
        # Simple text fallback
        if sources or citations_from_answer:
            all_sources = list(set(sources + citations_from_answer))
            result = "\n📚 RESOURCES:\n" + "─" * 40 + "\n"
            for i, source in enumerate(all_sources[:5], 1):
                cleaned = clean_source_name(source)
                result += f"📄 {i}. {cleaned}\n"
            result += "─" * 40
            return result
        return "\n📚 RESOURCES: No sources available"
    
    console = Console()
    
    # Combine sources from the sources list and citations found in answer
    all_sources = list(set(sources + citations_from_answer))
    
    if not all_sources:
        return Panel(
            "[dim]No sources available[/dim]",
            title="📚 Resources",
            border_style="dim",
            padding=(0, 1)
        )
    
    # Create table for sources
    table = Table(show_header=False, show_edge=False, padding=(0, 1))
    table.add_column("Icon", width=3)
    table.add_column("Source", style="cyan")
    
    for i, source in enumerate(all_sources[:5], 1):  # Limit to 5 sources
        cleaned = clean_source_name(source)
        table.add_row("📄", f"{i}. {cleaned}")
    
    if len(all_sources) > 5:
        table.add_row("⋮", f"[dim]... and {len(all_sources) - 5} more sources[/dim]")
    
    return Panel(
        table,
        title="📚 Resources",
        border_style="blue",
        padding=(0, 1)
    )

def pretty_print_answer_enhanced(result: Dict[str, Any], use_rich: bool = True) -> str:
    """Enhanced format answer output with concise text and resource boxes"""
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    confidence = result.get("confidence", 0.0)
    
    # Import and apply answer cleaning if available
    try:
        from ..rag.answer_processing import clean_and_format_answer, remove_meta_commentary, filter_irrelevant_content, create_concise_answer
        # Apply enhanced processing for concise output
        answer = remove_meta_commentary(answer)
        answer = filter_irrelevant_content(answer)
        answer = clean_and_format_answer(answer)
        answer = create_concise_answer(answer, max_sentences=3)
    except ImportError:
        pass  # Fallback if cleaning functions not available
    
    # Extract citations from answer text
    citations_from_answer = extract_sources_from_answer(answer)
    
    if not use_rich or not RICH_AVAILABLE:
        # Fallback to enhanced plain text
        formatted = format_equations(answer)
        formatted = format_citations(formatted)
        
        # Clean up formatting artifacts
        formatted = re.sub(r'\s*│\s*', ' ', formatted)
        formatted = re.sub(r'\s+', ' ', formatted)
        formatted = re.sub(r'^\s*[-*]\s*', '', formatted, flags=re.MULTILINE)
        
        # Add confidence if available
        if confidence > 0:
            formatted = f"🎯 Confidence: {confidence:.1%}\n\n{formatted}"
        
        # Add resource box
        resource_box = create_resource_box(sources, citations_from_answer)
        return f"{formatted}\n\n{resource_box}"
    
    console = Console()
    
    # Format main answer
    formatted_answer = format_equations(answer)
    formatted_answer = format_citations(formatted_answer)
    
    # Clean up formatting artifacts
    formatted_answer = re.sub(r'\s*│\s*', ' ', formatted_answer)
    formatted_answer = re.sub(r'\s+', ' ', formatted_answer)
    formatted_answer = re.sub(r'^\s*[-*]\s*', '', formatted_answer, flags=re.MULTILINE)
    
    # Create confidence indicator
    confidence_text = ""
    if confidence > 0:
        if confidence >= 0.8:
            confidence_text = f"[bold green]🎯 Confidence: {confidence:.1%}[/bold green]"
        elif confidence >= 0.6:
            confidence_text = f"[bold yellow]🎯 Confidence: {confidence:.1%}[/bold yellow]"
        else:
            confidence_text = f"[bold red]🎯 Confidence: {confidence:.1%}[/bold red]"
    
    # Create main answer panel
    answer_content = formatted_answer
    if confidence_text:
        answer_content = f"{confidence_text}\n\n{formatted_answer}"
    
    console.print(Panel(
        Markdown(answer_content),
        title="[bold cyan]💡 Answer[/bold cyan]",
        border_style="cyan",
        padding=(1, 2)
    ))
    
    # Create and display resource box
    resource_panel = create_resource_box(sources, citations_from_answer)
    console.print(resource_panel)
    
    return answer  # Return original for API compatibility

def pretty_print_answer(result: Dict[str, Any], use_rich: bool = True) -> str:
    """Format answer output with enhanced styling - updated to use enhanced version"""
    return pretty_print_answer_enhanced(result, use_rich)

def format_for_terminal(result: Dict[str, Any]) -> None:
    """Print formatted output to terminal using rich if available"""
    pretty_print_answer_enhanced(result, use_rich=RICH_AVAILABLE) 