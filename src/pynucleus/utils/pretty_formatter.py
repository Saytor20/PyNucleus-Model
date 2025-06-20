"""
Pretty formatting utilities for enhanced output display
"""
import re
from typing import Dict, Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.text import Text
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

def pretty_print_answer(result: Dict[str, Any], use_rich: bool = True) -> str:
    """Format answer output with enhanced styling"""
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    
    if not use_rich or not RICH_AVAILABLE:
        # Fallback to plain text with basic formatting
        formatted = format_equations(answer)
        formatted = format_citations(formatted)
        return formatted
    
    console = Console()
    
    # Split answer and references
    if "References:" in answer:
        parts = answer.split("References:", 1)
        main_answer = parts[0].strip()
        references = "References:" + parts[1]
    else:
        main_answer = answer.strip()
        references = ""
    
    # Format main answer
    formatted_answer = format_equations(main_answer)
    formatted_answer = format_citations(formatted_answer)
    
    # Create rich output
    console.print(Panel(
        Markdown(formatted_answer), 
        title="[bold cyan]Answer[/bold cyan]",
        border_style="cyan"
    ))
    
    if references:
        formatted_refs = format_citations(references)
        console.print(Panel(
            Markdown(formatted_refs),
            title="[bold green]References[/bold green]", 
            border_style="green"
        ))
    
    return answer  # Return original for API compatibility

def format_for_terminal(result: Dict[str, Any]) -> None:
    """Print formatted output to terminal using rich if available"""
    pretty_print_answer(result, use_rich=RICH_AVAILABLE) 