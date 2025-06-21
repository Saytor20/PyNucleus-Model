"""
Enhanced prompting system with concise, direct responses.
"""

from ..utils.logger import logger
from ..settings import settings

def build_prompt(context: str, question: str, max_context_chars=None) -> str:
    """
    Build a concise prompt that produces direct answers without showing reasoning.
    
    Args:
        context: The retrieved context documents
        question: The user's question
        max_context_chars: Optional context truncation limit (backward-compatible)
    
    Returns:
        Concise prompt designed for direct answers
    """
    # Handle backward compatibility - use setting if not provided
    if max_context_chars is None:
        max_context_chars = getattr(settings, 'MAX_CONTEXT_CHARS', 1200)
    
    # Truncate context if needed
    if len(context) > max_context_chars:
        context = context[:max_context_chars]
        logger.info(f"Context truncated to {max_context_chars} characters")
    
    # Ultra-direct prompt with strong constraints
    prompt = f"""You are a chemical engineering expert. Answer the question using only the provided context.

STRICT RULES:
- Give only the direct answer
- Use facts from the context
- Cite sources as [1], [2], etc.
- Do NOT explain your reasoning
- Do NOT mention what you're thinking
- Do NOT discuss how to answer
- Do NOT repeat text from the context
- Do NOT start with "The following" or similar phrases
- Give a single, clear sentence that directly answers the question
- STOP after giving the answer

CONTEXT:
{context}

QUESTION: {question}

DIRECT ANSWER:"""

    return prompt


def build_simple_prompt(context: str, question: str, max_context_chars=None) -> str:
    """
    Build a simpler prompt for basic queries (backward compatibility).
    
    Args:
        context: The retrieved context documents
        question: The user's question
        max_context_chars: Optional context truncation limit
    
    Returns:
        Simple prompt without step-by-step reasoning
    """
    # Handle backward compatibility
    if max_context_chars is None:
        max_context_chars = getattr(settings, 'MAX_CONTEXT_CHARS', 1200)
    
    # Truncate context if needed
    if len(context) > max_context_chars:
        context = context[:max_context_chars]
    
    return f"""You are an expert chemical process engineer.
Use the context below to answer the question clearly and precisely. Include citations [1], [2], etc. when referencing specific sources.

Context:
{context}

Question: {question}

Answer:"""


# Export the main function and backward compatibility
__all__ = ["build_prompt", "build_simple_prompt"] 