"""
Enhanced RAG Prompting System
"""

try:
    from ..utils.logger import logger
except ImportError:
    # Fallback for when module is run directly
    import logging
    logger = logging.getLogger(__name__)

from ..settings import settings

def build_enhanced_rag_prompt(context: str, question: str, max_context_chars=None) -> str:
    if max_context_chars is None:
        max_context_chars = getattr(settings, 'MAX_CONTEXT_CHARS', 5000)

    if len(context) > max_context_chars:
        context = context[:max_context_chars]
        logger.info(f"Context truncated to {max_context_chars} characters")

    # Detect if this is a complex question
    from ..rag.engine import is_complex_question
    is_complex = is_complex_question(question)
    
    if is_complex:
        # Enhanced prompt for complex questions
        prompt = f"""You are an expert chemical engineer assistant. Provide a comprehensive, structured answer using the given context. Summarize in your own words.

### QUESTION:
{question}

### RETRIEVED CONTEXT:
{context}

### INSTRUCTIONS:
- Provide a complete, structured answer with clear steps or components.
- For design questions, include key design principles, considerations, and methodology.
- Avoid repetition and stay focused on the specific question.
- Clearly enumerate sources like [Doc-XX] after your synthesized answer.
- If context is insufficient, state clearly that more information is needed.
- Keep your answer comprehensive but concise and well-organized.

### ANSWER:
"""
    else:
        # Standard prompt for simple questions
        prompt = f"""You are an expert chemical engineer assistant. Provide a concise, clear answer using the given context. Summarize in your own words.

### QUESTION:
{question}

### RETRIEVED CONTEXT:
{context}

### INSTRUCTIONS:
- Synthesize a short, clear, original answer based on the context.
- Avoid copying long phrases verbatim.
- Clearly enumerate sources like [Doc-XX] after your synthesized answer.
- If context is insufficient, state clearly that more information is needed.

### ANSWER:
"""
    
    return prompt

def build_prompt(context: str, question: str, max_context_chars=None) -> str:
    return build_enhanced_rag_prompt(context, question, max_context_chars)

def build_simple_prompt(context: str, question: str, max_context_chars=None) -> str:
    if max_context_chars is None:
        max_context_chars = getattr(settings, 'MAX_CONTEXT_CHARS', 5000)

    if len(context) > max_context_chars:
        context = context[:max_context_chars]

    return f"""
You are an experienced chemical engineer. Provide a clear answer based on the context provided.

Context:
{context}

Question:
{question}

Answer concisely with citations as [Source: name].

Answer:
"""

def build_detailed_prompt(context: str, question: str, max_context_chars=None) -> str:
    if max_context_chars is None:
        max_context_chars = getattr(settings, 'MAX_CONTEXT_CHARS', 5000)

    if len(context) > max_context_chars:
        context = context[:max_context_chars]

    return f"""
You are a senior chemical engineering consultant.

### TECHNICAL QUESTION:
{question}

### TECHNICAL CONTEXT:
{context}

### RESPONSE REQUIREMENTS:
- Provide detailed technical explanations with clarity.
- Structure logically with relevant technical details and terminology.
- Include citations clearly as [Source: name].
- Provide examples or practical applications if contextually available.
- If insufficient information is available, explicitly mention that.

### DETAILED ANSWER:
"""

__all__ = ["build_prompt", "build_enhanced_rag_prompt", "build_simple_prompt", "build_detailed_prompt"]