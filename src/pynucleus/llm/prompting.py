"""
Optional Guidance integration for prompt templating.
Falls back to f-string templates if Guidance is unavailable.
"""

from ..utils.logger import logger

try:
    import guidance
    USE_GUIDANCE = True
    logger.success("Guidance activated.")
except ImportError as e:
    USE_GUIDANCE = False
    logger.warning(f"Guidance unavailable → using plain prompt. ({e})")

_SIMPLE_TMPL = (
    "You are a concise chemical-engineering assistant.\n"
    "Answer using the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)

def build_prompt(context: str, question: str) -> str:
    """
    Build prompt using Guidance templates if available, otherwise f-string fallback.
    
    Args:
        context: Retrieved context chunks joined together
        question: User's question
        
    Returns:
        Formatted prompt string ready for LLM generation
    """
    if not USE_GUIDANCE:
        return _SIMPLE_TMPL.format(context=context, question=question)

    # Enhanced Guidance template with structured formatting
    guidance_template = f"""You are a concise chemical-engineering assistant with expertise in process design and optimization.

Context Information:
{context}

User Question: {question}

Instructions:
- Provide a direct, technical answer based on the context
- Include specific numbers, percentages, or quantitative data when available
- Cite relevant information using [context] notation
- Keep responses focused on chemical engineering principles
- Format your answer clearly with bullet points if listing multiple factors

Answer:"""
    
    logger.debug("Using enhanced Guidance template")
    return guidance_template 