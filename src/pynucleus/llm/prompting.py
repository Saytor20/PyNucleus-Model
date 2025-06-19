"""
Optional Guidance integration for prompt templating.
Falls back to f-string templates if Guidance is unavailable.
"""

from ..utils.logger import logger

try:
    import guidance
    guidance.llm = guidance.llms.Transformers("Qwen/Qwen1.5-0.5B-Chat")
    USE_GUIDANCE = True
    logger.success("Guidance activated.")
except Exception as e:
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

    @guidance
    def qa(g):
        g += "You are a concise chemical-engineering assistant.\n"
        g += "Context:\n{{context}}\n"
        g += "Question: {{question}}\n"
        g += "Answer (cite numbers in square brackets): "
        g += guidance.gen(name="answer", max_tokens=256)
    
    result = qa(context=context, question=question)
    return result["answer"] 