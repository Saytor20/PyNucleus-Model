"""
Optional Guidance integration for prompt templating.
Falls back to f-string templates if Guidance is unavailable.
"""

from ..utils.logger import logger
from ..settings import settings

try:
    import guidance
    _USE_GUIDANCE = True
    logger.success("Guidance available")
except Exception as e:
    _USE_GUIDANCE = False
    logger.warning(f"Guidance off → plain prompts. ({e})")

# Export for external use/testing
USE_GUIDANCE = _USE_GUIDANCE

def _grade(ctx:str)->str:
    if not ctx or "No relevant" in ctx: return "none"
    l = len(ctx)
    if l < 200:  return "low"
    if l < 800:  return "medium"
    return "high"

def build_prompt(context: str, question: str, max_context_chars=1200) -> str:
    context = context[:max_context_chars]
    return f"""You are an expert chemical-process engineer.
Use ONLY the context below to answer clearly and precisely. Cite explicitly when possible.

Context:
{context}

Question:
{question}

Answer:
"""

# Define public exports
__all__ = ["build_prompt", "USE_GUIDANCE"] 