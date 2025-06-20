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

def _grade(ctx:str)->str:
    if not ctx or "No relevant" in ctx: return "none"
    l = len(ctx)
    if l < 200:  return "low"
    if l < 800:  return "medium"
    return "high"

def build_prompt(ctx:str, q:str)->str:
    tier = _grade(ctx)
    return _simple(ctx, q, tier)

def _simple(ctx:str, q:str, tier:str)->str:
    if tier == "none":
        return f"Human: {q}\n\nAssistant:"
    
    # Truncate context to avoid exceeding token limits
    # Estimate ~2-3 chars per token, keep context under 300 tokens (~900 chars)
    max_context_chars = 900
    if len(ctx) > max_context_chars:
        ctx = ctx[:max_context_chars] + "..."
        logger.info(f"Context truncated to {max_context_chars} characters to fit context window")
    
    return f"Human: Use this context to answer: {ctx}\n\n{q}\n\nAssistant:" 