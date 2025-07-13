"""
Enhanced RAG Prompting System with Chemical Engineering Domain Expertise
"""

try:
    from ..utils.logger import logger
except ImportError:
    # Fallback for when module is run directly
    import logging
    logger = logging.getLogger(__name__)

from ..settings import settings

def build_enhanced_rag_prompt(context: str, question: str, max_context_chars=None) -> str:
    """
    Build enhanced RAG prompt for chemical engineering questions with context.
    
    Args:
        context: Retrieved document context
        question: User question
        max_context_chars: Maximum context length (optional)
        
    Returns:
        Formatted prompt string
    """
    if max_context_chars is None:
        max_context_chars = getattr(settings, 'MAX_CONTEXT_CHARS', 2000)
    
    # Truncate context if needed
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "..."
    
    # Detect question complexity and type
    from ..rag.engine import is_complex_question
    is_complex = is_complex_question(question)
    question_type = _detect_question_type(question)
    
    if is_complex:
        # Enhanced prompt for complex questions with domain expertise
        prompt = f"""You are a senior chemical engineer providing a comprehensive explanation.

### QUESTION:
{question}

### TECHNICAL INFORMATION PROVIDED:
{context}

### CRITICAL REQUIREMENTS:
- Write your OWN comprehensive explanation using your chemical engineering expertise
- DO NOT copy document headers, section numbers, page references, or formatting
- DO NOT copy any author names, contact information, or university affiliations
- DO NOT include phone numbers, emails, or personal details
- DO NOT start with "Technical Information" or any document markers
- SYNTHESIZE the information into a clear, professional explanation
- STRUCTURE your response with proper paragraphs and flow
- INCLUDE technical details, principles, and practical considerations
- WRITE as if explaining to a fellow engineer in plain professional language
- ONLY use the technical content, completely ignore any personal/contact information

### YOUR EXPLANATION:"""
    
    elif question_type == "process":
        prompt = f"""You are a process engineer explaining chemical engineering processes. Using the context provided, synthesize a clear explanation of the process.

### QUESTION:
{question}

### CONTEXT FROM DOCUMENTS:
{context}

### INSTRUCTIONS:
- SYNTHESIZE your own explanation using the context as reference
- DO NOT copy text directly - explain in your own professional words
- DESCRIBE the process steps clearly and logically
- INCLUDE key operating conditions and principles
- EXPLAIN the chemical/physical mechanisms involved
- CITE sources using [Doc-XX] format appropriately
- WRITE as a process engineering explanation

### YOUR PROCESS EXPLANATION:
"""
        
    elif question_type == "equipment":
        prompt = f"""You are an equipment design engineer. Using the provided context, synthesize a technical explanation about the equipment.

### QUESTION:
{question}

### CONTEXT FROM DOCUMENTS:
{context}

### INSTRUCTIONS:
- SYNTHESIZE your own explanation using the context as reference
- DO NOT copy text directly from documents
- EXPLAIN equipment function and design principles clearly
- INCLUDE key design parameters and considerations
- DESCRIBE operating principles and applications
- ADD citations using [Doc-XX] format where appropriate
- WRITE as a technical equipment explanation

### YOUR EQUIPMENT EXPLANATION:
"""
    else:
        # Concise general chemical engineering prompt
        prompt = f"""You are a chemical engineer answering a colleague's question.

Question: {question}

Context: {context}

Provide a direct answer in 2-3 sentences. Do not mention the context or reference information. Just answer the question directly as if you know the answer from your expertise.

Answer:"""
    
    return prompt

def _detect_question_type(question: str) -> str:
    """Detect the type of chemical engineering question for specialized prompting."""
    question_lower = question.lower()
    
    # Definition questions
    definition_keywords = ["what is", "define", "definition", "meaning of", "explain what"]
    if any(keyword in question_lower for keyword in definition_keywords):
        return "definition"
    
    # Process questions  
    process_keywords = ["process", "procedure", "steps", "how to", "method", "technique"]
    if any(keyword in question_lower for keyword in process_keywords):
        return "process"
    
    # Equipment questions
    equipment_keywords = ["equipment", "device", "apparatus", "design", "sizing", "selection"]
    if any(keyword in question_lower for keyword in equipment_keywords):
        return "equipment"
    
    # Default to general
    return "general"

def build_answer_validation_prompt(question: str, answer: str, expected_keywords: list) -> str:
    """Build prompt for validating answer quality and accuracy."""
    keywords_str = ", ".join(expected_keywords)
    
    prompt = f"""You are a chemical engineering expert reviewer. Evaluate the quality and accuracy of this answer.

### ORIGINAL QUESTION:
{question}

### GENERATED ANSWER:
{answer}

### EXPECTED TOPICS/KEYWORDS:
{keywords_str}

### EVALUATION CRITERIA:
1. Technical accuracy - Is the information correct?
2. Completeness - Does it address the question fully?
3. Relevance - Is the content relevant to the question?
4. Chemical engineering focus - Uses proper terminology and concepts?
5. Clarity - Is it well-structured and understandable?

### INSTRUCTIONS:
- Rate each criteria on a scale of 1-5 (5 = excellent)
- Provide specific feedback on what's good or needs improvement
- Suggest key missing information if any
- Flag any technical inaccuracies

### EVALUATION:
Technical Accuracy (1-5): 
Completeness (1-5):
Relevance (1-5): 
Chemical Engineering Focus (1-5):
Clarity (1-5):

### DETAILED FEEDBACK:
"""
    return prompt

def build_answer_improvement_prompt(question: str, poor_answer: str, context: str) -> str:
    """Build prompt for improving a poor quality answer."""
    prompt = f"""Please provide a clear, concise answer to this question using the context provided.

Question: {question}

Context: {context}

Instructions:
- Answer directly in 1-3 sentences
- Use technical terms appropriately
- Include citations as [Doc-XX]
- Be factual and concise

Answer:"""
    return prompt

# Backward compatibility function
def build_prompt(context: str, question: str) -> str:
    """Backward compatibility wrapper for build_enhanced_rag_prompt."""
    return build_enhanced_rag_prompt(context, question)

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