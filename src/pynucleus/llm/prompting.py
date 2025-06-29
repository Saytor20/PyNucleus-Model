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
    """Build enhanced RAG prompt with chemical engineering domain expertise."""
    if max_context_chars is None:
        max_context_chars = getattr(settings, 'MAX_CONTEXT_CHARS', 5000)

    if len(context) > max_context_chars:
        context = context[:max_context_chars]
        logger.info(f"Context truncated to {max_context_chars} characters")

    # Detect if this is a complex question
    from ..rag.engine import is_complex_question
    is_complex = is_complex_question(question)
    
    # Detect question type for specialized prompting
    question_type = _detect_question_type(question)
    
    if is_complex:
        # Enhanced prompt for complex questions with domain expertise
        prompt = f"""You are a senior chemical engineer with 15+ years of experience in process design, equipment selection, and plant operations. Provide a comprehensive, technically accurate answer using the provided context.

### QUESTION:
{question}

### RETRIEVED CONTEXT:
{context}

### INSTRUCTIONS:
- Answer as a chemical engineering expert would explain to a colleague
- Use proper chemical engineering terminology and principles
- Structure your answer with clear sections: definition, principles, applications, considerations
- For process questions: include key parameters, operating conditions, and design considerations  
- For equipment questions: include design principles, sizing factors, and operational aspects
- For separation processes: include mechanisms, driving forces, and efficiency factors
- Be technically precise but clearly explained
- Include relevant equations, principles, or design rules when applicable
- Cite sources using [Doc-XX] format at the end
- If context is insufficient, clearly state what additional information would be needed

### CHEMICAL ENGINEERING ANSWER:
"""
    else:
        # Enhanced prompt for simple questions with domain focus
        if question_type == "definition":
            prompt = f"""You are a chemical engineering professor explaining a fundamental concept. Provide a clear, accurate definition using the provided context.

### QUESTION:
{question}

### RETRIEVED CONTEXT:
{context}

### INSTRUCTIONS:
- Start with a clear, concise definition
- Explain the underlying principles or mechanisms
- Provide a practical example or application in chemical engineering
- Use proper technical terminology
- Keep the answer focused and well-structured
- Cite sources using [Doc-XX] format

### DEFINITION AND EXPLANATION:
"""
        elif question_type == "process":
            prompt = f"""You are a process engineer explaining a chemical process. Provide a clear, step-by-step explanation using the provided context.

### QUESTION:
{question}

### RETRIEVED CONTEXT:
{context}

### INSTRUCTIONS:
- Explain the process steps clearly and logically
- Include key operating conditions (temperature, pressure, flow rates)
- Mention important equipment used
- Explain the chemical/physical principles involved
- Include any safety or operational considerations
- Cite sources using [Doc-XX] format

### PROCESS EXPLANATION:
"""
        elif question_type == "equipment":
            prompt = f"""You are an equipment design engineer explaining chemical engineering equipment. Provide a technical but clear explanation using the provided context.

### QUESTION:
{question}

### RETRIEVED CONTEXT:
{context}

### INSTRUCTIONS:
- Describe the equipment function and design principles
- Include key design parameters and considerations
- Explain operating principles and typical applications
- Mention sizing factors or selection criteria
- Include any maintenance or operational aspects
- Cite sources using [Doc-XX] format

### EQUIPMENT EXPLANATION:
"""
        else:
            # General chemical engineering prompt
            prompt = f"""You are a chemical engineer providing technical guidance. Answer the question clearly and accurately using the provided context.

### QUESTION:
{question}

### RETRIEVED CONTEXT:
{context}

### INSTRUCTIONS:
- Provide a clear, technically accurate answer
- Use proper chemical engineering terminology
- Structure your response logically
- Include relevant technical details
- Be concise but comprehensive
- Cite sources using [Doc-XX] format at the end

### TECHNICAL ANSWER:
"""
    
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
    prompt = f"""You are a chemical engineering expert tasked with improving a poor quality answer.

### ORIGINAL QUESTION:
{question}

### POOR QUALITY ANSWER:
{poor_answer}

### AVAILABLE CONTEXT:
{context}

### INSTRUCTIONS:
- Identify what makes the original answer poor quality
- Rewrite the answer to be technically accurate and well-structured
- Use proper chemical engineering terminology
- Ensure the answer directly addresses the question
- Keep it concise but comprehensive
- Include relevant technical details from the context

### IMPROVED ANSWER:
"""
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