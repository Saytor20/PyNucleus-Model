"""
Simple Local Adapter - Direct transformers integration without LangChain
Fallback when LangChain has compatibility issues
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class SimpleRetrievedContext:
    """Represents retrieved context from RAG"""
    content: str
    source: str
    score: float


@dataclass 
class SimpleStructuredResponse:
    """Structured response from simple local adapter"""
    answer: str
    reasoning: str
    confidence: float
    citations: List[str]


class SimpleLocalAdapter:
    """
    Simple local adapter using transformers directly
    No LangChain dependency - works around version conflicts
    """
    
    def __init__(self, model_id: str, max_tokens: int = 500, device: str = "auto"):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.tokenizer = None
        self.configured = False
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_model()
        else:
            self.logger.warning("Transformers not available - SimpleLocalAdapter disabled")
    
    def _initialize_model(self):
        """Initialize the model using transformers directly"""
        try:
            self.logger.info(f"Initializing simple local model: {self.model_id}")
            
            # Set device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with minimal configuration to avoid cache issues
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
                low_cpu_mem_usage=True
            )
            
            self.configured = True
            self.logger.info("✅ SimpleLocalAdapter successfully configured")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize simple local model: {e}")
            self.configured = False
    
    def chain_of_thought(
        self, 
        question: str, 
        context: Optional[List[SimpleRetrievedContext]] = None
    ) -> SimpleStructuredResponse:
        """
        Simple Chain of Thought reasoning with direct model call
        """
        if not self.configured:
            raise RuntimeError("SimpleLocalAdapter not configured")
        
        # Build structured prompt
        prompt = self._build_cot_prompt(question, context)
        
        try:
            # Generate response using direct model call
            self.logger.debug(f"Generating response for prompt length: {len(prompt)}")
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.device)
            
            # Generate with careful parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(self.max_tokens, 500),
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # Disable cache to avoid version issues
                    num_return_sequences=1
                )
            
            # Decode only the new tokens
            input_length = inputs['input_ids'].shape[1]
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            self.logger.debug(f"Generated response length: {len(response)}")
            
            # Parse structured output
            parsed_response = self._parse_structured_response(response)
            
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"Simple chain of thought failed: {e}")
            self.logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
            
            return SimpleStructuredResponse(
                answer=f"I encountered a technical error while processing your question. Error: {str(e)[:100]}",
                reasoning="Technical error during model inference",
                confidence=0.1,
                citations=[]
            )
    
    def _build_cot_prompt(
        self, 
        question: str, 
        context: Optional[List[SimpleRetrievedContext]] = None
    ) -> str:
        """
        Build a structured Chain of Thought prompt
        """
        context_section = ""
        if context:
            context_texts = []
            for i, ctx in enumerate(context[:3], 1):  # Limit to top 3 contexts
                context_texts.append(f"[{i}] {ctx.content[:300]}... (Source: {ctx.source})")
            context_section = f"""
Context Information:
{chr(10).join(context_texts)}

"""
        
        prompt = f"""You are a helpful technical assistant. Please answer the question step by step.

{context_section}Question: {question}

Please provide your response in this format:

REASONING:
[Your step-by-step thinking process]

ANSWER:
[Your clear, direct answer]

CONFIDENCE:
[A number from 0.0 to 1.0]

CITATIONS:
[Any sources you referenced]

Response:"""

        return prompt
    
    def _parse_structured_response(self, response: str) -> SimpleStructuredResponse:
        """
        Parse the structured response from the model
        """
        try:
            # Initialize defaults
            reasoning = ""
            answer = ""
            confidence = 0.5
            citations = []
            
            # Parse response sections
            lines = response.split('\n')
            current_section = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.upper().startswith('REASONING:'):
                    if current_section and current_content:
                        self._assign_content(current_section, current_content, reasoning, answer, confidence, citations)
                    current_section = 'reasoning'
                    current_content = [line[10:].strip()] if len(line) > 10 else []
                elif line.upper().startswith('ANSWER:'):
                    if current_section == 'reasoning':
                        reasoning = '\n'.join(current_content)
                    current_section = 'answer'
                    current_content = [line[7:].strip()] if len(line) > 7 else []
                elif line.upper().startswith('CONFIDENCE:'):
                    if current_section == 'answer':
                        answer = '\n'.join(current_content)
                    current_section = 'confidence'
                    conf_text = line[11:].strip()
                    try:
                        confidence = float(conf_text)
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                    except ValueError:
                        confidence = 0.5
                    current_content = []
                elif line.upper().startswith('CITATIONS:'):
                    current_section = 'citations'
                    current_content = []
                else:
                    if current_section:
                        current_content.append(line)
            
            # Handle final section
            if current_section == 'reasoning':
                reasoning = '\n'.join(current_content)
            elif current_section == 'answer':
                answer = '\n'.join(current_content)
            elif current_section == 'citations':
                citations = [c.strip() for c in current_content if c.strip()]
            
            # Fallback if parsing failed
            if not answer and not reasoning:
                answer = response[:300] + "..." if len(response) > 300 else response
                reasoning = "Generated using simple local model"
            
            return SimpleStructuredResponse(
                answer=answer if answer else reasoning,
                reasoning=reasoning if reasoning else "Direct model response",
                confidence=confidence,
                citations=citations
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse structured response: {e}")
            return SimpleStructuredResponse(
                answer=response[:300] + "..." if len(response) > 300 else response,
                reasoning="Parsing failed, returning raw response",
                confidence=0.3,
                citations=[]
            )
    
    def _assign_content(self, section: str, content: List[str], reasoning: str, answer: str, confidence: float, citations: List[str]):
        """Helper to assign content to variables (placeholder for now)"""
        pass 