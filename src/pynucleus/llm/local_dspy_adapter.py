"""
Local DSPy Adapter - Provides DSPy-like structured prompting for local models
Compatible with HuggingFace models via LangChain
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

try:
    from langchain_huggingface import HuggingFacePipeline
    from langchain.schema import BaseMessage, HumanMessage, SystemMessage
    from langchain.prompts import PromptTemplate
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback to langchain-community if langchain-huggingface not available
        from langchain_community.llms import HuggingFacePipeline
        from langchain.schema import BaseMessage, HumanMessage, SystemMessage
        from langchain.prompts import PromptTemplate
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        try:
            # Final fallback to deprecated import
            from langchain.llms import HuggingFacePipeline
            from langchain.schema import BaseMessage, HumanMessage, SystemMessage
            from langchain.prompts import PromptTemplate
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            LANGCHAIN_AVAILABLE = True
        except ImportError:
            LANGCHAIN_AVAILABLE = False


@dataclass
class RetrievedContext:
    """Represents retrieved context from RAG"""
    content: str
    source: str
    score: float


@dataclass 
class StructuredResponse:
    """Structured response from local DSPy adapter"""
    answer: str
    reasoning: str
    confidence: float
    citations: List[str]


class LocalDSPyAdapter:
    """
    Local DSPy-like adapter that provides structured prompting for local models
    Mimics DSPy's ChainOfThought and RetrieveGenerateAnswer patterns
    """
    
    def __init__(self, model_id: str, max_tokens: int = 1000, device: str = "auto"):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        self.llm = None
        self.tokenizer = None
        self.configured = False
        
        if LANGCHAIN_AVAILABLE:
            self._initialize_model()
        else:
            self.logger.warning("LangChain not available - LocalDSPyAdapter disabled")
    
    def _initialize_model(self):
        """Initialize the local model using LangChain + HuggingFace"""
        try:
            self.logger.info(f"Initializing local model: {self.model_id}")
            
            # Determine the appropriate device
            device_map = None
            if self.device == "auto":
                device_map = "auto"
            elif self.device in ["cpu", "cuda", "mps"]:
                device_map = self.device
            else:
                # Default to auto for unrecognized devices
                device_map = "auto"
            
            # Create HuggingFace pipeline with improved configuration
            hf_pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                max_new_tokens=self.max_tokens,
                device_map=device_map,
                trust_remote_code=True,
                return_full_text=False,
                pad_token_id=50256,  # Set explicit pad token
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                # Disable problematic cache features
                use_cache=False
            )
            
            # Wrap with LangChain
            self.llm = HuggingFacePipeline(
                pipeline=hf_pipeline,
                model_kwargs={
                    "temperature": 0.7,
                    "max_new_tokens": self.max_tokens,
                    "do_sample": True
                }
            )
            
            # Initialize tokenizer separately for safety
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as tokenizer_error:
                self.logger.warning(f"Tokenizer initialization failed: {tokenizer_error}")
                self.tokenizer = None
            
            self.configured = True
            self.logger.info("✅ LocalDSPyAdapter successfully configured")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize local model: {e}")
            self.configured = False
    
    def chain_of_thought(
        self, 
        question: str, 
        context: Optional[List[RetrievedContext]] = None
    ) -> StructuredResponse:
        """
        DSPy-like Chain of Thought reasoning with local model
        """
        if not self.configured:
            raise RuntimeError("LocalDSPyAdapter not configured")
        
        # Build structured prompt
        prompt = self._build_cot_prompt(question, context)
        
        try:
            # Generate response with better error handling
            self.logger.debug(f"Generating response for prompt length: {len(prompt)}")
            
            # Use the pipeline directly to avoid LangChain wrapper issues
            if hasattr(self.llm, 'pipeline'):
                # Access the underlying pipeline
                pipeline_response = self.llm.pipeline(
                    prompt,
                    max_new_tokens=min(self.max_tokens, 500),
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.llm.pipeline.tokenizer.eos_token_id,
                    eos_token_id=self.llm.pipeline.tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                if isinstance(pipeline_response, list) and len(pipeline_response) > 0:
                    response = pipeline_response[0].get('generated_text', '')
                else:
                    response = str(pipeline_response)
            else:
                # Fallback to LangChain invoke
                response = self.llm.invoke(prompt)
            
            self.logger.debug(f"Generated response length: {len(response)}")
            
            # Parse structured output
            parsed_response = self._parse_structured_response(response)
            
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"Chain of thought failed: {e}")
            self.logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
            
            # Provide a more helpful fallback response
            return StructuredResponse(
                answer=f"I encountered a technical error while processing your question about: {question[:50]}{'...' if len(question) > 50 else ''}. This might be due to model compatibility issues.",
                reasoning="Technical error occurred during model inference. This could be related to model version compatibility or memory constraints.",
                confidence=0.1,
                citations=[]
            )
    
    def retrieve_generate_answer(
        self,
        question: str,
        contexts: List[RetrievedContext]
    ) -> StructuredResponse:
        """
        DSPy-like Retrieve-Generate-Answer pattern
        """
        if not self.configured:
            raise RuntimeError("LocalDSPyAdapter not configured")
        
        # Use chain of thought with context
        return self.chain_of_thought(question, contexts)
    
    def _build_cot_prompt(
        self, 
        question: str, 
        context: Optional[List[RetrievedContext]] = None
    ) -> str:
        """
        Build a structured Chain of Thought prompt
        """
        context_section = ""
        if context:
            context_texts = []
            for i, ctx in enumerate(context[:5], 1):  # Limit to top 5 contexts
                context_texts.append(f"[{i}] {ctx.content} (Source: {ctx.source})")
            context_section = f"""
Context Information:
{chr(10).join(context_texts)}

"""
        
        prompt = f"""You are an expert technical assistant. Please answer the question using the following structured format:

{context_section}Question: {question}

Please provide your response in the following format:

REASONING:
[Step-by-step reasoning process, citing specific sources when available]

ANSWER:
[Clear, concise answer to the question]

CONFIDENCE:
[Confidence level from 0.0 to 1.0]

CITATIONS:
[List any sources referenced, one per line]

Now, please respond:"""

        return prompt
    
    def _parse_structured_response(self, response: str) -> StructuredResponse:
        """
        Parse the structured response from the model
        """
        try:
            # Initialize defaults
            reasoning = ""
            answer = ""
            confidence = 0.5
            citations = []
            
            # Split response into sections
            sections = response.upper().split('\n\n')
            current_section = None
            current_content = []
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.upper().startswith('REASONING:'):
                    if current_section and current_content:
                        self._assign_section_content(current_section, current_content, 
                                                   reasoning, answer, confidence, citations)
                    current_section = 'reasoning'
                    current_content = [line[10:].strip()] if len(line) > 10 else []
                elif line.upper().startswith('ANSWER:'):
                    if current_section and current_content:
                        reasoning = self._assign_section_content('reasoning', current_content, 
                                                               reasoning, answer, confidence, citations)[0]
                    current_section = 'answer'
                    current_content = [line[7:].strip()] if len(line) > 7 else []
                elif line.upper().startswith('CONFIDENCE:'):
                    if current_section and current_content:
                        if current_section == 'reasoning':
                            reasoning = '\n'.join(current_content)
                        elif current_section == 'answer':
                            answer = '\n'.join(current_content)
                    current_section = 'confidence'
                    conf_text = line[11:].strip()
                    try:
                        confidence = float(conf_text)
                    except ValueError:
                        confidence = 0.5
                    current_content = []
                elif line.upper().startswith('CITATIONS:'):
                    if current_section and current_content:
                        if current_section == 'reasoning':
                            reasoning = '\n'.join(current_content)
                        elif current_section == 'answer':
                            answer = '\n'.join(current_content)
                    current_section = 'citations'
                    current_content = []
                else:
                    if current_section:
                        current_content.append(line)
            
            # Handle final section
            if current_section and current_content:
                if current_section == 'reasoning':
                    reasoning = '\n'.join(current_content)
                elif current_section == 'answer':
                    answer = '\n'.join(current_content)
                elif current_section == 'citations':
                    citations = [c.strip() for c in current_content if c.strip()]
            
            # Fallback if parsing failed
            if not answer:
                answer = response[:500] + "..." if len(response) > 500 else response
                reasoning = "Generated using local model"
            
            return StructuredResponse(
                answer=answer,
                reasoning=reasoning,
                confidence=confidence,
                citations=citations
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse structured response: {e}")
            return StructuredResponse(
                answer=response[:500] + "..." if len(response) > 500 else response,
                reasoning="Parsing failed, returning raw response",
                confidence=0.3,
                citations=[]
            )
    
    def _assign_section_content(self, section: str, content: List[str], 
                              reasoning: str, answer: str, confidence: float, 
                              citations: List[str]) -> Tuple[str, str, float, List[str]]:
        """Helper method to assign content to appropriate section"""
        content_text = '\n'.join(content)
        
        if section == 'reasoning':
            return content_text, answer, confidence, citations
        elif section == 'answer':
            return reasoning, content_text, confidence, citations
        elif section == 'citations':
            return reasoning, answer, confidence, [c.strip() for c in content if c.strip()]
        
        return reasoning, answer, confidence, citations 