"""
DSPy-based answer engine for PyNucleus system.
Replaces template-based prompt generation with structured DSPy programs.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

# Import Local DSPy Adapter
try:
    from .local_dspy_adapter import LocalDSPyAdapter, RetrievedContext, StructuredResponse
    LOCAL_DSPY_AVAILABLE = True
except ImportError:
    LOCAL_DSPY_AVAILABLE = False

# Import Simple Local Adapter as fallback
try:
    from .simple_local_adapter import SimpleLocalAdapter, SimpleRetrievedContext, SimpleStructuredResponse
    SIMPLE_LOCAL_AVAILABLE = True
except ImportError:
    SIMPLE_LOCAL_AVAILABLE = False

try:
    from .dspy_program import PyNucleusProgram, OptimizedProgram
    from .retriever_adapter import PyNucleusRetriever, DocumentRetriever
    CORE_MODULES_AVAILABLE = True
except ImportError:
    # Handle case when running as script directly
    PyNucleusProgram = None
    OptimizedProgram = None 
    PyNucleusRetriever = None
    DocumentRetriever = None
    CORE_MODULES_AVAILABLE = False


class DSPyAnswerEngine:
    """DSPy-based answer engine for structured LLM interactions."""
    
    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct", rag_pipeline=None):
        self.model_id = model_id
        self.rag_pipeline = rag_pipeline
        self.logger = logging.getLogger(__name__)
        
        # Initialize DSPy components
        self.dspy_configured = False
        self.program = None
        self.retriever = None
        self.document_retriever = None
        
        # Initialize Local DSPy Adapter
        self.local_dspy_adapter = None
        self.local_dspy_configured = False
        
        # Initialize Simple Local Adapter as fallback
        self.simple_local_adapter = None
        self.simple_local_configured = False
        
        self._initialize_dspy()
    
    def _is_openai_model(self, model_id: str) -> bool:
        """Check if the model is an OpenAI model."""
        openai_models = [
            'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-instruct',
            'gpt-4', 'gpt-4-turbo', 'gpt-4-turbo-preview', 'gpt-4o', 'gpt-4o-mini',
            'text-davinci-003', 'text-davinci-002', 'davinci-002', 'babbage-002'
        ]
        return any(model_id.startswith(model) or model_id == model for model in openai_models)
    
    def _is_anthropic_model(self, model_id: str) -> bool:
        """Check if the model is an Anthropic model."""
        return model_id.startswith('claude-')
    
    def _get_openai_model_name(self, model_id: str) -> str:
        """Extract the OpenAI model name, defaulting to gpt-3.5-turbo if not recognized."""
        if self._is_openai_model(model_id):
            return model_id
        return 'gpt-3.5-turbo'  # Default fallback
    
    def _initialize_dspy(self):
        """Initialize DSPy components."""
        if not DSPY_AVAILABLE:
            self.logger.warning("DSPy not available, falling back to simple responses")
            return
        
        try:
            # Configure DSPy with the specified language model
            self.logger.info(f"Configuring DSPy with model: {self.model_id}")
            
            # Detect model type and configure accordingly
            if self._is_openai_model(self.model_id):
                # OpenAI model configuration
                try:
                    openai_model = self._get_openai_model_name(self.model_id)
                    lm = dspy.LM(f'openai/{openai_model}', max_tokens=1000)
                    dspy.configure(lm=lm)
                    self.logger.info(f"DSPy configured with OpenAI model: {openai_model}")
                except Exception as openai_error:
                    self.logger.warning(f"OpenAI configuration failed: {openai_error}")
                    self.logger.info("Disabling DSPy - will use fallback RAG + LLM system")
                    self.dspy_configured = False
                    return
            
            elif self._is_anthropic_model(self.model_id):
                # Anthropic model configuration
                try:
                    lm = dspy.LM(f'anthropic/{self.model_id}', max_tokens=1000)
                    dspy.configure(lm=lm)
                    self.logger.info(f"DSPy configured with Anthropic model: {self.model_id}")
                except Exception as anthropic_error:
                    self.logger.warning(f"Anthropic configuration failed: {anthropic_error}")
                    self.logger.info("Disabling DSPy - will use fallback RAG + LLM system")
                    self.dspy_configured = False
                    return
            
            else:
                # Local/HuggingFace model - Try LocalDSPyAdapter first
                self.logger.info(f"Local model detected: {self.model_id}")
                
                # Try LocalDSPyAdapter first (LangChain + structured prompting)
                if LOCAL_DSPY_AVAILABLE:
                    try:
                        self.logger.info("Attempting LocalDSPyAdapter configuration...")
                        self.local_dspy_adapter = LocalDSPyAdapter(
                            model_id=self.model_id,
                            max_tokens=1000
                        )
                        
                        if self.local_dspy_adapter.configured:
                            self.local_dspy_configured = True
                            self.logger.info("✅ LocalDSPyAdapter configured successfully")
                            self.logger.info("Using local DSPy-like structured prompting")
                            # Don't configure regular DSPy for local models
                            self.dspy_configured = False
                            return
                        else:
                            self.logger.warning("LocalDSPyAdapter configuration failed")
                            self.local_dspy_adapter = None  # Clear failed adapter
                    except Exception as local_dspy_error:
                        self.logger.warning(f"LocalDSPyAdapter failed: {local_dspy_error}")
                        self.local_dspy_adapter = None  # Clear failed adapter
                else:
                    self.logger.warning("LocalDSPyAdapter not available (LangChain dependency missing)")
                
                # Try SimpleLocalAdapter as fallback if LocalDSPy failed
                if not self.local_dspy_configured and SIMPLE_LOCAL_AVAILABLE:
                    try:
                        self.logger.info("Attempting SimpleLocalAdapter configuration...")
                        self.simple_local_adapter = SimpleLocalAdapter(
                            model_id=self.model_id,
                            max_tokens=500
                        )
                        
                        if self.simple_local_adapter.configured:
                            self.simple_local_configured = True
                            self.logger.info("✅ SimpleLocalAdapter configured successfully")
                            self.logger.info("Using simple direct transformers integration")
                            # Don't configure regular DSPy for local models
                            self.dspy_configured = False
                            return
                        else:
                            self.logger.warning("SimpleLocalAdapter configuration failed")
                    except Exception as simple_local_error:
                        self.logger.warning(f"SimpleLocalAdapter failed: {simple_local_error}")
                elif not self.local_dspy_configured:
                    self.logger.warning("SimpleLocalAdapter not available")
                
                # Mark DSPy as not configured and continue with RAG system
                self.logger.info("Advanced DSPy features not available, using standard RAG + LLM system")
                self.dspy_configured = False
            
            # Initialize programs only if DSPy is properly configured and modules available
            if CORE_MODULES_AVAILABLE and PyNucleusProgram:
                self.program = PyNucleusProgram()
                self.logger.info("DSPy programs initialized")
                
                # Initialize retrievers if RAG pipeline is available
                if self.rag_pipeline and PyNucleusRetriever and DocumentRetriever:
                    self.retriever = PyNucleusRetriever(k=5, rag_pipeline=self.rag_pipeline)
                    self.document_retriever = DocumentRetriever(rag_pipeline=self.rag_pipeline)
                    self.logger.info("DSPy retrievers initialized with RAG pipeline")
                else:
                    self.logger.info("No RAG pipeline provided - retrievers not initialized")
            else:
                self.logger.warning("Core DSPy modules not available - programs not initialized")
            
            self.dspy_configured = True
            self.logger.info("DSPy answer engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"DSPy initialization failed: {e}")
            self.dspy_configured = False
    
    def analyze_report(
        self, 
        question: str, 
        report_path: Path,
        context_window: int = 2000
    ) -> Dict[str, Any]:
        """
        Analyze a report using DSPy structured prompting.
        
        Args:
            question: Question about the report
            report_path: Path to the report file
            context_window: Maximum context length
            
        Returns:
            Dictionary with analysis results and metadata
        """
        start_time = datetime.now()
        
        try:
            # Read report content
            if not report_path.exists():
                return self._create_error_response(
                    f"Report file not found: {report_path}",
                    question,
                    "report_analysis"
                )
            
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # Truncate if too long
            if len(report_content) > context_window:
                report_content = report_content[:context_window] + "..."
                self.logger.info(f"Report content truncated to {context_window} characters")
            
            # Get additional context from retriever if available
            additional_context = []
            if self.document_retriever:
                additional_context = self.document_retriever.retrieve_for_report_analysis(
                    question, 
                    report_type="simulation_report"
                )
            
            # Initialize context variable before conditional blocks
            context = None
            
            # Use DSPy, LocalDSPyAdapter, or SimpleLocalAdapter for structured analysis
            if self.local_dspy_configured and self.local_dspy_adapter:
                # Convert report content to RetrievedContext format
                context = [RetrievedContext(
                    content=report_content,
                    source=str(report_path),
                    score=1.0
                )]
                
                # Add additional context if available
                if additional_context:
                    for ctx in additional_context[:3]:  # Limit additional context
                        context.append(RetrievedContext(
                            content=ctx.get('content', ''),
                            source=ctx.get('source', 'unknown'),
                            score=ctx.get('score', 0.5)
                        ))
                
                structured_response = self.local_dspy_adapter.chain_of_thought(question, context)
                analysis = structured_response.answer
                
            elif self.simple_local_configured and self.simple_local_adapter:
                # Convert report content to SimpleRetrievedContext format if needed
                if context is None:
                    context = [RetrievedContext(
                        content=report_content,
                        source=str(report_path),
                        score=1.0
                    )]
                    
                    # Add additional context if available
                    if additional_context:
                        for ctx in additional_context[:3]:  # Limit additional context
                            context.append(RetrievedContext(
                                content=ctx.get('content', ''),
                                source=ctx.get('source', 'unknown'),
                                score=ctx.get('score', 0.5)
                            ))
                
                simple_context = []
                if context:
                    for ctx in context:
                        simple_context.append(SimpleRetrievedContext(
                            content=ctx.content,
                            source=ctx.source,
                            score=ctx.score
                        ))
                
                structured_response = self.simple_local_adapter.chain_of_thought(question, simple_context)
                analysis = structured_response.answer
                
            elif self.dspy_configured and self.program:
                analysis = self.program.analyze_report(report_content, question)
            else:
                # Fallback to simple analysis
                analysis = self._fallback_report_analysis(question, report_content)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "analysis": analysis,
                "question": question,
                "report_path": str(report_path),
                "report_length": len(report_content),
                "additional_context_count": len(additional_context),
                "query_type": "report_analysis",
                "model_id": self.model_id,
                "generation_time": generation_time,
                "timestamp": datetime.now().isoformat(),
                "dspy_used": self.dspy_configured
            }
            
        except Exception as e:
            self.logger.error(f"Report analysis failed: {e}")
            return self._create_error_response(str(e), question, "report_analysis")
    
    def query_simulation(
        self, 
        question: str, 
        simulation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Query about simulation results using DSPy.
        
        Args:
            question: Question about the simulation
            simulation_data: Simulation results data
            
        Returns:
            Dictionary with query response and metadata
        """
        start_time = datetime.now()
        
        try:
            # Convert simulation data to string format
            sim_data_str = json.dumps(simulation_data, indent=2, default=str)
            
            # Get additional context from retriever if available
            additional_context = []
            if self.document_retriever:
                case_name = simulation_data.get("case_name", "unknown")
                additional_context = self.document_retriever.retrieve_for_simulation(
                    question, 
                    case_name
                )
            
            # Initialize context variable before conditional blocks
            context = None
            
            # Use DSPy or LocalDSPyAdapter for structured querying
            if self.local_dspy_configured and self.local_dspy_adapter:
                # Convert simulation data to RetrievedContext format
                context = [RetrievedContext(
                    content=sim_data_str,
                    source=f"simulation_{simulation_data.get('case_name', 'unknown')}",
                    score=1.0
                )]
                
                # Add additional context if available
                if additional_context:
                    for ctx in additional_context[:3]:  # Limit additional context
                        context.append(RetrievedContext(
                            content=ctx.get('content', ''),
                            source=ctx.get('source', 'unknown'),
                            score=ctx.get('score', 0.5)
                        ))
                
                structured_response = self.local_dspy_adapter.chain_of_thought(question, context)
                response = structured_response.answer
                
            elif self.simple_local_configured and self.simple_local_adapter:
                # Convert simulation data to SimpleRetrievedContext format
                simple_context = []
                if context is None:
                    context = [RetrievedContext(
                        content=sim_data_str,
                        source=f"simulation_{simulation_data.get('case_name', 'unknown')}",
                        score=1.0
                    )]
                    
                    # Add additional context if available
                    if additional_context:
                        for ctx in additional_context[:3]:  # Limit additional context
                            context.append(RetrievedContext(
                                content=ctx.get('content', ''),
                                source=ctx.get('source', 'unknown'),
                                score=ctx.get('score', 0.5)
                            ))
                
                if context:
                    for ctx in context:
                        simple_context.append(SimpleRetrievedContext(
                            content=ctx.content,
                            source=ctx.source,
                            score=ctx.score
                        ))
                
                structured_response = self.simple_local_adapter.chain_of_thought(question, simple_context)
                response = structured_response.answer
                
            elif self.dspy_configured and self.program:
                response = self.program.query_simulation(sim_data_str, question)
            else:
                # Fallback to simple response
                response = self._fallback_simulation_query(question, simulation_data)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "response": response,
                "question": question,
                "simulation_case": simulation_data.get("case_name", "unknown"),
                "additional_context_count": len(additional_context),
                "query_type": "simulation_analysis",
                "model_id": self.model_id,
                "generation_time": generation_time,
                "timestamp": datetime.now().isoformat(),
                "dspy_used": self.dspy_configured
            }
            
        except Exception as e:
            self.logger.error(f"Simulation query failed: {e}")
            return self._create_error_response(str(e), question, "simulation_analysis")
    
    def answer_general(
        self, 
        question: str, 
        context: str = None
    ) -> Dict[str, Any]:
        """
        Answer general chemical engineering questions.
        
        Args:
            question: General question
            context: Optional context
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = datetime.now()
        
        try:
            # Get additional context from retriever if available
            if context is None:
                context = "General chemical engineering knowledge base."
            
            if self.retriever:
                retrieved_context = self.retriever.forward(question, k=3)
                if retrieved_context:
                    context = f"{context}\n\nAdditional context:\n" + "\n".join(retrieved_context)
            
            # Initialize context_list variable before conditional blocks
            context_list = None
            
            # Use DSPy or LocalDSPyAdapter for structured answering
            if self.local_dspy_configured and self.local_dspy_adapter:
                # Convert context to RetrievedContext format if available
                context_list = []
                if context:
                    context_list = [RetrievedContext(
                        content=context,
                        source="provided_context",
                        score=1.0
                    )]
                
                structured_response = self.local_dspy_adapter.chain_of_thought(question, context_list)
                answer = structured_response.answer
                
            elif self.simple_local_configured and self.simple_local_adapter:
                # Convert context to SimpleRetrievedContext format if available
                if context_list is None:
                    context_list = []
                    if context:
                        context_list = [RetrievedContext(
                            content=context,
                            source="provided_context",
                            score=1.0
                        )]
                
                simple_context_list = []
                if context_list:
                    for ctx in context_list:
                        simple_context_list.append(SimpleRetrievedContext(
                            content=ctx.content,
                            source=ctx.source,
                            score=ctx.score
                        ))
                
                structured_response = self.simple_local_adapter.chain_of_thought(question, simple_context_list)
                answer = structured_response.answer
                
            elif self.dspy_configured and self.program:
                answer = self.program.answer_general(context, question)
            else:
                # Fallback to simple answer
                answer = self._fallback_general_answer(question, context)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "answer": answer,
                "question": question,
                "context_provided": bool(context),
                "query_type": "general_query",
                "model_id": self.model_id,
                "generation_time": generation_time,
                "timestamp": datetime.now().isoformat(),
                "dspy_used": self.dspy_configured
            }
            
        except Exception as e:
            self.logger.error(f"General query failed: {e}")
            return self._create_error_response(str(e), question, "general_query")
    
    def batch_query(
        self, 
        questions: List[str], 
        context: str = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions
            context: Shared context for all questions
            
        Returns:
            List of response dictionaries
        """
        results = []
        
        for i, question in enumerate(questions):
            self.logger.info(f"Processing question {i+1}/{len(questions)}")
            
            result = self.answer_general(question, context)
            result.update({
                "batch_index": i,
                "total_questions": len(questions)
            })
            
            results.append(result)
        
        return results
    
    def _fallback_report_analysis(self, question: str, report_content: str) -> str:
        """Fallback analysis when DSPy is not available."""
        return f"Fallback analysis for question '{question}' based on report content (length: {len(report_content)} characters). DSPy analysis not available."
    
    def _fallback_simulation_query(self, question: str, simulation_data: Dict[str, Any]) -> str:
        """Fallback query when DSPy is not available."""
        case_name = simulation_data.get("case_name", "unknown")
        return f"Fallback response for simulation question '{question}' about case '{case_name}'. DSPy querying not available."
    
    def _fallback_general_answer(self, question: str, context: str) -> str:
        """Fallback answer when DSPy is not available."""
        return f"Fallback answer for question '{question}'. Context length: {len(context) if context else 0}. DSPy answering not available."
    
    def _create_error_response(self, error: str, question: str, query_type: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "error": error,
            "question": question,
            "query_type": query_type,
            "model_id": self.model_id,
            "timestamp": datetime.now().isoformat(),
            "dspy_used": False
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status information."""
        return {
            "dspy_available": DSPY_AVAILABLE,
            "dspy_configured": self.dspy_configured,
            "local_dspy_available": LOCAL_DSPY_AVAILABLE,
            "local_dspy_configured": self.local_dspy_configured,
            "simple_local_available": SIMPLE_LOCAL_AVAILABLE,
            "simple_local_configured": self.simple_local_configured,
            "rag_pipeline_connected": self.rag_pipeline is not None,
            "retriever_available": self.retriever is not None,
            "model_id": self.model_id,
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    print("❌ Error: answer_engine.py cannot be run directly due to relative imports")
    print("💡 To use the DSPy Answer Engine, run one of these instead:")
    print("   1. python run_answer_engine.py")
    print("   2. python demo_answer_engine.py")
    print("   3. python test_answer_engine.py")
    print("   4. python -m src.pynucleus.llm.answer_engine")
    print("\n📖 Or import it as a module:")
    print("   from src.pynucleus.llm.answer_engine import DSPyAnswerEngine")
    print("   engine = DSPyAnswerEngine()") 