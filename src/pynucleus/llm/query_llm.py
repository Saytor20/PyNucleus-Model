"""
LLM query manager for PyNucleus system.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

class LLMQueryManager:
    """Manage LLM queries and integrate with PyNucleus pipeline."""
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_id = model_id
        self.logger = logging.getLogger(__name__)
        self.llm_runner = None
        
        # Initialize LLM runner
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM runner."""
        try:
            from .llm_runner import LLMRunner
            self.llm_runner = LLMRunner(self.model_id)
            self.logger.info("LLM Query Manager initialized successfully")
        except Exception as e:
            self.logger.warning(f"LLM initialization failed: {e}")
            self.llm_runner = None
    
    def query_about_report(
        self, 
        question: str, 
        report_path: Path,
        context_window: int = 2000
    ) -> Dict[str, Any]:
        """
        Query LLM about a specific report file.
        
        Args:
            question: Question to ask about the report
            report_path: Path to the report file
            context_window: Maximum context length
            
        Returns:
            Dictionary with LLM response and metadata
        """
        try:
            # Read report content
            if not report_path.exists():
                return {
                    "error": f"Report file not found: {report_path}",
                    "question": question,
                    "timestamp": datetime.now().isoformat()
                }
            
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # Truncate if too long
            if len(report_content) > context_window:
                report_content = report_content[:context_window] + "..."
                self.logger.info(f"Report content truncated to {context_window} characters")
            
            # Create prompt
            prompt = self._create_report_query_prompt(question, report_content)
            
            # Query LLM
            if self.llm_runner:
                response = self.llm_runner.generate_response(prompt)
            else:
                response = {
                    "response": f"LLM not available. Mock response for: {question}",
                    "model_id": "mock",
                    "generation_time": 0,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Add query metadata
            response.update({
                "question": question,
                "report_path": str(report_path),
                "report_length": len(report_content),
                "query_type": "report_analysis"
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Report query failed: {e}")
            return {
                "error": str(e),
                "question": question,
                "report_path": str(report_path),
                "timestamp": datetime.now().isoformat()
            }
    
    def query_simulation_results(
        self, 
        question: str, 
        simulation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Query LLM about simulation results.
        
        Args:
            question: Question about the simulation
            simulation_data: Simulation results data
            
        Returns:
            Dictionary with LLM response and metadata
        """
        try:
            # Create prompt from simulation data
            prompt = self._create_simulation_query_prompt(question, simulation_data)
            
            # Query LLM
            if self.llm_runner:
                response = self.llm_runner.generate_response(prompt)
            else:
                response = {
                    "response": f"LLM not available. Mock response for simulation query: {question}",
                    "model_id": "mock",
                    "generation_time": 0,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Add query metadata
            response.update({
                "question": question,
                "simulation_case": simulation_data.get("case_name", "unknown"),
                "query_type": "simulation_analysis"
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Simulation query failed: {e}")
            return {
                "error": str(e),
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
    
    def batch_query(
        self, 
        questions: List[str], 
        context: str
    ) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions
            context: Context for all questions
            
        Returns:
            List of response dictionaries
        """
        results = []
        
        for i, question in enumerate(questions):
            self.logger.info(f"Processing question {i+1}/{len(questions)}")
            
            prompt = self._create_general_query_prompt(question, context)
            
            if self.llm_runner:
                response = self.llm_runner.generate_response(prompt)
            else:
                response = {
                    "response": f"Mock response for: {question}",
                    "model_id": "mock",
                    "generation_time": 0,
                    "timestamp": datetime.now().isoformat()
                }
            
            response.update({
                "question": question,
                "batch_index": i,
                "query_type": "batch_query"
            })
            
            results.append(response)
        
        return results
    
    def _create_report_query_prompt(self, question: str, report_content: str) -> str:
        """Create prompt for report analysis."""
        return f"""You are analyzing a PyNucleus chemical process simulation report. Please answer the following question based on the report content.

Report Content:
{report_content}

Question: {question}

Please provide a detailed and accurate answer based on the information in the report:"""
    
    def _create_simulation_query_prompt(self, question: str, simulation_data: Dict[str, Any]) -> str:
        """Create prompt for simulation analysis."""
        # Extract key information from simulation data
        case_name = simulation_data.get("case_name", "Unknown")
        status = simulation_data.get("status", "Unknown")
        results = simulation_data.get("results", {})
        
        results_text = "\n".join([f"- {k}: {v}" for k, v in results.items()])
        
        return f"""You are analyzing chemical process simulation results from PyNucleus. Please answer the following question based on the simulation data.

Simulation: {case_name}
Status: {status}

Results:
{results_text}

Question: {question}

Please provide a detailed analysis based on the simulation data:"""
    
    def _create_general_query_prompt(self, question: str, context: str) -> str:
        """Create general query prompt."""
        return f"""Context:
{context}

Question: {question}

Answer:"""
    
    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get query history (placeholder for future implementation)."""
        return []
    
    def clear_history(self):
        """Clear query history (placeholder for future implementation)."""
        pass 