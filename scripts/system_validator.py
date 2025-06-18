#!/usr/bin/env python3
"""
PyNucleus System Validator

FOCUSED VALIDATION TESTING - validates accuracy, citations, and ground-truth responses.
This script specifically focuses on validation aspects of the PyNucleus system:
- Ground-truth validation with known answers
- Citation accuracy and backtracking verification  
- Factual accuracy validation
- Response quality assessment
- RAG system validation

For comprehensive system diagnostics, use comprehensive_system_diagnostic.py instead.
"""

import sys
import warnings
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add src directory to Python path
root_dir = Path(__file__).parent.parent
src_path = str(root_dir / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

@dataclass
class ValidationResult:
    """Structure for validation test results."""
    test_name: str
    query: str
    expected_answer: str
    generated_answer: str
    sources_used: List[str]
    accuracy_score: float
    citation_accuracy: float
    response_time: float
    domain: str = ""
    difficulty_level: str = ""
    expert_rating: Optional[float] = None
    validation_notes: str = ""

class SystemValidator:
    """Focused system validator for PyNucleus accuracy and validation testing."""
    
    def __init__(self, quiet_mode: bool = False):
        self.quiet_mode = quiet_mode
        self.validation_results: List[ValidationResult] = []
        self.total_tests = 0
        self.passed_tests = 0
        self.start_time = datetime.now()
        
        # Validation test datasets
        self.ground_truth_tests = self._create_validation_datasets()
        
    def _create_validation_datasets(self) -> Dict[str, List[Dict]]:
        """Create comprehensive validation test datasets."""
        return {
            "chemical_engineering_concepts": [
                {
                    "query": "What are the main advantages of modular chemical plants?",
                    "expected_keywords": ["reduced capital costs", "faster construction", "quality control", "factory fabrication", "scalability"],
                    "domain": "modular_plants",
                    "difficulty": "intermediate"
                },
                {
                    "query": "How do distillation columns work in chemical separation processes?",
                    "expected_keywords": ["vapor", "liquid", "separation", "boiling points", "reflux", "reboiler"],
                    "domain": "separation_processes", 
                    "difficulty": "basic"
                },
                {
                    "query": "What factors affect reactor conversion efficiency in chemical processes?",
                    "expected_keywords": ["temperature", "pressure", "catalyst", "residence time", "mixing", "kinetics"],
                    "domain": "reactor_design",
                    "difficulty": "intermediate"
                }
            ],
            "process_safety": [
                {
                    "query": "What are the key principles of process safety management?",
                    "expected_keywords": ["hazard identification", "risk assessment", "management systems", "emergency response"],
                    "domain": "safety_management",
                    "difficulty": "advanced"
                },
                {
                    "query": "How do pressure relief systems work in chemical plants?",
                    "expected_keywords": ["relief valve", "pressure", "safety", "rupture disc", "vent"],
                    "domain": "safety_systems",
                    "difficulty": "intermediate"
                }
            ],
            "environmental_engineering": [
                {
                    "query": "What are the main wastewater treatment processes?",
                    "expected_keywords": ["primary treatment", "secondary treatment", "biological", "sedimentation", "activated sludge"],
                    "domain": "wastewater_treatment",
                    "difficulty": "basic"
                }
            ]
        }
    
    def log_message(self, message: str, level: str = "info"):
        """Log messages with appropriate formatting."""
        symbols = {"info": "ℹ️  ", "success": "✅ ", "warning": "⚠️  ", "error": "❌ "}
        symbol = symbols.get(level, "")
        
        if not self.quiet_mode or level in ["error", "warning"]:
            print(f"{symbol}{message}")
    
    def run_validation_suite(self, include_citations: bool = True, include_notebook: bool = False):
        """Run the complete validation suite."""
        self.log_message("🚀 Starting PyNucleus Validation Suite...")
        self.log_message(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("=" * 60)
        print("   PYNUCLEUS VALIDATION TESTING SUITE")
        print("=" * 60)
        print("Focus: Accuracy, Citations, and Ground-Truth Validation")
        print()
        
        try:
            # Core validation tests
            self._run_ground_truth_validation()
            
            if include_citations:
                self._run_citation_validation()
            
            self._run_rag_accuracy_tests()
            
            if include_notebook:
                self._run_notebook_validation()
            
            # Generate validation report
            self._generate_validation_report()
            self._save_validation_results()
            
        except Exception as e:
            self.log_message(f"Validation suite failed: {e}", "error")
            raise
    
    def _run_ground_truth_validation(self):
        """Run ground-truth validation tests."""
        print("\n" + "=" * 60)
        print("   GROUND-TRUTH VALIDATION TESTING")
        print("=" * 60)
        
        total_questions = 0
        successful_validations = 0
        
        for domain, questions in self.ground_truth_tests.items():
            self.log_message(f"\n--- Testing Domain: {domain.replace('_', ' ').title()} ---")
            
            for question_data in questions:
                total_questions += 1
                self.total_tests += 1
                
                try:
                    # Test with RAG system
                    start_time = time.time()
                    
                    # Try to use real RAG system
                    rag_result = self._query_rag_system(question_data["query"])
                    
                    # Calculate accuracy
                    accuracy = self._calculate_keyword_accuracy(
                        question_data["expected_keywords"], 
                        rag_result.get("answer", "")
                    )
                    
                    # Create validation result
                    result = ValidationResult(
                        test_name=f"Ground Truth - {domain}",
                        query=question_data["query"],
                        expected_answer=" ".join(question_data["expected_keywords"]),
                        generated_answer=rag_result.get("answer", ""),
                        sources_used=rag_result.get("sources", []),
                        accuracy_score=accuracy,
                        citation_accuracy=self._calculate_citation_quality(rag_result.get("sources", [])),
                        response_time=time.time() - start_time,
                        domain=question_data["domain"],
                        difficulty_level=question_data["difficulty"]
                    )
                    
                    self.validation_results.append(result)
                    
                    # Check success criteria
                    if accuracy >= 0.4:  # 40% accuracy threshold (more realistic)
                        self.log_message(f"✓ {question_data['query'][:60]}...", "success")
                        self.log_message(f"   Accuracy: {accuracy:.2f}, Time: {result.response_time:.1f}s")
                        successful_validations += 1
                        self.passed_tests += 1
                    else:
                        self.log_message(f"✗ {question_data['query'][:60]}...", "error")
                        self.log_message(f"   Accuracy: {accuracy:.2f} (below 0.40 threshold)", "warning")
                        
                except Exception as e:
                    self.log_message(f"Validation failed for: {question_data['query'][:40]}... - {e}", "error")
        
        # Summary
        success_rate = successful_validations / total_questions if total_questions > 0 else 0
        self.log_message(f"\nGround-Truth Validation Results:")
        self.log_message(f"  Total Questions: {total_questions}")
        self.log_message(f"  Successful: {successful_validations}")
        self.log_message(f"  Success Rate: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            self.log_message("Ground-Truth Validation: PASSED", "success")
        else:
            self.log_message("Ground-Truth Validation: FAILED", "error")
    
    def _run_citation_validation(self):
        """Run citation accuracy and backtracking validation."""
        print("\n" + "=" * 60)
        print("   CITATION VALIDATION TESTING")
        print("=" * 60)
        
        citation_tests = [
            {
                "query": "What are the benefits of process intensification?",
                "expected_sources": ["process_intensification.pdf", "chemical_engineering_handbook.pdf"],
                "domain": "process_engineering"
            },
            {
                "query": "How does heat integration work in chemical processes?",
                "expected_sources": ["heat_integration.pdf", "energy_efficiency.pdf"],
                "domain": "energy_systems"
            },
            {
                "query": "What are the main types of chemical reactors?",
                "expected_sources": ["reactor_design.pdf", "chemical_reactions.pdf"],
                "domain": "reactor_technology"
            }
        ]
        
        successful_citations = 0
        
        for test_data in citation_tests:
            self.total_tests += 1
            
            try:
                # Query RAG system
                rag_result = self._query_rag_system(test_data["query"])
                actual_sources = rag_result.get("sources", [])
                
                # Calculate citation accuracy
                citation_accuracy = self._calculate_citation_overlap(
                    test_data["expected_sources"], 
                    actual_sources
                )
                
                if citation_accuracy >= 0.3:  # 30% overlap threshold (more realistic)
                    self.log_message(f"✓ {test_data['query'][:60]}...", "success")
                    self.log_message(f"   Citation accuracy: {citation_accuracy:.2f}")
                    self.log_message(f"   Sources found: {len(actual_sources)}")
                    successful_citations += 1
                    self.passed_tests += 1
                else:
                    self.log_message(f"✗ {test_data['query'][:60]}...", "error")
                    self.log_message(f"   Citation accuracy: {citation_accuracy:.2f} (below 0.30 threshold)", "warning")
                    
            except Exception as e:
                self.log_message(f"Citation test failed for: {test_data['query'][:40]}... - {e}", "error")
        
        # Summary
        citation_rate = successful_citations / len(citation_tests)
        self.log_message(f"\nCitation Validation Results:")
        self.log_message(f"  Total Tests: {len(citation_tests)}")
        self.log_message(f"  Successful: {successful_citations}")
        self.log_message(f"  Success Rate: {citation_rate:.1%}")
        
        if citation_rate >= 0.7:
            self.log_message("Citation Validation: PASSED", "success")
        else:
            self.log_message("Citation Validation: FAILED", "error")
    
    def _run_rag_accuracy_tests(self):
        """Run RAG system accuracy tests."""
        print("\n" + "=" * 60)
        print("   RAG SYSTEM ACCURACY TESTING")
        print("=" * 60)
        
        # Test RAG system basic functionality
        try:
            from pynucleus.pipeline.pipeline_rag import RAGPipeline
            from pynucleus.rag.vector_store import RealFAISSVectorStore
            
            # Initialize components
            rag_pipeline = RAGPipeline(data_dir="data")
            vector_store = RealFAISSVectorStore()
            
            self.log_message("RAG Pipeline Components:")
            self.log_message(f"  RAG Pipeline: {'✓ Initialized' if rag_pipeline else '✗ Failed'}")
            self.log_message(f"  Vector Store: {'✓ Loaded' if vector_store.loaded else '✗ Not Loaded'}")
            
            # Test search functionality
            if vector_store.loaded:
                test_query = "What are modular chemical plants?"
                search_results = vector_store.search(test_query, top_k=3)
                
                self.total_tests += 1
                if search_results:
                    self.log_message(f"✓ Vector search working ({len(search_results)} results)", "success")
                    self.passed_tests += 1
                else:
                    self.log_message("✗ Vector search returned no results", "warning")
            
        except Exception as e:
            self.log_message(f"RAG accuracy testing failed: {e}", "error")
    
    def _run_notebook_validation(self):
        """Run notebook validation tests."""
        print("\n" + "=" * 60)
        print("   NOTEBOOK VALIDATION TESTING")
        print("=" * 60)
        
        notebooks_to_test = [
            "Capstone Project.ipynb",
            "Developer_Notebook.ipynb"
        ]
        
        for notebook_path in notebooks_to_test:
            self.total_tests += 1
            
            if Path(notebook_path).exists():
                self.log_message(f"✓ Found notebook: {notebook_path}", "success")
                # Could add actual notebook execution testing here
                self.passed_tests += 1
            else:
                self.log_message(f"✗ Missing notebook: {notebook_path}", "error")
    
    def _query_rag_system(self, query: str) -> Dict[str, Any]:
        """Query the RAG system with error handling."""
        try:
            from pynucleus.pipeline.pipeline_rag import RAGPipeline
            rag_pipeline = RAGPipeline(data_dir="data")
            return rag_pipeline.query(query)
        except Exception as e:
            self.log_message(f"RAG query failed: {e}", "warning")
            return {"answer": "", "sources": []}
    
    def _calculate_keyword_accuracy(self, expected_keywords: List[str], generated_answer: str) -> float:
        """Calculate accuracy based on keyword presence."""
        if not expected_keywords or not generated_answer:
            return 0.0
        
        answer_lower = generated_answer.lower()
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
        return matches / len(expected_keywords)
    
    def _calculate_citation_quality(self, sources: List[str]) -> float:
        """Calculate citation quality score."""
        if not sources:
            return 0.0
        # Simple quality based on number and format of sources
        return min(len(sources) / 3.0, 1.0)  # Max score for 3+ sources
    
    def _calculate_citation_overlap(self, expected: List[str], actual: List[str]) -> float:
        """Calculate overlap between expected and actual citations."""
        if not expected:
            return 1.0
        
        expected_set = set(s.lower() for s in expected)
        actual_set = set(s.lower() for s in actual)
        
        intersection = expected_set.intersection(actual_set)
        return len(intersection) / len(expected_set)
    
    def _generate_validation_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "=" * 60)
        print("   VALIDATION REPORT SUMMARY")
        print("=" * 60)
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.log_message(f"PYNUCLEUS VALIDATION REPORT")
        self.log_message(f"Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"Duration: {duration:.1f} seconds")
        
        # Test results summary
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        self.log_message(f"\nEXECUTIVE SUMMARY")
        self.log_message(f"Validation Health: {success_rate:.1f}%")
        self.log_message(f"Tests Performed: {self.total_tests}")
        self.log_message(f"Tests Passed: {self.passed_tests}")
        self.log_message(f"Tests Failed: {self.total_tests - self.passed_tests}")
        
        # Detailed breakdown
        if self.validation_results:
            avg_accuracy = sum(r.accuracy_score for r in self.validation_results) / len(self.validation_results)
            avg_citation = sum(r.citation_accuracy for r in self.validation_results) / len(self.validation_results)
            avg_response_time = sum(r.response_time for r in self.validation_results) / len(self.validation_results)
            
            self.log_message(f"\nDETAILED METRICS")
            self.log_message(f"Average Accuracy: {avg_accuracy:.2f}")
            self.log_message(f"Average Citation Quality: {avg_citation:.2f}")
            self.log_message(f"Average Response Time: {avg_response_time:.2f}s")
        
        # Final assessment
        if success_rate >= 90:
            self.log_message("Overall Validation Status: EXCELLENT 🎉", "success")
        elif success_rate >= 80:
            self.log_message("Overall Validation Status: GOOD ✅", "success")
        elif success_rate >= 70:
            self.log_message("Overall Validation Status: WARNING ⚠️", "warning")
        else:
            self.log_message("Overall Validation Status: CRITICAL ❌", "error")
    
    def _save_validation_results(self):
        """Save validation results to JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"data/validation/results/system_validation_{timestamp}.json"
            
            # Ensure directory exists
            Path(results_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare results data
            results_data = {
                "timestamp": timestamp,
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "success_rate": (self.passed_tests / self.total_tests) if self.total_tests > 0 else 0,
                "validation_results": [
                    {
                        "test_name": r.test_name,
                        "query": r.query,
                        "domain": r.domain,
                        "difficulty": r.difficulty_level,
                        "accuracy_score": r.accuracy_score,
                        "citation_accuracy": r.citation_accuracy,
                        "response_time": r.response_time,
                        "sources_used": r.sources_used
                    }
                    for r in self.validation_results
                ]
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            self.log_message(f"Validation results saved to: {results_file}")
            
        except Exception as e:
            self.log_message(f"Failed to save validation results: {e}", "error")

def main():
    """Main function for validation testing."""
    parser = argparse.ArgumentParser(description="PyNucleus System Validator - Focused Validation Testing")
    parser.add_argument('--quick', action='store_true', help='Quick validation mode (basic tests only)')
    parser.add_argument('--citations', action='store_true', help='Include citation accuracy testing')
    parser.add_argument('--notebook', action='store_true', help='Include notebook validation testing')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode with minimal output')
    parser.add_argument('--validation', action='store_true', help='Run full validation suite (default)')
    
    args = parser.parse_args()
    
    # Create validator
    validator = SystemValidator(quiet_mode=args.quiet)
    
    try:
        # Determine what to run
        if args.quick:
            validator.log_message("Running quick validation suite...")
            validator._run_ground_truth_validation()
        else:
            # Full validation is default
            validator.run_validation_suite(
                include_citations=args.citations or not args.quick,
                include_notebook=args.notebook
            )
        
        # Exit with appropriate code based on results
        success_rate = validator.passed_tests / validator.total_tests if validator.total_tests > 0 else 0
        exit_code = 0 if success_rate >= 0.8 else 1  # 80% threshold for success
        
        if exit_code == 0:
            validator.log_message("🎉 Validation completed successfully!", "success")
        else:
            validator.log_message("⚠️ Validation completed with issues!", "warning")
        
        sys.exit(exit_code)
        
    except Exception as e:
        validator.log_message(f"Validation suite failed: {e}", "error")
        sys.exit(2)

if __name__ == "__main__":
    main() 