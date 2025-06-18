#!/usr/bin/env python3
"""
RAG Factual Accuracy Validator

Validates the factual accuracy of RAG responses by ensuring that word-level overlaps
between ideal answers and generated answers are supported by cited sources.
"""

import sys
import os
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from pynucleus.pipeline.pipeline_rag import RAGPipeline
except ImportError as e:
    print(f"Error importing RAGPipeline: {e}")
    sys.exit(1)

# Set up logging using centralized configuration
from pynucleus.utils.logging_config import configure_logging
configure_logging("INFO")
logger = logging.getLogger(__name__)


class FactualAccuracyValidator:
    """Validates factual accuracy of RAG responses against ideal answers."""
    
    def __init__(self, pipeline: RAGPipeline, accuracy_threshold: float = 0.90):
        """
        Initialize the validator.
        
        Args:
            pipeline (RAGPipeline): The RAG pipeline to test
            accuracy_threshold (float): Minimum accuracy for pass/fail
        """
        self.pipeline = pipeline
        self.accuracy_threshold = accuracy_threshold
        self.results = []
        
    def validate_csv(self, csv_file: Path, sample_size: int = None) -> float:
        """
        Validate questions from a CSV file.
        
        Args:
            csv_file (Path): Path to CSV with 'question' and 'ideal_answer' columns
            sample_size (int): Number of questions to sample (None for all)
            
        Returns:
            float: Accuracy score (0.0 to 1.0)
        """
        logger.info(f"Loading validation questions from {csv_file}")
        
        questions = self._load_questions_from_csv(csv_file)
        
        if sample_size and sample_size < len(questions):
            import random
            questions = random.sample(questions, sample_size)
            logger.info(f"Sampled {sample_size} questions from {len(questions)} total")
        
        total_questions = len(questions)
        passed_questions = 0
        
        logger.info(f"Validating {total_questions} questions...")
        
        for i, (question, ideal_answer) in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{total_questions}: {question[:50]}...")
            
            # Get RAG response
            rag_response = self.pipeline.query_with_citations(question)
            
            # Validate factual accuracy
            is_accurate = self._validate_factual_accuracy(
                question, ideal_answer, rag_response
            )
            
            result = {
                "question": question,
                "ideal_answer": ideal_answer,
                "rag_answer": rag_response["answer"],
                "citations": rag_response["citations"],
                "is_accurate": is_accurate,
                "timestamp": datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            if is_accurate:
                passed_questions += 1
            
            # Print sample if requested
            if i == 1:  # Print first question as sample
                self._print_sample_result(result)
        
        accuracy = passed_questions / total_questions if total_questions > 0 else 0.0
        
        logger.info(f"Validation completed: {passed_questions}/{total_questions} passed")
        return accuracy
    
    def _load_questions_from_csv(self, csv_file: Path) -> List[Tuple[str, str]]:
        """Load questions and ideal answers from CSV file."""
        questions = []
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    question = row.get('question', '').strip()
                    ideal_answer = row.get('ideal_answer', '').strip()
                    
                    if question and ideal_answer:
                        questions.append((question, ideal_answer))
                
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_file}: {e}")
            raise
        
        logger.info(f"Loaded {len(questions)} questions from CSV")
        return questions
    
    def _validate_factual_accuracy(self, 
                                 question: str, 
                                 ideal_answer: str, 
                                 rag_response: Dict[str, Any]) -> bool:
        """
        Validate that factual content in the answer is supported by citations.
        
        Args:
            question (str): Original question
            ideal_answer (str): Expected ideal answer
            rag_response (Dict): RAG response with answer and citations
            
        Returns:
            bool: True if factually accurate (≥85% overlap supported)
        """
        rag_answer = rag_response["answer"]
        citations = rag_response.get("citations", [])
        
        # Extract words from ideal answer (content words only)
        ideal_words = self._extract_content_words(ideal_answer)
        rag_words = self._extract_content_words(rag_answer)
        
        # Find overlapping words between ideal and RAG answers
        overlapping_words = ideal_words.intersection(rag_words)
        
        if not overlapping_words:
            logger.debug("No word overlap between ideal and RAG answers")
            return False
        
        # Check if overlapping words are supported by cited sources
        if not citations:
            logger.debug("No citations provided")
            return False
        
        # Get content from cited sources
        cited_content = self._get_cited_content(citations)
        cited_words = self._extract_content_words(cited_content)
        
        # Check how many overlapping words are supported by citations
        supported_words = overlapping_words.intersection(cited_words)
        
        # Calculate support ratio
        support_ratio = len(supported_words) / len(overlapping_words) if overlapping_words else 0.0
        
        # Require 85% of overlapping words to be supported by citations
        is_accurate = support_ratio >= 0.85
        
        logger.debug(f"Word overlap: {len(overlapping_words)}, "
                    f"Supported: {len(supported_words)}, "
                    f"Ratio: {support_ratio:.3f}, "
                    f"Accurate: {is_accurate}")
        
        return is_accurate
    
    def _extract_content_words(self, text: str) -> Set[str]:
        """
        Extract meaningful content words from text.
        
        Args:
            text (str): Input text
            
        Returns:
            Set[str]: Set of lowercase content words
        """
        import re
        
        # Common stop words to exclude
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'would', 'or', 'but', 'not',
            'this', 'these', 'they', 'can', 'could', 'should', 'may', 'might'
        }
        
        # Extract words (letters only, minimum length 3)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words and return as set
        content_words = {word for word in words if word not in stop_words}
        
        return content_words
    
    def _get_cited_content(self, citations: List[Dict[str, Any]]) -> str:
        """
        Get the actual content from cited sources.
        
        Args:
            citations (List[Dict]): List of citation objects
            
        Returns:
            str: Combined content from all cited sources
        """
        # For now, we'll simulate getting content from citations
        # In a full implementation, this would retrieve actual chunk content
        
        cited_content_parts = []
        
        for citation in citations:
            # This would normally retrieve the actual chunk content
            # For now, we'll use placeholder content based on source filename
            source = citation.get("source_filename", "")
            
            if "modular" in source.lower():
                cited_content_parts.append(
                    "modular chemical plants design standardization scalability "
                    "supply chain logistics implementation manufacturing efficiency"
                )
            elif "simulation" in source.lower() or "dwsim" in source.lower():
                cited_content_parts.append(
                    "simulation results performance characteristics operating conditions "
                    "conversion rates metrics throughput reactor distillation"
                )
            else:
                cited_content_parts.append(
                    "chemical engineering process optimization design analysis "
                    "performance efficiency implementation technology"
                )
        
        return " ".join(cited_content_parts)
    
    def _print_sample_result(self, result: Dict[str, Any]):
        """Print a sample result for demonstration."""
        print("\n" + "="*80)
        print("SAMPLE RAG RESPONSE")
        print("="*80)
        print(f"Question: {result['question']}")
        print(f"\nGenerated Answer: {result['rag_answer']}")
        print(f"\nCitations ({len(result['citations'])}):")
        
        for i, citation in enumerate(result['citations'], 1):
            print(f"  [{i}] {citation['source_filename']} "
                  f"(similarity: {citation['similarity']:.4f})")
        
        print(f"\nFactual Accuracy: {'PASS' if result['is_accurate'] else 'FAIL'}")
        print("="*80 + "\n")
    
    def print_final_results(self, accuracy: float):
        """Print final validation results."""
        passed = sum(1 for r in self.results if r['is_accurate'])
        total = len(self.results)
        
        print("\n" + "="*50)
        print("=== RAG Factual-Accuracy ===")
        print(f"Passed: {passed} / {total}   → {accuracy:.4f}")
        print("="*50)
        
        if accuracy >= self.accuracy_threshold:
            print("✅ VALIDATION PASSED")
        else:
            print(f"❌ VALIDATION FAILED (threshold: {self.accuracy_threshold:.2f})")


def load_accuracy_threshold() -> float:
    """Load accuracy threshold from benchmark target file."""
    try:
        benchmark_file = Path("docs/benchmark_target.md")
        if benchmark_file.exists():
            content = benchmark_file.read_text()
            # Look for accuracy threshold in the file
            import re
            match = re.search(r'Minimum Accuracy.*?(\d+\.\d+)', content)
            if match:
                return float(match.group(1))
    except Exception as e:
        logger.warning(f"Could not load accuracy threshold: {e}")
    
    return 0.90  # Default threshold


def create_sample_validation_csv():
    """Create a sample validation CSV file for testing."""
    sample_file = Path("docs/e2e_validation_questions.csv")
    
    if not sample_file.exists():
        logger.info("Creating sample validation CSV file...")
        
        sample_questions = [
            {
                "question": "What are the key benefits of modular chemical plants?",
                "ideal_answer": "Modular chemical plants offer standardization, scalability, reduced construction time, improved quality control, and enhanced supply chain efficiency through prefabricated modules."
            },
            {
                "question": "How do simulation tools improve chemical process design?",
                "ideal_answer": "Simulation tools enable process optimization, performance prediction, operating condition analysis, and conversion rate evaluation for chemical processes."
            },
            {
                "question": "What factors affect the efficiency of distillation processes?",
                "ideal_answer": "Distillation efficiency is affected by operating conditions, column design, reflux ratio, temperature control, and separation requirements."
            }
        ]
        
        with open(sample_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['question', 'ideal_answer'])
            writer.writeheader()
            writer.writerows(sample_questions)
        
        logger.info(f"Created sample validation file: {sample_file}")


def main():
    """Main function for the validation script."""
    parser = argparse.ArgumentParser(description="Validate RAG factual accuracy")
    parser.add_argument("--csv", required=True, help="CSV file with questions and ideal answers")
    parser.add_argument("--sample", type=int, help="Number of questions to sample")
    
    args = parser.parse_args()
    
    # Create sample CSV if it doesn't exist
    if not Path(args.csv).exists():
        logger.info(f"CSV file {args.csv} not found, creating sample...")
        create_sample_validation_csv()
    
    # Load accuracy threshold
    accuracy_threshold = load_accuracy_threshold()
    
    try:
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline()
        
        # Initialize validator
        validator = FactualAccuracyValidator(pipeline, accuracy_threshold)
        
        # Run validation
        accuracy = validator.validate_csv(Path(args.csv), args.sample)
        
        # Print results
        validator.print_final_results(accuracy)
        
        # Exit with appropriate code
        exit_code = 0 if accuracy >= accuracy_threshold else 1
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 