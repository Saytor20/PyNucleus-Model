"""
DSPy Program Compilation Utility for PyNucleus

This module provides functionality to compile DSPy programs using development datasets
and save the compiled programs for production use. It supports both local development
and CI/CD pipeline integration.

Key Features:
- Load development examples from CSV files
- Compile PyNucleus DSPy programs using dsp.Compiler
- Save compiled artifacts to data/dspy_artifacts/
- Support for CI mode (artifact generation without commit)
- Fallback to uncompiled programs when artifacts unavailable
"""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import tempfile
import traceback

try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logging.warning("DSPy not available. Compilation will be skipped.")

from .dspy_program import PyNucleusProgram, OptimizedProgram
from .answer_engine import DSPyAnswerEngine


class DSPyCompiler:
    """DSPy program compiler for PyNucleus system."""
    
    def __init__(self, artifacts_dir: Path = None):
        """
        Initialize DSPy compiler.
        
        Args:
            artifacts_dir: Directory to save compiled artifacts
        """
        self.artifacts_dir = artifacts_dir or Path("data/dspy_artifacts")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.compilation_results = {}
        
    def load_dev_examples(self, csv_path: Path) -> List[Dict[str, Any]]:
        """
        Load development examples from CSV file.
        
        Args:
            csv_path: Path to CSV file with development examples
            
        Returns:
            List of example dictionaries
        """
        if not csv_path.exists():
            self.logger.error(f"Development examples file not found: {csv_path}")
            return []
        
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_columns = ['question', 'context', 'expected_answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns in CSV: {missing_columns}")
                return []
            
            # Convert to list of dictionaries
            examples = []
            for _, row in df.iterrows():
                example = {
                    'question': row['question'],
                    'context': row.get('context', ''),
                    'expected_answer': row['expected_answer'],
                    'domain': row.get('domain', 'general'),
                    'difficulty': row.get('difficulty', 'medium')
                }
                examples.append(example)
            
            self.logger.info(f"Loaded {len(examples)} development examples from {csv_path}")
            return examples
            
        except Exception as e:
            self.logger.error(f"Failed to load development examples: {e}")
            return []
    
    def compile_program(
        self, 
        examples: List[Dict[str, Any]], 
        program_class=PyNucleusProgram,
        optimization_metric: str = "accuracy"
    ) -> Optional[Any]:
        """
        Compile DSPy program using development examples.
        
        Args:
            examples: List of development examples
            program_class: DSPy program class to compile
            optimization_metric: Metric to optimize for
            
        Returns:
            Compiled program or None if compilation fails
        """
        if not DSPY_AVAILABLE:
            self.logger.warning("DSPy not available. Skipping compilation.")
            return None
        
        if not examples:
            self.logger.error("No examples provided for compilation")
            return None
        
        try:
            self.logger.info(f"Starting DSPy compilation with {len(examples)} examples")
            
            # Initialize program
            program = program_class()
            
            # Prepare examples for DSPy compiler
            dspy_examples = []
            for example in examples:
                # Create DSPy example format
                dspy_example = dspy.Example(
                    question=example['question'],
                    context=example['context'],
                    answer=example['expected_answer']
                ).with_inputs('question', 'context')
                dspy_examples.append(dspy_example)
            
            # Initialize optimizer
            # Create a simple metric for compilation
            def simple_metric(gold, pred, trace=None):
                """Simple metric that checks if answer is not empty."""
                return len(str(pred.answer).strip()) > 0
            
            # Use BootstrapFewShot optimizer for compilation
            optimizer = BootstrapFewShot(
                metric=simple_metric,
                max_bootstrapped_demos=3,
                max_labeled_demos=2,
                max_rounds=1
            )
            
            # Compile program
            self.logger.info("Compiling DSPy program...")
            compiled_program = optimizer.compile(
                program, 
                trainset=dspy_examples
            )
            
            self.logger.info("DSPy compilation completed successfully")
            
            # Store compilation metadata
            self.compilation_results = {
                "compilation_time": datetime.now().isoformat(),
                "examples_count": len(examples),
                "optimization_metric": optimization_metric,
                "program_class": program_class.__name__,
                "success": True
            }
            
            return compiled_program
            
        except Exception as e:
            self.logger.error(f"DSPy compilation failed: {e}")
            self.logger.debug(f"Compilation error details: {traceback.format_exc()}")
            
            self.compilation_results = {
                "compilation_time": datetime.now().isoformat(),
                "examples_count": len(examples),
                "error": str(e),
                "success": False
            }
            
            return None
    
    def save_compiled_program(
        self, 
        compiled_program: Any, 
        filename: str = "compiled_program.json"
    ) -> Path:
        """
        Save compiled program to artifacts directory.
        
        Args:
            compiled_program: Compiled DSPy program
            filename: Output filename
            
        Returns:
            Path to saved artifact
        """
        artifact_path = self.artifacts_dir / filename
        
        try:
            # Save compiled program state
            program_state = {
                "compiled_program": compiled_program.state_dict() if hasattr(compiled_program, 'state_dict') else str(compiled_program),
                "compilation_metadata": self.compilation_results,
                "saved_at": datetime.now().isoformat(),
                "pynucleus_version": "1.0.0",  # You might want to read this from a version file
                "dspy_version": getattr(dspy, '__version__', 'unknown') if DSPY_AVAILABLE else 'unavailable'
            }
            
            with open(artifact_path, 'w', encoding='utf-8') as f:
                json.dump(program_state, f, indent=2, default=str)
            
            self.logger.info(f"Compiled program saved to: {artifact_path}")
            return artifact_path
            
        except Exception as e:
            self.logger.error(f"Failed to save compiled program: {e}")
            raise
    
    def load_compiled_program(
        self, 
        filename: str = "compiled_program.json",
        program_class=PyNucleusProgram
    ) -> Optional[Any]:
        """
        Load compiled program from artifacts directory.
        
        Args:
            filename: Artifact filename
            program_class: Program class to instantiate
            
        Returns:
            Loaded compiled program or None if loading fails
        """
        artifact_path = self.artifacts_dir / filename
        
        if not artifact_path.exists():
            self.logger.warning(f"Compiled program artifact not found: {artifact_path}")
            return None
        
        try:
            with open(artifact_path, 'r', encoding='utf-8') as f:
                program_state = json.load(f)
            
            # Initialize program
            program = program_class()
            
            # Load state if available
            if hasattr(program, 'load_state_dict') and 'compiled_program' in program_state:
                program.load_state_dict(program_state['compiled_program'])
            
            self.logger.info(f"Compiled program loaded from: {artifact_path}")
            return program
            
        except Exception as e:
            self.logger.error(f"Failed to load compiled program: {e}")
            return None
    
    def compile_and_save(
        self, 
        csv_path: Path, 
        output_filename: str = "compiled_program.json",
        ci_mode: bool = False
    ) -> bool:
        """
        Complete compilation workflow: load examples, compile, and save.
        
        Args:
            csv_path: Path to development examples CSV
            output_filename: Output artifact filename
            ci_mode: If True, run in CI mode (no commit, just validate)
            
        Returns:
            True if compilation successful, False otherwise
        """
        self.logger.info(f"Starting DSPy compilation workflow (CI mode: {ci_mode})")
        
        # Load examples
        examples = self.load_dev_examples(csv_path)
        if not examples:
            self.logger.error("No examples loaded. Compilation aborted.")
            return False
        
        # Compile program
        compiled_program = self.compile_program(examples)
        if not compiled_program:
            self.logger.error("Program compilation failed.")
            return False
        
        # Save compiled program
        try:
            artifact_path = self.save_compiled_program(compiled_program, output_filename)
            
            if ci_mode:
                self.logger.info("CI mode: Compilation successful, artifact created but not committed")
            else:
                self.logger.info(f"Compilation successful! Artifact saved to: {artifact_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save compiled program: {e}")
            return False
    
    def create_fallback_program(self) -> PyNucleusProgram:
        """
        Create fallback uncompiled program when compiled artifacts are unavailable.
        
        Returns:
            Uncompiled PyNucleus program
        """
        self.logger.info("Creating fallback uncompiled program")
        return PyNucleusProgram()
    
    def get_program(self, prefer_compiled: bool = True) -> PyNucleusProgram:
        """
        Get DSPy program with fallback to uncompiled version.
        
        Args:
            prefer_compiled: If True, try to load compiled version first
            
        Returns:
            DSPy program (compiled or fallback)
        """
        if prefer_compiled:
            compiled_program = self.load_compiled_program()
            if compiled_program:
                self.logger.info("Using compiled DSPy program")
                return compiled_program
        
        self.logger.info("Using fallback uncompiled DSPy program")
        return self.create_fallback_program()
    
    def validate_compilation(self, test_questions: List[str] = None) -> Dict[str, Any]:
        """
        Validate compiled program functionality.
        
        Args:
            test_questions: Optional test questions for validation
            
        Returns:
            Validation results
        """
        if not test_questions:
            test_questions = [
                "What is the optimal temperature for distillation?",
                "How do you calculate heat transfer coefficient?",
                "What are the safety considerations for pressure vessels?"
            ]
        
        validation_results = {
            "validation_time": datetime.now().isoformat(),
            "test_questions": len(test_questions),
            "results": []
        }
        
        try:
            # Get program (compiled or fallback)
            program = self.get_program()
            
            for question in test_questions:
                try:
                    start_time = datetime.now()
                    answer = program.answer_general("Chemical engineering context", question)
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    result = {
                        "question": question,
                        "answer": answer,
                        "response_time": response_time,
                        "success": True
                    }
                    
                except Exception as e:
                    result = {
                        "question": question,
                        "error": str(e),
                        "success": False
                    }
                
                validation_results["results"].append(result)
            
            success_count = sum(1 for r in validation_results["results"] if r["success"])
            validation_results["success_rate"] = success_count / len(test_questions)
            
            self.logger.info(f"Validation completed: {success_count}/{len(test_questions)} successful")
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            validation_results["error"] = str(e)
        
        return validation_results


def create_sample_dev_csv(output_path: Path) -> bool:
    """
    Create a sample development dataset CSV for testing.
    
    Args:
        output_path: Path where to save the sample CSV
        
    Returns:
        True if successful, False otherwise
    """
    sample_data = [
        {
            "question": "What is the optimal temperature for ethanol distillation?",
            "context": "Chemical engineering process optimization for ethanol production",
            "expected_answer": "The optimal temperature for ethanol distillation is typically between 78-85°C, depending on pressure and purity requirements.",
            "domain": "distillation",
            "difficulty": "medium"
        },
        {
            "question": "How do you calculate heat transfer coefficient in a shell-and-tube heat exchanger?",
            "context": "Heat transfer calculations for chemical process equipment",
            "expected_answer": "The overall heat transfer coefficient is calculated using: 1/U = 1/hi + Rfi + x/k + Rfo + 1/ho, where hi and ho are inside and outside convection coefficients.",
            "domain": "heat_transfer",
            "difficulty": "hard"
        },
        {
            "question": "What are the main safety considerations for pressure vessel design?",
            "context": "Chemical process safety and equipment design",
            "expected_answer": "Key safety considerations include pressure rating, material selection, corrosion allowance, safety relief systems, and compliance with ASME codes.",
            "domain": "safety",
            "difficulty": "medium"
        }
    ]
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        
        logging.info(f"Sample development dataset created: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to create sample dataset: {e}")
        return False


# CLI Integration Functions
def compile_dspy_main(
    csv_path: str = "docs/devset/dspy_examples.csv",
    output_dir: str = "data/dspy_artifacts",
    ci_mode: bool = False,
    create_sample: bool = False
) -> int:
    """
    Main function for DSPy compilation CLI command.
    
    Args:
        csv_path: Path to development examples CSV
        output_dir: Output directory for artifacts
        ci_mode: Run in CI mode
        create_sample: Create sample dataset if CSV doesn't exist
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        csv_path = Path(csv_path)
        output_dir = Path(output_dir)
        
        # Create sample dataset if requested or if file doesn't exist
        if create_sample or not csv_path.exists():
            logger.info("Creating sample development dataset...")
            if not create_sample_dev_csv(csv_path):
                logger.error("Failed to create sample dataset")
                return 1
            
            if create_sample:
                logger.info(f"Sample dataset created at: {csv_path}")
                return 0
        
        # Initialize compiler
        compiler = DSPyCompiler(artifacts_dir=output_dir)
        
        # Run compilation workflow
        success = compiler.compile_and_save(
            csv_path=csv_path,
            ci_mode=ci_mode
        )
        
        if success:
            logger.info("DSPy compilation completed successfully!")
            
            # Run validation
            validation_results = compiler.validate_compilation()
            logger.info(f"Validation success rate: {validation_results.get('success_rate', 0):.2%}")
            
            return 0
        else:
            logger.error("DSPy compilation failed")
            return 1
    
    except Exception as e:
        logger.error(f"DSPy compilation error: {e}")
        logger.debug(f"Error details: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    import sys
    
    # Simple CLI for testing
    if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
        exit_code = compile_dspy_main(create_sample=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--ci":
        exit_code = compile_dspy_main(ci_mode=True)
    else:
        exit_code = compile_dspy_main()
    
    sys.exit(exit_code) 