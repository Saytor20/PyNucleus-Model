"""
PyNucleus Evaluation Module

This module provides comprehensive evaluation and validation capabilities including:
- Golden dataset evaluation
- Expert validation workflows
- Semantic validation
- Confidence calibration
"""

from .golden_eval import run_eval, run_enhanced_eval, EnhancedEvaluator
from .validation_manager import (
    ValidationManager,
    ExpertProfile,
    ExpertLevel,
    ValidationStatus,
    ValidationRecord,
    initialize_validation_system
)
from .semantic_validation import calculate_semantic_similarity, evaluate_answer_semantically
from .confidence_calibration import ConfidenceCalibrator

__all__ = [
    # Golden evaluation
    'run_eval',
    'run_enhanced_eval', 
    'EnhancedEvaluator',
    
    # Validation manager
    'ValidationManager',
    'ExpertProfile',
    'ExpertLevel',
    'ValidationStatus',
    'ValidationRecord',
    'initialize_validation_system',
    
    # Semantic validation
    'calculate_semantic_similarity',
    'evaluate_answer_semantically',
    
    # Confidence calibration
    'ConfidenceCalibrator'
] 