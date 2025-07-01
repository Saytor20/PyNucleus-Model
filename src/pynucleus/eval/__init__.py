"""
PyNucleus Evaluation Module

This module provides comprehensive evaluation and validation capabilities including:
- Golden dataset evaluation
- Expert validation workflows
- Semantic validation
- Confidence calibration
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, Callable

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

logger = logging.getLogger(__name__)


def load_latest_model() -> Optional[Callable[[float], float]]:
    """
    Load the latest trained isotonic regression calibration model.
    
    Returns:
        Callable that takes raw confidence (float) and returns calibrated confidence (float),
        or None if no model is available
    """
    model_path = Path("data/calibration/models/calibration_isotonic.pkl")
    
    if not model_path.exists():
        logger.warning(f"No calibration model found at {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        def calibrate_confidence(raw_confidence: float) -> float:
            """Calibrate a single confidence score using the trained model."""
            # Ensure input is in valid range
            raw_confidence = max(0.0, min(1.0, raw_confidence))
            
            # Get calibrated score
            calibrated = model.predict([raw_confidence])[0]
            
            # Ensure output is in valid range
            return max(0.0, min(1.0, float(calibrated)))
        
        logger.info(f"Loaded calibration model from {model_path}")
        return calibrate_confidence
        
    except Exception as e:
        logger.error(f"Failed to load calibration model: {e}")
        return None


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
    'ConfidenceCalibrator',
    'load_latest_model'
] 