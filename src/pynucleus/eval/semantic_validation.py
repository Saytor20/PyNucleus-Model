"""
Semantic Validation Module for PyNucleus Evaluation System
==========================================================

This module provides semantic similarity metrics to replace keyword-based evaluation
with more robust semantic similarity scoring using BLEU, ROUGE, and BERTScore.

Functions:
    - calculate_bleu_score: BLEU score for n-gram precision
    - calculate_rouge_scores: ROUGE-1, ROUGE-2, ROUGE-L scores
    - calculate_bert_score: BERTScore for semantic similarity
    - calculate_semantic_similarity: Combined semantic similarity score
    - evaluate_answer_semantically: Complete semantic evaluation
"""

import warnings
from typing import Dict, List, Any, Tuple, Optional
import logging
import re
from dataclasses import dataclass

# Suppress warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    import nltk
    
    # Download required NLTK data (only once)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        
    DEPENDENCIES_AVAILABLE = True
    MISSING_DEPS = None
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS = str(e)

from ..utils.logger import logger


@dataclass
class SemanticScores:
    """Container for semantic similarity scores."""
    bleu_score: float
    rouge_1_f: float
    rouge_2_f: float
    rouge_l_f: float
    bert_score_f1: float
    combined_score: float
    success: bool


def _check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    if not DEPENDENCIES_AVAILABLE:
        error_msg = MISSING_DEPS if MISSING_DEPS else "Unknown import error"
        logger.error(f"Semantic validation dependencies not available: {error_msg}")
        logger.error("Please install: pip install rouge-score bert-score nltk")
        return False
    return True


def _preprocess_text(text: str) -> str:
    """Preprocess text for semantic evaluation."""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Convert to lowercase for consistency
    text = text.lower()
    
    return text


def _tokenize_text(text: str) -> List[str]:
    """Tokenize text into words."""
    if not DEPENDENCIES_AVAILABLE:
        return text.split()
    
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text.lower())
    except:
        # Fallback to simple split
        return text.lower().split()


def calculate_bleu_score(generated_answer: str, expected_keywords: List[str]) -> float:
    """
    Calculate BLEU score between generated answer and expected keywords.
    
    Args:
        generated_answer: The generated answer text
        expected_keywords: List of expected keywords/phrases
        
    Returns:
        BLEU score (0.0 to 1.0)
    """
    if not _check_dependencies():
        return 0.0
        
    if not generated_answer or not expected_keywords:
        return 0.0
    
    try:
        # Preprocess texts
        generated = _preprocess_text(generated_answer)
        
        # Create reference from expected keywords
        reference_text = " ".join(expected_keywords).lower()
        reference = _tokenize_text(reference_text)
        candidate = _tokenize_text(generated)
        
        if not candidate or not reference:
            return 0.0
        
        # Use smoothing for better results with short texts
        smoothing = SmoothingFunction().method1
        
        # Calculate BLEU score with multiple n-grams
        bleu = sentence_bleu(
            [reference], 
            candidate,
            weights=(0.25, 0.25, 0.25, 0.25),  # Equal weights for 1-4 grams
            smoothing_function=smoothing
        )
        
        return round(float(bleu), 4)
        
    except Exception as e:
        logger.warning(f"BLEU score calculation failed: {e}")
        return 0.0


def calculate_rouge_scores(generated_answer: str, expected_keywords: List[str]) -> Dict[str, float]:
    """
    Calculate ROUGE scores between generated answer and expected keywords.
    
    Args:
        generated_answer: The generated answer text
        expected_keywords: List of expected keywords/phrases
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    """
    if not _check_dependencies():
        return {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}
        
    if not generated_answer or not expected_keywords:
        return {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}
    
    try:
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Preprocess texts
        generated = _preprocess_text(generated_answer)
        
        # Create reference from expected keywords
        reference_text = " ".join(expected_keywords).lower()
        
        if not generated or not reference_text:
            return {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}
        
        # Calculate ROUGE scores
        scores = scorer.score(reference_text, generated)
        
        return {
            "rouge_1_f": round(scores['rouge1'].fmeasure, 4),
            "rouge_2_f": round(scores['rouge2'].fmeasure, 4),
            "rouge_l_f": round(scores['rougeL'].fmeasure, 4)
        }
        
    except Exception as e:
        logger.warning(f"ROUGE score calculation failed: {e}")
        return {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}


def calculate_bert_score(generated_answer: str, expected_keywords: List[str]) -> float:
    """
    Calculate BERTScore between generated answer and expected keywords.
    
    Args:
        generated_answer: The generated answer text
        expected_keywords: List of expected keywords/phrases
        
    Returns:
        BERTScore F1 score (0.0 to 1.0)
    """
    if not _check_dependencies():
        return 0.0
        
    if not generated_answer or not expected_keywords:
        return 0.0
    
    try:
        # Preprocess texts
        generated = _preprocess_text(generated_answer)
        
        # Create reference from expected keywords
        reference_text = " ".join(expected_keywords).lower()
        
        if not generated or not reference_text:
            return 0.0
        
        # Calculate BERTScore
        # Use a lighter model for faster computation
        P, R, F1 = bert_score(
            [generated], 
            [reference_text], 
            lang='en',
            model_type='distilbert-base-uncased',  # Faster than full BERT
            verbose=False
        )
        
        # Return F1 score
        bert_f1 = float(F1[0])
        return round(bert_f1, 4)
        
    except Exception as e:
        logger.warning(f"BERTScore calculation failed: {e}")
        return 0.0


def calculate_semantic_similarity(generated_answer: str, expected_keywords: List[str]) -> SemanticScores:
    """
    Calculate comprehensive semantic similarity scores.
    
    Args:
        generated_answer: The generated answer text
        expected_keywords: List of expected keywords/phrases
        
    Returns:
        SemanticScores object with all similarity metrics
    """
    if not _check_dependencies():
        return SemanticScores(
            bleu_score=0.0,
            rouge_1_f=0.0,
            rouge_2_f=0.0,
            rouge_l_f=0.0,
            bert_score_f1=0.0,
            combined_score=0.0,
            success=False
        )
    
    try:
        logger.debug(f"Calculating semantic similarity for answer: {generated_answer[:100]}...")
        
        # Calculate individual scores
        bleu_score = calculate_bleu_score(generated_answer, expected_keywords)
        rouge_scores = calculate_rouge_scores(generated_answer, expected_keywords)
        bert_score_f1 = calculate_bert_score(generated_answer, expected_keywords)
        
        # Calculate combined score with weighted average
        # Weights: BLEU (20%), ROUGE-1 (25%), ROUGE-2 (15%), ROUGE-L (20%), BERTScore (20%)
        combined_score = (
            bleu_score * 0.20 +
            rouge_scores["rouge_1_f"] * 0.25 +
            rouge_scores["rouge_2_f"] * 0.15 +
            rouge_scores["rouge_l_f"] * 0.20 +
            bert_score_f1 * 0.20
        )
        
        # Determine success (threshold: 0.3 for combined score)
        success = combined_score >= 0.3
        
        scores = SemanticScores(
            bleu_score=bleu_score,
            rouge_1_f=rouge_scores["rouge_1_f"],
            rouge_2_f=rouge_scores["rouge_2_f"],
            rouge_l_f=rouge_scores["rouge_l_f"],
            bert_score_f1=bert_score_f1,
            combined_score=round(combined_score, 4),
            success=success
        )
        
        logger.debug(f"Semantic scores calculated: combined={combined_score:.4f}, success={success}")
        return scores
        
    except Exception as e:
        logger.error(f"Semantic similarity calculation failed: {e}")
        return SemanticScores(
            bleu_score=0.0,
            rouge_1_f=0.0,
            rouge_2_f=0.0,
            rouge_l_f=0.0,
            bert_score_f1=0.0,
            combined_score=0.0,
            success=False
        )


def evaluate_answer_semantically(
    generated_answer: str, 
    expected_keywords: List[str],
    threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Complete semantic evaluation of generated answer.
    
    Args:
        generated_answer: The generated answer text
        expected_keywords: List of expected keywords/phrases
        threshold: Success threshold for combined score (default: 0.3)
        
    Returns:
        Dictionary with detailed semantic evaluation results
    """
    if not _check_dependencies():
        return {
            "semantic_scores": None,
            "success": False,
            "error": "Semantic validation dependencies not available"
        }
    
    try:
        # Calculate semantic scores
        scores = calculate_semantic_similarity(generated_answer, expected_keywords)
        
        # Apply custom threshold if provided
        success = scores.combined_score >= threshold
        
        return {
            "semantic_scores": {
                "bleu_score": scores.bleu_score,
                "rouge_1_f": scores.rouge_1_f,
                "rouge_2_f": scores.rouge_2_f,
                "rouge_l_f": scores.rouge_l_f,
                "bert_score_f1": scores.bert_score_f1,
                "combined_score": scores.combined_score,
            },
            "success": success,
            "threshold": threshold,
            "methodology": {
                "bleu_weight": 0.20,
                "rouge1_weight": 0.25,
                "rouge2_weight": 0.15,
                "rougeL_weight": 0.20,
                "bert_weight": 0.20
            }
        }
        
    except Exception as e:
        logger.error(f"Semantic evaluation failed: {e}")
        return {
            "semantic_scores": None,
            "success": False,
            "error": str(e)
        }


def get_semantic_validation_info() -> Dict[str, Any]:
    """
    Get information about semantic validation capabilities.
    
    Returns:
        Dictionary with validation info and dependencies status
    """
    return {
        "dependencies_available": DEPENDENCIES_AVAILABLE,
        "missing_dependencies": MISSING_DEPS if not DEPENDENCIES_AVAILABLE and MISSING_DEPS else None,
        "supported_metrics": [
            "BLEU Score (n-gram precision)",
            "ROUGE-1 (unigram overlap)",
            "ROUGE-2 (bigram overlap)", 
            "ROUGE-L (longest common subsequence)",
            "BERTScore (semantic similarity)"
        ],
        "default_threshold": 0.3,
        "scoring_weights": {
            "bleu": 0.20,
            "rouge_1": 0.25,
            "rouge_2": 0.15,
            "rouge_l": 0.20,
            "bert_score": 0.20
        }
    } 