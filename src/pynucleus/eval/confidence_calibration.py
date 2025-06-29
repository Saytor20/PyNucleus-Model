"""
Confidence Calibration Module for PyNucleus RAG System

This module implements data-driven calibrated confidence scores using:
- Platt Scaling (Sigmoid calibration)
- Isotonic Regression
- Comprehensive validation and evaluation metrics

The calibration system uses user interaction data and model performance metrics
to provide more accurate confidence estimates for RAG responses.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Literal
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)

@dataclass
class CalibrationSample:
    """Individual calibration training sample"""
    question: str
    predicted_confidence: float
    true_quality: float  # Ground truth quality score (0-1)
    response_time: float
    sources_count: int
    has_citations: bool
    retrieval_score: float
    timestamp: datetime
    user_feedback: Optional[float] = None  # Optional user rating (0-1)
    context_length: int = 0
    model_used: str = ""
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML training"""
        return np.array([
            self.predicted_confidence,
            self.response_time,
            self.sources_count,
            int(self.has_citations),
            self.retrieval_score,
            self.context_length,
            self.user_feedback or 0.5  # Default neutral if no feedback
        ])

@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality"""
    brier_score: float
    log_loss: float
    auc_score: float
    reliability_curve: Tuple[np.ndarray, np.ndarray]
    expected_calibration_error: float
    maximum_calibration_error: float
    calibration_method: str
    n_samples: int
    
class ConfidenceCalibrator:
    """
    Advanced confidence calibration system for RAG responses.
    
    Supports multiple calibration methods and provides comprehensive
    evaluation metrics for confidence score reliability.
    """
    
    def __init__(self, 
                 method: Literal["platt", "isotonic", "both"] = "both",
                 data_dir: str = "data/calibration",
                 min_samples: int = 50):
        """
        Initialize confidence calibrator.
        
        Args:
            method: Calibration method ("platt", "isotonic", or "both")
            data_dir: Directory to store calibration data and models
            min_samples: Minimum samples required for calibration
        """
        self.method = method
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.min_samples = min_samples
        
        # Calibration models
        self.platt_calibrator = None
        self.isotonic_calibrator = None
        self.feature_scaler = None
        
        # Training data storage
        self.training_samples: List[CalibrationSample] = []
        self.is_trained = False
        
        # Performance tracking
        self.metrics_history: List[CalibrationMetrics] = []
        
        logger.info(f"Initialized ConfidenceCalibrator with method={method}")
    
    def add_training_sample(self, sample: CalibrationSample) -> None:
        """Add a new training sample for calibration"""
        self.training_samples.append(sample)
        
        # Auto-retrain if we have enough samples and significant new data
        if (len(self.training_samples) % 20 == 0 and 
            len(self.training_samples) >= self.min_samples):
            logger.info(f"Auto-retraining with {len(self.training_samples)} samples")
            self.train()
    
    def collect_interaction_data(self, 
                               question: str,
                               rag_result: Dict[str, Any],
                               user_feedback: Optional[float] = None) -> CalibrationSample:
        """
        Collect interaction data and create calibration sample.
        
        Args:
            question: User question
            rag_result: RAG system response dictionary
            user_feedback: Optional user rating (0-1)
            
        Returns:
            CalibrationSample for training
        """
        # Extract features from RAG result
        predicted_confidence = rag_result.get("confidence", 0.0)
        response_time = rag_result.get("response_time", 0.0)
        sources_count = len(rag_result.get("sources", []))
        has_citations = rag_result.get("has_citations", False)
        retrieval_score = rag_result.get("retrieval_score", 0.0)
        context_length = rag_result.get("context_length", 0)
        
        # Calculate true quality score (heuristic-based if no user feedback)
        true_quality = self._calculate_true_quality(rag_result, user_feedback)
        
        sample = CalibrationSample(
            question=question,
            predicted_confidence=predicted_confidence,
            true_quality=true_quality,
            response_time=response_time,
            sources_count=sources_count,
            has_citations=has_citations,
            retrieval_score=retrieval_score,
            timestamp=datetime.now(),
            user_feedback=user_feedback,
            context_length=context_length,
            model_used=rag_result.get("model_id", "unknown")
        )
        
        self.add_training_sample(sample)
        return sample
    
    def _calculate_true_quality(self, 
                              rag_result: Dict[str, Any], 
                              user_feedback: Optional[float]) -> float:
        """
        Calculate true quality score for training.
        
        If user feedback is available, use it directly.
        Otherwise, use heuristic based on system metrics.
        """
        if user_feedback is not None:
            return user_feedback
        
        # Heuristic quality calculation
        metrics = []
        
        # Citation quality
        if rag_result.get("has_citations", False):
            metrics.append(0.8)
        else:
            metrics.append(0.3)
        
        # Source quality
        sources_count = len(rag_result.get("sources", []))
        if sources_count >= 3:
            metrics.append(0.9)
        elif sources_count >= 1:
            metrics.append(0.7)
        else:
            metrics.append(0.2)
        
        # Response completeness (based on length)
        answer_length = len(rag_result.get("answer", ""))
        if answer_length >= 200:
            metrics.append(0.8)
        elif answer_length >= 100:
            metrics.append(0.6)
        else:
            metrics.append(0.4)
        
        # Retrieval quality
        retrieval_count = rag_result.get("retrieval_count", 0)
        if retrieval_count >= 5:
            metrics.append(0.8)
        elif retrieval_count >= 2:
            metrics.append(0.6)
        else:
            metrics.append(0.3)
        
        return np.mean(metrics)
    
    def train(self) -> bool:
        """
        Train calibration models on collected data.
        
        Returns:
            True if training successful, False otherwise
        """
        if len(self.training_samples) < self.min_samples:
            logger.warning(f"Insufficient samples for training: {len(self.training_samples)} < {self.min_samples}")
            return False
        
        logger.info(f"Training calibration models with {len(self.training_samples)} samples")
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(np.unique(y)) < 2:
                logger.warning("Insufficient class diversity for calibration training")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
            
            # Train calibration models
            if self.method in ["platt", "both"]:
                self._train_platt_calibration(X_train, y_train)
            
            if self.method in ["isotonic", "both"]:
                self._train_isotonic_calibration(X_train, y_train)
            
            # Evaluate models
            metrics = self._evaluate_calibration(X_test, y_test)
            self.metrics_history.append(metrics)
            
            self.is_trained = True
            self._save_models()
            
            logger.info(f"Calibration training completed. Brier Score: {metrics.brier_score:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Calibration training failed: {e}")
            return False
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target vector for training"""
        X = np.array([sample.to_feature_vector() for sample in self.training_samples])
        y = np.array([sample.true_quality for sample in self.training_samples])
        
        # Normalize features
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            X = self.feature_scaler.fit_transform(X)
        else:
            X = self.feature_scaler.transform(X)
        
        return X, y
    
    def _train_platt_calibration(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train Platt scaling (sigmoid) calibration"""
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        base_model.fit(X_train, y_train > 0.5)  # Binary classification
        
        # Use CalibratedClassifierCV for Platt scaling
        self.platt_calibrator = CalibratedClassifierCV(
            base_model, method='sigmoid', cv=3
        )
        self.platt_calibrator.fit(X_train, y_train > 0.5)
        
        logger.info("Platt calibration model trained")
    
    def _train_isotonic_calibration(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train isotonic regression calibration"""
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        base_model.fit(X_train, y_train > 0.5)  # Binary classification
        
        # Use CalibratedClassifierCV for isotonic regression
        self.isotonic_calibrator = CalibratedClassifierCV(
            base_model, method='isotonic', cv=3
        )
        self.isotonic_calibrator.fit(X_train, y_train > 0.5)
        
        logger.info("Isotonic calibration model trained")
    
    def calibrate_confidence(self, 
                           original_confidence: float,
                           features: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply calibration to original confidence score.
        
        Args:
            original_confidence: Original confidence from RAG system
            features: Additional features for calibration
            
        Returns:
            Dictionary with calibrated confidence scores
        """
        if not self.is_trained:
            logger.warning("Calibration models not trained, returning original confidence")
            return {"original": original_confidence, "calibrated": original_confidence}
        
        try:
            # Prepare feature vector
            feature_vector = np.array([[
                original_confidence,
                features.get("response_time", 1.0),
                features.get("sources_count", 1),
                int(features.get("has_citations", False)),
                features.get("retrieval_score", 0.5),
                features.get("context_length", 100),
                0.5  # Default user feedback
            ]])
            
            # Normalize features
            if self.feature_scaler:
                feature_vector = self.feature_scaler.transform(feature_vector)
            
            results = {"original": original_confidence}
            
            # Apply calibration methods
            if self.platt_calibrator:
                platt_prob = self.platt_calibrator.predict_proba(feature_vector)[0, 1]
                results["platt_calibrated"] = float(platt_prob)
            
            if self.isotonic_calibrator:
                isotonic_prob = self.isotonic_calibrator.predict_proba(feature_vector)[0, 1]
                results["isotonic_calibrated"] = float(isotonic_prob)
            
            # Use ensemble if both methods available
            if self.platt_calibrator and self.isotonic_calibrator:
                ensemble_confidence = (results["platt_calibrated"] + results["isotonic_calibrated"]) / 2
                results["calibrated"] = ensemble_confidence
            elif self.platt_calibrator:
                results["calibrated"] = results["platt_calibrated"]
            elif self.isotonic_calibrator:
                results["calibrated"] = results["isotonic_calibrated"]
            else:
                results["calibrated"] = original_confidence
            
            return results
            
        except Exception as e:
            logger.error(f"Confidence calibration failed: {e}")
            return {"original": original_confidence, "calibrated": original_confidence}
    
    def _evaluate_calibration(self, X_test: np.ndarray, y_test: np.ndarray) -> CalibrationMetrics:
        """Evaluate calibration quality with comprehensive metrics"""
        
        # Get predictions from best available model
        if self.platt_calibrator:
            y_pred_proba = self.platt_calibrator.predict_proba(X_test)[:, 1]
            method = "platt"
        elif self.isotonic_calibrator:
            y_pred_proba = self.isotonic_calibrator.predict_proba(X_test)[:, 1]
            method = "isotonic"
        else:
            raise ValueError("No calibration model available for evaluation")
        
        # Convert continuous targets to binary for evaluation
        y_binary = (y_test > 0.5).astype(int)
        
        # Calculate metrics
        brier_score = brier_score_loss(y_binary, y_pred_proba)
        log_loss_score = log_loss(y_binary, y_pred_proba)
        auc_score = roc_auc_score(y_binary, y_pred_proba)
        
        # Reliability curve
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_binary[in_bin].mean()
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracy_in_bin)
        
        reliability_curve = (np.array(bin_centers), np.array(bin_accuracies))
        
        # Expected Calibration Error (ECE)
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_binary[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Maximum Calibration Error (MCE)
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_binary[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return CalibrationMetrics(
            brier_score=brier_score,
            log_loss=log_loss_score,
            auc_score=auc_score,
            reliability_curve=reliability_curve,
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            calibration_method=method,
            n_samples=len(y_test)
        )
    
    def _save_models(self) -> None:
        """Save trained calibration models to disk"""
        models_dir = self.data_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        try:
            if self.platt_calibrator:
                with open(models_dir / "platt_calibrator.pkl", "wb") as f:
                    pickle.dump(self.platt_calibrator, f)
            
            if self.isotonic_calibrator:
                with open(models_dir / "isotonic_calibrator.pkl", "wb") as f:
                    pickle.dump(self.isotonic_calibrator, f)
            
            if self.feature_scaler:
                with open(models_dir / "feature_scaler.pkl", "wb") as f:
                    pickle.dump(self.feature_scaler, f)
            
            # Save training metadata
            metadata = {
                "method": self.method,
                "n_samples": len(self.training_samples),
                "last_trained": datetime.now().isoformat(),
                "is_trained": self.is_trained
            }
            
            with open(models_dir / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Calibration models saved to {models_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save calibration models: {e}")
    
    def load_models(self) -> bool:
        """Load trained calibration models from disk"""
        models_dir = self.data_dir / "models"
        
        if not models_dir.exists():
            logger.info("No saved calibration models found")
            return False
        
        try:
            # Load metadata
            metadata_path = models_dir / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                self.is_trained = metadata.get("is_trained", False)
            
            # Load models
            platt_path = models_dir / "platt_calibrator.pkl"
            if platt_path.exists():
                with open(platt_path, "rb") as f:
                    self.platt_calibrator = pickle.load(f)
                logger.info("Loaded Platt calibration model")
            
            isotonic_path = models_dir / "isotonic_calibrator.pkl"
            if isotonic_path.exists():
                with open(isotonic_path, "rb") as f:
                    self.isotonic_calibrator = pickle.load(f)
                logger.info("Loaded Isotonic calibration model")
            
            scaler_path = models_dir / "feature_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.feature_scaler = pickle.load(f)
                logger.info("Loaded feature scaler")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration models: {e}")
            return False
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """Generate comprehensive calibration report"""
        if not self.metrics_history:
            return {"status": "no_metrics", "message": "No calibration metrics available"}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            "status": "trained" if self.is_trained else "not_trained",
            "training_samples": len(self.training_samples),
            "method": self.method,
            "latest_metrics": {
                "brier_score": latest_metrics.brier_score,
                "log_loss": latest_metrics.log_loss,
                "auc_score": latest_metrics.auc_score,
                "expected_calibration_error": latest_metrics.expected_calibration_error,
                "maximum_calibration_error": latest_metrics.maximum_calibration_error,
                "n_samples": latest_metrics.n_samples
            },
            "calibration_quality": self._assess_calibration_quality(latest_metrics),
            "recommendations": self._generate_recommendations(latest_metrics)
        }
    
    def _assess_calibration_quality(self, metrics: CalibrationMetrics) -> str:
        """Assess overall calibration quality based on metrics"""
        if metrics.expected_calibration_error < 0.05:
            return "excellent"
        elif metrics.expected_calibration_error < 0.1:
            return "good"
        elif metrics.expected_calibration_error < 0.15:
            return "fair"
        else:
            return "poor"
    
    def _generate_recommendations(self, metrics: CalibrationMetrics) -> List[str]:
        """Generate recommendations for improving calibration"""
        recommendations = []
        
        if metrics.n_samples < 100:
            recommendations.append("Collect more training samples for better calibration")
        
        if metrics.expected_calibration_error > 0.1:
            recommendations.append("Consider collecting more user feedback for ground truth labels")
        
        if metrics.auc_score < 0.7:
            recommendations.append("Review feature engineering - current features may not be predictive enough")
        
        if metrics.brier_score > 0.25:
            recommendations.append("Consider different calibration methods or ensemble approaches")
        
        return recommendations

# Global calibrator instance
_global_calibrator: Optional[ConfidenceCalibrator] = None

def get_calibrator() -> ConfidenceCalibrator:
    """Get or create global calibrator instance"""
    global _global_calibrator
    if _global_calibrator is None:
        _global_calibrator = ConfidenceCalibrator()
        _global_calibrator.load_models()  # Try to load existing models
    return _global_calibrator

def calibrate_rag_confidence(rag_result: Dict[str, Any], 
                           question: str = "",
                           user_feedback: Optional[float] = None) -> Dict[str, Any]:
    """
    Convenience function to calibrate RAG confidence scores.
    
    Args:
        rag_result: Original RAG system result dictionary
        question: User question (for logging)
        user_feedback: Optional user feedback for training
        
    Returns:
        Enhanced RAG result with calibrated confidence
    """
    calibrator = get_calibrator()
    
    # Collect training data
    if question:
        calibrator.collect_interaction_data(question, rag_result, user_feedback)
    
    # Get calibrated confidence
    original_confidence = rag_result.get("confidence", 0.0)
    features = {
        "response_time": rag_result.get("response_time", 1.0),
        "sources_count": len(rag_result.get("sources", [])),
        "has_citations": rag_result.get("has_citations", False),
        "retrieval_score": rag_result.get("retrieval_score", 0.5),
        "context_length": rag_result.get("context_length", 100)
    }
    
    calibration_results = calibrator.calibrate_confidence(original_confidence, features)
    
    # Add calibration info to result
    rag_result["confidence_calibration"] = calibration_results
    rag_result["confidence"] = calibration_results.get("calibrated", original_confidence)
    
    return rag_result 