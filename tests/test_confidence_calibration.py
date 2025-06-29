"""
Tests for confidence calibration module.

This test suite validates:
- Calibration training and validation
- Platt scaling and isotonic regression
- Integration with RAG system
- Evaluation metrics accuracy
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.pynucleus.eval.confidence_calibration import (
    ConfidenceCalibrator,
    CalibrationSample,
    CalibrationMetrics,
    calibrate_rag_confidence,
    get_calibrator
)

class TestCalibrationSample:
    """Test CalibrationSample dataclass functionality"""
    
    def test_calibration_sample_creation(self):
        """Test creating a calibration sample"""
        sample = CalibrationSample(
            question="What is distillation?",
            predicted_confidence=0.8,
            true_quality=0.9,
            response_time=2.5,
            sources_count=3,
            has_citations=True,
            retrieval_score=0.85,
            timestamp=datetime.now(),
            user_feedback=0.9,
            context_length=1500,
            model_used="test_model"
        )
        
        assert sample.question == "What is distillation?"
        assert sample.predicted_confidence == 0.8
        assert sample.true_quality == 0.9
        assert sample.user_feedback == 0.9
    
    def test_feature_vector_conversion(self):
        """Test converting calibration sample to feature vector"""
        sample = CalibrationSample(
            question="Test question",
            predicted_confidence=0.7,
            true_quality=0.8,
            response_time=1.5,
            sources_count=2,
            has_citations=False,
            retrieval_score=0.6,
            timestamp=datetime.now(),
            user_feedback=0.75,
            context_length=1000
        )
        
        features = sample.to_feature_vector()
        expected = np.array([0.7, 1.5, 2, 0, 0.6, 1000, 0.75])
        
        np.testing.assert_array_equal(features, expected)
    
    def test_feature_vector_no_feedback(self):
        """Test feature vector with no user feedback"""
        sample = CalibrationSample(
            question="Test question",
            predicted_confidence=0.7,
            true_quality=0.8,
            response_time=1.5,
            sources_count=2,
            has_citations=True,
            retrieval_score=0.6,
            timestamp=datetime.now(),
            context_length=1000
        )
        
        features = sample.to_feature_vector()
        assert features[-1] == 0.5  # Default neutral feedback

class TestConfidenceCalibrator:
    """Test ConfidenceCalibrator functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def calibrator(self, temp_dir):
        """Create calibrator instance for testing"""
        return ConfidenceCalibrator(
            method="both",
            data_dir=temp_dir,
            min_samples=10  # Low threshold for testing
        )
    
    def test_calibrator_initialization(self, temp_dir):
        """Test calibrator initialization"""
        calibrator = ConfidenceCalibrator(
            method="platt",
            data_dir=temp_dir,
            min_samples=20
        )
        
        assert calibrator.method == "platt"
        assert calibrator.min_samples == 20
        assert len(calibrator.training_samples) == 0
        assert not calibrator.is_trained
        assert Path(temp_dir).exists()
    
    def test_add_training_sample(self, calibrator):
        """Test adding training samples"""
        sample = CalibrationSample(
            question="Test question",
            predicted_confidence=0.8,
            true_quality=0.9,
            response_time=2.0,
            sources_count=3,
            has_citations=True,
            retrieval_score=0.85,
            timestamp=datetime.now()
        )
        
        calibrator.add_training_sample(sample)
        assert len(calibrator.training_samples) == 1
        assert calibrator.training_samples[0] == sample
    
    def test_collect_interaction_data(self, calibrator):
        """Test collecting interaction data from RAG results"""
        rag_result = {
            "confidence": 0.75,
            "response_time": 1.8,
            "sources": ["source1", "source2"],
            "has_citations": True,
            "retrieval_score": 0.8,
            "context_length": 1200,
            "answer": "This is a comprehensive answer about distillation process.",
            "retrieval_count": 3
        }
        
        sample = calibrator.collect_interaction_data(
            question="What is distillation?",
            rag_result=rag_result,
            user_feedback=0.9
        )
        
        assert sample.question == "What is distillation?"
        assert sample.predicted_confidence == 0.75
        assert sample.user_feedback == 0.9
        assert sample.sources_count == 2
        assert sample.has_citations == True
        assert len(calibrator.training_samples) == 1
    
    def test_calculate_true_quality_with_feedback(self, calibrator):
        """Test true quality calculation with user feedback"""
        rag_result = {"answer": "test"}
        quality = calibrator._calculate_true_quality(rag_result, user_feedback=0.8)
        assert quality == 0.8
    
    def test_calculate_true_quality_heuristic(self, calibrator):
        """Test heuristic-based true quality calculation"""
        rag_result = {
            "has_citations": True,
            "sources": ["source1", "source2", "source3"],
            "answer": "This is a detailed answer with sufficient length for quality assessment.",
            "retrieval_count": 5
        }
        
        quality = calibrator._calculate_true_quality(rag_result, user_feedback=None)
        assert 0.0 <= quality <= 1.0
        assert quality > 0.7  # Should be high quality based on heuristics
    
    @pytest.fixture
    def trained_calibrator(self, calibrator):
        """Create a calibrator with training data"""
        # Generate synthetic training data
        np.random.seed(42)
        for i in range(50):  # Minimum samples for training
            # Create realistic synthetic data
            base_confidence = np.random.uniform(0.1, 0.9)
            noise = np.random.normal(0, 0.1)
            true_quality = np.clip(base_confidence + noise, 0, 1)
            
            sample = CalibrationSample(
                question=f"Question {i}",
                predicted_confidence=base_confidence,
                true_quality=true_quality,
                response_time=np.random.uniform(0.5, 5.0),
                sources_count=np.random.randint(1, 5),
                has_citations=np.random.choice([True, False]),
                retrieval_score=np.random.uniform(0.3, 0.9),
                timestamp=datetime.now(),
                user_feedback=true_quality,
                context_length=np.random.randint(100, 2000)
            )
            calibrator.add_training_sample(sample)
        
        # Train the calibrator
        success = calibrator.train()
        assert success, "Calibrator training should succeed with sufficient data"
        
        return calibrator
    
    def test_training_insufficient_samples(self, calibrator):
        """Test training with insufficient samples"""
        # Add only a few samples
        for i in range(5):
            sample = CalibrationSample(
                question=f"Question {i}",
                predicted_confidence=0.5,
                true_quality=0.6,
                response_time=1.0,
                sources_count=2,
                has_citations=True,
                retrieval_score=0.7,
                timestamp=datetime.now()
            )
            calibrator.add_training_sample(sample)
        
        success = calibrator.train()
        assert not success
        assert not calibrator.is_trained
    
    def test_training_success(self, trained_calibrator):
        """Test successful training"""
        assert trained_calibrator.is_trained
        assert len(trained_calibrator.metrics_history) > 0
        
        # Check that models were created
        if trained_calibrator.method in ["platt", "both"]:
            assert trained_calibrator.platt_calibrator is not None
        if trained_calibrator.method in ["isotonic", "both"]:
            assert trained_calibrator.isotonic_calibrator is not None
        assert trained_calibrator.feature_scaler is not None
    
    def test_calibrate_confidence_untrained(self, calibrator):
        """Test confidence calibration without training"""
        features = {
            "response_time": 2.0,
            "sources_count": 3,
            "has_citations": True,
            "retrieval_score": 0.8,
            "context_length": 1500
        }
        
        result = calibrator.calibrate_confidence(0.7, features)
        
        # Should return original confidence when not trained
        assert result["original"] == 0.7
        assert result["calibrated"] == 0.7
    
    def test_calibrate_confidence_trained(self, trained_calibrator):
        """Test confidence calibration with trained model"""
        features = {
            "response_time": 2.0,
            "sources_count": 3,
            "has_citations": True,
            "retrieval_score": 0.8,
            "context_length": 1500
        }
        
        result = trained_calibrator.calibrate_confidence(0.7, features)
        
        assert "original" in result
        assert "calibrated" in result
        assert result["original"] == 0.7
        assert 0.0 <= result["calibrated"] <= 1.0
        
        # Should have method-specific results
        if trained_calibrator.method in ["platt", "both"]:
            assert "platt_calibrated" in result
        if trained_calibrator.method in ["isotonic", "both"]:
            assert "isotonic_calibrated" in result
    
    def test_model_persistence(self, trained_calibrator):
        """Test saving and loading calibration models"""
        # Save models
        trained_calibrator._save_models()
        
        # Create new calibrator and load models
        new_calibrator = ConfidenceCalibrator(
            method="both",
            data_dir=trained_calibrator.data_dir,
            min_samples=10
        )
        
        success = new_calibrator.load_models()
        assert success
        assert new_calibrator.is_trained
        
        # Test that loaded models work
        features = {
            "response_time": 2.0,
            "sources_count": 3,
            "has_citations": True,
            "retrieval_score": 0.8,
            "context_length": 1500
        }
        
        result = new_calibrator.calibrate_confidence(0.7, features)
        assert "calibrated" in result
        assert 0.0 <= result["calibrated"] <= 1.0
    
    def test_calibration_report(self, trained_calibrator):
        """Test calibration report generation"""
        report = trained_calibrator.get_calibration_report()
        
        assert report["status"] == "trained"
        assert report["training_samples"] >= 50
        assert report["method"] == "both"
        assert "latest_metrics" in report
        assert "calibration_quality" in report
        assert "recommendations" in report
        
        # Check metrics
        metrics = report["latest_metrics"]
        assert "brier_score" in metrics
        assert "auc_score" in metrics
        assert "expected_calibration_error" in metrics
    
    def test_evaluation_metrics(self, trained_calibrator):
        """Test calibration evaluation metrics"""
        # Get the latest metrics
        assert len(trained_calibrator.metrics_history) > 0
        metrics = trained_calibrator.metrics_history[-1]
        
        # Validate metric ranges
        assert 0.0 <= metrics.brier_score <= 1.0
        assert 0.0 <= metrics.auc_score <= 1.0
        assert metrics.expected_calibration_error >= 0.0
        assert metrics.maximum_calibration_error >= 0.0
        assert metrics.n_samples > 0
        
        # Check reliability curve
        bin_centers, bin_accuracies = metrics.reliability_curve
        assert len(bin_centers) <= 10  # Maximum number of bins
        assert len(bin_centers) == len(bin_accuracies)

class TestCalibrationIntegration:
    """Test integration with RAG system"""
    
    def test_calibrate_rag_confidence_function(self):
        """Test the convenience function for RAG confidence calibration"""
        rag_result = {
            "confidence": 0.8,
            "response_time": 2.0,
            "sources": ["source1", "source2"],
            "has_citations": True,
            "retrieval_score": 0.75,
            "context_length": 1200,
            "answer": "Test answer"
        }
        
        # Mock the global calibrator to avoid training requirements
        with patch('src.pynucleus.eval.confidence_calibration.get_calibrator') as mock_get_calibrator:
            mock_calibrator = MagicMock()
            mock_calibrator.collect_interaction_data.return_value = None
            mock_calibrator.calibrate_confidence.return_value = {
                "original": 0.8,
                "calibrated": 0.75,
                "platt_calibrated": 0.74,
                "isotonic_calibrated": 0.76
            }
            mock_get_calibrator.return_value = mock_calibrator
            
            result = calibrate_rag_confidence(
                rag_result=rag_result,
                question="Test question",
                user_feedback=0.9
            )
            
            assert "confidence_calibration" in result
            assert result["confidence"] == 0.75  # Should use calibrated confidence
            assert result["confidence_calibration"]["original"] == 0.8
            
            # Verify calibrator was called correctly
            mock_calibrator.collect_interaction_data.assert_called_once()
            mock_calibrator.calibrate_confidence.assert_called_once()
    
    def test_get_calibrator_singleton(self):
        """Test global calibrator singleton behavior"""
        # Clear any existing global calibrator
        import src.pynucleus.eval.confidence_calibration as calib_module
        calib_module._global_calibrator = None
        
        # Get calibrator instances
        calibrator1 = get_calibrator()
        calibrator2 = get_calibrator()
        
        # Should be the same instance
        assert calibrator1 is calibrator2
        assert isinstance(calibrator1, ConfidenceCalibrator)

class TestCalibrationMethods:
    """Test specific calibration methods"""
    
    @pytest.fixture
    def calibration_data(self):
        """Generate synthetic calibration data for testing"""
        np.random.seed(42)
        n_samples = 100
        
        # Generate features and targets with known relationship
        X = np.random.randn(n_samples, 7)
        # Create targets that have some relationship to features
        y = (X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.1, n_samples)) > 0
        
        return X.astype(float), y.astype(float)
    
    def test_platt_calibration_method(self, temp_dir, calibration_data):
        """Test Platt scaling calibration method"""
        X, y = calibration_data
        
        calibrator = ConfidenceCalibrator(
            method="platt",
            data_dir=temp_dir,
            min_samples=10
        )
        
        # Manually set up training data
        for i in range(len(X)):
            sample = CalibrationSample(
                question=f"Question {i}",
                predicted_confidence=float(X[i, 0]),
                true_quality=float(y[i]),
                response_time=float(X[i, 1]),
                sources_count=int(abs(X[i, 2]) + 1),
                has_citations=bool(X[i, 3] > 0),
                retrieval_score=float(abs(X[i, 4])),
                timestamp=datetime.now(),
                context_length=int(abs(X[i, 5]) * 100 + 100)
            )
            calibrator.add_training_sample(sample)
        
        success = calibrator.train()
        assert success
        assert calibrator.platt_calibrator is not None
        assert calibrator.isotonic_calibrator is None  # Only Platt method
    
    def test_isotonic_calibration_method(self, temp_dir, calibration_data):
        """Test isotonic regression calibration method"""
        X, y = calibration_data
        
        calibrator = ConfidenceCalibrator(
            method="isotonic",
            data_dir=temp_dir,
            min_samples=10
        )
        
        # Manually set up training data
        for i in range(len(X)):
            sample = CalibrationSample(
                question=f"Question {i}",
                predicted_confidence=float(X[i, 0]),
                true_quality=float(y[i]),
                response_time=float(X[i, 1]),
                sources_count=int(abs(X[i, 2]) + 1),
                has_citations=bool(X[i, 3] > 0),
                retrieval_score=float(abs(X[i, 4])),
                timestamp=datetime.now(),
                context_length=int(abs(X[i, 5]) * 100 + 100)
            )
            calibrator.add_training_sample(sample)
        
        success = calibrator.train()
        assert success
        assert calibrator.isotonic_calibrator is not None
        assert calibrator.platt_calibrator is None  # Only isotonic method

@pytest.mark.integration
class TestFullCalibrationWorkflow:
    """Integration tests for complete calibration workflow"""
    
    def test_end_to_end_calibration_workflow(self):
        """Test complete calibration workflow from data collection to prediction"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize calibrator
            calibrator = ConfidenceCalibrator(
                method="both",
                data_dir=temp_dir,
                min_samples=20
            )
            
            # Simulate collecting data over time
            np.random.seed(42)
            for i in range(30):
                # Simulate RAG result
                predicted_conf = np.random.uniform(0.2, 0.9)
                rag_result = {
                    "confidence": predicted_conf,
                    "response_time": np.random.uniform(0.5, 3.0),
                    "sources": [f"source_{j}" for j in range(np.random.randint(1, 4))],
                    "has_citations": np.random.choice([True, False]),
                    "retrieval_score": np.random.uniform(0.4, 0.9),
                    "context_length": np.random.randint(100, 1500),
                    "answer": "Generated answer content"
                }
                
                # Simulate user feedback (correlated with predicted confidence)
                user_feedback = np.clip(
                    predicted_conf + np.random.normal(0, 0.15), 0, 1
                )
                
                # Collect interaction data
                calibrator.collect_interaction_data(
                    question=f"Question {i}",
                    rag_result=rag_result,
                    user_feedback=user_feedback
                )
            
            # Training should trigger automatically
            assert calibrator.is_trained
            
            # Test calibration on new data
            test_features = {
                "response_time": 1.5,
                "sources_count": 2,
                "has_citations": True,
                "retrieval_score": 0.7,
                "context_length": 800
            }
            
            result = calibrator.calibrate_confidence(0.6, test_features)
            
            assert "calibrated" in result
            assert 0.0 <= result["calibrated"] <= 1.0
            
            # Generate report
            report = calibrator.get_calibration_report()
            assert report["status"] == "trained"
            assert report["training_samples"] >= 20

if __name__ == "__main__":
    pytest.main([__file__]) 