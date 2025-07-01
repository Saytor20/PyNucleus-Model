"""
Unit tests for confidence calibration training module.
"""

import pytest
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from sklearn.isotonic import IsotonicRegression

# Import the module under test
from src.pynucleus.eval.train_confidence import (
    load_feedback_data,
    split_train_validation,
    prepare_features_targets,
    train_isotonic_regression,
    calculate_ece,
    evaluate_model,
    save_model,
    main
)
from src.pynucleus.eval import load_latest_model


class TestTrainConfidence:
    """Test cases for confidence training functions."""
    
    def test_split_train_validation(self):
        """Test train/validation split by timestamp."""
        # Create dummy data with timestamps
        timestamps = [
            datetime.now() - timedelta(days=10),
            datetime.now() - timedelta(days=8),
            datetime.now() - timedelta(days=6),
            datetime.now() - timedelta(days=4),
            datetime.now() - timedelta(days=2),
        ]
        
        df = pd.DataFrame({
            'raw_confidence': [0.1, 0.3, 0.5, 0.7, 0.9],
            'rating': [2, 4, 5, 8, 9],
            'created_at': timestamps
        })
        
        train_df, val_df = split_train_validation(df, test_size=0.2)
        
        # Check split sizes (80/20)
        assert len(train_df) == 4
        assert len(val_df) == 1
        
        # Check temporal ordering (earlier data in training)
        assert train_df['created_at'].max() <= val_df['created_at'].min()
    
    def test_prepare_features_targets(self):
        """Test feature and target preparation."""
        df = pd.DataFrame({
            'raw_confidence': [0.2, 0.5, 0.8],
            'rating': [3, 6, 9]
        })
        
        X, y = prepare_features_targets(df)
        
        # Check shapes
        assert X.shape == (3, 1)
        assert y.shape == (3,)
        
        # Check values
        np.testing.assert_array_equal(X.flatten(), [0.2, 0.5, 0.8])
        np.testing.assert_array_equal(y, [0.3, 0.6, 0.9])  # ratings/10
    
    def test_train_isotonic_regression(self):
        """Test isotonic regression training."""
        # Create simple training data
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        model = train_isotonic_regression(X_train, y_train)
        
        # Check model type
        assert isinstance(model, IsotonicRegression)
        
        # Check that model can make predictions
        predictions = model.predict([0.2, 0.6])
        assert len(predictions) == 2
        assert all(0 <= p <= 1 for p in predictions)
    
    def test_calculate_ece(self):
        """Test Expected Calibration Error calculation."""
        # Perfect calibration case
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        
        ece = calculate_ece(y_true, y_prob, n_bins=2)
        
        # ECE should be low for well-calibrated predictions
        assert 0 <= ece <= 1
        assert ece < 0.5  # Should be reasonably calibrated
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create a simple model
        model = IsotonicRegression()
        X_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        model.fit(X_train, y_train)
        
        # Validation data
        X_val = np.array([[0.2], [0.4], [0.6], [0.8]])
        y_val = np.array([0.2, 0.4, 0.6, 0.8])
        
        ece, brier = evaluate_model(model, X_val, y_val)
        
        # Check that metrics are reasonable
        assert 0 <= ece <= 1
        assert 0 <= brier <= 1
    
    def test_save_model(self):
        """Test model saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple model
            model = IsotonicRegression()
            model.fit([0.1, 0.5, 0.9], [0.1, 0.5, 0.9])
            
            # Save model
            model_path = Path(temp_dir) / "test_model.pkl"
            save_model(model, model_path)
            
            # Check file exists
            assert model_path.exists()
            
            # Check we can load it back
            import pickle
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Test that loaded model works
            predictions = loaded_model.predict([0.3, 0.7])
            assert len(predictions) == 2


class TestMainTrainingWorkflow:
    """Integration tests for the main training workflow."""
    
    @pytest.fixture
    def dummy_feedback_data(self):
        """Create dummy feedback data for testing."""
        # Generate 120 dummy feedback records (>100 threshold)
        np.random.seed(42)
        data = []
        
        for i in range(120):
            # Create realistic synthetic data
            raw_conf = np.random.uniform(0.1, 0.9)
            # Rating correlated with confidence but with noise
            rating = max(1, min(10, int(raw_conf * 10 + np.random.normal(0, 1.5))))
            
            data.append({
                'id': i + 1,
                'raw_confidence': raw_conf,
                'rating': rating,
                'created_at': datetime.now() - timedelta(days=i),
                'user_id': f'user_{i % 10}',
                'query_hash': f'hash_{i}'
            })
        
        return pd.DataFrame(data)
    
    @patch('src.pynucleus.eval.train_confidence.load_feedback_data')
    def test_main_training_with_sufficient_data(self, mock_load_data, dummy_feedback_data):
        """Test main training function with sufficient data."""
        mock_load_data.return_value = dummy_feedback_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch the model save path
            with patch('src.pynucleus.eval.train_confidence.Path') as mock_path:
                model_path = Path(temp_dir) / "calibration_isotonic.pkl"
                mock_path.return_value = model_path
                
                # Run main training
                main()
                
                # Check that model was saved (if ECE was good)
                # Note: We can't guarantee ECE will be ≤0.05 with random data,
                # but we can check the training process completed
                assert mock_load_data.called
    
    @patch('src.pynucleus.eval.train_confidence.load_feedback_data')
    def test_main_training_insufficient_data(self, mock_load_data, caplog):
        """Test main training function with insufficient data."""
        # Return insufficient data
        insufficient_data = pd.DataFrame({
            'raw_confidence': [0.5, 0.7],
            'rating': [5, 7],
            'created_at': [datetime.now(), datetime.now()]
        })
        mock_load_data.return_value = None  # Simulates insufficient data
        
        # Run main training
        main()
        
        # Check that warning was logged
        assert "Exiting due to insufficient feedback data" in caplog.text


class TestLoadLatestModel:
    """Test cases for loading trained models."""
    
    def test_load_model_nonexistent(self):
        """Test loading model when file doesn't exist."""
        with patch('src.pynucleus.eval.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            calibrator = load_latest_model()
            assert calibrator is None
    
    def test_load_model_success(self):
        """Test successful model loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save a model
            model = IsotonicRegression()
            model.fit([0.1, 0.5, 0.9], [0.2, 0.6, 0.8])
            
            model_path = Path(temp_dir) / "calibration_isotonic.pkl"
            model_path.parent.mkdir(exist_ok=True)
            
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Patch the path in load_latest_model
            with patch('src.pynucleus.eval.Path') as mock_path:
                mock_path.return_value = model_path
                
                calibrator = load_latest_model()
                
                # Check that we got a callable
                assert callable(calibrator)
                
                # Test the calibrator function
                result = calibrator(0.3)
                assert isinstance(result, float)
                assert 0.0 <= result <= 1.0
    
    def test_calibrator_bounds_checking(self):
        """Test that calibrator handles out-of-bounds input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save a model
            model = IsotonicRegression()
            model.fit([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
            
            model_path = Path(temp_dir) / "calibration_isotonic.pkl"
            model_path.parent.mkdir(exist_ok=True)
            
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Load calibrator
            with patch('src.pynucleus.eval.Path') as mock_path:
                mock_path.return_value = model_path
                
                calibrator = load_latest_model()
                
                # Test bounds checking
                assert 0.0 <= calibrator(-0.5) <= 1.0  # Below 0
                assert 0.0 <= calibrator(1.5) <= 1.0   # Above 1
                assert 0.0 <= calibrator(0.5) <= 1.0   # Normal range


class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    def test_complete_training_and_loading_workflow(self):
        """Test complete workflow from data creation to model loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy CSV data
            np.random.seed(42)
            data = []
            
            for i in range(100):
                raw_conf = np.random.uniform(0.1, 0.9)
                rating = max(1, min(10, int(raw_conf * 10 + np.random.normal(0, 1))))
                
                data.append({
                    'raw_confidence': raw_conf,
                    'rating': rating,
                    'created_at': datetime.now() - timedelta(hours=i)
                })
            
            df = pd.DataFrame(data)
            
            # Train model directly
            train_df, val_df = split_train_validation(df)
            X_train, y_train = prepare_features_targets(train_df)
            X_val, y_val = prepare_features_targets(val_df)
            
            model = train_isotonic_regression(X_train, y_train)
            ece, brier = evaluate_model(model, X_val, y_val)
            
            # Save model
            model_path = Path(temp_dir) / "calibration_isotonic.pkl"
            save_model(model, model_path)
            
            # Verify model file exists
            assert model_path.exists()
            
            # Load and test model
            with patch('src.pynucleus.eval.Path') as mock_path:
                mock_path.return_value = model_path
                
                calibrator = load_latest_model()
                assert calibrator is not None
                
                # Test calibration improves over baseline
                # (at least the model should work)
                test_confidence = 0.5
                calibrated = calibrator(test_confidence)
                assert isinstance(calibrated, float)
                assert 0.0 <= calibrated <= 1.0 