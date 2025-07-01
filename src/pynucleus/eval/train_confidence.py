"""
Confidence Calibration Training Module

This module trains isotonic regression models to calibrate confidence scores
using user feedback data. It can be run from CLI as:
python -m pynucleus.eval.train_confidence
"""

import os
import sys
import pickle
import logging
import warnings
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pynucleus.db import DATABASE_URL, SessionLocal
from src.pynucleus.feedback.models import UserFeedback

logger = logging.getLogger(__name__)


def load_feedback_data() -> Optional[pd.DataFrame]:
    """
    Load feedback data from database.
    
    Returns:
        DataFrame with feedback data or None if insufficient data
    """
    try:
        session = SessionLocal()
        
        # Query feedback with non-null ratings
        query = session.query(UserFeedback).filter(UserFeedback.rating.isnot(None))
        feedback_records = query.all()
        
        if len(feedback_records) < 100:
            logger.warning(f"Insufficient feedback data: {len(feedback_records)} rows, need ≥100")
            return None
        
        # Convert to DataFrame
        data = []
        for record in feedback_records:
            data.append({
                'id': record.id,
                'raw_confidence': record.raw_confidence,
                'rating': record.rating,
                'created_at': record.created_at,
                'user_id': record.user_id,
                'query_hash': record.query_hash
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} feedback records with ratings")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load feedback data: {e}")
        return None
    finally:
        session.close()


def split_train_validation(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation by timestamp (80/20).
    
    Args:
        df: DataFrame with feedback data
        test_size: Fraction for validation set
        
    Returns:
        Tuple of (train_df, val_df)
    """
    # Sort by timestamp
    df_sorted = df.sort_values('created_at')
    
    # Split by timestamp (earlier data for training)
    split_idx = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:split_idx].copy()
    val_df = df_sorted.iloc[split_idx:].copy()
    
    logger.info(f"Split data: {len(train_df)} train, {len(val_df)} validation")
    return train_df, val_df


def prepare_features_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features (raw_confidence) and targets (normalized_rating/10).
    
    Args:
        df: DataFrame with feedback data
        
    Returns:
        Tuple of (X, y) arrays
    """
    X = df['raw_confidence'].values.reshape(-1, 1)
    y = df['rating'].values / 10.0  # Normalize 1-10 ratings to 0-1
    
    return X, y


def train_isotonic_regression(X_train: np.ndarray, y_train: np.ndarray) -> IsotonicRegression:
    """
    Train isotonic regression model.
    
    Args:
        X_train: Training features (raw confidence scores)
        y_train: Training targets (normalized ratings)
        
    Returns:
        Trained IsotonicRegression model
    """
    # Flatten X for isotonic regression (expects 1D)
    X_train_flat = X_train.flatten()
    
    model = IsotonicRegression(out_of_bounds='clip')
    model.fit(X_train_flat, y_train)
    
    logger.info("Trained isotonic regression model")
    return model


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels (0/1)
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve
        
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def evaluate_model(model: IsotonicRegression, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate calibration model on validation set.
    
    Args:
        model: Trained isotonic regression model
        X_val: Validation features
        y_val: Validation targets
        
    Returns:
        Tuple of (ECE, Brier score)
    """
    # Get predictions
    X_val_flat = X_val.flatten()
    y_pred = model.predict(X_val_flat)
    
    # Convert continuous targets to binary for ECE calculation
    # Use median split for binary classification
    y_binary = (y_val > np.median(y_val)).astype(int)
    
    # Calculate ECE and Brier score
    ece = calculate_ece(y_binary, y_pred)
    brier = brier_score_loss(y_binary, y_pred)
    
    logger.info(f"Validation metrics - ECE: {ece:.4f}, Brier: {brier:.4f}")
    return ece, brier


def save_model(model: IsotonicRegression, model_path: Path) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained isotonic regression model
        model_path: Path to save model file
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved calibration model to {model_path}")


def main():
    """Main training function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting confidence calibration training")
    
    # Load feedback data
    df = load_feedback_data()
    if df is None:
        logger.warning("Exiting due to insufficient feedback data")
        return
    
    # Split train/validation
    train_df, val_df = split_train_validation(df)
    
    # Prepare features and targets
    X_train, y_train = prepare_features_targets(train_df)
    X_val, y_val = prepare_features_targets(val_df)
    
    # Train model
    model = train_isotonic_regression(X_train, y_train)
    
    # Evaluate model
    ece, brier = evaluate_model(model, X_val, y_val)
    
    # Save model if ECE is good enough
    model_path = Path("data/calibration/models/calibration_isotonic.pkl")
    
    if ece <= 0.05:
        save_model(model, model_path)
        logger.info(f"Model saved successfully with ECE: {ece:.4f}")
    else:
        logger.warning(f"Model ECE too high ({ece:.4f} > 0.05), skipping save")
    
    logger.info("Training completed")


if __name__ == "__main__":
    main() 