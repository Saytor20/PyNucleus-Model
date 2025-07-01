"""
Feedback models for user interaction data collection.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Index
from sqlalchemy.sql import func
from ..db.models import BaseModel


class UserFeedback(BaseModel):
    """
    Model for storing user feedback on RAG responses.
    
    This model collects both explicit (ratings, thumbs up/down) and implicit 
    (confidence scores, query patterns) feedback for RLHF training data.
    """
    __tablename__ = "user_feedback"
    
    # User identification (nullable for anonymous feedback)
    user_id = Column(String(255), nullable=True, index=True)
    
    # Query identification for deduplication and analysis
    query_hash = Column(String(64), nullable=False, index=True)
    
    # Original system confidence score (before calibration)
    raw_confidence = Column(Float, nullable=False)
    
    # Explicit feedback scores
    rating = Column(Integer, nullable=True)  # 1-10 rating scale
    thumb_up = Column(Boolean, nullable=True)  # Boolean thumbs up/down
    
    # Implicit feedback score (calculated from user behavior)
    implicit_score = Column(Float, nullable=True)
    
    # Timestamp override (uses BaseModel's created_at by default)
    # created_at is inherited from BaseModel with UTC default
    
    # Indexes for efficient queries
    __table_args__ = (
        Index('idx_query_hash_created', 'query_hash', 'created_at'),
        Index('idx_user_feedback_created', 'user_id', 'created_at'),
        Index('idx_rating_confidence', 'rating', 'raw_confidence'),
    ) 