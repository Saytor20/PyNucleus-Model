"""
Feedback module for PyNucleus user feedback collection and management.
"""

from .models import UserFeedback
from ..db import SessionLocal


def save_feedback(**kwargs) -> UserFeedback:
    """
    Helper function to save user feedback to the database.
    
    Args:
        **kwargs: Keyword arguments for UserFeedback model fields
        
    Returns:
        UserFeedback: The saved feedback record
    """
    db = SessionLocal()
    try:
        feedback = UserFeedback(**kwargs)
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        return feedback
    finally:
        db.close()


__all__ = ["UserFeedback", "save_feedback"] 