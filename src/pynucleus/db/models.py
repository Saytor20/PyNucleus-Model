"""
SQLAlchemy models for PyNucleus database.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, Index
from sqlalchemy.sql import func
from . import Base


class BaseModel(Base):
    """Base model with common fields."""
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


# Additional models can be added here as the system evolves 