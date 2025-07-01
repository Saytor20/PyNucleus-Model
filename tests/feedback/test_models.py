"""
Unit tests for feedback models.
"""

import pytest
import tempfile
import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.pynucleus.db import Base
from src.pynucleus.feedback.models import UserFeedback
from src.pynucleus.feedback import save_feedback


class TestUserFeedback:
    """Test cases for UserFeedback model."""
    
    @pytest.fixture
    def db_session(self):
        """Create an in-memory SQLite database for testing."""
        # Create in-memory SQLite database
        engine = create_engine("sqlite:///:memory:", echo=False)
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        # Create session
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        yield session
        
        session.close()
    
    def test_create_user_feedback(self, db_session):
        """Test creating a UserFeedback record with all fields."""
        # Create feedback record
        feedback = UserFeedback(
            user_id="test_user_123",
            query_hash="abc123def456",
            raw_confidence=0.85,
            rating=8,
            thumb_up=True,
            implicit_score=0.78
        )
        
        # Add to session and commit
        db_session.add(feedback)
        db_session.commit()
        db_session.refresh(feedback)
        
        # Assert record was created with auto-generated fields
        assert feedback.id is not None
        assert feedback.user_id == "test_user_123"
        assert feedback.query_hash == "abc123def456"
        assert feedback.raw_confidence == 0.85
        assert feedback.rating == 8
        assert feedback.thumb_up is True
        assert feedback.implicit_score == 0.78
        assert feedback.created_at is not None
        assert isinstance(feedback.created_at, datetime)
    
    def test_create_minimal_feedback(self, db_session):
        """Test creating feedback with only required fields."""
        feedback = UserFeedback(
            query_hash="minimal_test_hash",
            raw_confidence=0.65
        )
        
        db_session.add(feedback)
        db_session.commit()
        db_session.refresh(feedback)
        
        # Assert required fields are set
        assert feedback.query_hash == "minimal_test_hash"
        assert feedback.raw_confidence == 0.65
        
        # Assert nullable fields are None
        assert feedback.user_id is None
        assert feedback.rating is None
        assert feedback.thumb_up is None
        assert feedback.implicit_score is None
        
        # Assert auto-generated fields
        assert feedback.id is not None
        assert feedback.created_at is not None
    
    def test_create_anonymous_feedback(self, db_session):
        """Test creating feedback without user_id (anonymous)."""
        feedback = UserFeedback(
            query_hash="anonymous_query_hash",
            raw_confidence=0.72,
            rating=6,
            thumb_up=False
        )
        
        db_session.add(feedback)
        db_session.commit()
        db_session.refresh(feedback)
        
        assert feedback.user_id is None
        assert feedback.query_hash == "anonymous_query_hash"
        assert feedback.rating == 6
        assert feedback.thumb_up is False
    
    def test_retrieval_integrity(self, db_session):
        """Test that committed records can be retrieved with integrity."""
        # Create multiple feedback records
        feedback_data = [
            {
                "user_id": "user_1",
                "query_hash": "hash_1",
                "raw_confidence": 0.90,
                "rating": 9,
                "thumb_up": True,
                "implicit_score": 0.88
            },
            {
                "user_id": "user_2", 
                "query_hash": "hash_2",
                "raw_confidence": 0.45,
                "rating": 3,
                "thumb_up": False,
                "implicit_score": 0.42
            },
            {
                "user_id": None,  # Anonymous
                "query_hash": "hash_3",
                "raw_confidence": 0.67,
                "rating": None,
                "thumb_up": None,
                "implicit_score": None
            }
        ]
        
        created_ids = []
        for data in feedback_data:
            feedback = UserFeedback(**data)
            db_session.add(feedback)
            db_session.commit()
            db_session.refresh(feedback)
            created_ids.append(feedback.id)
        
        # Retrieve all records and verify integrity
        all_feedback = db_session.query(UserFeedback).all()
        assert len(all_feedback) == 3
        
        # Verify specific record retrieval
        for i, expected_data in enumerate(feedback_data):
            retrieved = db_session.query(UserFeedback).filter(
                UserFeedback.id == created_ids[i]
            ).first()
            
            assert retrieved is not None
            assert retrieved.user_id == expected_data["user_id"]
            assert retrieved.query_hash == expected_data["query_hash"]
            assert retrieved.raw_confidence == expected_data["raw_confidence"]
            assert retrieved.rating == expected_data["rating"]
            assert retrieved.thumb_up == expected_data["thumb_up"]
            assert retrieved.implicit_score == expected_data["implicit_score"]
    
    def test_query_hash_index(self, db_session):
        """Test querying by query_hash index."""
        # Create feedback records with different query hashes
        feedback1 = UserFeedback(query_hash="test_hash_1", raw_confidence=0.8)
        feedback2 = UserFeedback(query_hash="test_hash_2", raw_confidence=0.7)
        feedback3 = UserFeedback(query_hash="test_hash_1", raw_confidence=0.9)  # Same hash
        
        db_session.add_all([feedback1, feedback2, feedback3])
        db_session.commit()
        
        # Query by hash
        results = db_session.query(UserFeedback).filter(
            UserFeedback.query_hash == "test_hash_1"
        ).all()
        
        assert len(results) == 2
        assert all(r.query_hash == "test_hash_1" for r in results)
    
    def test_user_id_index(self, db_session):
        """Test querying by user_id index."""
        # Create feedback records for different users
        feedback1 = UserFeedback(user_id="user_1", query_hash="hash_1", raw_confidence=0.8)
        feedback2 = UserFeedback(user_id="user_2", query_hash="hash_2", raw_confidence=0.7)
        feedback3 = UserFeedback(user_id="user_1", query_hash="hash_3", raw_confidence=0.9)
        
        db_session.add_all([feedback1, feedback2, feedback3])
        db_session.commit()
        
        # Query by user_id
        user1_feedback = db_session.query(UserFeedback).filter(
            UserFeedback.user_id == "user_1"
        ).all()
        
        assert len(user1_feedback) == 2
        assert all(f.user_id == "user_1" for f in user1_feedback)


class TestSaveFeedbackHelper:
    """Test cases for save_feedback helper function."""
    
    @pytest.fixture(autouse=True)
    def setup_test_db(self):
        """Setup test database for save_feedback function."""
        # Create temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp()
        
        # Set environment variable for test database
        os.environ["DATABASE_URL"] = f"sqlite:///{self.db_path}"
        
        # Import after setting env var to ensure correct database is used
        from src.pynucleus.db import engine, Base
        Base.metadata.create_all(engine)
        
        yield
        
        # Cleanup
        os.close(self.db_fd)
        os.unlink(self.db_path)
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
    
    def test_save_feedback_function(self):
        """Test save_feedback helper function."""
        # Use save_feedback helper
        feedback = save_feedback(
            user_id="helper_test_user",
            query_hash="helper_test_hash",
            raw_confidence=0.75,
            rating=7,
            thumb_up=True,
            implicit_score=0.72
        )
        
        # Verify returned object
        assert isinstance(feedback, UserFeedback)
        assert feedback.id is not None
        assert feedback.user_id == "helper_test_user"
        assert feedback.query_hash == "helper_test_hash"
        assert feedback.raw_confidence == 0.75
        assert feedback.rating == 7
        assert feedback.thumb_up is True
        assert feedback.implicit_score == 0.72
        assert feedback.created_at is not None
    
    def test_save_feedback_minimal(self):
        """Test save_feedback with minimal required fields."""
        feedback = save_feedback(
            query_hash="minimal_helper_hash",
            raw_confidence=0.55
        )
        
        assert feedback.query_hash == "minimal_helper_hash"
        assert feedback.raw_confidence == 0.55
        assert feedback.user_id is None
        assert feedback.rating is None
        assert feedback.thumb_up is None
        assert feedback.implicit_score is None 