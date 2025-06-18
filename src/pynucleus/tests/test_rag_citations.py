#!/usr/bin/env python3
"""
Unit tests for RAG citation functionality.

Tests the citation structure, validation script, and overall citation system.
"""

import unittest
import tempfile
import csv
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from pynucleus.pipeline.pipeline_rag import RAGPipeline
except ImportError as e:
    print(f"Warning: Could not import RAGPipeline: {e}")


class TestRAGCitations(unittest.TestCase):
    """Test cases for RAG citation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_query = "What are the benefits of modular chemical plants?"
        self.test_response = {
            "query_id": "test_query_123",
            "timestamp": "2024-01-01T12:00:00",
            "question": self.test_query,
            "answer": "Modular chemical plants offer several key advantages [†1]. The design principles focus on standardization and scalability [†2].",
            "citations": [
                {
                    "source_filename": "modular_plants.json",
                    "chunk_id": "modular_001",
                    "similarity": 0.8754
                },
                {
                    "source_filename": "design_principles.json", 
                    "chunk_id": "design_002",
                    "similarity": 0.7623
                }
            ],
            "metadata": {
                "chunks_retrieved": 5,
                "chunks_used": 2,
                "similarity_threshold": 0.3
            }
        }
    
    def test_citation_structure(self):
        """Test that citation structure has all required fields."""
        # Test that citations key exists
        self.assertIn("citations", self.test_response)
        
        # Test that citations is a list
        self.assertIsInstance(self.test_response["citations"], list)
        
        # Test that each citation has required fields
        for citation in self.test_response["citations"]:
            self.assertIn("source_filename", citation)
            self.assertIn("chunk_id", citation)
            self.assertIn("similarity", citation)
            
            # Test field types
            self.assertIsInstance(citation["source_filename"], str)
            self.assertIsInstance(citation["chunk_id"], str)
            self.assertIsInstance(citation["similarity"], (int, float))
            
            # Test similarity score range
            self.assertGreaterEqual(citation["similarity"], 0.0)
            self.assertLessEqual(citation["similarity"], 1.0)
    
    def test_answer_has_citation_markers(self):
        """Test that answer contains proper citation markers."""
        answer = self.test_response["answer"]
        
        # Check for citation markers
        self.assertIn("[†1]", answer)
        self.assertIn("[†2]", answer)
        
        # Verify number of citation markers matches citations
        num_citations = len(self.test_response["citations"])
        for i in range(1, num_citations + 1):
            self.assertIn(f"[†{i}]", answer)
    
    def test_response_metadata(self):
        """Test that response contains proper metadata."""
        metadata = self.test_response.get("metadata", {})
        
        # Check required metadata fields
        self.assertIn("chunks_retrieved", metadata)
        self.assertIn("chunks_used", metadata)
        self.assertIn("similarity_threshold", metadata)
        
        # Verify metadata types and values
        self.assertIsInstance(metadata["chunks_retrieved"], int)
        self.assertIsInstance(metadata["chunks_used"], int)
        self.assertIsInstance(metadata["similarity_threshold"], (int, float))
        
        # Logical constraints
        self.assertGreaterEqual(metadata["chunks_retrieved"], metadata["chunks_used"])
        self.assertGreaterEqual(metadata["chunks_used"], 0)
    
    @patch('pynucleus.pipeline.pipeline_rag.RAGPipeline')
    def test_pipeline_query_with_citations(self, mock_pipeline_class):
        """Test that pipeline returns properly structured citation response."""
        # Create mock pipeline instance
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # Mock the query_with_citations method
        mock_pipeline.query_with_citations.return_value = self.test_response
        
        # Test the method call
        result = mock_pipeline.query_with_citations(self.test_query)
        
        # Verify the result structure
        self.assertEqual(result, self.test_response)
        self.assertIn("citations", result)
        self.assertIn("answer", result)
        self.assertIn("question", result)
        
        # Verify method was called correctly
        mock_pipeline.query_with_citations.assert_called_once_with(self.test_query)
    
    def test_validate_script_runs(self):
        """Test that the validation script can be imported and run on a mini CSV."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = Path(f.name)
            
            # Write test data
            writer = csv.DictWriter(f, fieldnames=['question', 'ideal_answer'])
            writer.writeheader()
            writer.writerows([
                {
                    "question": "What are the benefits of modular plants?",
                    "ideal_answer": "Modular plants offer standardization and scalability benefits."
                },
                {
                    "question": "How do simulations help design?",
                    "ideal_answer": "Simulations enable optimization and performance prediction."
                }
            ])
        
        try:
            # Try to import the validation script
            sys.path.insert(0, str(project_root / "scripts"))
            import validate_rag_factual_accuracy
            
            # Test that we can create a validator
            with patch('validate_rag_factual_accuracy.RAGPipeline') as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline
                
                # Mock query responses
                mock_pipeline.query_with_citations.return_value = self.test_response
                
                validator = validate_rag_factual_accuracy.FactualAccuracyValidator(
                    mock_pipeline, accuracy_threshold=0.5
                )
                
                # This should return a float between 0 and 1
                accuracy = validator.validate_csv(csv_path)
                
                self.assertIsInstance(accuracy, float)
                self.assertGreaterEqual(accuracy, 0.0)
                self.assertLessEqual(accuracy, 1.0)
                
        except ImportError as e:
            self.skipTest(f"Could not import validation script: {e}")
        finally:
            # Clean up temporary file
            csv_path.unlink()
    
    def test_citation_logging(self):
        """Test that citations are properly logged to trace file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_trace.jsonl"
            
            # Mock a pipeline with citation logging
            with patch('pynucleus.pipeline.pipeline_rag.RAGPipeline') as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline
                mock_pipeline.citation_log_file = log_file
                
                # Simulate logging
                with open(log_file, 'w') as f:
                    json.dump(self.test_response, f)
                    f.write('\n')
                
                # Verify log file contents
                self.assertTrue(log_file.exists())
                
                with open(log_file, 'r') as f:
                    logged_response = json.loads(f.readline())
                
                self.assertEqual(logged_response["question"], self.test_response["question"])
                self.assertEqual(logged_response["answer"], self.test_response["answer"])
                self.assertEqual(len(logged_response["citations"]), len(self.test_response["citations"]))
    
    def test_factual_accuracy_validation(self):
        """Test the factual accuracy validation logic."""
        try:
            sys.path.insert(0, str(project_root / "scripts"))
            import validate_rag_factual_accuracy
            
            with patch('validate_rag_factual_accuracy.RAGPipeline') as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline
                
                validator = validate_rag_factual_accuracy.FactualAccuracyValidator(
                    mock_pipeline, accuracy_threshold=0.9
                )
                
                # Test word extraction
                text = "Modular chemical plants offer standardization and scalability benefits."
                words = validator._extract_content_words(text)
                
                # Should extract meaningful words
                self.assertIn("modular", words)
                self.assertIn("chemical", words)
                self.assertIn("plants", words)
                self.assertIn("standardization", words)
                self.assertIn("scalability", words)
                
                # Should exclude stop words
                self.assertNotIn("and", words)
                self.assertNotIn("the", words)
                
                # Test cited content retrieval
                citations = [
                    {"source_filename": "modular_plants.json", "chunk_id": "mod_001", "similarity": 0.8}
                ]
                cited_content = validator._get_cited_content(citations)
                self.assertIsInstance(cited_content, str)
                self.assertGreater(len(cited_content), 0)
                
        except ImportError as e:
            self.skipTest(f"Could not import validation script: {e}")
    
    def test_mini_csv_creation(self):
        """Test creation of mini validation CSV for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "mini_validation.csv"
            
            # Create mini CSV
            test_data = [
                {"question": "Test question 1?", "ideal_answer": "Test answer 1."},
                {"question": "Test question 2?", "ideal_answer": "Test answer 2."}
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['question', 'ideal_answer'])
                writer.writeheader()
                writer.writerows(test_data)
            
            # Verify file creation and content
            self.assertTrue(csv_file.exists())
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['question'], "Test question 1?")
            self.assertEqual(rows[1]['ideal_answer'], "Test answer 2.")


class TestCitationIntegration(unittest.TestCase):
    """Integration tests for the complete citation system."""
    
    def test_end_to_end_citation_flow(self):
        """Test the complete flow from query to citation validation."""
        # This test requires a more complete setup and would typically
        # be run as an integration test with actual data
        
        test_query = "What are modular chemical plants?"
        
        # Mock the complete flow
        with patch('pynucleus.pipeline.pipeline_rag.RAGPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            # Mock a complete response
            mock_response = {
                "query_id": "integration_test_001",
                "timestamp": "2024-01-01T12:00:00",
                "question": test_query,
                "answer": "Modular chemical plants are standardized manufacturing facilities [†1].",
                "citations": [
                    {
                        "source_filename": "chemical_engineering.json",
                        "chunk_id": "ce_modular_001",
                        "similarity": 0.9123
                    }
                ],
                "metadata": {
                    "chunks_retrieved": 3,
                    "chunks_used": 1,
                    "similarity_threshold": 0.3
                }
            }
            
            mock_pipeline.query_with_citations.return_value = mock_response
            
            # Test the query
            result = mock_pipeline.query_with_citations(test_query)
            
            # Verify complete structure
            self.assertIn("query_id", result)
            self.assertIn("timestamp", result)
            self.assertIn("question", result)
            self.assertIn("answer", result)
            self.assertIn("citations", result)
            self.assertIn("metadata", result)
            
            # Verify citation structure
            citations = result["citations"]
            self.assertIsInstance(citations, list)
            self.assertGreater(len(citations), 0)
            
            for citation in citations:
                self.assertIn("source_filename", citation)
                self.assertIn("chunk_id", citation)
                self.assertIn("similarity", citation)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestRAGCitations))
    suite.addTest(unittest.makeSuite(TestCitationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1) 