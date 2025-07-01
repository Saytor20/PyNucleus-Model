"""
Unit tests for confidence calibration integration in RAG engine.
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest

# Import the RAG engine functions
from src.pynucleus.rag.engine import ask, _load_confidence_calibrator


class TestConfidenceCalibration(unittest.TestCase):
    """Test confidence calibration integration in RAG engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset the global calibrator state
        import src.pynucleus.rag.engine as engine_module
        engine_module._confidence_calibrator = None
    
    @patch('src.pynucleus.rag.engine.retrieve_enhanced')
    @patch('src.pynucleus.rag.engine.generate')
    @patch('src.pynucleus.rag.engine.process_answer_quality')
    @patch('src.pynucleus.eval.load_latest_model')
    def test_confidence_calibration_applied(self, mock_load_model, mock_quality, mock_generate, mock_retrieve):
        """Test that confidence calibration is properly applied to responses."""
        # Mock calibrator function that reduces confidence by 20%
        mock_calibrator = lambda c: c * 0.8
        mock_load_model.return_value = mock_calibrator
        
        # Mock retrieval
        mock_retrieve.return_value = (
            ["Sample document content"],
            ["doc_1"],
            [{"source": "test.pdf"}]
        )
        
        # Mock generation
        mock_generate.return_value = "This is a test answer."
        
        # Mock quality assessment with 0.9 confidence
        mock_quality.return_value = {
            "processed_answer": "This is a test answer with citations [doc_1].",
            "quality_score": 0.9,
            "has_citations": True,
            "citations_found": ["doc_1"],
            "sentence_count": 1,
            "deduplication_applied": False
        }
        
        # Call the ask function
        result = ask("What is chemical engineering?")
        
        # Verify confidence scores
        self.assertIn("confidence_raw", result)
        self.assertIn("confidence_cal", result)
        self.assertEqual(result["confidence_raw"], 0.9)
        self.assertAlmostEqual(result["confidence_cal"], 0.72, places=2)  # 0.9 * 0.8
        self.assertAlmostEqual(result["confidence"], 0.72, places=2)  # Should use calibrated
        
        # Verify calibrator was loaded
        mock_load_model.assert_called_once()
    
    @patch('src.pynucleus.rag.engine.retrieve_enhanced')
    @patch('src.pynucleus.rag.engine.generate')  
    @patch('src.pynucleus.rag.engine.process_answer_quality')
    @patch('src.pynucleus.eval.load_latest_model')
    @patch('src.pynucleus.metrics.prometheus.record_confidence_calibration')
    def test_prometheus_metrics_recorded(self, mock_record_metrics, mock_load_model, 
                                       mock_quality, mock_generate, mock_retrieve):
        """Test that Prometheus metrics are properly recorded."""
        # Mock calibrator
        mock_calibrator = lambda c: c * 0.8
        mock_load_model.return_value = mock_calibrator
        
        # Mock other dependencies
        mock_retrieve.return_value = (["doc"], ["source"], [{}])
        mock_generate.return_value = "Answer"
        mock_quality.return_value = {
            "processed_answer": "Answer [source]",
            "quality_score": 0.7,
            "has_citations": True,
            "citations_found": ["source"],
            "sentence_count": 1,
            "deduplication_applied": False
        }
        
        # Call ask function
        result = ask("Test question")
        
        # Verify metrics were recorded (with floating point tolerance)
        expected_cal = 0.7 * 0.8  # 0.56
        mock_record_metrics.assert_called_once()
        call_args = mock_record_metrics.call_args[0]
        self.assertEqual(call_args[0], 0.7)  # raw confidence
        self.assertAlmostEqual(call_args[1], expected_cal, places=2)  # calibrated confidence
        self.assertEqual(call_args[2], 'success')  # status
        
        # Verify response structure
        self.assertEqual(result["confidence_raw"], 0.7)
        self.assertAlmostEqual(result["confidence_cal"], expected_cal, places=2)
    
    @patch('src.pynucleus.rag.engine.retrieve_enhanced')
    @patch('src.pynucleus.eval.load_latest_model')
    @patch('src.pynucleus.metrics.prometheus.record_confidence_calibration')
    def test_no_documents_case(self, mock_record_metrics, mock_load_model, mock_retrieve):
        """Test confidence calibration when no documents are retrieved."""
        # Mock calibrator
        mock_calibrator = lambda c: c * 0.8
        mock_load_model.return_value = mock_calibrator
        
        # Mock empty retrieval
        mock_retrieve.return_value = ([], [], [])
        
        # Call ask function
        result = ask("Test question")
        
        # Verify zero confidence is calibrated
        self.assertEqual(result["confidence_raw"], 0.0)
        self.assertEqual(result["confidence_cal"], 0.0)  # 0.0 * 0.8 = 0.0
        self.assertEqual(result["confidence"], 0.0)
        
        # Verify metrics were recorded
        mock_record_metrics.assert_called_once_with(0.0, 0.0, 'success')
    
    @patch('src.pynucleus.rag.engine.retrieve_enhanced')
    @patch('src.pynucleus.eval.load_latest_model')
    @patch('src.pynucleus.metrics.prometheus.record_confidence_calibration')
    def test_no_calibration_model_fallback(self, mock_record_metrics, mock_load_model, mock_retrieve):
        """Test fallback behavior when no calibration model is available."""
        # Mock no calibration model
        mock_load_model.return_value = None
        
        # Mock empty retrieval for simplicity
        mock_retrieve.return_value = ([], [], [])
        
        # Call ask function
        result = ask("Test question")
        
        # Verify identity function is used (no change)
        self.assertEqual(result["confidence_raw"], 0.0)
        self.assertEqual(result["confidence_cal"], 0.0)
        self.assertEqual(result["confidence"], 0.0)
        
        # Verify metrics recorded identity transformation
        mock_record_metrics.assert_called_once_with(0.0, 0.0, 'success')
    
    @patch('src.pynucleus.rag.engine.retrieve_enhanced')
    @patch('src.pynucleus.rag.engine.generate')
    @patch('src.pynucleus.rag.engine.process_answer_quality')
    @patch('src.pynucleus.eval.load_latest_model')
    @patch('src.pynucleus.metrics.prometheus.record_confidence_calibration')
    def test_calibration_error_handling(self, mock_record_metrics, mock_load_model,
                                      mock_quality, mock_generate, mock_retrieve):
        """Test error handling when calibration fails."""
        # Mock calibrator that raises exception
        def failing_calibrator(c):
            raise ValueError("Calibration failed")
        mock_load_model.return_value = failing_calibrator
        
        # Mock other dependencies  
        mock_retrieve.return_value = (["doc"], ["source"], [{}])
        mock_generate.return_value = "Answer"
        mock_quality.return_value = {
            "processed_answer": "Answer [source]",
            "quality_score": 0.8,
            "has_citations": True,
            "citations_found": ["source"],
            "sentence_count": 1,
            "deduplication_applied": False
        }
        
        # Call ask function
        result = ask("Test question")
        
        # Verify fallback to raw confidence
        self.assertEqual(result["confidence_raw"], 0.8)
        self.assertEqual(result["confidence_cal"], 0.8)  # Should fallback to raw
        self.assertEqual(result["confidence"], 0.8)
        
        # Verify failure metrics were recorded
        mock_record_metrics.assert_called_once_with(0.8, 0.8, 'failure')
    
    @patch('src.pynucleus.rag.engine.retrieve_enhanced')
    @patch('src.pynucleus.rag.engine.generate')
    @patch('src.pynucleus.rag.engine.process_answer_quality')
    @patch('src.pynucleus.eval.load_latest_model')
    def test_confidence_bounds_enforcement(self, mock_load_model, mock_quality, mock_generate, mock_retrieve):
        """Test that calibrated confidence is properly bounded to [0, 1]."""
        # Mock calibrator that could produce out-of-bounds values
        mock_calibrator = lambda c: c * 2.0  # Could exceed 1.0
        mock_load_model.return_value = mock_calibrator
        
        # Mock dependencies
        mock_retrieve.return_value = (["doc"], ["source"], [{}])
        mock_generate.return_value = "Answer"
        mock_quality.return_value = {
            "processed_answer": "Answer [source]",
            "quality_score": 0.8,
            "has_citations": True,
            "citations_found": ["source"],
            "sentence_count": 1,
            "deduplication_applied": False
        }
        
        # Call ask function
        result = ask("Test question")
        
        # Verify bounds are enforced
        self.assertEqual(result["confidence_raw"], 0.8)
        self.assertEqual(result["confidence_cal"], 1.0)  # Should be clamped to 1.0
        self.assertEqual(result["confidence"], 1.0)
        
        # Test lower bound with negative calibrator
        # Reset global state first
        import src.pynucleus.rag.engine as engine_module
        engine_module._confidence_calibrator = None
        
        mock_calibrator = lambda c: c - 1.0  # Could go below 0.0
        mock_load_model.return_value = mock_calibrator
        
        result = ask("Test question")
        self.assertEqual(result["confidence_cal"], 0.0)  # Should be clamped to 0.0
    
    @patch('src.pynucleus.rag.engine.retrieve_enhanced')
    @patch('src.pynucleus.eval.load_latest_model')
    def test_load_calibrator_caching(self, mock_load_model, mock_retrieve):
        """Test that calibrator is loaded once and cached."""
        # Mock calibrator
        mock_calibrator = lambda c: c * 0.9
        mock_load_model.return_value = mock_calibrator
        
        # Mock empty retrieval for simplicity
        mock_retrieve.return_value = ([], [], [])
        
        # Call ask function multiple times
        ask("Question 1")
        ask("Question 2")
        ask("Question 3")
        
        # Verify load_latest_model was called only once (cached)
        mock_load_model.assert_called_once()
    
    def test_load_confidence_calibrator_function(self):
        """Test the _load_confidence_calibrator function directly."""
        with patch('src.pynucleus.eval.load_latest_model') as mock_load:
            # Test successful loading
            mock_calibrator = lambda c: c * 0.9
            mock_load.return_value = mock_calibrator
            
            calibrator = _load_confidence_calibrator()
            self.assertEqual(calibrator(0.5), 0.45)
            
            # Test no model case
            mock_load.return_value = None
            
            # Reset global state
            import src.pynucleus.rag.engine as engine_module
            engine_module._confidence_calibrator = None
            
            calibrator = _load_confidence_calibrator()
            self.assertEqual(calibrator(0.5), 0.5)  # Identity function
            
            # Test error case
            mock_load.side_effect = Exception("Load failed")
            
            # Reset global state
            engine_module._confidence_calibrator = None
            
            calibrator = _load_confidence_calibrator()
            self.assertEqual(calibrator(0.5), 0.5)  # Identity function fallback


if __name__ == '__main__':
    unittest.main() 