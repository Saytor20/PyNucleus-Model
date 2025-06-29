"""
Unit Tests for Semantic Validation Module
=========================================

Tests for the semantic similarity evaluation functions including BLEU, ROUGE, 
BERTScore, and combined semantic scoring.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path for testing
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pynucleus.eval.semantic_validation import (
    calculate_bleu_score,
    calculate_rouge_scores,
    calculate_bert_score,
    calculate_semantic_similarity,
    evaluate_answer_semantically,
    get_semantic_validation_info,
    _preprocess_text,
    _tokenize_text,
    _check_dependencies,
    SemanticScores
)


class TestSemanticValidation(unittest.TestCase):
    """Test cases for semantic validation functions."""
    
    def setUp(self):
        """Set up test cases."""
        self.sample_answer = "Distillation is a separation process based on differences in boiling points of components in a mixture."
        self.sample_keywords = ["distillation", "separation", "boiling points", "mixture"]
        
        self.sample_answer_2 = "Heat exchangers transfer thermal energy between fluids at different temperatures."
        self.sample_keywords_2 = ["heat exchanger", "thermal energy", "temperature", "fluids"]
        
        # Edge cases
        self.empty_answer = ""
        self.empty_keywords = []
        self.short_answer = "Yes"
        self.unrelated_answer = "The weather is nice today and I like ice cream."
    
    def test_preprocess_text(self):
        """Test text preprocessing function."""
        # Test normal text
        text = "  This is   a TEST   with extra    spaces  "
        expected = "this is a test with extra spaces"
        self.assertEqual(_preprocess_text(text), expected)
        
        # Test empty text
        self.assertEqual(_preprocess_text(""), "")
        self.assertEqual(_preprocess_text(None), "")
        
        # Test text with special characters
        text_special = "Hello, World! How are you? Fine."
        expected_special = "hello, world! how are you? fine."
        self.assertEqual(_preprocess_text(text_special), expected_special)
    
    def test_tokenize_text(self):
        """Test text tokenization function."""
        text = "Hello world how are you"
        tokens = _tokenize_text(text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
    
    @patch('pynucleus.eval.semantic_validation.DEPENDENCIES_AVAILABLE', True)
    def test_calculate_bleu_score_success(self):
        """Test BLEU score calculation with dependencies available."""
        with patch('pynucleus.eval.semantic_validation.sentence_bleu') as mock_bleu:
            mock_bleu.return_value = 0.75
            
            score = calculate_bleu_score(self.sample_answer, self.sample_keywords)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    @patch('pynucleus.eval.semantic_validation.DEPENDENCIES_AVAILABLE', False)
    def test_calculate_bleu_score_no_dependencies(self):
        """Test BLEU score calculation without dependencies."""
        score = calculate_bleu_score(self.sample_answer, self.sample_keywords)
        self.assertEqual(score, 0.0)
    
    def test_calculate_bleu_score_edge_cases(self):
        """Test BLEU score calculation with edge cases."""
        # Empty answer
        score = calculate_bleu_score(self.empty_answer, self.sample_keywords)
        self.assertEqual(score, 0.0)
        
        # Empty keywords
        score = calculate_bleu_score(self.sample_answer, self.empty_keywords)
        self.assertEqual(score, 0.0)
        
        # Both empty
        score = calculate_bleu_score(self.empty_answer, self.empty_keywords)
        self.assertEqual(score, 0.0)
    
    @patch('pynucleus.eval.semantic_validation.DEPENDENCIES_AVAILABLE', True)
    def test_calculate_rouge_scores_success(self):
        """Test ROUGE scores calculation with dependencies available."""
        with patch('pynucleus.eval.semantic_validation.rouge_scorer') as mock_rouge:
            mock_scorer = MagicMock()
            mock_scorer.score.return_value = {
                'rouge1': MagicMock(fmeasure=0.6),
                'rouge2': MagicMock(fmeasure=0.4),
                'rougeL': MagicMock(fmeasure=0.5)
            }
            mock_rouge.RougeScorer.return_value = mock_scorer
            
            scores = calculate_rouge_scores(self.sample_answer, self.sample_keywords)
            
            self.assertIsInstance(scores, dict)
            self.assertIn("rouge_1_f", scores)
            self.assertIn("rouge_2_f", scores)
            self.assertIn("rouge_l_f", scores)
            self.assertEqual(scores["rouge_1_f"], 0.6)
            self.assertEqual(scores["rouge_2_f"], 0.4)
            self.assertEqual(scores["rouge_l_f"], 0.5)
    
    @patch('pynucleus.eval.semantic_validation.DEPENDENCIES_AVAILABLE', False)
    def test_calculate_rouge_scores_no_dependencies(self):
        """Test ROUGE scores calculation without dependencies."""
        scores = calculate_rouge_scores(self.sample_answer, self.sample_keywords)
        expected = {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}
        self.assertEqual(scores, expected)
    
    def test_calculate_rouge_scores_edge_cases(self):
        """Test ROUGE scores calculation with edge cases."""
        expected_zero = {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}
        
        # Empty answer
        scores = calculate_rouge_scores(self.empty_answer, self.sample_keywords)
        self.assertEqual(scores, expected_zero)
        
        # Empty keywords
        scores = calculate_rouge_scores(self.sample_answer, self.empty_keywords)
        self.assertEqual(scores, expected_zero)
    
    @patch('pynucleus.eval.semantic_validation.DEPENDENCIES_AVAILABLE', True)
    def test_calculate_bert_score_success(self):
        """Test BERTScore calculation with dependencies available."""
        with patch('pynucleus.eval.semantic_validation.bert_score') as mock_bert:
            # Mock BERTScore return values (P, R, F1 tensors)
            mock_bert.return_value = (
                MagicMock(),  # Precision
                MagicMock(),  # Recall  
                [0.8]  # F1 scores
            )
            
            score = calculate_bert_score(self.sample_answer, self.sample_keywords)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    @patch('pynucleus.eval.semantic_validation.DEPENDENCIES_AVAILABLE', False)
    def test_calculate_bert_score_no_dependencies(self):
        """Test BERTScore calculation without dependencies."""
        score = calculate_bert_score(self.sample_answer, self.sample_keywords)
        self.assertEqual(score, 0.0)
    
    def test_calculate_bert_score_edge_cases(self):
        """Test BERTScore calculation with edge cases."""
        # Empty answer
        score = calculate_bert_score(self.empty_answer, self.sample_keywords)
        self.assertEqual(score, 0.0)
        
        # Empty keywords
        score = calculate_bert_score(self.sample_answer, self.empty_keywords)
        self.assertEqual(score, 0.0)
    
    @patch('pynucleus.eval.semantic_validation.DEPENDENCIES_AVAILABLE', True)
    @patch('pynucleus.eval.semantic_validation.calculate_bleu_score')
    @patch('pynucleus.eval.semantic_validation.calculate_rouge_scores')
    @patch('pynucleus.eval.semantic_validation.calculate_bert_score')
    def test_calculate_semantic_similarity_success(self, mock_bert, mock_rouge, mock_bleu):
        """Test combined semantic similarity calculation."""
        # Mock individual score functions
        mock_bleu.return_value = 0.6
        mock_rouge.return_value = {
            "rouge_1_f": 0.7,
            "rouge_2_f": 0.5,
            "rouge_l_f": 0.6
        }
        mock_bert.return_value = 0.8
        
        scores = calculate_semantic_similarity(self.sample_answer, self.sample_keywords)
        
        self.assertIsInstance(scores, SemanticScores)
        self.assertEqual(scores.bleu_score, 0.6)
        self.assertEqual(scores.rouge_1_f, 0.7)
        self.assertEqual(scores.rouge_2_f, 0.5)
        self.assertEqual(scores.rouge_l_f, 0.6)
        self.assertEqual(scores.bert_score_f1, 0.8)
        
        # Check combined score calculation
        # Weights: BLEU (20%), ROUGE-1 (25%), ROUGE-2 (15%), ROUGE-L (20%), BERTScore (20%)
        expected_combined = 0.6*0.2 + 0.7*0.25 + 0.5*0.15 + 0.6*0.2 + 0.8*0.2
        self.assertAlmostEqual(scores.combined_score, expected_combined, places=3)
        
        # Check success determination (threshold = 0.3)
        self.assertEqual(scores.success, expected_combined >= 0.3)
    
    @patch('pynucleus.eval.semantic_validation.DEPENDENCIES_AVAILABLE', False)
    def test_calculate_semantic_similarity_no_dependencies(self):
        """Test semantic similarity calculation without dependencies."""
        scores = calculate_semantic_similarity(self.sample_answer, self.sample_keywords)
        
        self.assertIsInstance(scores, SemanticScores)
        self.assertEqual(scores.bleu_score, 0.0)
        self.assertEqual(scores.rouge_1_f, 0.0)
        self.assertEqual(scores.rouge_2_f, 0.0)
        self.assertEqual(scores.rouge_l_f, 0.0)
        self.assertEqual(scores.bert_score_f1, 0.0)
        self.assertEqual(scores.combined_score, 0.0)
        self.assertFalse(scores.success)
    
    @patch('pynucleus.eval.semantic_validation.calculate_semantic_similarity')
    def test_evaluate_answer_semantically_success(self, mock_semantic):
        """Test complete semantic evaluation function."""
        # Mock semantic similarity calculation
        mock_scores = SemanticScores(
            bleu_score=0.6,
            rouge_1_f=0.7,
            rouge_2_f=0.5,
            rouge_l_f=0.6,
            bert_score_f1=0.8,
            combined_score=0.65,
            success=True
        )
        mock_semantic.return_value = mock_scores
        
        result = evaluate_answer_semantically(
            self.sample_answer, 
            self.sample_keywords, 
            threshold=0.5
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("semantic_scores", result)
        self.assertIn("success", result)
        self.assertIn("threshold", result)
        self.assertIn("methodology", result)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["threshold"], 0.5)
        self.assertEqual(result["semantic_scores"]["combined_score"], 0.65)
    
    @patch('pynucleus.eval.semantic_validation.DEPENDENCIES_AVAILABLE', False)
    def test_evaluate_answer_semantically_no_dependencies(self):
        """Test semantic evaluation without dependencies."""
        result = evaluate_answer_semantically(self.sample_answer, self.sample_keywords)
        
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertFalse(result["success"])
        self.assertIn("dependencies not available", result["error"])
    
    def test_get_semantic_validation_info(self):
        """Test semantic validation info function."""
        info = get_semantic_validation_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("dependencies_available", info)
        self.assertIn("supported_metrics", info)
        self.assertIn("default_threshold", info)
        self.assertIn("scoring_weights", info)
        
        # Check supported metrics
        metrics = info["supported_metrics"]
        self.assertIn("BLEU Score", str(metrics))
        self.assertIn("ROUGE-1", str(metrics))
        self.assertIn("BERTScore", str(metrics))
        
        # Check default threshold
        self.assertEqual(info["default_threshold"], 0.3)
        
        # Check scoring weights
        weights = info["scoring_weights"]
        self.assertEqual(weights["bleu"], 0.20)
        self.assertEqual(weights["rouge_1"], 0.25)
        self.assertEqual(weights["bert_score"], 0.20)
    
    def test_semantic_scores_dataclass(self):
        """Test SemanticScores dataclass."""
        scores = SemanticScores(
            bleu_score=0.5,
            rouge_1_f=0.6,
            rouge_2_f=0.4,
            rouge_l_f=0.5,
            bert_score_f1=0.7,
            combined_score=0.55,
            success=True
        )
        
        self.assertEqual(scores.bleu_score, 0.5)
        self.assertEqual(scores.rouge_1_f, 0.6)
        self.assertEqual(scores.rouge_2_f, 0.4)
        self.assertEqual(scores.rouge_l_f, 0.5)
        self.assertEqual(scores.bert_score_f1, 0.7)
        self.assertEqual(scores.combined_score, 0.55)
        self.assertTrue(scores.success)
    
    def test_threshold_variations(self):
        """Test semantic evaluation with different thresholds."""
        with patch('pynucleus.eval.semantic_validation.calculate_semantic_similarity') as mock_semantic:
            mock_scores = SemanticScores(
                bleu_score=0.4,
                rouge_1_f=0.5,
                rouge_2_f=0.3,
                rouge_l_f=0.4,
                bert_score_f1=0.6,
                combined_score=0.45,
                success=True  # This is based on default threshold 0.3
            )
            mock_semantic.return_value = mock_scores
            
            # Test with low threshold
            result_low = evaluate_answer_semantically(
                self.sample_answer, self.sample_keywords, threshold=0.2
            )
            self.assertTrue(result_low["success"])  # 0.45 >= 0.2
            
            # Test with high threshold
            result_high = evaluate_answer_semantically(
                self.sample_answer, self.sample_keywords, threshold=0.8
            )
            self.assertFalse(result_high["success"])  # 0.45 < 0.8
    
    def test_error_handling(self):
        """Test error handling in semantic validation functions."""
        with patch('pynucleus.eval.semantic_validation.calculate_semantic_similarity') as mock_semantic:
            mock_semantic.side_effect = Exception("Test error")
            
            result = evaluate_answer_semantically(self.sample_answer, self.sample_keywords)
            
            self.assertIsInstance(result, dict)
            self.assertIn("error", result)
            self.assertFalse(result["success"])
            self.assertIn("Test error", result["error"])


class TestSemanticValidationIntegration(unittest.TestCase):
    """Integration tests for semantic validation with realistic scenarios."""
    
    def setUp(self):
        """Set up integration test cases."""
        # Chemical engineering test cases
        self.distillation_cases = [
            {
                "answer": "Distillation is a separation process that uses differences in boiling points to separate components of a liquid mixture.",
                "keywords": ["distillation", "separation", "boiling point", "liquid mixture"],
                "expected_high_score": True
            },
            {
                "answer": "It's a method to purify substances by heating and cooling.",
                "keywords": ["distillation", "separation", "boiling point", "liquid mixture"],
                "expected_high_score": False  # Less specific
            },
            {
                "answer": "Cats are cute animals that like to sleep.",
                "keywords": ["distillation", "separation", "boiling point", "liquid mixture"],
                "expected_high_score": False  # Completely unrelated
            }
        ]
    
    def test_realistic_evaluation_scenarios(self):
        """Test with realistic chemical engineering Q&A scenarios."""
        # Skip if dependencies not available
        try:
            from pynucleus.eval.semantic_validation import DEPENDENCIES_AVAILABLE
            if not DEPENDENCIES_AVAILABLE:
                self.skipTest("Semantic validation dependencies not available")
        except ImportError:
            self.skipTest("Cannot import semantic validation module")
        
        for i, case in enumerate(self.distillation_cases):
            with self.subTest(case_index=i):
                result = evaluate_answer_semantically(
                    case["answer"], 
                    case["keywords"],
                    threshold=0.3
                )
                
                # Check that result structure is correct
                self.assertIn("semantic_scores", result)
                self.assertIn("success", result)
                
                if case["expected_high_score"]:
                    # For good answers, expect reasonable scores
                    scores = result["semantic_scores"]
                    if scores:  # Only check if semantic scoring worked
                        self.assertGreater(scores["combined_score"], 0.1)
                else:
                    # For poor answers, expect lower scores
                    if result["semantic_scores"]:
                        scores = result["semantic_scores"]
                        # Very unrelated answers should have very low scores
                        if "cats" in case["answer"]:
                            self.assertLess(scores["combined_score"], 0.2)


if __name__ == "__main__":
    unittest.main() 