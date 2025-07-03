"""
Unit tests for PyNucleus system statistics module.

Tests RAG retrieval metrics computation, aggregation, and Prometheus integration
with synthetic data and exact precision/recall assertions.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path for imports
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pynucleus.metrics.system_statistics import (  # noqa: E402
    RAGRetrievalMetrics,
    compute_retrieval_metrics,
    aggregate_metrics,
    log_metrics_prometheus
)


class TestRAGRetrievalMetrics:
    """Test cases for RAGRetrievalMetrics dataclass."""

    def test_dataclass_creation(self):
        """Test basic dataclass instantiation."""
        metrics = RAGRetrievalMetrics(
            k=5,
            precision=0.6,
            recall=0.5,
            f1=0.545,
            num_relevant=10,
            num_retrieved=5
        )

        assert metrics.k == 5
        assert metrics.precision == 0.6
        assert metrics.recall == 0.5
        assert metrics.f1 == 0.545
        assert metrics.num_relevant == 10
        assert metrics.num_retrieved == 5


class TestComputeRetrievalMetrics:
    """Test cases for compute_retrieval_metrics function."""

    def test_perfect_precision_recall(self):
        """Test case where all retrieved items are relevant."""
        retrieved_ids = ["doc1", "doc2", "doc3"]
        relevant_ids = {"doc1", "doc2", "doc3", "doc4", "doc5"}

        metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k=3)

        assert metrics.k == 3
        assert metrics.precision == 1.0  # 3/3 retrieved are relevant
        assert metrics.recall == 0.6     # 3/5 relevant are retrieved
        assert abs(metrics.f1 - 0.75) < 1e-10  # 2 * (1.0 * 0.6) / (1.0 + 0.6)
        assert metrics.num_relevant == 5
        assert metrics.num_retrieved == 3

    def test_exact_precision_recall_at_k5(self):
        """Test exact precision/recall values at k=5 with synthetic data."""
        retrieved_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant_ids = {"doc1", "doc3", "doc5", "doc7", "doc9"}

        metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k=5)

        # Exact assertions as requested
        assert metrics.k == 5
        assert metrics.precision == 0.6   # 3 relevant out of 5 retrieved
        assert metrics.recall == 0.6      # 3 found out of 5 relevant
        assert metrics.f1 == 0.6          # 2 * (0.6 * 0.6) / (0.6 + 0.6) = 0.6
        assert metrics.num_relevant == 5
        assert metrics.num_retrieved == 5

    def test_k_smaller_than_retrieved(self):
        """Test when k is smaller than available retrieved documents."""
        retrieved_ids = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
        relevant_ids = {"doc1", "doc2"}  # Only first 2 are relevant

        metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k=3)

        assert metrics.k == 3
        assert abs(metrics.precision - 2/3) < 1e-10  # 2/3 at k=3 are relevant
        assert metrics.recall == 1.0      # All 2 relevant docs found
        assert abs(metrics.f1 - 0.8) < 1e-10  # F1 for precision=2/3, recall=1.0
        assert metrics.num_relevant == 2
        assert metrics.num_retrieved == 3

    def test_no_relevant_found(self):
        """Test case where no retrieved items are relevant."""
        retrieved_ids = ["doc1", "doc2", "doc3"]
        relevant_ids = {"doc4", "doc5", "doc6"}

        metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k=3)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0

    def test_empty_relevant_set(self):
        """Test handling of empty relevant set."""
        retrieved_ids = ["doc1", "doc2", "doc3"]
        relevant_ids = set()

        metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k=3)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.num_relevant == 0

    def test_invalid_k_value(self):
        """Test error handling for invalid k values."""
        retrieved_ids = ["doc1", "doc2"]
        relevant_ids = {"doc1"}

        with pytest.raises(ValueError, match="k must be positive"):
            compute_retrieval_metrics(retrieved_ids, relevant_ids, k=0)

        with pytest.raises(ValueError, match="k must be positive"):
            compute_retrieval_metrics(retrieved_ids, relevant_ids, k=-1)

    def test_empty_retrieved_list(self):
        """Test error handling for empty retrieved list."""
        retrieved_ids = []
        relevant_ids = {"doc1", "doc2"}

        with pytest.raises(ValueError, match="retrieved_ids cannot be empty"):
            compute_retrieval_metrics(retrieved_ids, relevant_ids, k=5)


class TestAggregateMetrics:
    """Test cases for aggregate_metrics function."""

    def test_micro_averaging_two_queries(self):
        """Test micro-averaging across two queries with exact values."""
        # Query 1: 3 relevant out of 5 retrieved, 5 total relevant
        metrics1 = RAGRetrievalMetrics(
            k=5, precision=0.6, recall=0.6, f1=0.6,
            num_relevant=5, num_retrieved=5
        )

        # Query 2: 4 relevant out of 5 retrieved, 10 total relevant
        metrics2 = RAGRetrievalMetrics(
            k=5, precision=0.8, recall=0.4, f1=0.533,
            num_relevant=10, num_retrieved=5
        )

        aggregated = aggregate_metrics([metrics1, metrics2])

        # Calculate expected micro-averages:
        # Total true positives: (0.6 * 5) + (0.8 * 5) = 3 + 4 = 7
        # Total retrieved: 5 + 5 = 10
        # Total relevant: 5 + 10 = 15
        # Micro precision: 7/10 = 0.7
        # Micro recall: 7/15 = 0.4667
        # Micro F1: 2 * (0.7 * 0.4667) / (0.7 + 0.4667) = 0.56

        assert aggregated['precision'] == 0.7
        assert abs(aggregated['recall'] - 7/15) < 1e-10
        assert abs(aggregated['f1'] - 0.56) < 1e-10
        assert aggregated['num_queries'] == 2
        assert aggregated['total_retrieved'] == 10
        assert aggregated['total_relevant'] == 15

    def test_single_query_aggregation(self):
        """Test aggregation with single query (should return same values)."""
        # Query with 2.25 true positives (0.75 * 3), 6 total relevant
        metrics = RAGRetrievalMetrics(
            k=3, precision=0.75, recall=0.375, f1=0.5,
            num_relevant=6, num_retrieved=3
        )

        aggregated = aggregate_metrics([metrics])

        assert aggregated['precision'] == 0.75
        assert aggregated['recall'] == 0.375  # 2.25/6 = 0.375
        assert abs(aggregated['f1'] - 0.5) < 1e-10
        assert aggregated['num_queries'] == 1

    def test_perfect_metrics_aggregation(self):
        """Test aggregation when all queries have perfect metrics."""
        perfect_metrics = RAGRetrievalMetrics(
            k=5, precision=1.0, recall=1.0, f1=1.0,
            num_relevant=5, num_retrieved=5
        )

        aggregated = aggregate_metrics([perfect_metrics, perfect_metrics])

        assert aggregated['precision'] == 1.0
        assert aggregated['recall'] == 1.0
        assert aggregated['f1'] == 1.0

    def test_zero_metrics_aggregation(self):
        """Test aggregation when all queries have zero metrics."""
        zero_metrics = RAGRetrievalMetrics(
            k=5, precision=0.0, recall=0.0, f1=0.0,
            num_relevant=5, num_retrieved=5
        )

        aggregated = aggregate_metrics([zero_metrics, zero_metrics])

        assert aggregated['precision'] == 0.0
        assert aggregated['recall'] == 0.0
        assert aggregated['f1'] == 0.0

    def test_empty_batch_error(self):
        """Test error handling for empty batch."""
        with pytest.raises(ValueError, match="batch cannot be empty"):
            aggregate_metrics([])

    def test_invalid_metrics_type_error(self):
        """Test error handling for invalid metrics type in batch."""
        invalid_metrics = {"not": "a RAGRetrievalMetrics object"}

        with pytest.raises(ValueError, match="Expected RAGRetrievalMetrics"):
            aggregate_metrics([invalid_metrics])


class TestLogMetricsPrometheus:
    """Test cases for log_metrics_prometheus function."""

    @patch('pynucleus.metrics.system_statistics.PROMETHEUS_AVAILABLE', True)
    @patch('pynucleus.metrics.system_statistics.rag_precision_gauge')
    @patch('pynucleus.metrics.system_statistics.rag_recall_gauge')
    @patch('pynucleus.metrics.system_statistics.rag_f1_gauge')
    def test_successful_prometheus_logging(self, mock_f1, mock_recall, mock_precision):
        """Test successful logging to Prometheus gauges."""
        metrics = {
            'precision': 0.75,
            'recall': 0.60,
            'f1': 0.67
        }

        # Mock the labels().set() chain
        mock_precision.labels.return_value.set = MagicMock()
        mock_recall.labels.return_value.set = MagicMock()
        mock_f1.labels.return_value.set = MagicMock()

        log_metrics_prometheus(metrics, query_type='technical')

        # Verify gauges were called correctly
        mock_precision.labels.assert_called_once_with(query_type='technical')
        mock_precision.labels.return_value.set.assert_called_once_with(0.75)

        mock_recall.labels.assert_called_once_with(query_type='technical')
        mock_recall.labels.return_value.set.assert_called_once_with(0.60)

        mock_f1.labels.assert_called_once_with(query_type='technical')
        mock_f1.labels.return_value.set.assert_called_once_with(0.67)

    @patch('pynucleus.metrics.system_statistics.PROMETHEUS_AVAILABLE', False)
    def test_prometheus_not_available(self):
        """Test graceful handling when Prometheus is not available."""
        metrics = {'precision': 0.75, 'recall': 0.60, 'f1': 0.67}

        # Should not raise any errors
        log_metrics_prometheus(metrics)

    @patch('pynucleus.metrics.system_statistics.PROMETHEUS_AVAILABLE', True)
    def test_missing_required_metrics(self):
        """Test error handling for missing required metrics."""
        incomplete_metrics = {'precision': 0.75, 'recall': 0.60}  # Missing 'f1'

        with pytest.raises(ValueError, match="Missing required metrics"):
            log_metrics_prometheus(incomplete_metrics)

    @patch('pynucleus.metrics.system_statistics.PROMETHEUS_AVAILABLE', True)
    def test_invalid_metric_values(self):
        """Test error handling for invalid metric values."""
        # Test negative value
        with pytest.raises(ValueError, match="Invalid precision value"):
            log_metrics_prometheus({
                'precision': -0.1,
                'recall': 0.5,
                'f1': 0.4
            })

        # Test value > 1
        with pytest.raises(ValueError, match="Invalid recall value"):
            log_metrics_prometheus({
                'precision': 0.5,
                'recall': 1.5,
                'f1': 0.4
            })

        # Test non-numeric value
        with pytest.raises(ValueError, match="Invalid f1 value"):
            log_metrics_prometheus({
                'precision': 0.5,
                'recall': 0.6,
                'f1': "invalid"
            })


# Integration test
class TestIntegrationWorkflow:
    """Integration tests for complete workflow."""

    def test_complete_metrics_workflow(self):
        """Test complete workflow from computation to Prometheus logging."""
        # Step 1: Compute metrics for multiple queries
        query1_retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        query1_relevant = {"doc1", "doc3", "doc5", "doc7", "doc9"}

        query2_retrieved = ["doc10", "doc11", "doc12", "doc13", "doc14"]
        query2_relevant = {"doc10", "doc11", "doc12", "doc15", "doc16"}

        metrics1 = compute_retrieval_metrics(query1_retrieved, query1_relevant, k=5)
        metrics2 = compute_retrieval_metrics(query2_retrieved, query2_relevant, k=5)

        # Step 2: Aggregate metrics
        aggregated = aggregate_metrics([metrics1, metrics2])

        # Step 3: Verify aggregated values are reasonable
        assert 0 <= aggregated['precision'] <= 1
        assert 0 <= aggregated['recall'] <= 1
        assert 0 <= aggregated['f1'] <= 1
        assert aggregated['num_queries'] == 2

        # Step 4: Test Prometheus logging (with mocked gauges)
        with patch('pynucleus.metrics.system_statistics.PROMETHEUS_AVAILABLE', True):
            with patch('pynucleus.metrics.system_statistics.rag_precision_gauge') as mock_p:
                with patch('pynucleus.metrics.system_statistics.rag_recall_gauge') as mock_r:
                    with patch('pynucleus.metrics.system_statistics.rag_f1_gauge') as mock_f:
                        mock_p.labels.return_value.set = MagicMock()
                        mock_r.labels.return_value.set = MagicMock()
                        mock_f.labels.return_value.set = MagicMock()

                        log_metrics_prometheus(aggregated, query_type='integration_test')

                        # Verify all gauges were updated
                        mock_p.labels.assert_called_once()
                        mock_r.labels.assert_called_once()
                        mock_f.labels.assert_called_once()
