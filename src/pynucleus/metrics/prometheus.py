"""
Prometheus metrics for PyNucleus confidence calibration monitoring.
"""

import logging
from typing import Optional
try:
    from prometheus_client import Histogram, Counter, Gauge, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create mock classes if prometheus_client is not available
    class MockMetric:
        def __init__(self, *args, **kwargs): pass
        def observe(self, value): pass
        def inc(self, value=1): pass
        def set(self, value): pass
        def labels(self, **kwargs): return self
    
    Histogram = Counter = Gauge = MockMetric

logger = logging.getLogger(__name__)

# Confidence calibration metrics
confidence_cal_error = Histogram(
    'confidence_calibration_error',
    'Absolute difference between raw and calibrated confidence scores',
    buckets=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
)

# Additional metrics for monitoring
confidence_calibration_requests = Counter(
    'confidence_calibration_requests_total',
    'Total number of confidence calibration requests',
    ['status']  # success, failure, not_available
)

raw_confidence_distribution = Histogram(
    'raw_confidence_distribution',
    'Distribution of raw confidence scores',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

calibrated_confidence_distribution = Histogram(
    'calibrated_confidence_distribution', 
    'Distribution of calibrated confidence scores',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)


def record_confidence_calibration(raw_confidence: float, calibrated_confidence: float, status: str = 'success'):
    """
    Record confidence calibration metrics.
    
    Args:
        raw_confidence: Original confidence score
        calibrated_confidence: Calibrated confidence score  
        status: Status of calibration ('success', 'failure', 'not_available')
    """
    if not PROMETHEUS_AVAILABLE:
        return
        
    try:
        # Record calibration error (absolute difference)
        calibration_error = abs(raw_confidence - calibrated_confidence)
        confidence_cal_error.observe(calibration_error)
        
        # Record request status
        confidence_calibration_requests.labels(status=status).inc()
        
        # Record confidence distributions
        raw_confidence_distribution.observe(raw_confidence)
        calibrated_confidence_distribution.observe(calibrated_confidence)
        
        logger.debug(f"Recorded calibration metrics: raw={raw_confidence:.3f}, "
                    f"calibrated={calibrated_confidence:.3f}, error={calibration_error:.3f}")
                    
    except Exception as e:
        logger.warning(f"Failed to record confidence calibration metrics: {e}")


def start_metrics_server(port: int = 8000) -> Optional[object]:
    """
    Start Prometheus metrics server.
    
    Args:
        port: Port to serve metrics on
        
    Returns:
        Server object or None if failed
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus client not available, metrics server not started")
        return None
        
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start Prometheus metrics server: {e}")
        return None


def get_metrics_summary() -> dict:
    """
    Get a summary of current metrics.
    
    Returns:
        Dictionary with metrics summary
    """
    if not PROMETHEUS_AVAILABLE:
        return {"error": "Prometheus client not available"}
        
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        
        # Get latest metrics
        metrics_data = generate_latest(REGISTRY)
        
        return {
            "metrics_available": True,
            "content_type": CONTENT_TYPE_LATEST,
            "data_size": len(metrics_data)
        }
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        return {"error": str(e)} 