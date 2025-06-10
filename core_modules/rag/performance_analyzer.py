#!/usr/bin/env python3
"""
Performance analysis and metrics tracking for RAG system
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class PerformanceAnalyzer:
    """Analyzes and tracks performance metrics for RAG system."""
    
    def __init__(self, log_dir: str = "data/analysis"):
        """Initialize performance analyzer.
        
        Args:
            log_dir: Directory to store performance logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, Any] = {}
        self.start_time = time.time()
    
    def log_metric(self, name: str, value: Any) -> None:
        """Log a single metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        self.metrics.update(metrics)
    
    def save(self, filename: str = None) -> Path:
        """Save metrics to file.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved metrics file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"
        
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        return filepath
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since initialization.
        
        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time 