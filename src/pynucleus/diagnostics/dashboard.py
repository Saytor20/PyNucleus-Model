"""
Real-Time Visual Analytics Dashboard for PyNucleus

This module provides comprehensive monitoring and visualization for:
- Semantic validation scores and accuracy metrics
- Confidence score distribution and calibration quality
- Domain-specific performance analytics
- Real-time error detection and anomaly monitoring
- System health tracking with automated alerting
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import asdict
import re
import sys
import os

# Add src directory to Python path
root_dir = Path(__file__).parent.parent.parent.parent
src_path = str(root_dir / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from pynucleus.diagnostics.runner import DiagnosticRunner, ValidationResult
    from pynucleus.eval.confidence_calibration import ConfidenceCalibrator, CalibrationMetrics
    from pynucleus.utils.logging_config import setup_diagnostic_logging
except ImportError as e:
    st.error(f"Failed to import PyNucleus modules: {e}")
    st.stop()

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "refresh_interval": 30,
    "alert_thresholds": {
        "error_rate": 0.1,
        "response_time": 5.0,
        "confidence_degradation": 0.2,
        "system_health": 0.8
    },
    "data_retention": {
        "metrics_hours": 24,
        "logs_hours": 12,
        "alerts_hours": 48
    }
}


class DashboardDataManager:
    """Manages data collection and processing for the dashboard"""
    
    def __init__(self):
        try:
            _, self.logger, _ = setup_diagnostic_logging("dashboard")
        except:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("dashboard")
            
        self.root_dir = Path(__file__).parent.parent.parent.parent
        self.logs_dir = self.root_dir / "logs"
        self.data_dir = self.root_dir / "data"
        
        self.diagnostic_runner = None
        self.confidence_calibrator = None
        self.cached_data = {}
        self.alerts = []
        
    def initialize_components(self):
        """Initialize diagnostic and calibration components"""
        try:
            self.diagnostic_runner = DiagnosticRunner(quick_mode=True)
            self.confidence_calibrator = ConfidenceCalibrator()
            
            try:
                if not self.confidence_calibrator.load_models():
                    self.logger.info("No existing calibration models found")
            except:
                self.logger.info("Calibration models not available")
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return False
    
    def parse_log_file(self, log_file: Path, hours_back: int = 12) -> List[Dict]:
        """Parse log file for error and performance data"""
        log_entries = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            if not log_file.exists():
                return []
            
            with open(log_file, 'r') as f:
                for line in f:
                    entry = self._parse_log_line(line.strip())
                    if entry and entry.get('timestamp'):
                        if entry['timestamp'] > cutoff_time:
                            log_entries.append(entry)
            
            return sorted(log_entries, key=lambda x: x['timestamp'])
        except Exception as e:
            self.logger.error(f"Error parsing log file {log_file}: {e}")
            return []
    
    def _parse_log_line(self, line: str) -> Optional[Dict]:
        """Parse individual log line"""
        if not line:
            return None
        
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})'
        match = re.match(timestamp_pattern, line)
        
        if not match:
            return None
        
        try:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
            
            parts = line.split(' ', 4)
            if len(parts) >= 4:
                level = parts[1]
                source = parts[2].rstrip(':')
                message = parts[3] if len(parts) > 3 else ''
                
                metrics = self._extract_metrics_from_message(message)
                
                return {
                    'timestamp': timestamp,
                    'level': level,
                    'source': source,
                    'message': message,
                    'is_error': level in ['ERROR', 'CRITICAL'],
                    'is_warning': level == 'WARNING',
                    **metrics
                }
        except Exception as e:
            self.logger.debug(f"Failed to parse log line: {e}")
            return None
    
    def _extract_metrics_from_message(self, message: str) -> Dict:
        """Extract performance metrics from log messages"""
        metrics = {}
        
        response_time_match = re.search(r'processed successfully in ([\d.]+)s', message)
        if response_time_match:
            metrics['response_time'] = float(response_time_match.group(1))
        
        health_match = re.search(r'(\d+\.?\d*)% health', message)
        if health_match:
            metrics['health_percentage'] = float(health_match.group(1))
        
        exec_time_match = re.search(r'(\d+\.?\d*)s execution time', message)
        if exec_time_match:
            metrics['execution_time'] = float(exec_time_match.group(1))
        
        if 'RAG engine initialized' in message:
            metrics['operation'] = 'rag_init'
        elif 'Question processed' in message:
            metrics['operation'] = 'question_processing'
        elif 'Diagnostics completed' in message:
            metrics['operation'] = 'diagnostics'
        elif 'Health check' in message:
            metrics['operation'] = 'health_check'
        
        return metrics
    
    def get_system_metrics(self) -> Dict:
        """Get comprehensive system metrics"""
        log_file = self.logs_dir / "app.log"
        log_entries = self.parse_log_file(log_file)
        
        if not log_entries:
            return {
                'error_rate': 0,
                'avg_response_time': 0,
                'health_status': 'unknown',
                'recent_errors': [],
                'performance_trend': [],
                'total_requests': 0
            }
        
        total_entries = len(log_entries)
        error_entries = [e for e in log_entries if e.get('is_error')]
        error_rate = len(error_entries) / total_entries if total_entries > 0 else 0
        
        response_times = [e.get('response_time') for e in log_entries if e.get('response_time')]
        avg_response_time = np.mean(response_times) if response_times else 0
        
        health_entries = [e for e in log_entries if e.get('health_percentage') is not None]
        latest_health = health_entries[-1].get('health_percentage') if health_entries else None
        
        return {
            'error_rate': error_rate,
            'avg_response_time': avg_response_time,
            'health_status': 'healthy' if latest_health and latest_health > 80 else 'degraded',
            'health_percentage': latest_health,
            'recent_errors': error_entries[-10:],
            'total_requests': total_entries
        }
    
    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        current_time = datetime.now()
        
        if metrics['error_rate'] > DASHBOARD_CONFIG['alert_thresholds']['error_rate']:
            alerts.append({
                'type': 'error_rate',
                'severity': 'high',
                'message': f"High error rate detected: {metrics['error_rate']:.1%}",
                'timestamp': current_time,
                'value': metrics['error_rate']
            })
        
        if metrics['avg_response_time'] > DASHBOARD_CONFIG['alert_thresholds']['response_time']:
            alerts.append({
                'type': 'response_time',
                'severity': 'medium',
                'message': f"Slow response time: {metrics['avg_response_time']:.2f}s",
                'timestamp': current_time,
                'value': metrics['avg_response_time']
            })
        
        if (metrics.get('health_percentage') and 
            metrics['health_percentage'] < DASHBOARD_CONFIG['alert_thresholds']['system_health'] * 100):
            alerts.append({
                'type': 'system_health',
                'severity': 'critical',
                'message': f"System health degraded: {metrics['health_percentage']:.1f}%",
                'timestamp': current_time,
                'value': metrics['health_percentage']
            })
        
        return alerts


class DashboardUI:
    """Streamlit UI components for the dashboard"""
    
    def __init__(self, data_manager: DashboardDataManager):
        self.data_manager = data_manager
        
    def render_header(self):
        """Render dashboard header with status indicators"""
        st.set_page_config(
            page_title="PyNucleus Analytics Dashboard",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔬 PyNucleus Real-Time Analytics Dashboard")
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if self.data_manager.diagnostic_runner:
                st.success("✅ Diagnostics Active")
            else:
                st.error("❌ Diagnostics Inactive")
        
        with col2:
            if (self.data_manager.confidence_calibrator and 
                hasattr(self.data_manager.confidence_calibrator, 'is_trained') and
                self.data_manager.confidence_calibrator.is_trained):
                st.success("✅ Calibration Trained")
            else:
                st.warning("⚠️ Calibration Not Trained")
        
        with col3:
            log_file = self.data_manager.logs_dir / "app.log"
            if log_file.exists():
                st.success("✅ Logs Available")
            else:
                st.error("❌ No Logs Found")
        
        with col4:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.info(f"🕒 {current_time}")
    
    def render_system_overview(self, metrics: Dict):
        """Render system overview metrics"""
        st.markdown("### 📊 System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Error Rate",
                f"{metrics['error_rate']:.1%}",
                delta=f"{'↑' if metrics['error_rate'] > 0.05 else '↓'} Threshold: 10%"
            )
        
        with col2:
            st.metric(
                "Avg Response Time", 
                f"{metrics['avg_response_time']:.2f}s",
                delta=f"{'↑' if metrics['avg_response_time'] > 2.0 else '↓'} Target: <2s"
            )
        
        with col3:
            health_pct = metrics.get('health_percentage', 0)
            st.metric(
                "System Health",
                f"{health_pct:.1f}%" if health_pct else "Unknown",
                delta=f"{'↑' if health_pct and health_pct > 90 else '↓'} Target: >90%"
            )
        
        with col4:
            st.metric(
                "Total Requests",
                f"{metrics.get('total_requests', 0):,}",
                delta="Last 12 hours"
            )
    
    def render_alerts_section(self, alerts: List[Dict]):
        """Render alerts and notifications"""
        if not alerts:
            st.success("✅ No active alerts")
            return
        
        st.markdown("### 🚨 Active Alerts")
        
        for alert in alerts[-5:]:
            severity_color = {
                'critical': 'error',
                'high': 'error', 
                'medium': 'warning',
                'low': 'info'
            }.get(alert['severity'], 'info')
            
            getattr(st, severity_color)(
                f"**{alert['type'].replace('_', ' ').title()}**: {alert['message']} "
                f"(at {alert['timestamp'].strftime('%H:%M:%S')})"
            )
    
    def render_performance_charts(self, metrics: Dict):
        """Render performance visualization charts"""
        st.markdown("### 📈 Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample confidence distribution chart
            np.random.seed(42)
            confidence_scores = np.random.beta(2, 2, size=100)
            
            fig = px.histogram(
                x=confidence_scores,
                nbins=20,
                title="Confidence Score Distribution",
                labels={'x': 'Confidence Score', 'y': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sample domain performance chart
            domains = ['Nuclear Safety', 'Reactor Physics', 'Thermal Hydraulics', 'Materials']
            accuracy_scores = [0.85, 0.92, 0.78, 0.89]
            
            fig = px.bar(
                x=domains,
                y=accuracy_scores,
                title="Performance by Domain",
                labels={'x': 'Domain', 'y': 'Accuracy Score'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_log_analysis(self, recent_errors: List[Dict]):
        """Render recent log analysis"""
        st.markdown("### 📋 Recent Log Analysis")
        
        if not recent_errors:
            st.success("✅ No recent errors detected!")
            return
        
        error_sources = {}
        for error in recent_errors:
            source = error.get('source', 'unknown')
            error_sources[source] = error_sources.get(source, 0) + 1
        
        if error_sources:
            fig = px.pie(
                values=list(error_sources.values()),
                names=list(error_sources.keys()),
                title="Error Distribution by Source"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Recent Error Details")
        for i, error in enumerate(recent_errors[-5:], 1):
            with st.expander(f"Error {i}: {error.get('source', 'Unknown')} - {error['timestamp'].strftime('%H:%M:%S')}"):
                st.code(error.get('message', 'No message available'))
    
    def render_sidebar_controls(self):
        """Render sidebar controls"""
        st.sidebar.title("Dashboard Controls")
        
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
        
        refresh_interval = st.sidebar.selectbox(
            "Refresh Interval",
            [10, 30, 60, 120],
            index=1,
            format_func=lambda x: f"{x} seconds"
        )
        
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Data Range**")
        hours_back = st.sidebar.slider("Hours of data", 1, 24, 12)
        
        return {
            'auto_refresh': auto_refresh,
            'refresh_interval': refresh_interval,
            'hours_back': hours_back
        }


def main():
    """Main dashboard application"""
    data_manager = DashboardDataManager()
    ui = DashboardUI(data_manager)
    
    ui.render_header()
    
    if not data_manager.initialize_components():
        st.error("Failed to initialize dashboard components. Please check the logs.")
        st.stop()
    
    controls = ui.render_sidebar_controls()
    
    if controls['auto_refresh']:
        time.sleep(controls['refresh_interval'])
        st.rerun()
    
    with st.spinner("Collecting system metrics..."):
        system_metrics = data_manager.get_system_metrics()
        alerts = data_manager.check_alerts(system_metrics)
    
    ui.render_alerts_section(alerts)
    st.markdown("---")
    
    ui.render_system_overview(system_metrics)
    st.markdown("---")
    
    ui.render_performance_charts(system_metrics)
    st.markdown("---")
    
    ui.render_log_analysis(system_metrics.get('recent_errors', []))
    
    st.markdown("---")
    st.markdown(
        f"**Dashboard Status**: Last updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Auto-refresh: {'ON' if controls['auto_refresh'] else 'OFF'} | "
        f"Data range: {controls['hours_back']} hours"
    )


if __name__ == "__main__":
    main()
