# PyNucleus Real-Time Analytics Dashboard

## Overview

The PyNucleus Real-Time Analytics Dashboard provides comprehensive monitoring and visualization for the PyNucleus system, including:

- **Semantic validation scores** and accuracy metrics
- **Confidence score distribution** and calibration quality
- **Domain-specific performance analytics** across nuclear engineering domains
- **Real-time error detection** and anomaly monitoring
- **System health tracking** with automated alerting
- **Performance trends** and operational insights

## Features

### 🔍 System Monitoring
- Real-time error rate tracking
- Response time analytics
- System health percentage monitoring
- Request volume analysis

### 📊 Performance Analytics
- Confidence score distribution visualization
- Domain-specific accuracy breakdowns
- Performance trend analysis
- Validation result tracking

### 🚨 Intelligent Alerting
- Configurable alert thresholds
- Multiple severity levels (Critical, High, Medium, Low)
- Real-time notifications for:
  - High error rates (>10%)
  - Slow response times (>5s)
  - System health degradation (<80%)

### 📈 Interactive Visualizations
- Real-time charts using Plotly
- Historical trend analysis
- Error distribution pie charts
- Performance comparison across domains

## Installation & Setup

### Prerequisites

Ensure you have the PyNucleus environment set up with all dependencies:

```bash
# Install visualization dependencies (already added to requirements.txt)
pip install streamlit plotly
```

### Quick Start

1. **Navigate to the project root:**
   ```bash
   cd /path/to/PyNucleus-Model
   ```

2. **Launch the dashboard:**
   ```bash
   streamlit run src/pynucleus/diagnostics/dashboard.py
   ```

3. **Access the dashboard:**
   - Open your browser to `http://localhost:8501`
   - The dashboard will automatically detect and connect to your PyNucleus logs and data

### Advanced Configuration

#### Environment Variables

Set these optional environment variables for enhanced functionality:

```bash
# Dashboard refresh interval (seconds)
export PYNUCLEUS_DASHBOARD_REFRESH=30

# Log retention period (hours)
export PYNUCLEUS_LOG_RETENTION=24

# Alert notification settings
export PYNUCLEUS_ALERT_EMAIL="admin@your-org.com"
```

#### Custom Alert Thresholds

Modify alert thresholds in the dashboard configuration:

```python
DASHBOARD_CONFIG = {
    "alert_thresholds": {
        "error_rate": 0.1,      # 10% error rate threshold
        "response_time": 5.0,   # 5 second response time threshold
        "system_health": 0.8    # 80% health threshold
    }
}
```

## Dashboard Components

### Header Status Indicators

The dashboard header displays real-time status of core components:

- **✅ Diagnostics Active**: PyNucleus diagnostics runner is operational
- **✅ Calibration Trained**: Confidence calibration model is trained and ready
- **✅ Logs Available**: Application logs are accessible for analysis
- **🕒 Current Time**: Real-time timestamp for dashboard updates

### System Overview Metrics

Four key performance indicators:

1. **Error Rate**: Percentage of failed requests/operations
2. **Average Response Time**: Mean response time for operations
3. **System Health**: Overall system health percentage
4. **Total Requests**: Volume of requests in the selected timeframe

### Active Alerts Section

Real-time alert monitoring with color-coded severity:

- **🔴 Critical/High**: System health issues, high error rates
- **🟡 Medium**: Performance degradation, moderate issues
- **🔵 Low/Info**: Informational alerts and warnings

### Performance Analytics

Visual analytics including:

- **Confidence Score Distribution**: Histogram of model confidence scores
- **Domain Performance**: Bar chart showing accuracy by nuclear engineering domain
- **Error Analysis**: Pie chart of error distribution by source
- **Trend Analysis**: Time-series performance trends

### Sidebar Controls

Interactive controls for dashboard customization:

- **Auto Refresh**: Toggle automatic dashboard updates
- **Refresh Interval**: Set update frequency (10-120 seconds)
- **Data Range**: Select historical data timeframe (1-24 hours)
- **Manual Refresh**: Force immediate data refresh

## Data Sources

The dashboard integrates with multiple PyNucleus data sources:

### Application Logs
- **Location**: `logs/app.log`
- **Data**: Error tracking, response times, system events
- **Updates**: Real-time log parsing and analysis

### Diagnostic Results
- **Source**: `DiagnosticRunner` validation results
- **Data**: Accuracy scores, validation metrics, system health
- **Updates**: On-demand diagnostic execution

### Confidence Calibration
- **Source**: `ConfidenceCalibrator` metrics
- **Data**: Calibration quality, confidence distributions
- **Updates**: Model training and validation cycles

### RAG Pipeline Metrics
- **Source**: RAG system performance data
- **Data**: Domain-specific accuracy, citation quality
- **Updates**: Query processing and evaluation results

## Operational Usage

### Daily Monitoring

1. **Morning Health Check**:
   - Review overnight alerts
   - Check system health percentage
   - Verify error rates are within normal ranges

2. **Performance Analysis**:
   - Monitor confidence score trends
   - Review domain-specific performance
   - Identify any degradation patterns

3. **Troubleshooting**:
   - Use error analysis to identify problematic components
   - Review recent error details for debugging
   - Monitor response time trends for performance issues

### Alert Response

#### Critical Alerts (System Health < 80%)
1. Immediate investigation required
2. Check system resources and dependencies
3. Review recent configuration changes
4. Consider activating incident response procedures

#### High Alerts (Error Rate > 10%)
1. Review recent error logs
2. Identify common error patterns
3. Check for external service dependencies
4. Monitor for cascading failures

#### Medium Alerts (Slow Response Times)
1. Analyze performance bottlenecks
2. Review system resource utilization
3. Check database and storage performance
4. Consider scaling or optimization

### Best Practices

1. **Regular Monitoring**: Check dashboard at least twice daily
2. **Threshold Tuning**: Adjust alert thresholds based on operational experience
3. **Data Retention**: Archive important metrics for long-term analysis
4. **Documentation**: Log significant events and resolutions for future reference

## Troubleshooting

### Common Issues

#### Dashboard Won't Start
```bash
# Check Streamlit installation
streamlit --version

# Verify PyNucleus imports
python -c "from pynucleus.diagnostics.runner import DiagnosticRunner; print('OK')"

# Check log file permissions
ls -la logs/app.log
```

#### No Data Displayed
1. Verify PyNucleus application is running and generating logs
2. Check log file location and permissions
3. Ensure diagnostic runner can access required components
4. Verify data directory structure

#### Import Errors
```bash
# Add PyNucleus to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/PyNucleus-Model/src"

# Verify all dependencies are installed
pip install -r requirements.txt
```

#### Performance Issues
1. Reduce auto-refresh frequency
2. Limit data range for large log files
3. Check system resource utilization
4. Consider running dashboard on dedicated server

### Log Analysis

The dashboard automatically parses PyNucleus logs with these patterns:

- **Timestamps**: `2024-01-01 12:00:00,000`
- **Levels**: `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Sources**: Component names (e.g., `app`, `rag`, `diagnostics`)
- **Metrics**: Extracted from log messages (response times, health percentages)

### Support and Maintenance

For ongoing support:

1. **Monitor Dashboard Performance**: Regular checks on dashboard responsiveness
2. **Update Dependencies**: Keep Streamlit and Plotly versions current
3. **Log Rotation**: Implement log rotation to prevent disk space issues
4. **Backup Configuration**: Save custom dashboard configurations
5. **Security**: Restrict dashboard access to authorized personnel

## Integration with PyNucleus Ecosystem

The dashboard seamlessly integrates with:

- **[[memory:1284172327240342256]]**: Production-ready Flask API with application factory pattern
- **[[memory:482210044065853938]]**: Auto-restart scripts for development workflow
- **Diagnostic System**: Comprehensive validation and health monitoring
- **Confidence Calibration**: ML-based confidence scoring system
- **RAG Pipeline**: Domain-specific nuclear engineering knowledge system

## Future Enhancements

Planned improvements include:

- **Email/Slack Notifications**: Automated alert delivery
- **Historical Data Export**: CSV/JSON export functionality
- **Custom Dashboards**: User-configurable dashboard layouts
- **Predictive Analytics**: ML-based anomaly detection
- **Mobile Responsiveness**: Optimized mobile dashboard views
- **Multi-tenant Support**: Organization-specific dashboards 