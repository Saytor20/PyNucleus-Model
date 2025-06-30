# 🔬 PyNucleus Dashboard - Quick Start

## Instant Launch (3 Commands)

```bash
# 1. Install dependencies (already done)
pip install streamlit plotly

# 2. Launch dashboard
./scripts/launch_dashboard.sh

# 3. Open browser
# Go to: http://localhost:8501
```

## What You Get

✅ **Real-time system monitoring**
✅ **Error rate & response time tracking** 
✅ **Confidence score analytics**
✅ **Domain performance insights**
✅ **Automated alerting system**
✅ **Interactive visualizations**

## Alternative Launch Methods

### Direct Streamlit
```bash
streamlit run src/pynucleus/diagnostics/dashboard.py
```

### Background Service
```bash
nohup streamlit run src/pynucleus/diagnostics/dashboard.py --server.headless true &
```

### Custom Port
```bash
streamlit run src/pynucleus/diagnostics/dashboard.py --server.port 8502
```

## Test Installation
```bash
python scripts/test_dashboard.py
```

## Full Documentation
- **Setup Guide**: `docs/dashboard_setup_guide.md`
- **Implementation Summary**: `docs/dashboard_implementation_summary.md`

---
**Ready to monitor PyNucleus in real-time!** 🚀 