# PyNucleus-Model: Google Colab Setup Guide

## 🚀 Quick Start for Google Colab

### Option 1: One-Click Colab Setup (Recommended)

Click the button below to open PyNucleus directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohammadalmusaiteer/PyNucleus-Model/blob/main/Capstone%20Project.ipynb)

### Option 2: Manual Setup

#### Step 1: Clone Repository
```python
# Run this in a Colab cell
!git clone https://github.com/mohammadalmusaiteer/PyNucleus-Model.git
%cd PyNucleus-Model
```

#### Step 2: Install Dependencies
```python
# Install Colab-optimized requirements
!pip install -r requirements-colab.txt

# Alternative: Install from PyPI (when available)
# !pip install pynucleus
```

#### Step 3: Setup Environment
```python
import os
import sys

# Add src to Python path
sys.path.insert(0, '/content/PyNucleus-Model/src')

# Set environment variables
os.environ['PYTHONPATH'] = '/content/PyNucleus-Model/src'
os.environ['PYNUCLEUS_MODE'] = 'colab'
```

#### Step 4: Initialize PyNucleus
```python
# Import and initialize PyNucleus
from pynucleus.pipeline import PipelineUtils
from pynucleus.integration.llm_output_generator import LLMOutputGenerator

# Initialize with Colab-specific settings
pipeline = PipelineUtils(
    results_dir="/content/PyNucleus-Model/data/05_output/results",
    mode='colab'  # Enables Colab optimizations
)

print("✅ PyNucleus initialized successfully in Google Colab!")
```

## 🔧 Colab-Specific Configuration

### DWSIM Integration in Colab

Since DWSIM requires Windows/.NET and is not available in Colab, PyNucleus automatically switches to **mock simulation mode** when running in Colab:

```python
# Mock DWSIM mode is automatically enabled in Colab
# This provides realistic sample data for testing and development
pipeline.run_complete_pipeline()  # Uses mock simulations
```

### GPU Acceleration (Optional)

To use GPU acceleration in Colab:

1. **Enable GPU Runtime**:
   - Go to Runtime → Change runtime type
   - Set Hardware accelerator to "GPU"
   - Click Save

2. **Install GPU-optimized dependencies**:
```python
# Uncomment to use GPU acceleration
# !pip uninstall faiss-cpu -y
# !pip install faiss-gpu>=1.7.0,<1.9.0
```

### Memory Management

Colab has memory limits. For large document processing:

```python
# Optimize for Colab memory constraints
pipeline = PipelineUtils(
    results_dir="/content/results",
    chunk_size=256,  # Smaller chunks for memory efficiency
    max_documents=100,  # Limit document processing
    mode='colab'
)
```

## 📁 File Management in Colab

### Persistent Storage

To save results between Colab sessions:

```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Use Drive for results storage
pipeline = PipelineUtils(
    results_dir="/content/drive/MyDrive/PyNucleus-Results"
)
```

### Download Results

```python
# Download results locally
from google.colab import files

# Package results for download
!zip -r pynucleus_results.zip data/05_output/
files.download('pynucleus_results.zip')
```

## 🎯 Complete Colab Workflow

### Full Analysis in Colab

```python
# Complete setup and analysis in one cell
import os
import sys

# Setup
!git clone https://github.com/mohammadalmusaiteer/PyNucleus-Model.git
%cd PyNucleus-Model
!pip install -r requirements-colab.txt

# Initialize
sys.path.insert(0, '/content/PyNucleus-Model/src')
from pynucleus.pipeline import PipelineUtils

# Run analysis
pipeline = PipelineUtils(mode='colab')
results = pipeline.run_complete_pipeline()

# Display results
pipeline.view_results_summary()
print(f"✅ Analysis complete! Generated {len(results.get('exported_files', []))} result files")
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**:
   ```python
   # Reduce memory usage
   pipeline = PipelineUtils(
       chunk_size=128,
       max_documents=50,
       mode='colab'
   )
   ```

2. **Module Import Errors**:
   ```python
   # Verify path setup
   import sys
   print("Python path:", sys.path)
   
   # Re-add if needed
   sys.path.insert(0, '/content/PyNucleus-Model/src')
   ```

3. **DWSIM Errors**:
   ```python
   # Ensure mock mode is enabled
   os.environ['PYNUCLEUS_MODE'] = 'colab'
   # This automatically enables DWSIM mock simulations
   ```

### Performance Tips

1. **Use Colab Pro** for higher memory limits and faster GPUs
2. **Batch Processing**: Process documents in smaller batches
3. **Persistent Sessions**: Use Google Drive for storing intermediate results

## 📚 Next Steps

After setup in Colab:

1. **Run the Pipeline**: Execute the complete analysis workflow
2. **Explore Results**: Use the built-in dashboard for result visualization
3. **Customize**: Modify configurations for your specific use case
4. **Export**: Download results for external analysis

## 🆘 Support

If you encounter issues:

1. Check the [main README](../README.md) for general troubleshooting
2. Review the [comprehensive documentation](../docs/COMPREHENSIVE_SYSTEM_DOCUMENTATION.md)
3. Open an issue on [GitHub](https://github.com/mohammadalmusaiteer/PyNucleus-Model/issues)

---

**🎉 You're ready to run PyNucleus in Google Colab!**

For the best experience, use the pre-configured notebook: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohammadalmusaiteer/PyNucleus-Model/blob/main/Capstone%20Project.ipynb) 