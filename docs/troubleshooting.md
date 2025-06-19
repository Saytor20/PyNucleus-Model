# PyNucleus Troubleshooting Guide

## 🚨 Common Issues and Solutions

### **pip/Requirements Issues**

#### **1. Invalid pip specifier error (like your original issue)**
```
ERROR: Invalid requirement: 'faiss-cpu>=1.7.*,<1.9': .* suffix can only be used with `==` or `!=` operators
```

**Solution:**
```bash
# This is now fixed in all our requirements files, but if you see this:
# Use the corrected requirements files:
pip install -r requirements-colab.txt  # For Colab
pip install -r requirements-minimal.txt  # For basic functionality
pip install -r requirements.txt  # For full installation
```

#### **2. Version conflicts**
```bash
# Clean install approach:
pip uninstall -y torch torchvision transformers
pip install --no-deps -r requirements.txt
pip check  # Verify no conflicts
```

### **Google Colab Issues**

#### **1. Module not found errors**
```python
# Ensure path is set correctly:
import sys
sys.path.insert(0, '/content/PyNucleus-Model/src')

# Verify:
print(sys.path)
```

#### **2. Memory errors in Colab**
```python
# Use memory-optimized settings:
from pynucleus.pipeline import PipelineUtils

pipeline = PipelineUtils(
    chunk_size=128,        # Smaller chunks
    max_documents=50,      # Limit documents
    mode='colab'          # Colab optimizations
)
```

#### **3. DWSIM not available in Colab**
```python
# This is expected and handled automatically:
import os
os.environ['PYNUCLEUS_MODE'] = 'colab'  # Enables mock DWSIM mode
```

### **Docker Issues**

#### **1. Docker build failures**
```bash
# Clean Docker environment:
docker system prune -f
docker volume prune -f

# Rebuild:
./docker/build.sh
```

#### **2. Permission denied on scripts**
```bash
# Make scripts executable:
chmod +x docker/build.sh
chmod +x scripts/validate_infrastructure.py
```

#### **3. Docker compose issues**
```bash
# Validate compose file:
cd docker
docker-compose config

# Check service health:
docker-compose ps
docker-compose logs api
```

### **Local Installation Issues**

#### **1. Python version compatibility**
```bash
# Check Python version:
python --version  # Should be 3.10+

# If using older Python:
pip install -r requirements-minimal.txt  # More compatible
```

#### **2. Virtual environment issues**
```bash
# Clean virtual environment:
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

# Upgrade pip first:
pip install --upgrade pip
pip install -r requirements.txt
```

#### **3. FAISS installation issues**
```bash
# For CPU-only systems:
pip uninstall faiss-gpu
pip install faiss-cpu>=1.7.0

# For GPU systems:
pip uninstall faiss-cpu
pip install faiss-gpu>=1.7.0
```

### **Import/Module Issues**

#### **1. PyNucleus modules not found**
```python
# Add to Python path:
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

# Or set environment variable:
import os
os.environ['PYTHONPATH'] = str(Path.cwd() / "src")
```

#### **2. Missing dependencies**
```bash
# Install missing packages:
pip install --upgrade transformers torch sentence-transformers

# Check what's installed:
pip list | grep -E "(torch|transformers|faiss)"
```

### **Performance Issues**

#### **1. Slow vector store operations**
```python
# Optimize FAISS settings:
import faiss
faiss.omp_set_num_threads(4)  # Adjust based on CPU cores
```

#### **2. Memory usage optimization**
```python
# Use smaller models and batches:
pipeline = PipelineUtils(
    embedding_model="all-MiniLM-L6-v2",  # Smaller model
    chunk_size=256,                       # Smaller chunks
    batch_size=16                         # Smaller batches
)
```

### **Data/File Issues**

#### **1. Missing data directories**
```bash
# Create required directories:
mkdir -p data/{01_raw,02_processed,03_intermediate,04_models,05_output}
mkdir -p logs configs
```

#### **2. Permission issues**
```bash
# Fix permissions:
chmod -R 755 data/
chmod -R 755 logs/
```

### **Validation and Diagnostics**

#### **Run comprehensive checks:**
```bash
# Infrastructure validation:
python scripts/validate_infrastructure.py

# System diagnostics:
python scripts/comprehensive_system_diagnostic.py

# Component validation:
python scripts/system_validator.py
```

## 🔧 **Advanced Troubleshooting**

### **Debug Mode**
```python
# Enable verbose logging:
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug mode:
pipeline = PipelineUtils(debug=True)
```

### **Check System Resources**
```python
import psutil

print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
print(f"CPU cores: {psutil.cpu_count()}")
print(f"Python version: {sys.version}")
```

### **Environment Validation**
```python
# Complete environment check:
def validate_environment():
    checks = {
        'Python 3.10+': sys.version_info >= (3, 10),
        'PyTorch available': True,
        'Transformers available': True,
        'FAISS available': True,
        'Sufficient memory': psutil.virtual_memory().available > 2 * 1024**3  # 2GB
    }
    
    try:
        import torch
        checks['PyTorch available'] = True
    except ImportError:
        checks['PyTorch available'] = False
        
    try:
        import transformers
        checks['Transformers available'] = True
    except ImportError:
        checks['Transformers available'] = False
        
    try:
        import faiss
        checks['FAISS available'] = True
    except ImportError:
        checks['FAISS available'] = False
    
    for check, status in checks.items():
        print(f"{'✅' if status else '❌'} {check}")
    
    return all(checks.values())

validate_environment()
```

## 🆘 **Getting Help**

If you're still experiencing issues:

1. **Run the validation script**: `python scripts/validate_infrastructure.py`
2. **Check the logs**: Look in the `logs/` directory for error details
3. **Review requirements**: Ensure you're using the correct requirements file for your environment
4. **Open an issue**: [GitHub Issues](https://github.com/mohammadalmusaiteer/PyNucleus-Model/issues) with:
   - Error message
   - Operating system
   - Python version
   - Installation method used
   - Output of validation script

## 📚 **Related Documentation**

- [Colab Setup Guide](colab_setup.md)
- [Main README](../README.md)
- [Comprehensive Documentation](COMPREHENSIVE_SYSTEM_DOCUMENTATION.md)
- [Project Roadmap](PROJECT_ROADMAP.md)

---

**Most issues can be resolved by using the correct requirements file for your environment and running the infrastructure validation script.** 