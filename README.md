# PyNucleus-Model: Chemical Process Simulation & RAG System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Support](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)

**PyNucleus** is a comprehensive system that integrates **Retrieval-Augmented Generation (RAG)** with **chemical process simulation** for advanced chemical engineering applications.

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.11+ (tested with 3.11, 3.12, and 3.13)
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ free space

### One-Click Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/mohammadalmusaiteer/PyNucleus-Model.git
cd PyNucleus-Model

# Run complete setup (installs everything + sample data)
bash scripts/quick-start.sh
```

### Alternative: Manual Installation

```bash
# Clone repository
git clone https://github.com/mohammadalmusaiteer/PyNucleus-Model.git
cd PyNucleus-Model

# Create virtual environment
python -m venv pynucleus_env
source pynucleus_env/bin/activate  # On Windows: pynucleus_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Docker Installation

```bash
# Clone repository
git clone https://github.com/mohammadalmusaiteer/PyNucleus-Model.git
cd PyNucleus-Model

# Build and run with Docker
docker build -f deployment/Dockerfile -t pynucleus .
docker run --rm -it pynucleus --help

# Or use Docker Compose
cd deployment
docker-compose up
```

## 🎯 Verify Installation

```bash
# Check CLI is working
pynucleus --help

# Run system health check
pynucleus health quick

# Test with a simple question
pynucleus chat --model Qwen/Qwen2.5-1.5B-Instruct
```

## 📋 Basic Usage

### CLI Commands

```bash
# View available commands
pynucleus --help

# Interactive chat session
pynucleus chat

# Run pipeline with configuration
pynucleus run --config configs/development_config.json

# Build chemical plant simulation
pynucleus build --interactive

# System diagnostics
pynucleus health full

# Process documents
pynucleus ingest auto --source-dir data/01_raw
```

### Python API

```python
from pynucleus.pipeline import PipelineOrchestrator
from pynucleus.rag import QueryEngine

# Initialize pipeline
pipeline = PipelineOrchestrator()
results = pipeline.run_full_pipeline()

# Query system
engine = QueryEngine()
answer = engine.query("What are the parameters for distillation column design?")
```

## ⚙️ Configuration

### Environment Setup

Create a `.env` file from the template:

```bash
cp configs/config_template.env .env
```

Edit `.env` with your settings:

```bash
# Model Configuration
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
EMB_MODEL=all-MiniLM-L6-v2
MAX_TOKENS=8192
USE_CUDA=false

# Database Configuration
CHROMA_PATH=data/03_intermediate/vector_db
```

### Configuration Files

- `configs/development_config.json` - Development settings
- `configs/production_config.json` - Production settings
- `configs/mock_data_modular_plants.json` - Plant templates

## 🗂️ Project Structure

```
PyNucleus-Model/
├── src/pynucleus/          # Main Python package
│   ├── cli.py              # Command-line interface
│   ├── pipeline/           # Core processing pipeline
│   ├── rag/                # Document processing & retrieval
│   ├── llm/                # Language model integration
│   ├── metrics/            # Performance monitoring
│   └── terminal_dashboard.py # Terminal dashboard
├── configs/                # Configuration files
├── data/                   # Data directories
│   ├── 01_raw/            # Source documents
│   ├── 02_processed/      # Processed text
│   └── 05_output/         # Results and reports
├── scripts/               # Utility scripts
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
└── pyproject.toml        # Package configuration
```

## 🧪 Testing & Validation

```bash
# Run comprehensive system validation
python scripts/run_comprehensive_validation.py

# Quick health check
pynucleus health quick

# Full system diagnostics
pynucleus diagnostics --comprehensive
```

## 🔧 Troubleshooting

### Common Issues

**Installation Problems:**
```bash
# Update pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Runtime Errors:**
```bash
# Validate system infrastructure
python scripts/validate_infrastructure.py

# Run diagnostics
pynucleus health full --verbose
```

**Performance Issues:**
```bash
# Check system resources
pynucleus stats --mode system

# Optimize memory usage
python scripts/optimize_memory.py
```

### Getting Help

1. Run system diagnostics: `pynucleus diagnostics`
2. Check logs in `logs/` directory
3. View detailed help: `pynucleus [command] --help`
4. Report issues: [GitHub Issues](https://github.com/mohammadalmusaiteer/PyNucleus-Model/issues)

## 📚 Documentation

- **[Features & Capabilities](docs/features.md)** - Detailed feature documentation
- **[CLI Reference](docs/cli-reference.md)** - Complete command reference
- **[Configuration Guide](docs/configuration.md)** - Setup and customization
- **[Development Guide](docs/development.md)** - Contributing and development

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DWSIM**: Open-source chemical process simulator
- **ChromaDB**: Vector database for embeddings
- **HuggingFace**: Transformers and model hub
- **Typer**: Modern CLI framework

---

**Ready for production use with comprehensive monitoring and enterprise-grade reliability!** 🎉

## 🆕 Recent Updates

### v2.0.0 (2025-01-14)
- **CLI-First Architecture**: Streamlined CLI-only interface for better performance
- **Python 3.11+**: Updated to support latest Python versions (3.11, 3.12, 3.13)
- **Latest Dependencies**: Updated all dependencies to latest stable versions
- **Cross-Platform Support**: Enhanced compatibility across macOS, Windows, and Linux
- **Docker Optimization**: Improved Docker builds with better caching and smaller images
- **Code Cleanup**: Removed duplicated files and web interface components
- **Performance Improvements**: Faster startup times and reduced memory usage