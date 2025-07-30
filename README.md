# PySaytor

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Health Score](https://img.shields.io/badge/Health%20Score-100%25-brightgreen.svg)](https://github.com/mohammadalmusaiteer/PySaytor)

**PySaytor** is a production-ready Chemical Process Simulation & RAG System designed for African industrial markets. Combines advanced language models with chemical engineering expertise for intelligent Q&A, plant design simulation, and economic analysis.

📖 **[Complete System Documentation](PySaytor.md)** | 🏭 **54 Documents Indexed** | 🧠 **SmolLM2-1.7B Model**

## ⚡ Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/mohammadalmusaiteer/PySaytor.git
cd PySaytor

# Set up environment
python -m venv pysaytor_env
source pysaytor_env/bin/activate  # Windows: pysaytor_env\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

**Requirements:** Python 3.11+, 8GB+ RAM, 2GB+ storage

### Verify Installation

```bash
pysaytor --help           # Show available commands
pysaytor health quick     # Run system health check
pysaytor chat             # Start interactive chat
```

## 💬 Core Features

### Interactive Chat
```bash
# Chat with chemical engineering knowledge base
pysaytor chat
> What is mass transfer zone?
> How do modular chemical plants work?
```

### Plant Design & Simulation
```bash
# Build chemical plants with economic analysis
pysaytor build --template 1 --feedstock natural_gas --capacity 1000 --location Nigeria
```

### Document Processing
```bash
# Process technical documents into knowledge base
pysaytor ingest auto --source-dir data/raw/documents
```

## 🏗️ System Architecture

- **LLM Engine:** HuggingFaceTB/SmolLM2-1.7B-Instruct with fallback models
- **RAG System:** ChromaDB vector store with 54 technical documents
- **Plant Simulation:** 22+ modular chemical plant templates
- **Knowledge Base:** Chemical engineering + African industrialization focus

### Configuration

```bash
# Quick setup - copy and edit environment template
cp configs/config_template.env .env
```

**📖 [Complete Documentation](PySaytor.md)** - System architecture, components, and advanced configuration

## 🔧 Troubleshooting

```bash
# System health check
pysaytor health full

# Check system diagnostics
python scripts/comprehensive_health_check.py

# View logs
tail -f logs/pysaytor_*.log
```

**Common Issues:**
- Install issues: Update pip and try `pip install -r requirements.txt` again
- Memory errors: Ensure 8GB+ RAM available
- GPU issues: Set `USE_CUDA=false` in `.env` for CPU-only mode

📋 [Report Issues](https://github.com/mohammadalmusaiteer/PySaytor/issues)

---

## 📊 System Status

**Health Score:** 100% ✅ | **Production Ready** | **54 Documents Indexed** | **22+ Plant Templates**

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**PySaytor v2.0.0** - Built for African industrialization with ❤️