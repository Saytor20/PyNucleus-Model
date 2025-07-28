# PyNucleus-Model

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Health Score](https://img.shields.io/badge/Health%20Score-100%25-brightgreen.svg)](https://github.com/mohammadalmusaiteer/PyNucleus-Model)

**PyNucleus** is a production-ready Chemical Process Simulation & RAG System designed for African industrial markets. Combines advanced language models with chemical engineering expertise for intelligent Q&A, plant design simulation, and economic analysis.

📖 **[Complete System Documentation](PyNucleus.md)** | 🏭 **54 Documents Indexed** | 🧠 **SmolLM2-1.7B Model**

## ⚡ Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/mohammadalmusaiteer/PyNucleus-Model.git
cd PyNucleus-Model

# Set up environment
python -m venv pynucleus_env
source pynucleus_env/bin/activate  # Windows: pynucleus_env\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

**Requirements:** Python 3.11+, 8GB+ RAM, 2GB+ storage

### Verify Installation

```bash
pynucleus --help           # Show available commands
pynucleus health quick     # Run system health check
pynucleus chat             # Start interactive chat
```

## 💬 Core Features

### Interactive Chat
```bash
# Chat with chemical engineering knowledge base
pynucleus chat
> What is mass transfer zone?
> How do modular chemical plants work?
```

### Plant Design & Simulation
```bash
# Build chemical plants with economic analysis
pynucleus build --template 1 --feedstock natural_gas --capacity 1000 --location Nigeria
```

### Document Processing
```bash
# Process technical documents into knowledge base
pynucleus ingest auto --source-dir data/raw/documents
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

**📖 [Complete Documentation](PyNucleus.md)** - System architecture, components, and advanced configuration

## 🔧 Troubleshooting

```bash
# System health check
pynucleus health full

# Check system diagnostics
python scripts/comprehensive_health_check.py

# View logs
tail -f logs/pynucleus_*.log
```

**Common Issues:**
- Install issues: Update pip and try `pip install -r requirements.txt` again
- Memory errors: Ensure 8GB+ RAM available
- GPU issues: Set `USE_CUDA=false` in `.env` for CPU-only mode

📋 [Report Issues](https://github.com/mohammadalmusaiteer/PyNucleus-Model/issues)

---

## 📊 System Status

**Health Score:** 100% ✅ | **Production Ready** | **54 Documents Indexed** | **22+ Plant Templates**

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**PyNucleus v2.0.0** - Built for African industrialization with ❤️