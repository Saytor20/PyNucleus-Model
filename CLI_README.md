# PyNucleus CLI Usage Guide

## 🚀 Quick Start

The PyNucleus CLI can be run in several ways. Choose the method that works best for you:

### Method 1: Automatic Wrapper (Recommended)
```bash
./activate_and_run.sh
```
This automatically:
- Activates the virtual environment
- Checks and installs missing dependencies
- Runs the CLI

### Method 2: Manual Virtual Environment Activation
```bash
source pynucleus_env/bin/activate
./pynucleus
```

### Method 3: Direct Python Execution
```bash
source pynucleus_env/bin/activate
python pynucleus
```

## ❌ Why `./pynucleus` Alone Doesn't Work

The `./pynucleus` script uses `#!/usr/bin/env python3` which points to your system Python, not the virtual environment Python where all dependencies are installed.

## 🔧 Available Commands

When you run the CLI, you'll see an interactive menu with these options:

1. **Execute pipeline** - Run the main PyNucleus pipeline
2. **Chat with LLM** - Interactive chat with the language model
3. **Build simulation** - Build chemical process simulations
4. **System monitoring** - Monitor system performance
5. **Auto document ingest** - Automatically ingest documents
6. **Ask question** - Ask questions using RAG (Retrieval-Augmented Generation)
7. **Health check** - Check system health
8. **Ingest documents** - Manually ingest documents
9. **Show version** - Display version and dependency status
10. **Run evaluations** - Run system evaluations
11. **Web server** - Start the web server

## 💡 Tips

- Use `./activate_and_run.sh --help` to see command help
- Use `./activate_and_run.sh --verbose` for detailed output
- You can also run commands directly: `./activate_and_run.sh ask "What is modular plants?"`

## 🛠 Troubleshooting

If you get "ModuleNotFoundError", it means the virtual environment isn't activated. Always use one of the methods above that includes virtual environment activation.

## 📦 Dependencies

The CLI automatically checks for and installs these key dependencies:
- `typer` - CLI framework
- `rich` - Terminal formatting
- `chromadb` - Vector database
- `torch` - PyTorch for ML
- `transformers` - HuggingFace transformers
- `pandas` - Data processing
- `rank-bm25` - Text ranking
- `sentence-transformers` - Text embeddings
- `pydantic-settings` - Settings management
- `rapidfuzz` - Text deduplication

## 🧪 A comprehensive command-line interface for the PyNucleus chemical process simulation and RAG system.

## Available Commands

### 🚀 Pipeline Execution

**Run the full PyNucleus pipeline:**
```bash
./pynucleus run --config configs/production_config.json
./pynucleus run --config configs/development_config.json --verbose --dry-run
```

**Options:**
- `--config, -c`: Configuration file path (required)
- `--output, -o`: Output directory (default: `data/05_output`)
- `--verbose, -v`: Verbose logging
- `--log-file`: Custom log file path
- `--dry-run`: Validate configuration without execution

### 💬 Interactive Chat

**Start interactive chat with RAG system:**
```bash
./pynucleus chat
./pynucleus chat --model Qwen/Qwen2.5-1.5B-Instruct --top-k 10
```

**Options:**
- `--model, -m`: LLM model ID (default: `Qwen/Qwen2.5-1.5B-Instruct`)
- `--top-k, -k`: Number of RAG results to retrieve (default: 5)
- `--temperature, -t`: Model temperature (default: 0.7)
- `--max-tokens`: Maximum response tokens (default: 512)
- `--verbose, -v`: Verbose logging

### 🏗️ Chemical Plant Building

**Build and simulate chemical plants:**
```bash
./pynucleus build --template 1 --feedstock natural_gas --capacity 100000
./pynucleus build --interactive  # Interactive mode for missing parameters
```

**Options:**
- `--template, -t`: African plant template ID (1-22)
- `--feedstock, -f`: Feedstock type
- `--capacity, -c`: Production capacity (tons/year)
- `--location, -l`: Plant location
- `--hours`: Operating hours per year
- `--output, -o`: Save results to JSON file
- `--interactive/--no-interactive`: Interactive mode for missing parameters
- `--verbose, -v`: Verbose logging

### 📊 System Statistics

**Comprehensive system monitoring and analytics:**
```bash
./pynucleus stats --mode menu          # Interactive menu
./pynucleus stats --mode system --live # Live system dashboard
./pynucleus stats --mode chat          # Chat mode analysis
./pynucleus stats --mode metrics       # RAG metrics analysis
```

**Options:**
- `--mode, -m`: Statistics mode: `menu`, `system`, `chat`, `metrics` (default: `menu`)
- `--output, -o`: Save results to JSON file
- `--live, -l`: Live updating dashboard
- `--hours, -h`: Hours of historical data to analyze (default: 24)
- `--verbose, -v`: Verbose logging

### 🔄 Auto-Ingest Documents

**Automatically ingest new documents with file watching:**
```bash
./pynucleus auto-ingest --watch-dir data/01_raw/source_documents
./pynucleus auto-ingest --types .pdf .txt .md --recursive
```

**Options:**
- `--watch-dir, -w`: Directory to watch for new files (default: `data/01_raw/source_documents`)
- `--types, -t`: File extensions to monitor (default: `.pdf`, `.txt`, `.md`)
- `--recursive, -r`: Watch subdirectories recursively
- `--daemon, -d`: Run as background daemon
- `--verbose, -v`: Verbose logging

### ❓ Single Question Answering

**Ask a single question to the RAG system:**
```bash
./pynucleus ask "What is chemical engineering?"
./pynucleus ask "Explain distillation" --model Qwen/Qwen2.5-1.5B-Instruct --top-k 10
```

**Options:**
- `question`: Question to ask the RAG system (required)
- `--pretty/--plain`: Use enhanced formatting (default: `pretty`)
- `--model, -m`: LLM model ID (default: `Qwen/Qwen2.5-1.5B-Instruct`)
- `--top-k, -k`: Number of results to retrieve (default: 5)
- `--verbose, -v`: Verbose logging

### 🧪 Evaluation and Testing

**Run golden dataset evaluation:**
```bash
./pynucleus eval golden --threshold 0.8
```

**Compute RAG retrieval metrics:**
```bash
./pynucleus eval metrics --retrieved retrieved.txt --relevant relevant.txt --k 5
```

**Options for golden evaluation:**
- `--threshold, -t`: Minimum passing threshold (default: 0.8)
- `--verbose, -v`: Verbose logging

**Options for metrics computation:**
- `--retrieved, -r`: File with retrieved document IDs (required)
- `--relevant, -g`: File with ground truth relevant IDs (required)
- `--k`: Number of top documents to consider (default: 5)
- `--output, -o`: Output file for metrics
- `--verbose, -v`: Verbose logging

### 🌐 Web Server Management

**Start the web server:**
```bash
./pynucleus serve start --port 5001 --workers 4
./pynucleus serve start --reload  # Development mode with auto-reload
```

**Stop the web server:**
```bash
./pynucleus serve stop --port 5001
```

**Restart the web server:**
```bash
./pynucleus serve restart --port 5001
```

**Options for start:**
- `--port, -p`: Server port (default: 5001)
- `--host`: Server host (default: `0.0.0.0`)
- `--workers, -w`: Number of worker processes (default: 1)
- `--reload`: Enable auto-reload for development
- `--verbose, -v`: Verbose logging

**Options for stop/restart:**
- `--port, -p`: Server port (default: 5001)
- `--verbose, -v`: Verbose logging

### 🩺 System Diagnostics

**Run system health checks:**
```bash
./pynucleus diagnostics
./pynucleus diagnostics --comprehensive --output report.json
```

**Options:**
- `--comprehensive, -c`: Run comprehensive system diagnostics
- `--output, -o`: Save diagnostics report
- `--verbose, -v`: Verbose logging

### 📚 Document Ingestion

**Ingest documents into the RAG system:**
```bash
./pynucleus ingest --source data/01_raw
./pynucleus ingest --source data/01_raw --extract-tables --backend chroma
```

**Options:**
- `--source, -s`: Source directory (default: `data/01_raw`)
- `--extract-tables/--skip-tables`: Extract PDF tables (default: `extract-tables`)
- `--backend, -b`: Vector store backend (default: `chroma`)
- `--verbose, -v`: Verbose logging

### 📋 Version Information

**Show PyNucleus version and system info:**
```bash
./pynucleus version
```

## Error Handling

The CLI includes robust error handling with:

- **Graceful error messages** with rich formatting
- **Proper exit codes** for different error types
- **Verbose mode** for debugging with `--verbose`
- **Logging** to both console and files
- **Keyboard interrupt handling** (Ctrl+C)

## Logging

All commands support logging with:

- **Console output** with rich formatting
- **File logging** to `logs/pynucleus_*.log`
- **Verbose mode** for detailed debugging
- **Custom log file** support with `--log-file`

## Examples

### Complete Workflow Example

```bash
# 1. Check system health
./pynucleus diagnostics

# 2. Ingest documents
./pynucleus ingest --source data/01_raw --extract-tables

# 3. Run pipeline
./pynucleus run --config configs/production_config.json --verbose

# 4. Start auto-ingest for new documents
./pynucleus auto-ingest --watch-dir data/01_raw/source_documents &

# 5. Start web server
./pynucleus serve start --port 5001 --workers 4

# 6. Monitor system performance
./pynucleus stats --mode system --live

# 7. Chat with the system
./pynucleus chat --model Qwen/Qwen2.5-1.5B-Instruct
```

### Development Workflow

```bash
# Development mode with auto-reload
./pynucleus serve start --reload --port 5001

# Test specific functionality
./pynucleus ask "What is the purpose of a heat exchanger?" --verbose

# Run evaluation
./pynucleus eval golden --threshold 0.8

# Monitor performance
./pynucleus stats --mode metrics
```

## Dependencies

The CLI requires:

- **Python 3.8+**
- **Typer** for CLI framework
- **Rich** for beautiful terminal output
- **Watchdog** for auto-ingest functionality
- **Gunicorn** for web server management

Install with:
```bash
pip install typer[all] rich watchdog gunicorn
```

## Configuration

The CLI uses the same configuration system as the main PyNucleus application:

- **Configuration files** in `configs/` directory
- **Environment variables** for sensitive settings
- **Logging configuration** in `configs/logging.yaml`

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running from the project root
2. **Permission errors**: Check file/directory permissions
3. **Port conflicts**: Use `--port` to specify different ports
4. **Missing dependencies**: Install required packages

### Debug Mode

Use `--verbose` flag for detailed debugging:
```bash
./pynucleus run --config config.json --verbose
```

### Log Files

Check log files in `logs/` directory for detailed error information:
```bash
tail -f logs/pynucleus_*.log
```

## Contributing

To add new commands to the CLI:

1. Add the command function to `src/pynucleus/cli.py`
2. Use the `@app.command()` decorator
3. Add proper error handling with `@handle_errors`
4. Include comprehensive help text
5. Test with `--help` and `--verbose` flags

## License

This CLI is part of the PyNucleus project and follows the same license terms. 