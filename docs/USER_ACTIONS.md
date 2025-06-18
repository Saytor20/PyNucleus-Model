# PyNucleus User Actions Guide

This guide provides step-by-step instructions for setting up and running the PyNucleus system.

## Prerequisites

- Python 3.8+ installed
- Git installed
- (Optional) Docker installed for containerized deployment

## Initial Setup

### 1. Export API Key

The PyNucleus system requires an API key for secure access. Set your API key as an environment variable:

**Linux/macOS:**
```bash
export PYNUCLEUS_API_KEY="your-secure-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set PYNUCLEUS_API_KEY=your-secure-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:PYNUCLEUS_API_KEY = "your-secure-api-key-here"
```

**Alternative: .env File (Recommended)**

Create a `.env` file in the project root:
```bash
echo "PYNUCLEUS_API_KEY=your-secure-api-key-here" > .env
```

Additional optional environment variables:
```bash
# Device configuration (cpu, cuda, auto)
PYNUCLEUS_DEVICE=auto

# OpenAI API key (if using OpenAI models)
OPENAI_API_KEY=your-openai-key

# HuggingFace API key (if needed)
HUGGINGFACE_API_KEY=your-hf-key

# Custom model configurations
PYNUCLEUS_DEFAULT_MODEL=microsoft/DialoGPT-large
PYNUCLEUS_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 2. Install Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Build Your First Index

Before using the RAG system, you need to build a document index from your data:

**Basic Index Build:**
```bash
# Using CLI
python -m pynucleus.cli build-index --data-dir data/01_raw --output-dir data/04_models

# Using the pipeline directly
python run_pipeline.py --mode rag --data-dir data/01_raw
```

**Advanced Index Build with Custom Settings:**
```bash
python -m pynucleus.cli build-index \
    --data-dir data/01_raw \
    --output-dir data/04_models \
    --chunk-size 1000 \
    --chunk-overlap 200 \
    --embedding-model all-MiniLM-L6-v2
```

## Running the System

### 4. Start the Flask Server

Use the provided bash script to start the server:

```bash
# Make the script executable
chmod +x scripts/run_flask.sh

# Run the server
./scripts/run_flask.sh
```

The server will start on `http://localhost:5000`. You should see output like:
```
Starting PyNucleus Flask API server...
Server will be available at: http://localhost:5000
Health check: http://localhost:5000/health

* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:5000
* Running on http://[::1]:5000
```

### 5. Open Browser and Test

**Health Check:**
Open your browser and navigate to: `http://localhost:5000/health`

You should see:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "service": "PyNucleus API"
}
```

**Web Interface:**
Navigate to: `http://localhost:5000`

This will load the browser-based UI for asking questions.

**API Testing:**
You can test the API using curl:
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secure-api-key-here" \
  -d '{"question": "What is the optimal temperature for chemical process X?"}'
```

## Command Line Interface

### Common CLI Commands

**Ask a Question:**
```bash
python -m pynucleus.cli ask "What is the optimal reactor temperature?"
```

**Run Full Pipeline:**
```bash
python -m pynucleus.cli pipeline --mode integrated --question "Your question here"
```

**System Diagnostics:**
```bash
python -m pynucleus.cli diagnose
```

**Validate System:**
```bash
python scripts/system_validator.py
```

## Docker Deployment (Optional)

For production or isolated environments, you can use Docker:

### Build Docker Image
```bash
docker build -t pynucleus .
```

### Run with Docker Compose
```bash
# Ensure your .env file is configured
docker-compose up -d
```

### Manual Docker Run
```bash
docker run -d \
  --name pynucleus-api \
  -p 5000:5000 \
  -e PYNUCLEUS_API_KEY=your-secure-api-key-here \
  -e PYNUCLEUS_DEVICE=cpu \
  -v $(pwd)/data:/app/data \
  pynucleus
```

## Troubleshooting

### Common Issues

1. **API Key Not Set:**
   - Error: `API key authentication not configured`
   - Solution: Ensure `PYNUCLEUS_API_KEY` is set in environment or `.env` file

2. **Port Already in Use:**
   - Error: `Address already in use`
   - Solution: Change port in `.env`: `PYNUCLEUS_PORT=5001`

3. **No Index Found:**
   - Error: `No index found` or similar
   - Solution: Build index first using `build-index` command

4. **CUDA Issues:**
   - Error: CUDA-related errors on CPU systems
   - Solution: Set `PYNUCLEUS_DEVICE=cpu` in `.env`

### Get Help

- Check logs in the `logs/` directory
- Run system diagnostics: `python scripts/comprehensive_system_diagnostic.py`
- View detailed documentation in `docs/COMPREHENSIVE_SYSTEM_DOCUMENTATION.md`

## Next Steps

1. **Add Your Documents**: Place PDF, TXT, or DOCX files in `data/01_raw/`
2. **Rebuild Index**: Run `build-index` command after adding new documents
3. **Configure Models**: Update `.env` with your preferred LLM models
4. **Production Setup**: Use Docker for production deployments
5. **Monitor Performance**: Check logs and use validation scripts

---

For more advanced configuration and development information, see the comprehensive documentation in the `docs/` directory. 