# PyNucleus Model

A Python-based system for chemical process simulation and RAG (Retrieval-Augmented Generation) integration.

## 🚀 Features

- DWSIM Chemical Process Simulation Integration
- RAG System for Document Processing
- Vector Database for Efficient Retrieval
- Docker-based Deployment
- Comprehensive Testing Suite

## 📋 Prerequisites

- Python 3.10+
- Docker and Docker Compose
- DWSIM Chemical Process Simulator
- Required Python packages (see requirements.txt)

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PyNucleus-Model.git
   cd PyNucleus-Model
   ```

2. Set up Python environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up DWSIM:
   - Copy DWSIM DLLs to `dwsim_libs/` directory
   - Set environment variable:
     ```bash
     export DWSIM_DLL_PATH="$(pwd)/dwsim_libs"
     ```

## 🏗️ Project Structure

```
PyNucleus-Model/
├── data/
│   ├── raw/              # Raw input documents
│   ├── processed/        # Processed text files
│   ├── vector_store/     # Vector database files
│   └── analysis/         # Analysis results
├── src/
│   ├── rag/             # RAG system components
│   ├── simulation/      # DWSIM integration
│   └── utils/           # Utility functions
├── tests/
│   ├── rag/             # RAG system tests
│   └── simulation/      # Simulation tests
├── scripts/             # Utility scripts
├── dwsim_libs/          # DWSIM DLL files
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🚀 Usage

1. Run system diagnostics:
   ```bash
   python scripts/system_diagnostic.py
   ```

2. Run with Docker:
   ```bash
   docker-compose up
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

## 🧪 Testing

- System diagnostics: `python scripts/system_diagnostic.py`
- DWSIM integration: `python scripts/test_dwsim_integration.py`
- RAG system: `pytest tests/rag/`

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 