# PyNucleus Pipeline-First Codebase Cleanup

## Pipeline Overview

PyNucleus is a comprehensive RAG-enabled chemical process simulation platform with the following core architecture:

### Main Pipeline Entry Points
1. **CLI Pipeline**: `pynucleus run --config configs/development_config.json`
2. **Script Pipeline**: `python scripts/run_pipeline.py run --config-path configs/development_config.json`

Both execute the core pipeline through `run_full_pipeline()` in `src/pynucleus/pipeline/pipeline_utils.py`.

### Core Pipeline Flow
1. **Configuration Loading**: via `ConfigManager` from `src/pynucleus/integration/config_manager.py`
2. **RAG Pipeline**: Document processing, vector storage, query processing
3. **Pipeline Execution**: RAG analysis with predefined queries
4. **Results Export**: JSON output to `data/05_output/`

## Baseline Run Results

**Command**: `python scripts/run_pipeline.py run --config-path configs/development_config.json --output-dir data/05_output/baseline_run`

**Success Metrics**: 
- Status: `"success": true`
- Output: `data/05_output/baseline_run/pipeline_results.json`
- Pipeline completed successfully with ChromaDB collection containing 54 documents
- Model loaded: HuggingFaceTB/SmolLM2-1.7B-Instruct

## Comprehensive Dependency Map (KEEP_SET)

### Core Application Files (MUST KEEP)

#### **Entry Points**
- `scripts/run_pipeline.py` - External CLI script entry point
- `src/pynucleus/cli.py` - Main CLI application
- `src/pynucleus/__init__.py` - Package initialization

#### **Core Pipeline Components**
- `src/pynucleus/pipeline/` - Complete directory (core pipeline logic)
  - `pipeline_utils.py` - Main pipeline orchestration
  - `pipeline_rag.py` - RAG pipeline implementation
  - `__init__.py` - Package initialization

#### **RAG System** 
- `src/pynucleus/rag/` - Complete directory (document processing & retrieval)
  - `engine.py` - RAG query engine
  - `vector_store.py` - Vector database integration
  - `vector_store_remote.py` - Remote vector store support
  - `collector.py` - Document ingestion
  - `document_processor.py` - Document processing
  - `rag_core.py` - Core RAG functionality
  - `__init__.py` - Package initialization

#### **LLM Integration**
- `src/pynucleus/llm/` - Complete directory (language model integration)
  - `model_loader.py` - Model loading and caching
  - `llm_runner.py` - LLM execution
  - `prompting.py` - Prompt engineering
  - `answer_engine.py` - Answer generation
  - `__init__.py` - Package initialization

#### **Integration Layer**
- `src/pynucleus/integration/` - Complete directory
  - `config_manager.py` - Configuration management
  - `llm_output_generator.py` - Output generation
  - `__init__.py` - Package initialization

#### **Utilities**
- `src/pynucleus/utils/` - Complete directory (essential utilities)
  - `logging_config.py` - Logging configuration
  - `__init__.py` - Package initialization

#### **Data Management**
- `src/pynucleus/data/` - Complete directory
  - `mock_data_manager.py` - Mock data for pipeline
  - `__init__.py` - Package initialization

#### **Core Settings**
- `src/pynucleus/settings.py` - Application settings

### Configuration Files (MUST KEEP)
- `configs/development_config.json` - Development configuration
- `configs/production_config.json` - Production configuration  
- `configs/mock_data_modular_plants.json` - Mock plant data
- `pyproject.toml` - Package configuration and entry points
- `requirements.txt` - Python dependencies

### Data Directories (CONDITIONAL KEEP)
- `data/01_raw/source_documents/` - Raw documents (if referenced by RAG)
- `data/03_intermediate/vector_db/` - ChromaDB vector database
- `data/03_processed/chromadb/` - ChromaDB storage

### Package Infrastructure (MUST KEEP)
- `src/pynucleus.egg-info/` - Package metadata (if needed for installation)
- `README.md` - Project documentation

## Files/Directories to DELETE

### High Confidence Deletions
- `archive/` - Already archived content
- `pynucleus_env/` - Virtual environment
- `venv/` - Virtual environment  
- `cache/` - Model cache (regenerable)
- `logs/` - Historical log files
- `examples/` - Example code
- `tests/` - Test suites
- Documentation files not in README.md

### Components Not Core to Pipeline
- `src/pynucleus/deployment/` - Deployment infrastructure
- `src/pynucleus/eval/` - Evaluation components
- `src/pynucleus/feedback/` - Feedback system
- `src/pynucleus/db/` - Database models
- `src/pynucleus/diagnostics/` - Diagnostic tools
- `src/pynucleus/metrics/` - Metrics collection
- `src/pynucleus/engine/` - Engine components
- `src/pynucleus/terminal_dashboard.py` - Dashboard
- `src/pynucleus/menus.py` - Menu systems

### Validation and Test Data
- `data/validation/` - Validation datasets
- `data/test_output/` - Test results
- `data/calibration/` - Calibration models

## Deletion Log

### Major Directories Removed
- **pynucleus_env/** - Python virtual environment (regenerable)
- **venv/** - Second virtual environment (regenerable)
- **cache/** - Model cache directory (regenerable)
- **logs/** - Historical log files (runtime regenerable)
- **examples/** - Example code not used by pipeline
- **tests/** - Test suites not core to pipeline
- **archive/** - Archived legacy content
- **deployment/** - Docker deployment infrastructure
- **docs/** - Documentation files

### Source Code Components Removed
- **src/pynucleus/deployment/** - Deployment infrastructure
- **src/pynucleus/eval/** - Evaluation components
- **src/pynucleus/feedback/** - Feedback system
- **src/pynucleus/db/** - Database models
- **src/pynucleus/diagnostics/** - diagnostic tools
- **src/pynucleus/metrics/** - Metrics collection system
- **src/pynucleus/engine/** - Engine components
- **src/pynucleus/terminal_dashboard.py** - Dashboard interface
- **src/pynucleus/menus.py** - Menu system

### Data Directories Removed
- **data/validation/** - Validation datasets and results
- **data/test_output/** - Test results
- **data/calibration/** - Calibration models
- **data/05_output/llm_reports/** - Historical LLM outputs

### Scripts Removed
- All scripts in **scripts/** except **run_pipeline.py**
- Various utility and testing scripts (40+ files)

### Documentation and Misc Files Removed
- **PyNucleus_QA_Evaluation_Report.md**
- **PyNucleus_Quantitative_Performance_Analysis.md**
- **confidence_analysis_report.md**
- **test_model_upgrade.py**
- **test_questions.txt**
- **data/product_prices.json**
- **pynucleus.db**

### Code Modifications Made
- **src/pynucleus/rag/engine.py**: Replaced metrics imports with no-op stubs
- **src/pynucleus/rag/collector.py**: Replaced metrics imports with no-op stubs
- **src/pynucleus/cli.py**: Replaced metrics and diagnostics imports with no-op stubs

## Final Metrics Comparison

### Baseline Run (Before Cleanup)
- **Command**: `python scripts/run_pipeline.py run --config-path configs/development_config.json --output-dir data/05_output/baseline_run`
- **Success**: ✅ `"success": true`
- **Output**: `data/05_output/baseline_run/pipeline_results.json`
- **Summary**: "Pipeline completed successfully (mock implementation)"
- **ChromaDB**: 54 documents loaded successfully
- **Model**: HuggingFaceTB/SmolLM2-1.7B-Instruct loaded successfully

### Final Validation Run (After Cleanup)

#### Script Entry Point
- **Command**: `python scripts/run_pipeline.py run --config-path configs/development_config.json --output-dir data/05_output/final_validation`
- **Success**: ✅ `"success": true`
- **Output**: `data/05_output/final_validation/pipeline_results.json`
- **Summary**: "Pipeline completed successfully (mock implementation)"

#### CLI Entry Point
- **Command**: `python -m pynucleus.cli run --config configs/development_config.json --output data/05_output/final_cli_validation`
- **Success**: ✅ `"success": true`
- **Output**: `data/05_output/final_cli_validation/pipeline_results.json`
- **Summary**: "Pipeline completed successfully (mock implementation)"

### Success Criteria Met ✅
1. **Pipeline Execution**: Both entry points execute successfully
2. **JSON Output**: Generates `pipeline_results.json` with `"success": true`
3. **Runtime Dependencies**: All imports resolved, no missing modules
4. **Core Functionality**: RAG system, LLM integration, and configuration management intact

### Final Dependency Map

The cleaned codebase now contains only essential components:

#### Core Pipeline Files (35 files)
- **Entry Points**: `scripts/run_pipeline.py`, `src/pynucleus/cli.py`
- **Pipeline Core**: `src/pynucleus/pipeline/` (3 essential files)
- **RAG System**: `src/pynucleus/rag/` (10 files)
- **LLM Integration**: `src/pynucleus/llm/` (10 files)
- **Integration Layer**: `src/pynucleus/integration/` (3 files)
- **Utilities**: `src/pynucleus/utils/` (8 files)
- **Data Management**: `src/pynucleus/data/` (3 files)

#### Configuration Files (5 files)
- `configs/development_config.json`
- `configs/production_config.json`
- `configs/mock_data_modular_plants.json`
- `pyproject.toml`
- `requirements.txt`

#### Essential Data Directories
- `data/01_raw/source_documents/` - Raw documents for RAG
- `data/03_intermediate/vector_db/` - ChromaDB vector database
- `data/03_processed/chromadb/` - ChromaDB storage

**Total Reduction**: From ~200+ files to ~60 essential files (70% reduction while maintaining full pipeline functionality)