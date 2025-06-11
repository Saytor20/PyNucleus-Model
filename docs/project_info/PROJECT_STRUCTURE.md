# PyNucleus-Model Project Structure

## 📁 Directory Organization

```
PyNucleus-Model/
├── core_modules/                # Main Python package
│   ├── __init__.py             # Package initialization
│   ├── pipeline/               # Core pipeline components
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py     # RAG pipeline orchestration
│   │   ├── dwsim_pipeline.py   # DWSIM simulation pipeline
│   │   ├── results_exporter.py # CSV export functionality
│   │   └── pipeline_utils.py   # Complete pipeline orchestration
│   ├── rag/                    # RAG pipeline components
│   │   ├── __init__.py
│   │   ├── wiki_scraper.py     # Wikipedia article scraper
│   │   ├── document_processor.py # Document conversion
│   │   ├── data_chunking.py    # Text chunking
│   │   ├── vector_store.py     # FAISS vector store
│   │   └── performance_analyzer.py # Evaluation metrics
│   ├── integration/            # Enhanced pipeline integration
│   │   ├── __init__.py
│   │   ├── config_manager.py   # Configuration management
│   │   ├── dwsim_rag_integrator.py # DWSIM-RAG integration
│   │   └── llm_output_generator.py # Enhanced LLM output
│   ├── simulation/             # DWSIM simulation components
│   │   ├── __init__.py
│   │   └── dwsim_bridge.py     # DWSIM interface
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   └── file_utils.py       # File handling utilities
│   └── tests/                  # Unit tests
│       ├── __init__.py
│       └── test_*.py           # Test files
│
├── source_documents/           # Original documents (PDF, DOCX, TXT)
├── converted_to_txt/          # Converted text files
├── web_sources/               # Scraped Wikipedia articles
├── converted_chunked_data/    # Chunked documents for vector store
├── chunk_reports/            # FAISS analysis reports & vector store
├── results/                  # Standard pipeline results (CSV files)
├── llm_reports/              # Enhanced LLM output files (TXT, MD)
├── simulation_input_config/  # Configuration templates & files
├── automation_tools/         # Helper scripts
├── project_info/             # Project documentation
├── scripts/                  # Utility scripts
├── dwsim_libs/              # DWSIM integration files
└── .venv/                   # Python virtual environment

## 📝 Directory Purposes

### Core Application (`core_modules/`)
- **pipeline/**: Core pipeline orchestration (RAG + DWSIM + Export)
- **rag/**: RAG components (document processing, vector store, retrieval)
- **integration/**: Enhanced pipeline features (configuration, LLM output)
- **simulation/**: DWSIM integration and simulation management
- **utils/**: Shared utility functions
- **tests/**: Unit tests for all components

### Data Processing Directories
- `source_documents/`: Original input files (PDF, DOCX, TXT)
- `converted_to_txt/`: Text files converted from source documents
- `web_sources/`: Wikipedia articles and other web content
- `converted_chunked_data/`: Chunked text data ready for vector indexing

### Output Directories
- `results/`: Standard pipeline outputs (CSV files with simulation results)
- `llm_reports/`: Enhanced LLM outputs with detailed feed conditions
- `chunk_reports/`: FAISS index files and processing analysis reports

### Configuration & Setup
- `simulation_input_config/`: Configuration templates and simulation setups
- `automation_tools/`: Helper scripts for automation tasks
- `scripts/`: Utility scripts for various operations
- `project_info/`: Project documentation and structure info

### System Directories
- `dwsim_libs/`: DWSIM simulation integration files
- `.venv/`: Python virtual environment
- `.git/`: Git version control

## 🔄 Enhanced Data Flow

### Basic Pipeline Flow
1. **Input**: Documents → `source_documents/`
2. **Conversion**: Process → `converted_to_txt/`
3. **Web Content**: Scrape → `web_sources/`
4. **Chunking**: Combine & chunk → `converted_chunked_data/`
5. **Vector Store**: Index → `chunk_reports/` (FAISS files)
6. **Simulation**: DWSIM processing → simulation results
7. **Export**: Standard results → `results/` (CSV files)

### Enhanced Pipeline Flow
1. **Configuration**: Templates & configs → `simulation_input_config/`
2. **Integration**: DWSIM + RAG enhanced analysis
3. **LLM Output**: Enhanced reports → `llm_reports/` (TXT, MD files)
   - Detailed feed conditions with mole fractions
   - Flow rates (kmol/hr), temperatures (°C), pressures (kPa)
   - Component-specific mass flow rates
   - Process operating conditions (reflux ratios, residence times)
   - Performance metrics, conversion, selectivity, yield percentages
   - Financial analysis with ROI calculations

## Enhanced Pipeline Features

### 1. Separate Output Directories
- **Standard Results**: `results/` → CSV files for data analysis
- **LLM Reports**: `llm_reports/` → Human-readable enhanced summaries

### 2. Enhanced Content Generation
- **Feed Conditions**: Total feed rates, temperatures, pressures
- **Component Breakdown**: Mole fractions and mass flow rates
- **Operating Conditions**: Process-specific parameters
- **Performance Metrics**: Conversion, selectivity, yield percentages
- **Financial Analysis**: Recovery rates, ROI, profit calculations

### 3. Configuration Management
- **Smart Templates**: Only create if they don't exist
- **JSON & CSV Support**: Flexible configuration formats
- **Simulation Setup**: Pre-configured simulation parameters

### 4. Integration Capabilities
- **DWSIM-RAG Integration**: Enhanced analysis with knowledge insights
- **Performance Analytics**: Automated performance evaluation
- **Optimization Recommendations**: AI-powered suggestions

## Testing Structure

- **Location**: `core_modules/tests/`
- **Coverage**: All pipeline components tested
- **Types**: Unit tests, integration tests, functionality tests
- **Naming**: `test_*.py` pattern for easy discovery

## Development Workflow

### Standard Pipeline Usage
1. Run basic pipeline → generates CSV results in `results/`
2. Use for data analysis and standard processing

### Enhanced Pipeline Usage
1. Initialize enhanced components
2. Configure simulation templates
3. Run DWSIM-RAG integration
4. Generate enhanced LLM reports → `llm_reports/`
5. Access detailed feed conditions and financial analysis

## Monitoring & Analysis

### Processing Reports
- **Location**: `chunk_reports/`
- **Contents**: FAISS analysis, vector store performance
- **Format**: Text logs with timestamps

### System Diagnostics
- **Script**: `comprehensive_system_diagnostic.py`
- **Coverage**: Environment, directories, components, functionality
- **Health Check**: 100% system operational status

### Enhanced Analytics
- **Financial Metrics**: ROI, recovery rates, profit calculations
- **Performance Tracking**: Conversion, selectivity, yield monitoring
- **Process Optimization**: Automated recommendations and insights

## Key Enhancements Implemented

1. ** Separate LLM Folder**: LLM outputs in dedicated `llm_reports/` directory
2. ** Enhanced Content**: Detailed feed conditions with mole fractions, flow rates, temperatures, pressures
3. ** Folder Rename**: Configuration folder renamed to `simulation_input_config/`
4. ** Smart Templates**: Intelligent template creation and management
5. ** Enhanced Integration**: DWSIM-RAG integration with performance analytics
6. ** Financial Analysis**: Comprehensive ROI and profit calculations 