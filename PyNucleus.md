# PyNucleus System Architecture & Documentation

*Comprehensive System Reference - Version 2.0.0*

---

## 🏗️ System Overview

**PyNucleus** is a production-ready Chemical Process Simulation & RAG (Retrieval-Augmented Generation) System designed specifically for African industrial markets. It combines advanced language models with chemical engineering expertise to provide intelligent question-answering, plant design simulation, and economic analysis capabilities.

### System Status
- **Health Score:** 100% ✅
- **Architecture:** CLI-First Production System
- **Target Markets:** African Industrialization
- **Knowledge Base:** 54+ Technical Documents Indexed

---

## 🧠 Language Model (LLM) Component

### Primary Model Configuration
- **Model:** `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Performance:** Optimized for chemical engineering tasks
- **Size:** 1.7B parameters - balanced performance/resource usage
- **Quantization:** 8-bit GPU quantization for memory efficiency
- **Context Window:** 8192 tokens maximum

### Fallback Models
- **Secondary:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Legacy Support:** Qwen1.5-1.8B-Chat (cached)
- **Auto-detection:** Intelligent model switching based on availability

### Model Settings
```python
# Optimized for consistency and quality
TEMPERATURE: 0.2          # Lower for consistent outputs
TOP_P: 0.85               # Focused probability sampling
MAX_TOKENS: 200           # Standard response length
MAX_TOKENS_COMPLEX: 250   # Extended for complex queries
REPETITION_PENALTY: 1.15  # Prevent repetitive responses
```

### GPU & Hardware Support
- **CUDA Detection:** Automatic GPU utilization when available
- **Fallback:** CPU-only mode for compatibility
- **Memory Management:** Dynamic quantization and optimization
- **Device Manager:** `src/pynucleus/llm/device_manager.py`

---

## 🔍 RAG (Retrieval-Augmented Generation) Engine

### Vector Database
- **Backend:** ChromaDB (persistent storage)
- **Location:** `data/03_intermediate/vector_db/chroma.sqlite3`
- **Documents Indexed:** 54 technical documents
- **Storage Format:** SQLite with HNSW indexing

### Embedding Models
- **Primary:** `BAAI/bge-small-en-v1.5` (high-performance)
- **Fallback:** `all-MiniLM-L6-v2` (compatibility)
- **Embedding Dimensions:** 384 (bge-small-en-v1.5)

### Retrieval Configuration
```python
# Optimized for ≥90% retrieval recall
CHUNK_SIZE: 300                    # Optimal for relevance
CHUNK_OVERLAP: 50                  # Reduced for cleaner chunks
RETRIEVE_TOP_K: 3                  # Top 3 most relevant chunks
RAG_SIMILARITY_THRESHOLD: 0.05     # Low threshold for better recall
MAX_CONTEXT_CHUNKS: 3              # Limit context to top chunks
```

### Knowledge Base Content
- **Chemical Engineering:** Process design, thermodynamics, mass transfer
- **African Industrialization:** Economic factors, infrastructure challenges
- **Modular Plants:** 22+ plant templates and configurations
- **Safety & Regulations:** Process safety management guidelines
- **Wikipedia Articles:** Curated chemical engineering concepts

---

## 🏭 Chemical Process Simulation

### Plant Builder System
- **Templates:** 22+ modular chemical plant configurations
- **Feedstock Types:** Natural gas, crude oil, biomass, etc.
- **Economic Analysis:** African market-specific cost factors
- **Location Factors:** Country-specific economic multipliers

### Plant Configuration Options
```python
# Available plant templates (examples)
1. Ammonia Production Plant
2. Methanol Synthesis Unit
3. Ethylene Production Facility
4. Distillation Column Systems
5. Heat Exchanger Networks
...22 total configurations
```

### Economic Analysis Features
- **Capital Cost Estimation:** Equipment and installation costs
- **Operating Cost Analysis:** Utilities, labor, maintenance
- **Location Factors:** Nigeria, Ghana, Kenya, etc.
- **Financial Metrics:** ROI, payback period, NPV calculations

### Operational Parameters
- **Capacity Range:** 100 - 100,000 tons/year
- **Operating Hours:** 4,000 - 8,760 hours/year
- **Efficiency Factors:** Location-specific adjustments
- **Risk Assessment:** Quantitative risk analysis

---

## 💻 CLI Interface

### Core Commands
```bash
# Interactive chat with RAG system
pynucleus chat [--model MODEL] [--temperature TEMP]

# Chemical plant simulation
pynucleus build [--template ID] [--feedstock TYPE] [--capacity NUM]

# Full pipeline execution
pynucleus run [--config FILE]

# System health monitoring
pynucleus health [quick|full]

# Document processing
pynucleus ingest [auto|manual] [--source-dir DIR]

# RAG system operations
pynucleus rag [status|reset|reindex]

# System statistics
pynucleus stats [--mode system|rag|models]
```

### Interactive Features
- **Streaming Responses:** Real-time typewriter effect
- **Rich Formatting:** Colored output with progress bars
- **Auto-completion:** Command and parameter suggestions
- **Error Handling:** Comprehensive error reporting and recovery

### Configuration Management
- **Environment Files:** `.env` support with templates
- **JSON Configs:** Development and production configurations
- **Runtime Settings:** Dynamic parameter adjustment

---

## 📁 Directory Structure & Purpose

### Root Directory
```
PyNucleus-Model/
├── README.md                    # Quick start guide
├── PyNucleus.md                # This comprehensive documentation
├── pyproject.toml              # Python package configuration
├── requirements.txt            # Production dependencies
├── requirements_web.txt        # Web interface dependencies (legacy)
└── .env                        # Environment configuration
```

### Source Code (`src/pynucleus/`)
```
src/pynucleus/
├── cli.py                      # Main CLI interface
├── settings.py                 # System configuration
├── menus.py                   # Interactive menu systems
│
├── llm/                       # Language Model Components
│   ├── model_loader.py        # Model loading and management
│   ├── device_manager.py      # GPU/CPU device handling
│   ├── answer_engine.py       # Response generation
│   ├── prompting.py           # Prompt engineering
│   ├── llm_runner.py          # Model execution engine
│   ├── query_llm.py           # Query processing
│   ├── qwen_loader.py         # Qwen model specific loader
│   ├── retriever_adapter.py   # RAG integration adapter
│   └── simple_local_adapter.py # Local model adapter
│
├── rag/                       # Retrieval-Augmented Generation
│   ├── engine.py              # Main RAG engine
│   ├── rag_core.py            # Core RAG functionality
│   ├── vector_store.py        # Vector database operations
│   ├── document_processor.py  # Document ingestion
│   ├── answer_processing.py   # Response post-processing
│   ├── collector.py           # Data collection utilities
│   ├── embedding_monitor.py   # Embedding performance monitoring
│   ├── wiki_scraper.py        # Wikipedia content scraper
│   └── vector_store_remote.py # Remote vector store support
│
├── pipeline/                  # Processing Pipeline
│   ├── plant_builder.py       # Chemical plant simulation
│   ├── pipeline_rag.py        # RAG pipeline orchestration
│   ├── financial_analyzer.py  # Economic analysis
│   ├── feedstock_validator.py # Input validation
│   ├── location_factor_analyzer.py # Geographic cost factors
│   ├── operational_hours_validator.py # Operating schedule validation
│   ├── quantitative_risk_assessor.py # Risk analysis
│   ├── economic_benchmarking.py # Benchmark comparisons
│   ├── expert_review_system.py # Quality assurance
│   ├── comprehensive_validation_system.py # System validation
│   ├── results_exporter.py    # Output formatting
│   └── pipeline_utils.py      # Utility functions
│
├── data/                      # Data Management
│   ├── mock_data_manager.py   # Plant template management
│   └── table_cleaner.py       # Data cleaning utilities
│
├── eval/                      # Evaluation & Validation
│   ├── confidence_calibration.py # Confidence scoring
│   ├── semantic_validation.py # Answer quality validation
│   ├── golden_eval.py         # Golden dataset evaluation
│   ├── train_confidence.py    # Confidence training
│   └── validation_manager.py  # Validation orchestration
│
├── metrics/                   # Performance Monitoring
│   ├── system_statistics.py   # System performance metrics
│   └── prometheus.py          # Prometheus metrics export
│
├── integration/               # System Integration
│   ├── config_manager.py      # Configuration management
│   └── llm_output_generator.py # Output generation
│
├── diagnostics/               # System Diagnostics
│   └── runner.py              # Diagnostic test runner
│
└── utils/                     # Utilities
    ├── env.py                 # Environment management
    ├── logger.py              # Logging utilities
    ├── logging_config.py      # Logging configuration
    ├── error_handler.py       # Error handling
    ├── pretty_formatter.py    # Output formatting
    ├── telemetry_patch.py     # Telemetry management
    └── token_utils.py         # Token management utilities
```

### Data Pipeline (`data/`)
```
data/
├── 01_raw/                    # Raw Input Data (DVC Stage 1)
│   ├── source_documents/      # PDF/DOC technical documents
│   └── wikipedia/             # Scraped Wikipedia articles
│       ├── chemical_reactor/
│       ├── distillation/
│       ├── heat_exchanger/
│       ├── mass_transfer/
│       └── process_safety/
│
├── 02_processed/              # Processed Data (DVC Stage 2)
│   ├── cleaned_txt/           # Text extracted from PDFs
│   └── tables/                # Structured data tables
│
├── 03_intermediate/           # Vector Database (DVC Stage 3)
│   └── vector_db/             # ChromaDB persistent storage
│       └── chroma.sqlite3     # Vector embeddings (54 documents)
│
├── 04_models/                 # Model Artifacts (DVC Stage 4)
│   └── chunk_reports/         # Processing reports
│
├── 05_output/                 # Results & Reports (DVC Stage 5)
│   ├── pipeline_results.json  # Main pipeline outputs
│   └── baseline_run/          # Baseline comparison data
│
├── backups/                   # Database Backups
│   └── chromadb_*/           # Timestamped backups
│
└── validation_reports/        # System Validation
    └── comprehensive_health_*.json # Health check reports
```

### Configuration (`configs/`)
```
configs/
├── config_template.env           # Environment template
├── development_config.json       # Development settings
├── production_config.json        # Production settings
├── logging.yaml                  # Logging configuration
└── mock_data_modular_plants.json # Plant templates
```

### Scripts & Utilities (`scripts/`)
```
scripts/
├── run_pipeline.py              # Pipeline execution
├── comprehensive_health_check.py # System health validation
├── identify_current_model.py     # Model detection
├── reorganize_directories.py     # Directory reorganization
└── validate_smol_primary.py      # Model validation
```

### Documentation (`docs/`)
```
docs/
├── SYSTEM_OVERVIEW.md           # System architecture overview
├── MODEL_CONFIGURATION.md       # Model setup and configuration
└── WEB_INTERFACE_GUIDE.md      # Web interface documentation (legacy)
```

### Cache & Logs
```
cache/
└── models/                      # Cached model states
    ├── HuggingFaceTB_SmolLM2-1.7B-Instruct_state.pkl
    └── Qwen_Qwen1.5-1.8B-Chat_state.pkl

logs/
├── pynucleus_*.log             # Application logs
└── pynucleus_PyNucleus-Model.log # Main system log
```

---

## ⚙️ Core Functionalities

### 1. Interactive Chat System
- **Real-time Q&A:** Chemical engineering questions with cited sources
- **Streaming Responses:** Live typewriter effect for better UX
- **Context Awareness:** Maintains conversation context
- **Source Attribution:** Automatic citation of knowledge base sources

### 2. Chemical Plant Design
- **Template-based Design:** 22+ pre-configured plant types
- **Economic Analysis:** African market-specific cost modeling
- **Risk Assessment:** Quantitative risk analysis and mitigation
- **Performance Optimization:** Efficiency and capacity optimization

### 3. Document Processing
- **Multi-format Support:** PDF, DOC, TXT document ingestion
- **Intelligent Chunking:** Semantic text segmentation
- **Metadata Extraction:** Technical terms and section indexing
- **Vector Embedding:** High-quality embedding generation

### 4. System Monitoring
- **Health Checks:** Comprehensive system validation
- **Performance Metrics:** Real-time performance monitoring
- **Error Tracking:** Detailed error logging and recovery
- **Resource Monitoring:** Memory, CPU, and GPU utilization

### 5. African Market Focus
- **Location Factors:** Country-specific economic multipliers
- **Infrastructure Considerations:** African industrial challenges
- **Economic Modeling:** Local market conditions and constraints
- **Regulatory Compliance:** Regional safety and environmental standards

---

## 🔧 Configuration Management

### Environment Variables (.env)
```bash
# Model Configuration
MODEL_ID=HuggingFaceTB/SmolLM2-1.7B-Instruct
EMB_MODEL=BAAI/bge-small-en-v1.5
USE_CUDA=true
USE_GPU_QUANTIZATION=true

# Database Configuration
CHROMA_PATH=data/03_intermediate/vector_db

# Performance Settings
MAX_TOKENS=200
TEMPERATURE=0.2
RETRIEVE_TOP_K=3

# Logging
LOG_LEVEL=INFO
```

### Development Configuration
```json
{
  "model_settings": {
    "primary_model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "fallback_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "use_gpu": true,
    "quantization": true
  },
  "rag_settings": {
    "chunk_size": 300,
    "overlap": 50,
    "top_k": 3,
    "similarity_threshold": 0.05
  },
  "pipeline_settings": {
    "enable_caching": true,
    "validation_mode": "comprehensive",
    "output_format": "json"
  }
}
```

### Production Configuration
- Enhanced error handling and recovery
- Optimized memory usage and performance
- Comprehensive logging and monitoring
- Automatic failover and redundancy

---

## 🚀 Performance & Optimization

### Response Times
- **Simple Queries:** 3-5 seconds
- **Complex Analysis:** 8-12 seconds
- **Plant Simulation:** 15-30 seconds
- **Document Ingestion:** 1-3 minutes per document

### Memory Usage
- **Base System:** ~2GB RAM
- **With GPU:** ~4GB VRAM (recommended)
- **Vector Database:** ~500MB for 54 documents
- **Model Cache:** ~2GB for primary model

### Scalability
- **Document Limit:** 1000+ documents supported
- **Concurrent Users:** 10+ simultaneous CLI sessions
- **Vector Database:** Horizontal scaling with ChromaDB
- **Model Inference:** GPU acceleration for faster responses

---

## 🔒 Security & Best Practices

### Security Features
- **No External API Calls:** Fully local operation
- **Data Privacy:** All processing happens locally
- **Secure Storage:** Encrypted vector database storage
- **Access Control:** File system-based security

### Development Practices
- **Type Safety:** Full Python type hints
- **Error Handling:** Comprehensive exception management
- **Testing:** Automated validation and health checks
- **Documentation:** Inline code documentation

### Monitoring & Telemetry
- **Health Monitoring:** Continuous system health checks
- **Performance Metrics:** Response time and accuracy tracking
- **Error Logging:** Detailed error reporting and stack traces
- **Usage Analytics:** Local usage statistics and patterns

---

## 🔮 Future Roadmap

### Planned Enhancements
- **Model Upgrades:** Support for larger specialized models
- **Enhanced RAG:** Multi-modal document processing (images, tables)
- **Advanced Analytics:** Predictive modeling and optimization
- **API Integration:** REST API for external system integration

### Potential Extensions
- **Web Interface Revival:** Modern React-based frontend
- **Multi-language Support:** African language processing
- **Cloud Deployment:** Docker and Kubernetes support
- **Enterprise Features:** User management and access control

---
