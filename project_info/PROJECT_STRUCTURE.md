# PyNucleus-Model Project Structure

## 📁 Directory Organization

```
PyNucleus-Model/
├── core_modules/                # Main Python package
│   ├── __init__.py             # Package initialization
│   ├── rag/                    # RAG pipeline components
│   │   ├── __init__.py
│   │   ├── wiki_scraper.py     # Wikipedia article scraper
│   │   ├── document_processor.py # Document conversion
│   │   ├── data_chunking.py    # Text chunking
│   │   ├── vector_store.py     # FAISS vector store
│   │   └── performance_analyzer.py # Evaluation metrics
│   └── tests/                  # Unit tests
│       ├── __init__.py
│       └── test_*.py           # Test files
│
├── source_documents/           # Original documents (PDF, DOCX)
├── converted_to_txt/          # Converted text files
├── web_sources/               # Scraped Wikipedia articles
├── converted_chunked_data/    # Chunked documents
├── chunk_reports/            # Processing reports
├── vector_db/                # FAISS vector store files
├── automation_tools/         # Helper scripts
├── docker_config/            # Docker configuration
└── dwsim_libs/              # DWSIM integration files

## 📝 Directory Purposes

### Core Application (`core_modules/`)
- Contains all Python source code
- Follows standard Python package structure
- Includes both application code and tests
- Organized by functionality (RAG, tests, etc.)

### Data Directories
- `source_documents/`: Original input files (PDF, DOCX)
- `converted_to_txt/`: Text files converted from source documents
- `web_sources/`: Wikipedia articles and other web content
- `converted_chunked_data/`: Chunked text data for vector store
- `vector_db/`: FAISS index and embeddings

### Support Directories
- `chunk_reports/`: Processing logs and analysis reports
- `automation_tools/`: Helper scripts for automation
- `docker_config/`: Docker configuration files
- `dwsim_libs/`: DWSIM simulation integration files

## 🔄 Data Flow

1. **Input**: Documents go into `source_documents/`
2. **Conversion**: Processed into `converted_to_txt/`
3. **Web Content**: Scraped into `web_sources/`
4. **Chunking**: Combined and chunked into `converted_chunked_data/`
5. **Vector Store**: Indexed into `vector_db/`
6. **Reports**: Processing logs in `chunk_reports/`

## 🧪 Testing

- All tests are now in `core_modules/tests/`
- Follows Python's standard testing structure
- Tests are organized by component
- Each test file corresponds to a module

## 🚀 Development

- Main code is in `core_modules/`
- Each module has a clear purpose
- Tests are co-located with code
- Follows Python best practices

## 📊 Monitoring

- Processing reports in `chunk_reports/`
- Performance metrics tracked
- Logs maintained for debugging
- Analysis results documented 