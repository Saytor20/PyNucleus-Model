# PyNucleus-Model

A Python-based project for modular plant analysis and RAG (Retrieval-Augmented Generation) implementation.

## Project Structure

### Core Files
- `Capstone Project.ipynb`: Main Jupyter notebook containing the project workflow
- `update_log.txt`: Project update log and milestones

### Modules
- `document_processor.py`: Handles processing of various document types (PDF, DOCX, TXT)
- `wiki_scraper.py`: Scrapes Wikipedia articles for additional content
- `data_processor.py`: Processes and chunks documents for vector storage
- `faiss_manager.py`: Manages FAISS vector store operations

### Directories
- `source_documents/`: Input directory for raw documents
- `processed_txt_files/`: Contains processed text files
- `data_sources/`: Stores scraped Wikipedia articles
- `Chuncked_Data/`: Contains chunked and processed data
- `faiss_store/`: FAISS vector store files
- `vectordb_outputs/`: Vector database analysis logs

## Features

### 1. Document Processing
- Supports multiple file formats (PDF, DOCX, TXT)
- Automatic text extraction and conversion
- Organized output structure
- Error handling and logging

### 2. Wikipedia Integration
- Automated article scraping
- Customizable search keywords
- Content extraction and formatting
- Error handling and retry mechanisms

### 3. Data Processing
- Document chunking with configurable parameters
- Metadata preservation
- Statistical analysis of chunks
- Multiple output formats (JSON, TXT)

### 4. Vector Store Management
- FAISS-based vector storage
- GPU acceleration support
- Similarity search functionality
- Performance evaluation metrics
- Detailed logging and analysis

## Setup

1. Install required dependencies:
```bash
pip install langchain-unstructured PyPDF2 beautifulsoup4 requests langchain-community faiss-cpu torch sentence-transformers python-dotenv
```

2. Set up environment variables:
Create a `.env` file with:
```
GITHUB_USERNAME=your_username
GITHUB_TOKEN=your_token
```

3. Directory Structure:
```
PyNucleus-Model/
├── source_documents/     # Place your input files here
├── processed_txt_files/  # Processed text files
├── data_sources/        # Wikipedia articles
├── Chuncked_Data/       # Chunked documents
├── faiss_store/         # Vector store files
└── vectordb_outputs/    # Analysis logs
```

## Usage

1. Place your source documents in the `source_documents/` directory
2. Run the workflow in `Capstone Project.ipynb`
3. Check the outputs in respective directories
4. View analysis logs in `vectordb_outputs/`

## Workflow

1. Document Processing:
   - Converts various file formats to text
   - Preserves document structure
   - Handles errors gracefully

2. Wikipedia Integration:
   - Scrapes relevant articles
   - Extracts and formats content
   - Saves to data sources

3. Data Processing:
   - Chunks documents
   - Generates metadata
   - Creates analysis reports

4. Vector Store:
   - Builds FAISS index
   - Enables similarity search
   - Evaluates performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 