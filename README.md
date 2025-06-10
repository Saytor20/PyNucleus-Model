# PyNucleus-Model: A Modular RAG Pipeline

![PyNucleus Logo](https://raw.githubusercontent.com/m-a-i-n-s/PyNucleus-Model/main/automation_tools/PyNucleus_logo.png)

This project provides a complete, end-to-end RAG (Retrieval-Augmented Generation) pipeline for processing various document types, scraping web content, and building a searchable vector knowledge base. It is designed to be modular, configurable, and easy to evaluate.

---

## 🚀 Features

-   **Multi-Source Ingestion**: Process local files (`.pdf`, `.docx`, `.txt`) and scrape Wikipedia articles.
-   **Configurable Pipeline**: All settings (paths, models, keywords) are managed in a central `config.py`.
-   **Efficient Vector Store**: Uses FAISS for fast and scalable similarity search.
-   **Built-in Evaluation**: Includes a performance analyzer to measure recall and retrieval quality.
-   **Clean & Modular Code**: Organized into a clear Python package (`core_modules`).
-   **DWSIM Integration**: Includes tools for interacting with DWSIM process simulations.

---

## 🔧 Getting Started

### 1. Prerequisites

-   Python 3.9+
-   (Optional) NVIDIA GPU with CUDA for accelerated performance.

### 2. Installation

Clone the repository and install the required dependencies.

```bash
# Clone the repo
git clone https://github.com/m-a-i-n-s/PyNucleus-Model.git
cd PyNucleus-Model

# Install dependencies
pip install -r requirements.txt
```

**For GPU users**: Before running `pip install`, open `requirements.txt`, comment out `faiss-cpu`, and uncomment `faiss-gpu`.

### 3. Add Your Documents

Place your source files (e.g., PDFs, DOCX files) into the `source_documents/` directory. The pipeline will automatically discover and process them.

### 4. Run the Pipeline

The entire RAG pipeline can be executed by running the main Jupyter Notebook.

1.  **Open the notebook**: `Capstone Project.ipynb`
2.  **Run the main cell**: The second cell in the notebook will execute all the steps of the pipeline:
    -   Process source documents.
    -   Scrape Wikipedia articles (based on keywords in `config.py`).
    -   Chunk all content.
    -   Build and evaluate the FAISS vector store.

---

## 📁 Project Structure

The project is organized into a clean and logical directory structure:

```
PyNucleus-Model/
├── core_modules/           # Main Python package for the RAG pipeline
│   ├── rag/                # Core RAG components
│   └── config.py           # Central configuration file
├── source_documents/       # Place your PDF, DOCX files here
├── web_sources/            # OUTPUT: Scraped Wikipedia articles
├── converted_to_txt/       # OUTPUT: Documents converted to text
├── converted_chunked_data/ # OUTPUT: Chunked data ready for indexing
├── vector_db/              # OUTPUT: FAISS index and embeddings
├── chunk_reports/          # OUTPUT: Logs and evaluation reports
├── automation_tools/       # Helper scripts and assets
└── Capstone Project.ipynb  # Main notebook to run the pipeline
```

---

## ⚙️ Configuration

All pipeline settings can be modified in `core_modules/config.py`:

-   `SOURCE_DOCS_DIR`: Directory for your input files.
-   `WIKI_SEARCH_KEYWORDS`: Keywords for Wikipedia scraping.
-   `CHUNK_SIZE` / `CHUNK_OVERLAP`: Settings for text chunking.
-   `EMBEDDING_MODEL`: The sentence-transformer model to use.
-   `GROUND_TRUTH_DATA`: Questions and expected source documents for evaluation.

---
Love, A.I Assistant 

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 