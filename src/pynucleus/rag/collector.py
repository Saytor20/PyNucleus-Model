import pathlib, tqdm, chromadb
from chromadb.config import Settings
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging
from ..settings import settings
from ..utils.logger import logger
from .document_processor import DocumentProcessor  # Import enhanced document processor
import tiktoken  # for byte-pair encoding token counts

# disable HF user warnings
hf_logging.set_verbosity_error()

# choose a tokenizer compatible with your embedder
enc = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS_PER_CHUNK = 512

def chunk_text(text: str):
    """Legacy chunking function - kept for backward compatibility."""
    tokens = enc.encode(text)
    for i in range(0, len(tokens), MAX_TOKENS_PER_CHUNK):
        chunk = enc.decode(tokens[i : i + MAX_TOKENS_PER_CHUNK])
        yield chunk

def extract_pdf_text(pdf_path):
    """Extract text from PDF file."""
    try:
        import pypdf
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except ImportError:
        logger.warning("pypdf not installed, skipping PDF processing")
        return f"PDF file: {pdf_path.name} (text extraction not available)"
    except Exception as e:
        logger.warning(f"Failed to extract text from {pdf_path}: {e}")
        return f"PDF file: {pdf_path.name} (extraction failed)"

def _get_chromadb_client():
    """Get or create ChromaDB client with consistent settings."""
    try:
        # Ensure directory exists
        Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        
        # Create client with consistent settings (same as engine.py)
        client_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            chroma_client_auth_provider=None,
            chroma_server_host=None,
            chroma_server_http_port=None
        )
        
        client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH,
            settings=client_settings
        )
        
        # Test client connectivity
        client.list_collections()
        logger.info("ChromaDB client initialized successfully in collector")
        return client
        
    except Exception as e:
        if "already exists" in str(e).lower():
            # Handle existing instance conflict by using engine's archive function
            logger.warning(f"ChromaDB instance conflict detected in collector, archiving old DB...")
            try:
                from .engine import _archive_and_recreate_chromadb
                _archive_and_recreate_chromadb()
                
                # Recreate client
                client = chromadb.PersistentClient(
                    path=settings.CHROMA_PATH,
                    settings=client_settings
                )
                logger.info("ChromaDB client recreated after archiving in collector")
                return client
                
            except Exception as reset_error:
                logger.error(f"Failed to archive/recreate ChromaDB in collector: {reset_error}")
                return None
        else:
            logger.error(f"ChromaDB initialization failed in collector: {e}")
            return None

def ingest(source_dir: str):
    """Ingest documents into ChromaDB using enhanced document processor and enriched metadata."""
    client = _get_chromadb_client()
    if client is None:
        logger.error("Failed to get ChromaDB client for ingestion")
        return
    
    # Use consistent collection name with other modules
    coll = client.get_or_create_collection("pynucleus_documents")
    embedder = SentenceTransformer(settings.EMB_MODEL)
    
    # Initialize enhanced document processor with Step 4 parameters
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=150)

    # Support broader corpus coverage
    source_path = pathlib.Path(source_dir)
    
    # Process .txt, .pdf, and .md files from the specified directory
    txt_files = list(source_path.rglob("*.txt"))
    pdf_files = list(source_path.rglob("*.pdf"))
    md_files = list(source_path.rglob("*.md"))
    files = txt_files + pdf_files + md_files
    
    logger.info(f"Found {len(files)} files to ingest from {source_dir}")
    logger.info(f"  - {len(txt_files)} .txt files")
    logger.info(f"  - {len(pdf_files)} .pdf files") 
    logger.info(f"  - {len(md_files)} .md files")
    
    total = 0
    processed_files = 0
    failed_files = 0
    
    for f in tqdm.tqdm(files, desc="Enhanced ingestion & chunking"):
        try:
            # Use enhanced document processor
            result = processor.process_document(f)
            
            if result["status"] == "error":
                logger.warning(f"Document processor failed for {f}: {result.get('error', 'Unknown error')}")
                failed_files += 1
                continue
            
            # Process enhanced chunks with enriched metadata
            chunks = result.get("chunks", [])
            
            for chunk_data in chunks:
                try:
                    chunk_text = chunk_data["text"]
                    if not chunk_text.strip():
                        continue
                    
                    # Generate embedding
                    emb = embedder.encode(chunk_text).tolist()
                    
                    # Create unique document ID
                    doc_id = f"{f.stem}__{chunk_data['chunk_id']}"
                    
                    # Enriched metadata from enhanced chunking
                    metadata = {
                        "source": str(f.name),
                        "source_path": str(f),
                        "section_header": chunk_data.get("section_header", "Unknown"),
                        "section_index": chunk_data.get("section_index", 0),
                        "chunk_index_in_section": chunk_data.get("chunk_index_in_section", 0),
                        "estimated_page": chunk_data.get("estimated_page", 1),
                        "document_type": chunk_data.get("document_type", "general"),
                        "chunk_type": chunk_data.get("chunk_type", "narrative"),
                        "word_count": chunk_data.get("word_count", 0),
                        "character_count": chunk_data.get("character_count", 0),
                        "contains_technical_terms": chunk_data.get("contains_technical_terms", False),
                        "readability_score": chunk_data.get("readability_score", 0.0),
                        # Additional processing metadata
                        "processed_with": "enhanced_chunking_v1",
                        "chunk_size_setting": processor.chunk_size,
                        "chunk_overlap_setting": processor.chunk_overlap
                    }
                    
                    # Store in ChromaDB with enriched metadata
                    coll.add(
                        documents=[chunk_text], 
                        embeddings=[emb], 
                        ids=[doc_id],
                        metadatas=[metadata]
                    )
                    total += 1
                    
                except Exception as chunk_error:
                    logger.warning(f"Failed to process chunk {chunk_data.get('chunk_id', 'unknown')} from {f}: {chunk_error}")
                    continue
                    
            processed_files += 1
            
            # Log additional processing details
            if result.get("tables_extracted", 0) > 0:
                logger.info(f"Extracted {result['tables_extracted']} tables from {f.name}")
                
        except Exception as e:
            logger.warning(f"Failed to process file {f}: {e}")
            failed_files += 1
            continue
            
    logger.info(f"Enhanced ingestion completed:")
    logger.info(f"  - Total chunks ingested: {total}")
    logger.info(f"  - Files processed successfully: {processed_files}")
    logger.info(f"  - Files failed: {failed_files}")
    logger.info(f"  - Enhanced metadata fields: section_header, document_type, chunk_type, readability_score, etc.")

def ingest_legacy(source_dir: str):
    """Legacy ingestion function using simple chunking - kept for backward compatibility."""
    client = _get_chromadb_client()
    if client is None:
        logger.error("Failed to get ChromaDB client for ingestion")
        return
    
    # Use consistent collection name with other modules
    coll = client.get_or_create_collection("pynucleus_documents")
    embedder = SentenceTransformer(settings.EMB_MODEL)

    # Support broader corpus coverage
    source_path = pathlib.Path(source_dir)
    
    # Process both .txt and .pdf files from the specified directory
    txt_files = list(source_path.rglob("*.txt"))
    pdf_files = list(source_path.rglob("*.pdf"))
    md_files = list(source_path.rglob("*.md"))
    files = txt_files + pdf_files + md_files
    
    logger.info(f"Found {len(files)} files to ingest from {source_dir} (legacy mode)")
    logger.info(f"  - {len(txt_files)} .txt files")
    logger.info(f"  - {len(pdf_files)} .pdf files") 
    logger.info(f"  - {len(md_files)} .md files")
    
    total = 0
    for f in tqdm.tqdm(files, desc="Legacy ingesting & chunking"):
        try:
            if f.suffix.lower() == '.pdf':
                # Extract text from PDF
                text = extract_pdf_text(f)
            else:
                text = f.read_text(errors="ignore")
                
            # Add source metadata (legacy format)
            for idx, chunk in enumerate(chunk_text(text)):
                emb = embedder.encode(chunk).tolist()
                doc_id = f"{f.stem}__{idx}"
                metadata = {"source": str(f.name)}
                coll.add(
                    documents=[chunk], 
                    embeddings=[emb], 
                    ids=[doc_id],
                    metadatas=[metadata]
                )
                total += 1
        except Exception as e:
            logger.warning(f"Failed to process file {f}: {e}")
            continue
            
    logger.info(f"Legacy ingestion: {total} chunks from {len(files)} files.") 