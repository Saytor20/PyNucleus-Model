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
            logger.warning(f"ChromaDB instance conflict detected in collector, reinitializing...")
            try:
                # Clear any existing ChromaDB instances
                import time
                time.sleep(0.1)
                
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

def _is_already_processed(file_path: pathlib.Path) -> bool:
    """Check if a file has been successfully processed by looking for meaningful content in ChromaDB."""
    try:
        client = _get_chromadb_client()
        if client is None:
            return False
        
        coll = client.get_or_create_collection("pynucleus_documents")
        
        # Query for chunks from this specific file
        file_stem = file_path.stem
        
        # Get all documents with metadata to check content quality
        all_docs = coll.get(include=['documents', 'metadatas'])
        if not all_docs or not all_docs.get('ids'):
            return False
        
        matching_chunks = []
        
        # Check for chunks from this file and evaluate their quality
        for i, doc_id in enumerate(all_docs['ids']):
            if doc_id.startswith(f"{file_stem}__"):
                # Check the content quality
                document_content = all_docs['documents'][i] if all_docs['documents'] else ""
                metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                
                # Skip failed/placeholder chunks
                if (document_content.startswith("No tables extracted") or 
                    document_content.startswith("Failed to") or
                    document_content.startswith("Error:") or
                    len(document_content.strip()) < 50):  # Less than 50 chars = probably failed
                    continue
                    
                matching_chunks.append(doc_id)
        
        # Consider processed only if we have multiple meaningful chunks
        # (Most successful documents produce several chunks)
        if len(matching_chunks) >= 2:
            logger.debug(f"File {file_path.name} already processed with {len(matching_chunks)} meaningful chunks")
            return True
        elif len(matching_chunks) == 1:
            logger.info(f"File {file_path.name} has only 1 meaningful chunk, may need reprocessing")
            return False
        else:
            logger.info(f"File {file_path.name} has no meaningful chunks or failed processing, needs reprocessing")
            return False
        
    except Exception as e:
        logger.warning(f"Could not check if {file_path.name} is already processed: {e}")
        return False  # If we can't check, assume not processed

def ingest_single_file(file_path: str):
    """Ingest a single new document into ChromaDB without reprocessing existing ones."""
    client = _get_chromadb_client()
    if client is None:
        logger.error("Failed to get ChromaDB client for single file ingestion")
        return {"status": "error", "message": "ChromaDB client unavailable"}
    
    file_path_obj = pathlib.Path(file_path)
    
    # Check if already processed
    if _is_already_processed(file_path_obj):
        logger.info(f"File {file_path_obj.name} already processed, skipping")
        return {
            "status": "skipped", 
            "message": f"File {file_path_obj.name} already processed",
            "chunks_added": 0
        }
    
    # Use consistent collection name with other modules
    coll = client.get_or_create_collection("pynucleus_documents")
    embedder = SentenceTransformer(settings.EMB_MODEL)
    
    # Initialize enhanced document processor
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=150)
    
    logger.info(f"Processing new file: {file_path_obj.name}")
    
    try:
        # Use enhanced document processor
        result = processor.process_document(file_path_obj)
        
        if result["status"] == "error":
            error_msg = f"Document processor failed for {file_path_obj.name}: {result.get('error', 'Unknown error')}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg, "chunks_added": 0}
        
        # Process enhanced chunks with enriched metadata
        chunks = result.get("chunks", [])
        chunks_added = 0
        
        for chunk_data in chunks:
            try:
                chunk_text = chunk_data["text"]
                if not chunk_text.strip():
                    continue
                
                # Generate embedding
                emb = embedder.encode(chunk_text).tolist()
                
                # Create unique document ID
                doc_id = f"{file_path_obj.stem}__{chunk_data['chunk_id']}"
                
                # Enriched metadata from enhanced chunking
                metadata = {
                    "source": str(file_path_obj.name),
                    "source_path": str(file_path_obj),
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
                chunks_added += 1
                
            except Exception as chunk_error:
                logger.warning(f"Failed to process chunk {chunk_data.get('chunk_id', 'unknown')} from {file_path_obj.name}: {chunk_error}")
                continue
        
        # Log processing results
        success_msg = f"Successfully processed {file_path_obj.name}: {chunks_added} chunks added"
        if result.get("tables_extracted", 0) > 0:
            success_msg += f", {result['tables_extracted']} tables extracted"
        
        logger.info(success_msg)
        return {
            "status": "success", 
            "message": success_msg,
            "chunks_added": chunks_added,
            "tables_extracted": result.get("tables_extracted", 0)
        }
        
    except Exception as e:
        error_msg = f"Failed to process file {file_path_obj.name}: {e}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg, "chunks_added": 0}

def ingest(source_dir: str):
    """Ingest documents into ChromaDB using enhanced document processor and enriched metadata with incremental processing."""
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
    
    # Filter out already processed files
    unprocessed_files = [f for f in files if not _is_already_processed(f)]
    
    logger.info(f"Found {len(files)} files total in {source_dir}")
    logger.info(f"  - {len(txt_files)} .txt files")
    logger.info(f"  - {len(pdf_files)} .pdf files") 
    logger.info(f"  - {len(md_files)} .md files")
    logger.info(f"Files already processed: {len(files) - len(unprocessed_files)}")
    logger.info(f"Files to process: {len(unprocessed_files)}")
    
    if not unprocessed_files:
        logger.info("No new files to process - all files already processed")
        return
    
    total = 0
    processed_files = 0
    failed_files = 0
    
    for f in tqdm.tqdm(unprocessed_files, desc="Enhanced ingestion & chunking"):
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