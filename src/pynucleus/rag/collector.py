# Apply telemetry patch before any ChromaDB imports
from ..utils.telemetry_patch import apply_telemetry_patch
apply_telemetry_patch()

import pathlib, tqdm, chromadb
from chromadb.config import Settings
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging
from ..settings import settings
from ..utils.logger import logger
from .document_processor import DocumentProcessor  # Import enhanced document processor
from ..metrics import Metrics, inc

# disable HF user warnings
hf_logging.set_verbosity_error()

# Use enhanced chunking settings from configuration
MAX_TOKENS_PER_CHUNK = getattr(settings, 'CHUNK_SIZE', 400)

def count_tokens(text: str) -> int:
    """Count tokens in text using transformers tokenizer or fallback to word count."""
    if not text:
        return 0
    try:
        # Try to use transformers tokenizer if available
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception:
        # Fallback to word count (rough approximation: ~0.75 tokens per word)
        words = len(text.split())
        return max(1, int(words * 0.75))

def chunk_text(text: str):
    """Legacy chunking function - kept for backward compatibility."""
    # Simple sentence-based chunking as fallback
    sentences = text.split('. ')
    current_chunk = ""
    
    for sentence in sentences:
        test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
        
        if count_tokens(test_chunk) <= MAX_TOKENS_PER_CHUNK:
            current_chunk = test_chunk
        else:
            if current_chunk:
                yield current_chunk
            current_chunk = sentence
    
    if current_chunk:
        yield current_chunk

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
    """Get or create ChromaDB client with robust error handling."""
    try:
        # Disable telemetry at environment level
        import os
        os.environ["ANONYMIZED_TELEMETRY"] = "false"
        os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
        
        # Ensure directory exists
        Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        
        # Use consistent settings across all modules (match vector_store & collector)
        client_settings = Settings(
            anonymized_telemetry=settings.CHROMA_TELEMETRY_ENABLED,
            allow_reset=True,
            chroma_client_auth_provider=None,
            chroma_server_host=None,
            chroma_server_http_port=None
        )
        client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH,
            settings=client_settings
        )
        return client
        
    except Exception as e:
        logger.error(f"ChromaDB initialization failed in collector: {e}")
        return None

def _is_already_processed(file_path: pathlib.Path) -> bool:
    """Check if a file has already been processed by checking ChromaDB."""
    try:
        client = _get_chromadb_client()
        if client is None:
            return False
        
        # Use consistent collection name with other modules
        coll = client.get_or_create_collection("pynucleus_documents")
        
        # Search for documents from this file
        results = coll.get(where={"source": str(file_path.name)})
        return len(results['ids']) > 0
        
    except Exception as e:
        logger.warning(f"Could not check if file {file_path.name} is already processed: {e}")
        return False

def _get_file_stats(file_path: pathlib.Path) -> dict:
    """Get comprehensive file statistics."""
    try:
        stat = file_path.stat()
        return {
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": stat.st_mtime,
            "extension": file_path.suffix.lower()
        }
    except Exception as e:
        logger.warning(f"Could not get stats for {file_path}: {e}")
        return {"size_bytes": 0, "size_mb": 0, "modified_time": 0, "extension": ""}

def ingest_single_file(file_path: str):
    """Ingest a single new document into ChromaDB using enhanced chunking and metadata."""
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
    
    # Initialize enhanced document processor with enhanced settings
    chunk_size = getattr(settings, 'CHUNK_SIZE', 400)
    chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 100)
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    logger.info(f"Processing new file: {file_path_obj.name} with enhanced chunking (size: {chunk_size}, overlap: {chunk_overlap})")
    
    try:
        # Process document with enhanced chunking
        processing_result = processor.process_document(file_path_obj)
        
        if processing_result["status"] != "success":
            return {
                "status": "error",
                "message": f"Document processing failed: {processing_result.get('error', 'Unknown error')}",
                "chunks_added": 0
            }
        
        chunks_data = processing_result.get("chunks", [])
        file_stats = _get_file_stats(file_path_obj)
        
        chunks_added = 0
        
        # Process each chunk with enhanced metadata
        for chunk_data in tqdm.tqdm(chunks_data, desc=f"Processing {file_path_obj.name}"):
            try:
                chunk_text = chunk_data.get("text", "")  # DocumentProcessor uses "text" key
                if not chunk_text or len(chunk_text.strip()) < 10:
                    continue
                
                # Generate embeddings
                emb = embedder.encode(chunk_text).tolist()
                
                # Create unique document ID
                doc_id = f"{file_path_obj.stem}_{chunk_data.get('chunk_id', chunks_added)}"
                
                # Enhanced metadata with more indexable fields
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
                    
                    # Enhanced indexable metadata for improved retrieval
                    "token_count": count_tokens(chunk_text),
                    "file_size_mb": file_stats["size_mb"],
                    "file_extension": file_stats["extension"],
                    
                    # Processing metadata with enhanced settings
                    "processed_with": "enhanced_chunking_v2",
                    "chunk_size_setting": chunk_size,
                    "chunk_overlap_setting": chunk_overlap,
                    "enhanced_metadata": getattr(settings, 'ENHANCED_METADATA', True),
                    "index_section_titles": getattr(settings, 'INDEX_SECTION_TITLES', True),
                    "index_page_numbers": getattr(settings, 'INDEX_PAGE_NUMBERS', True),
                    "index_technical_terms": getattr(settings, 'INDEX_TECHNICAL_TERMS', True),
                    
                    # Quality metrics for retrieval optimization
                    "content_density": len(chunk_text) / max(1, chunk_data.get("word_count", 1)),
                    "semantic_coherence_score": chunk_data.get("readability_score", 0.0),
                }
                
                # Store in ChromaDB with enhanced metadata
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
        success_msg = f"Successfully processed {file_path_obj.name}: {chunks_added} chunks added with enhanced metadata"
        logger.info(success_msg)
        
        return {
            "status": "success",
            "message": success_msg,
            "chunks_added": chunks_added,
            "processing_result": processing_result
        }
        
    except Exception as e:
        error_msg = f"Failed to process file {file_path_obj.name}: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "chunks_added": 0
        }

def ingest(source_dir: str):
    """Ingest documents into ChromaDB using enhanced document processor and enriched metadata with incremental processing."""
    client = _get_chromadb_client()
    if client is None:
        logger.error("Failed to get ChromaDB client for ingestion")
        return
    
    # Use consistent collection name with other modules
    coll = client.get_or_create_collection("pynucleus_documents")
    embedder = SentenceTransformer(settings.EMB_MODEL)
    
    # Initialize enhanced document processor with enhanced settings
    chunk_size = getattr(settings, 'CHUNK_SIZE', 400)
    chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 100)
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    logger.info(f"Starting enhanced ingestion with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

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
    
    # Get total file size for progress tracking
    total_size_mb = sum(_get_file_stats(f)["size_mb"] for f in unprocessed_files)
    logger.info(f"Total size to process: {total_size_mb:.2f} MB")
    
    for f in tqdm.tqdm(unprocessed_files, desc="Processing files"):
        try:
            # Process document with enhanced chunking
            processing_result = processor.process_document(f)
            
            if processing_result["status"] != "success":
                logger.warning(f"Skipping {f.name}: {processing_result.get('error', 'Processing failed')}")
                failed_files += 1
                continue
                
            chunks_data = processing_result.get("chunks", [])
            file_stats = _get_file_stats(f)
            
            # Process each chunk with enhanced metadata
            for chunk_data in chunks_data:
                try:
                    chunk_text = chunk_data.get("text", "")  # DocumentProcessor uses "text" key
                    if not chunk_text or len(chunk_text.strip()) < 10:
                        continue
                    
                    # Generate embeddings
                    emb = embedder.encode(chunk_text).tolist()
                    
                    # Create unique document ID
                    doc_id = f"{f.stem}_{chunk_data.get('chunk_id', total)}"
                    
                    # Enhanced metadata with improved indexing capabilities
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
                        
                        # Enhanced indexable metadata for retrieval optimization
                        "token_count": count_tokens(chunk_text),
                        "file_size_mb": file_stats["size_mb"],
                        "file_extension": file_stats["extension"],
                        
                        # Processing metadata with enhanced settings
                        "processed_with": "enhanced_chunking_v2",
                        "chunk_size_setting": chunk_size,
                        "chunk_overlap_setting": chunk_overlap,
                        "enhanced_metadata": getattr(settings, 'ENHANCED_METADATA', True),
                        "index_section_titles": getattr(settings, 'INDEX_SECTION_TITLES', True),
                        "index_page_numbers": getattr(settings, 'INDEX_PAGE_NUMBERS', True),
                        "index_technical_terms": getattr(settings, 'INDEX_TECHNICAL_TERMS', True),
                        
                        # Advanced retrieval optimization metadata
                        "content_density": len(chunk_text) / max(1, chunk_data.get("word_count", 1)),
                        "semantic_coherence_score": chunk_data.get("readability_score", 0.0),
                        "chunk_position_in_document": chunk_data.get("section_index", 0),
                        "total_chunks_in_document": len(chunks_data),
                    }
                    
                    # Store in ChromaDB with enhanced metadata
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
            logger.info(f"✅ Processed {f.name}: {len(chunks_data)} chunks with enhanced metadata")
            
        except Exception as file_error:
            failed_files += 1
            logger.error(f"❌ Failed to process {f.name}: {file_error}")
            continue
    
    # Final report with enhanced statistics
    logger.info(f"""
    ======== ENHANCED INGESTION COMPLETE ========
    📁 Files processed: {processed_files}/{len(unprocessed_files)}
    📊 Total chunks created: {total}
    ❌ Failed files: {failed_files}
    🔧 Chunk settings: size={chunk_size}, overlap={chunk_overlap}
    📈 Average chunks per file: {total/max(1, processed_files):.1f}
    💾 Total size processed: {total_size_mb:.2f} MB
    ✨ Enhanced metadata enabled: {getattr(settings, 'ENHANCED_METADATA', True)}
    ============================================
    """)
    
    return {
        "total_chunks": total,
        "processed_files": processed_files,
        "failed_files": failed_files,
        "chunk_settings": {"size": chunk_size, "overlap": chunk_overlap},
        "enhanced_metadata": True
    }

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