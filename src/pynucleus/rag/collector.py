import pathlib, tqdm, chromadb
from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging
from ..settings import settings
from ..utils.logger import logger
import tiktoken  # for byte-pair encoding token counts

# disable HF user warnings
hf_logging.set_verbosity_error()

# choose a tokenizer compatible with your embedder
enc = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS_PER_CHUNK = 512

def chunk_text(text: str):
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

def ingest(source_dir: str):
    client = chromadb.PersistentClient(settings.CHROMA_PATH)
    coll = client.get_or_create_collection("pynucleus_documents")
    embedder = SentenceTransformer(settings.EMB_MODEL)

    # Process both .txt and .pdf files
    txt_files = list(pathlib.Path(source_dir).rglob("*.txt"))
    pdf_files = list(pathlib.Path(source_dir).rglob("*.pdf"))
    files = txt_files + pdf_files
    
    total = 0
    for f in tqdm.tqdm(files, desc="Ingesting & chunking"):
        if f.suffix.lower() == '.pdf':
            # Extract text from PDF
            text = extract_pdf_text(f)
        else:
            text = f.read_text(errors="ignore")
            
        for idx, chunk in enumerate(chunk_text(text)):
            emb = embedder.encode(chunk).tolist()
            doc_id = f"{f.stem}__{idx}"
            coll.add(documents=[chunk], embeddings=[emb], ids=[doc_id])
            total += 1
    logger.info(f"Ingested {total} chunks from {len(files)} files.") 