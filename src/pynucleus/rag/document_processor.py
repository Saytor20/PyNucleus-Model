#!/usr/bin/env python3
"""
Document Processor for RAG Pipeline

Handles document loading, conversion, and preprocessing for the PyNucleus RAG system.
Enhanced with OCR capabilities, image extraction, and drawing detection.
"""

import sys
import os
from pathlib import Path
import json
import logging
import logging.config
import warnings
import yaml
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import base64
from io import BytesIO
import re
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Configure logging from YAML
def setup_logging():
    """Setup logging configuration from YAML file."""
    config_path = project_root / "configs" / "logging.yaml"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
        except Exception as e:
            # Fallback to basic logging if YAML config fails
            logging.basicConfig(level=logging.INFO)
            logging.error(f"Failed to load logging config: {e}")
    else:
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO)
        
setup_logging()
logger = logging.getLogger(__name__)

# Try importing document processing libraries
try:
    from unstructured.partition.auto import partition
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.image import partition_image
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    warnings.warn("unstructured not available. Document processing limited.")

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    warnings.warn("python-docx not available. DOCX processing limited.")

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        import PyPDF2
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False
        warnings.warn("PDF processing not available.")

# Advanced PDF processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# OCR engines
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    TESSERACT_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    try:
        from PIL import Image, ImageEnhance, ImageFilter
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

# Image processing
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Table extraction
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False

# Import from absolute paths
try:
    from pynucleus.rag.config import RAGConfig, SOURCE_DOCS_DIR, CONVERTED_DIR
    # Use the correct path where PDF files are located
    SOURCE_DOCS_DIR = "data/01_raw/source_documents"
except ImportError:
    # Fallback configuration
    class RAGConfig:
        def __init__(self):
            self.input_dir = "data/01_raw/source_documents"
            self.output_dir = "data/02_processed/converted_to_txt"
    
    SOURCE_DOCS_DIR = "data/01_raw/source_documents"
    CONVERTED_DIR = "data/02_processed/converted_to_txt"

# Import langchain components
try:
    from langchain_unstructured import UnstructuredLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    warnings.warn("langchain_unstructured not available. Using fallback document processing.")

warnings.filterwarnings("ignore")

from tqdm import tqdm

# Section headers that indicate real content has started
SECTION_HEADERS = {
    'abstract', 'introduction', 'background', 'literature review',
    'methodology', 'methods', 'experimental', 'results', 'discussion',
    'analysis', 'findings', 'conclusion', 'summary', 'overview', 'purpose',
    'objective', 'research', 'study', 'investigation'
}

# Pre-compiled regexes for efficient pattern matching
RE_PAGE_FULL   = re.compile(r'^page\s+\d+\s+of\s+\d+$', re.I)
RE_STANDALONE  = re.compile(r'^\d+$')
RE_NUM_HDR     = re.compile(r'^\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}$')
RE_PUNCT_ONLY  = re.compile(r'^[\d\s\.\-\(\)]+$')
RE_DOI_COPY    = re.compile(r'(doi:|preprint|copyright|cc by|license)', re.I)

def looks_like_sentence(line: str) -> bool:
    """Check if a line looks like a proper sentence."""
    return line.endswith('.') and len(line.split()) >= 8

def strip_document_metadata(text: str) -> str:
    """Strip document metadata and formatting artifacts from raw text.
    
    This function efficiently removes common document artifacts like:
    - Title lines and author names
    - Institutional affiliations and email addresses
    - Page numbers and headers/footers
    - Copyright notices and DOI references
    - Repeated headers/footers
    - Standalone numbers and punctuation-only lines
    
    Args:
        text: Raw document text containing titles, headers, page numbers, etc.
        
    Returns:
        Cleaned text with metadata stripped, preserving actual content
    """
    if text is None:
        return None
    if not text or not text.strip():
        return ""

    lines          = text.splitlines()
    # Strip leading whitespace from all lines first to handle indented text
    lines          = [line.strip() for line in lines]
    norm_lines     = [re.sub(r'\s+', ' ', ln.lower().strip()) for ln in lines]
    freq           = Counter(norm_lines)
    cleaned_lines  = []
    content_started = False

    for raw, norm in zip(lines, norm_lines):
        if not norm:
            # Always keep blank lines once content has started (paragraph breaks)
            if content_started:
                cleaned_lines.append('')
            continue

        # -- Global skips (always remove these patterns) -----------------
        if (
            '@' in norm or
            'department of' in norm or 'university' in norm or
            'college of'   in norm or 'institute' in norm or
            RE_PAGE_FULL.match(norm) or RE_STANDALONE.match(norm) or
            RE_NUM_HDR.match(raw.strip()) or
            RE_PUNCT_ONLY.match(norm) or
            RE_DOI_COPY.search(norm) or
            freq[norm] >= 3                               # repeated header/footer
        ):
            continue

        # -- Identify start of real content ------------------------------
        if not content_started:
            if (
                norm in SECTION_HEADERS or
                looks_like_sentence(raw.strip())
            ):
                content_started = True
            else:
                # probably still in title / author block
                continue

        # -- After content start: keep unless obvious skip ---------------
        cleaned_lines.append(raw)

    # Collapse >2 consecutive blank lines
    cleaned_text = re.sub(r'(?:\n\s*){3,}', '\n\n', '\n'.join(cleaned_lines)).strip()
    return cleaned_text

# Global OCR engines - initialized once
_ocr_engines = {}
_ocr_initialized = False

def _init_ocr_engines():
    """Initialize available OCR engines globally."""
    global _ocr_engines, _ocr_initialized
    
    if _ocr_initialized:
        return _ocr_engines
    
    if TESSERACT_AVAILABLE:
        _ocr_engines['tesseract'] = _tesseract_ocr
        
    if EASYOCR_AVAILABLE:
        try:
            _ocr_engines['easyocr_reader'] = easyocr.Reader(['en'])
            _ocr_engines['easyocr'] = _easyocr_ocr
        except Exception as e:
            warnings.warn(f"Failed to initialize EasyOCR: {e}")
            
    if PADDLEOCR_AVAILABLE:
        try:
            _ocr_engines['paddle_reader'] = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')
            _ocr_engines['paddle'] = _paddle_ocr
        except Exception as e:
            warnings.warn(f"Failed to initialize PaddleOCR: {e}")
    
    _ocr_initialized = True
    return _ocr_engines

def _tesseract_ocr(image) -> str:
    """Extract text using Tesseract OCR."""
    try:
        # Configure Tesseract for better accuracy
        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;:()[]{}+-=/*%$@#&_|\\~`"\'<> \t\n'
        text = pytesseract.image_to_string(image, config=config)
        return text.strip()
    except Exception as e:
        print(f"⚠️ Tesseract OCR failed: {e}")
        return ""

def _easyocr_ocr(image) -> str:
    """Extract text using EasyOCR."""
    try:
        reader = _ocr_engines.get('easyocr_reader')
        if not reader:
            return ""
            
        # Convert PIL image to numpy array if needed
        if hasattr(image, 'save'):
            img_array = np.array(image)
        else:
            img_array = image
            
        results = reader.readtext(img_array)
        text = ' '.join([result[1] for result in results])
        return text.strip()
    except Exception as e:
        print(f"⚠️ EasyOCR failed: {e}")
        return ""

def _paddle_ocr(image) -> str:
    """Extract text using PaddleOCR."""
    try:
        paddle_reader = _ocr_engines.get('paddle_reader')
        if not paddle_reader:
            return ""
            
        # Convert PIL image to numpy array if needed
        if hasattr(image, 'save'):
            img_array = np.array(image)
        else:
            img_array = image
            
        results = paddle_reader.ocr(img_array, cls=True)
        text_lines = []
        for line in results:
            if line:
                for word_info in line:
                    text_lines.append(word_info[1][0])
        return ' '.join(text_lines).strip()
    except Exception as e:
        print(f"⚠️ PaddleOCR failed: {e}")
        return ""

def _preprocess_image(image) -> Image.Image:
    """Preprocess image for better OCR accuracy."""
    if not PIL_AVAILABLE:
        return image
        
    try:
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Sharpen image
        image = image.filter(ImageFilter.SHARPEN)
        
        # Resize if too small (OCR works better on larger images)
        width, height = image.size
        if width < 300 or height < 300:
            scale_factor = max(300/width, 300/height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        print(f"⚠️ Image preprocessing failed: {e}")
        return image

def _extract_text_from_pdf_images(pdf_path: str, doc_name: str, extract_image_text: bool = True) -> str:
    """Extract text from PDF images using OCR without saving image files."""
    ocr_text = ""
    
    if not extract_image_text:
        return ocr_text
    
    # Initialize OCR engines
    ocr_engines = _init_ocr_engines()
    
    # Choose best available OCR engine
    ocr_engine = None
    if 'easyocr' in ocr_engines:
        ocr_engine = 'easyocr'
    elif 'tesseract' in ocr_engines:
        ocr_engine = 'tesseract'
    elif 'paddle' in ocr_engines:
        ocr_engine = 'paddle'
    
    if not ocr_engine:
        return ocr_text
    
    # Method 1: PyMuPDF (best for vector graphics and embedded images)
    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract embedded images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        
                        # Perform OCR on extracted image (without saving)
                        img_ocr_text = ""
                        if ocr_engine and ocr_engine in ocr_engines:
                            try:
                                pil_image = Image.open(BytesIO(img_data))
                                preprocessed = _preprocess_image(pil_image)
                                img_ocr_text = ocr_engines[ocr_engine](preprocessed)
                                if img_ocr_text.strip():
                                    ocr_text += f"\n--- Image Text from Page {page_num+1} (Image {img_index+1}) ---\n{img_ocr_text}\n"
                            except Exception as e:
                                print(f"⚠️ OCR failed for image on page {page_num+1}: {e}")
                    
                    pix = None
            
            doc.close()
        except Exception as e:
            print(f"⚠️ PyMuPDF image text extraction failed: {e}")
    
    # Method 2: Convert PDF pages to images for full-page OCR (only if no embedded images found)
    if PDF2IMAGE_AVAILABLE and ocr_engine and not ocr_text.strip():
        try:
            pages = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=3)  # Limit to first 3 pages
            for page_num, page_image in enumerate(pages):
                # Perform OCR on full page (without saving image)
                img_ocr_text = ""
                if ocr_engine in ocr_engines:
                    try:
                        preprocessed = _preprocess_image(page_image)
                        img_ocr_text = ocr_engines[ocr_engine](preprocessed)
                        if img_ocr_text.strip():
                            ocr_text += f"\n--- Full Page {page_num+1} OCR Text ---\n{img_ocr_text}\n"
                    except Exception as e:
                        print(f"⚠️ OCR failed for page {page_num+1}: {e}")
                
        except Exception as e:
            print(f"⚠️ PDF2Image conversion failed: {e}")
    
    return ocr_text

def _extract_tables_from_pdf(pdf_path: str, doc_name: str, extract_tables: bool = True) -> List[Dict]:
    """Extract tables from PDF using multiple methods."""
    if not extract_tables:
        return []
    
    tables = []
    table_output_dir = Path("data/02_processed/extracted_tables")
    table_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Method 1: PDFPlumber (most reliable)
    if PDFPLUMBER_AVAILABLE:
        try:
            import pandas as pd
            with pdfplumber.open(pdf_path) as pdf:
                all_tables = []
                for page in pdf.pages:
                    tables_on_page = page.extract_tables()
                    all_tables.extend(tables_on_page)
                
                for i, table in enumerate(all_tables):
                    if table and len(table) > 1:  # Skip empty or single-row tables
                        df = pd.DataFrame(table[1:], columns=table[0])
                        table_name = f"{doc_name}_table_{i+1}.csv"
                        table_path = table_output_dir / table_name
                        df.to_csv(table_path, index=False)
                        
                        tables.append({
                            'filename': table_name,
                            'path': str(table_path),
                            'method': 'pdfplumber',
                            'shape': df.shape
                        })
        except Exception as e:
            print(f"⚠️ PDFPlumber table extraction failed: {e}")
    
    # Method 2: Camelot (if PDFPlumber didn't find tables)
    if CAMELOT_AVAILABLE and len(tables) == 0:
        try:
            camelot_tables = camelot.read_pdf(pdf_path, pages='1-3')  # Limit to first 3 pages
            for i, table in enumerate(camelot_tables):
                if table.accuracy > 50:  # Only keep tables with decent accuracy
                    table_name = f"{doc_name}_table_camelot_{i+1}.csv"
                    table_path = table_output_dir / table_name
                    table.to_csv(str(table_path))
                    
                    tables.append({
                        'filename': table_name,
                        'path': str(table_path),
                        'method': 'camelot',
                        'accuracy': table.accuracy,
                        'shape': table.shape
                    })
        except Exception as e:
            print(f"⚠️ Camelot table extraction failed: {e}")
    
    return tables

def _process_pdf_enhanced(input_path: str, filename: str, extract_image_text: bool = False, extract_tables: bool = False) -> str:
    """Enhanced PDF processing with text extraction from images and tables."""
    full_text = ""
    doc_name = os.path.splitext(filename)[0]
    
    # Method 1: Try PyMuPDF first (best for complex PDFs)
    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(input_path)
            for page in doc:
                full_text += page.get_text() + "\n\n"
            doc.close()
            if full_text.strip():
                # Extract text from images if requested
                if extract_image_text:
                    ocr_text = _extract_text_from_pdf_images(input_path, doc_name, extract_image_text)
                    if ocr_text.strip():
                        full_text += "\n\n--- EXTRACTED IMAGE TEXT ---\n" + ocr_text
                
                return full_text
        except Exception as e:
            print(f"⚠️ PyMuPDF failed: {e}")
    
    # Method 2: Try PDFPlumber (good for tables and layout)
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(input_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n\n"
            if full_text.strip():
                return full_text
        except Exception as e:
            print(f"⚠️ PDFPlumber failed: {e}")
    
    # Method 3: Fallback to pypdf
    if PYPDF_AVAILABLE:
        try:
            if hasattr(pypdf, "PdfReader"):
                reader = pypdf.PdfReader(input_path)
            else:
                reader = PyPDF2.PdfReader(input_path)
            
            for page in reader.pages:
                full_text += page.extract_text() + "\n\n"
        except Exception as e:
            print(f"⚠️ PyPDF failed: {e}")
    
    return full_text

def _process_image_with_ocr(input_path: str) -> str:
    """Process image files with OCR (returns text only, no image saving)."""
    ocr_engines = _init_ocr_engines()
    
    # Choose best available OCR engine
    ocr_engine = None
    if 'easyocr' in ocr_engines:
        ocr_engine = 'easyocr'
    elif 'tesseract' in ocr_engines:
        ocr_engine = 'tesseract'
    elif 'paddle' in ocr_engines:
        ocr_engine = 'paddle'
    
    if not ocr_engine:
        return ""
    
    try:
        image = Image.open(input_path)
        preprocessed = _preprocess_image(image)
        return ocr_engines[ocr_engine](preprocessed)
    except Exception as e:
        print(f"⚠️ Image OCR failed: {e}")
        return ""

def process_documents(
    input_dir: str = SOURCE_DOCS_DIR,
    output_dir: str = CONVERTED_DIR,
    use_progress_bar: bool = True,
    extract_image_text: bool = True,     # Extract text from images via OCR (no image files saved)
    extract_tables: bool = True,         # Extract tables to text format
    enable_ocr: bool = True             # Enable OCR for image files
) -> None:
    """
    Process all documents in the input directory and convert them to simple text files.
    Enhanced with OCR text extraction and table extraction - NO IMAGE FILES ARE CREATED.
    
    Args:
        input_dir: Input directory for documents
        output_dir: Output directory for text files
        use_progress_bar: Whether to show progress bar
        extract_image_text: Whether to extract text from images in PDFs via OCR (no image files saved)
        extract_tables: Whether to extract tables from PDFs and include as text
        enable_ocr: Whether to enable OCR for image files
    """
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"📂 Creating directory: '{input_dir}'")
        os.makedirs(input_dir, exist_ok=True)
        print(
            f"ℹ Please place your files (PDF, DOCX, TXT, etc.) in the '{input_dir}' directory and run the script again."
        )
        return

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    files_to_process = [
        f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))
    ]

    if not files_to_process:
        print(f"ℹ The '{input_dir}' directory is empty. Nothing to process.")
        return

    # Initialize OCR engines only if needed
    if extract_image_text or enable_ocr:
        ocr_engines = _init_ocr_engines()
        if ocr_engines:
            available_engines = [k for k in ocr_engines.keys() if not k.endswith('_reader')]
            print(f"🔍 OCR engines available: {', '.join(available_engines)}")
        else:
            print("⚠️ No OCR engines available. Install pytesseract, easyocr, or paddleocr for OCR capabilities.")

    print(
        f"--- 📄 Starting simple text processing for {len(files_to_process)} file(s) in '{input_dir}' ---"
    )
    print("💡 Processing mode: Simple text output (no image files created)")
    
    logger.info(f"Starting document processing: {len(files_to_process)} files from {input_dir}")

    for filename in tqdm(
        files_to_process, desc="Processing files", disable=not use_progress_bar
    ):
        # Skip hidden files like .DS_Store
        if filename.startswith("."):
            continue

        input_path = os.path.join(input_dir, filename)
        output_filename = f"{os.path.splitext(os.path.basename(filename))[0]}.txt"
        output_path = os.path.join(output_dir, output_filename)

        print(f" ▶ Processing: {filename}")

        try:
            # Initialize simple content structure
            content_parts = []
            doc_name = os.path.splitext(filename)[0]
            
            # Add simple document header
            content_parts.append(f"=" * 60)
            content_parts.append(f"Document: {filename}")
            content_parts.append(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            content_parts.append(f"=" * 60)
            content_parts.append("")

            # Handle different file types
            if filename.lower().endswith(".pdf"):
                print(f"      📄 Processing PDF...")
                
                try:
                    # Extract main text and image text
                    main_text = _process_pdf_enhanced(input_path, filename, extract_image_text, extract_tables)
                    if main_text.strip():
                        # Clean up the text
                        cleaned_text = strip_document_metadata(main_text)
                        content_parts.append(cleaned_text)
                        logger.info(f"Successfully extracted PDF text from {filename}")
                except Exception as e:
                    logger.error(f"Failed to extract PDF text from {filename}: {str(e)}\n{traceback.format_exc()}")
                    print(f"      ❌ PDF text extraction failed: {e}")
                    content_parts.append(f"Error processing PDF: {e}")
                
                # Extract tables and include as text
                if extract_tables:
                    tables = _extract_tables_from_pdf(input_path, doc_name, extract_tables)
                    if tables:
                        print(f"      📊 Extracted {len(tables)} tables")
                        content_parts.append("\n" + "="*40)
                        content_parts.append("EXTRACTED TABLES")
                        content_parts.append("="*40)
                        
                        for i, table_info in enumerate(tables, 1):
                            content_parts.append(f"\nTable {i} (Method: {table_info['method']}):")
                            content_parts.append("-" * 30)
                            # Read and include table content as text
                            try:
                                import pandas as pd
                                df = pd.read_csv(table_info['path'])
                                content_parts.append(df.to_string(index=False))
                                content_parts.append("")
                                # Clean up the temporary CSV file
                                Path(table_info['path']).unlink(missing_ok=True)
                            except Exception as e:
                                content_parts.append(f"Error reading table: {e}")
                                content_parts.append("")
                    
            elif filename.lower().endswith(".docx"):
                print(f"      📄 Processing DOCX...")
                try:
                    if DOCX_AVAILABLE:
                        doc = DocxDocument(input_path)
                        full_text = "\n\n".join([para.text for para in doc.paragraphs])
                        cleaned_text = strip_document_metadata(full_text)
                        content_parts.append(cleaned_text)
                        logger.info(f"Successfully processed DOCX file {filename}")
                    else:
                        raise ImportError("DOCX processing not available")
                except Exception as e:
                    logger.error(f"Failed to process DOCX file {filename}: {str(e)}\n{traceback.format_exc()}")
                    print(f"      ❌ DOCX processing failed: {e}")
                    content_parts.append(f"Error processing DOCX: {e}")
            
            elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")) and enable_ocr:
                print(f"      🔍 Processing image with OCR...")
                ocr_text = _process_image_with_ocr(input_path)
                if ocr_text:
                    print(f"      🔍 OCR extracted {len(ocr_text)} characters")
                    cleaned_text = strip_document_metadata(ocr_text)
                    content_parts.append(cleaned_text)
                else:
                    content_parts.append("No text could be extracted from this image.")
                        
            else:
                print(f"      📄 Processing text/other format...")
                try:
                    if LANGCHAIN_AVAILABLE:
                        loader = UnstructuredLoader(input_path)
                        documents = loader.load()
                        full_text = "\n\n".join([doc.page_content for doc in documents])
                    else:
                        # Fallback to basic text reading
                        with open(input_path, 'r', encoding='utf-8') as f:
                            full_text = f.read()
                    
                    cleaned_text = strip_document_metadata(full_text)
                    content_parts.append(cleaned_text)
                    logger.info(f"Successfully processed text file {filename}")
                except Exception as e:
                    logger.error(f"Failed to process text file {filename}: {str(e)}\n{traceback.format_exc()}")
                    print(f"      ❌ Text processing failed: {e}")
                    content_parts.append(f"Error processing file: {e}")

            # Combine all content into final document
            final_content = "\n".join(content_parts)
            
            # Save the simple text content
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_content)

            print(f"   ✅ Success! Simple text saved to: {output_path}")

        except Exception as e:
            logger.error(f"Critical error processing {filename}: {str(e)}\n{traceback.format_exc()}")
            print(f"   ❌ Error processing {filename}: {e}")
            # Continue with next file instead of crashing
            continue

    print(f"\n🎉 All files processed as simple text.")
    
    # Clean up any remaining temporary table files
    table_dir = Path("data/02_processed/extracted_tables")
    if table_dir.exists():
        for temp_file in table_dir.glob("*.csv"):
            temp_file.unlink(missing_ok=True)
        # Remove the directory if it's empty
        try:
            table_dir.rmdir()
        except OSError:
            pass  # Directory not empty, that's fine
    
    print(f"\n📊 Simple Text Processing Summary:")
    print(f"   📄 All content converted to simple text format")
    print(f"   🚫 No image files created")
    print(f"   📊 Tables included as formatted text")
    print(f"   🔍 OCR text extraction: {'Enabled' if extract_image_text else 'Disabled'}")

def main():
    """Simple document processing - converts everything to text without creating image files."""
    # Process all documents with simple text extraction
    process_documents()

if __name__ == "__main__":
    main()
