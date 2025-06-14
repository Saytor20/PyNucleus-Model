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
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import base64
from io import BytesIO

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

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

def _extract_images_from_pdf(pdf_path: str, doc_name: str, extract_images: bool = True) -> Tuple[List[Dict], str]:
    """Extract images from PDF and perform OCR."""
    images = []
    ocr_text = ""
    
    if not extract_images:
        return images, ocr_text
    
    # Create image output directory
    image_output_dir = Path("data/02_processed/extracted_images")
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
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
                        img_name = f"{doc_name}_page{page_num+1}_img{img_index+1}.png"
                        img_path = image_output_dir / img_name
                        
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        
                        # Perform OCR on extracted image
                        img_ocr_text = ""
                        if ocr_engine and ocr_engine in ocr_engines:
                            try:
                                pil_image = Image.open(BytesIO(img_data))
                                preprocessed = _preprocess_image(pil_image)
                                img_ocr_text = ocr_engines[ocr_engine](preprocessed)
                                ocr_text += f"\n--- Image {img_name} ---\n{img_ocr_text}\n"
                            except Exception as e:
                                print(f"⚠️ OCR failed for {img_name}: {e}")
                        
                        images.append({
                            'filename': img_name,
                            'path': str(img_path),
                            'page': page_num + 1,
                            'type': 'embedded_image',
                            'ocr_text': img_ocr_text,
                            'size': pix.width * pix.height
                        })
                    
                    pix = None
            
            doc.close()
        except Exception as e:
            print(f"⚠️ PyMuPDF image extraction failed: {e}")
    
    # Method 2: Convert PDF pages to images for full-page OCR (if no embedded images found)
    if PDF2IMAGE_AVAILABLE and ocr_engine and len(images) == 0:
        try:
            pages = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=3)  # Limit to first 3 pages
            for page_num, page_image in enumerate(pages):
                img_name = f"{doc_name}_fullpage{page_num+1}.png"
                img_path = image_output_dir / img_name
                page_image.save(img_path, "PNG")
                
                # Perform OCR on full page
                img_ocr_text = ""
                if ocr_engine in ocr_engines:
                    try:
                        preprocessed = _preprocess_image(page_image)
                        img_ocr_text = ocr_engines[ocr_engine](preprocessed)
                        ocr_text += f"\n--- Full Page {page_num+1} OCR ---\n{img_ocr_text}\n"
                    except Exception as e:
                        print(f"⚠️ OCR failed for {img_name}: {e}")
                
                images.append({
                    'filename': img_name,
                    'path': str(img_path),
                    'page': page_num + 1,
                    'type': 'full_page',
                    'ocr_text': img_ocr_text,
                    'size': page_image.size[0] * page_image.size[1]
                })
                
        except Exception as e:
            print(f"⚠️ PDF2Image conversion failed: {e}")
    
    return images, ocr_text

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

def _detect_drawings_and_diagrams(image_path: str) -> Dict:
    """Detect and analyze technical drawings and diagrams."""
    if not OPENCV_AVAILABLE:
        return {'has_drawings': False, 'analysis': 'OpenCV not available'}
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {'has_drawings': False, 'analysis': 'Could not load image'}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect lines (common in technical drawings)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        # Detect circles (common in process diagrams)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=10, maxRadius=100)
        
        # Detect contours (shapes and components)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze findings
        analysis = {
            'has_drawings': False,
            'line_count': len(lines) if lines is not None else 0,
            'circle_count': len(circles[0]) if circles is not None else 0,
            'contour_count': len(contours),
            'analysis': 'Basic geometric analysis completed'
        }
        
        # Heuristic: if many lines and shapes, likely a technical drawing
        if analysis['line_count'] > 20 or analysis['circle_count'] > 3:
            analysis['has_drawings'] = True
            analysis['analysis'] = 'Likely contains technical drawings or diagrams'
        
        return analysis
        
    except Exception as e:
        print(f"⚠️ Drawing detection failed: {e}")
        return {'has_drawings': False, 'analysis': f'Error: {e}'}

def _process_pdf_enhanced(input_path: str, filename: str, extract_images: bool = False, extract_tables: bool = False) -> str:
    """Enhanced PDF processing with multiple extraction methods."""
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
                # Extract images and tables if requested
                if extract_images or extract_tables:
                    images, ocr_text = _extract_images_from_pdf(input_path, doc_name, extract_images)
                    tables = _extract_tables_from_pdf(input_path, doc_name, extract_tables)
                    
                    if ocr_text:
                        full_text += "\n\n--- EXTRACTED IMAGE TEXT ---\n" + ocr_text
                    
                    if images:
                        print(f"      📷 Extracted {len(images)} images")
                        # Analyze images for drawings
                        for img_info in images:
                            if img_info['type'] == 'embedded_image':
                                drawing_analysis = _detect_drawings_and_diagrams(img_info['path'])
                                if drawing_analysis['has_drawings']:
                                    print(f"         🎨 Drawing detected in {img_info['filename']}")
                    
                    if tables:
                        print(f"      📊 Extracted {len(tables)} tables")
                
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
    """Process image files with OCR."""
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
    extract_images: bool = True,      # ENABLED by default
    extract_tables: bool = True,      # ENABLED by default
    enable_ocr: bool = True          # ENABLED by default
) -> None:
    """
    Process all documents in the input directory and convert them to comprehensive text files.
    Enhanced with OCR, image extraction, and table extraction capabilities - ALL ENABLED by default.
    
    Args:
        input_dir: Input directory for documents
        output_dir: Output directory for text files
        use_progress_bar: Whether to show progress bar
        extract_images: Whether to extract and OCR images from PDFs
        extract_tables: Whether to extract tables from PDFs
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

    # Initialize OCR engines
    ocr_engines = _init_ocr_engines()
    if ocr_engines:
        available_engines = [k for k in ocr_engines.keys() if not k.endswith('_reader')]
        print(f"🔍 OCR engines available: {', '.join(available_engines)}")
    else:
        print("⚠️ No OCR engines available. Install pytesseract, easyocr, or paddleocr for OCR capabilities.")

    print(
        f"--- 📄 Starting comprehensive processing for {len(files_to_process)} file(s) in '{input_dir}' ---"
    )

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
            # Initialize comprehensive content structure
            comprehensive_content = []
            doc_name = os.path.splitext(filename)[0]
            
            # Add document header
            comprehensive_content.append(f"=" * 80)
            comprehensive_content.append(f"COMPREHENSIVE DOCUMENT EXTRACTION")
            comprehensive_content.append(f"Source File: {filename}")
            comprehensive_content.append(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            comprehensive_content.append(f"=" * 80)
            comprehensive_content.append("")

            # Handle different file types with intelligent detection
            if filename.lower().endswith(".pdf"):
                print(f"      📄 Processing PDF with full extraction...")
                
                # Extract main text
                main_text = _process_pdf_enhanced(input_path, filename, extract_images, extract_tables)
                if main_text.strip():
                    comprehensive_content.append("📄 MAIN DOCUMENT TEXT")
                    comprehensive_content.append("-" * 40)
                    comprehensive_content.append(main_text)
                    comprehensive_content.append("")
                
                # Extract images and OCR
                if extract_images:
                    images, ocr_text = _extract_images_from_pdf(input_path, doc_name, extract_images)
                    if images:
                        print(f"      📷 Extracted {len(images)} images with OCR")
                        comprehensive_content.append("📷 EXTRACTED IMAGES & OCR TEXT")
                        comprehensive_content.append("-" * 40)
                        for img_info in images:
                            comprehensive_content.append(f"Image: {img_info['filename']} (Page {img_info['page']})")
                            if img_info['ocr_text']:
                                comprehensive_content.append(f"OCR Text: {img_info['ocr_text']}")
                            comprehensive_content.append("")
                            
                            # Analyze for drawings
                            if img_info['type'] == 'embedded_image':
                                drawing_analysis = _detect_drawings_and_diagrams(img_info['path'])
                                if drawing_analysis['has_drawings']:
                                    print(f"         🎨 Drawing detected in {img_info['filename']}")
                                    comprehensive_content.append(f"🎨 Drawing Analysis: {drawing_analysis['analysis']}")
                                    comprehensive_content.append(f"   Lines: {drawing_analysis['line_count']}, Circles: {drawing_analysis['circle_count']}")
                                    comprehensive_content.append("")
                
                # Extract tables
                if extract_tables:
                    tables = _extract_tables_from_pdf(input_path, doc_name, extract_tables)
                    if tables:
                        print(f"      📊 Extracted {len(tables)} tables")
                        comprehensive_content.append("📊 EXTRACTED TABLES")
                        comprehensive_content.append("-" * 40)
                        for table_info in tables:
                            comprehensive_content.append(f"Table: {table_info['filename']} (Method: {table_info['method']})")
                            # Read and include table content
                            try:
                                import pandas as pd
                                df = pd.read_csv(table_info['path'])
                                comprehensive_content.append("Table Content:")
                                comprehensive_content.append(df.to_string(index=False))
                                comprehensive_content.append("")
                            except Exception as e:
                                comprehensive_content.append(f"Table saved to: {table_info['path']}")
                                comprehensive_content.append("")
                    
            elif filename.lower().endswith(".docx"):
                print(f"      📄 Processing DOCX...")
                if DOCX_AVAILABLE:
                    doc = DocxDocument(input_path)
                    full_text = "\n\n".join([para.text for para in doc.paragraphs])
                    comprehensive_content.append("📄 DOCUMENT TEXT")
                    comprehensive_content.append("-" * 40)
                    comprehensive_content.append(full_text)
                else:
                    raise ImportError("DOCX processing not available")
            
            elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                print(f"      🔍 Processing image with OCR...")
                ocr_text = _process_image_with_ocr(input_path)
                if ocr_text:
                    print(f"      🔍 OCR extracted {len(ocr_text)} characters")
                    comprehensive_content.append("🔍 OCR EXTRACTED TEXT")
                    comprehensive_content.append("-" * 40)
                    comprehensive_content.append(ocr_text)
                    
                    # Analyze for drawings
                    drawing_analysis = _detect_drawings_and_diagrams(input_path)
                    if drawing_analysis['has_drawings']:
                        print(f"         🎨 Drawing detected")
                        comprehensive_content.append("")
                        comprehensive_content.append("🎨 DRAWING ANALYSIS")
                        comprehensive_content.append("-" * 40)
                        comprehensive_content.append(f"Analysis: {drawing_analysis['analysis']}")
                        comprehensive_content.append(f"Lines: {drawing_analysis['line_count']}, Circles: {drawing_analysis['circle_count']}")
                        
            else:
                print(f"      📄 Processing text/other format...")
                if LANGCHAIN_AVAILABLE:
                    loader = UnstructuredLoader(input_path)
                    documents = loader.load()
                    full_text = "\n\n".join([doc.page_content for doc in documents])
                else:
                    # Fallback to basic text reading
                    with open(input_path, 'r', encoding='utf-8') as f:
                        full_text = f.read()
                
                comprehensive_content.append("📄 DOCUMENT TEXT")
                comprehensive_content.append("-" * 40)
                comprehensive_content.append(full_text)

            # Combine all content into final document
            final_content = "\n".join(comprehensive_content)
            
            # Save the comprehensive extracted content
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_content)

            print(f"   ✅ Success! Comprehensive extraction saved to: {output_path}")

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

    print(f"\n🎉 All files processed with comprehensive extraction.")
    
    # Print comprehensive summary
    print(f"\n📊 Comprehensive Processing Summary:")
    image_dir = Path("data/02_processed/extracted_images")
    if image_dir.exists():
        image_count = len(list(image_dir.glob("*")))
        if image_count > 0:
            print(f"   📷 Images extracted: {image_count}")
    
    table_dir = Path("data/02_processed/extracted_tables")
    if table_dir.exists():
        table_count = len(list(table_dir.glob("*")))
        if table_count > 0:
            print(f"   📊 Tables extracted: {table_count}")
    
    print(f"   📄 All content organized into comprehensive text files")

def main():
    """Comprehensive document processing with all features enabled."""
    # Process all documents with full extraction enabled by default
    process_documents()

if __name__ == "__main__":
    main()
