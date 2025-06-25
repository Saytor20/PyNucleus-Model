from pydantic_settings import BaseSettings
from pydantic import ValidationError
from typing import List

class Settings(BaseSettings):
    CHROMA_PATH: str = "data/03_intermediate/vector_db"
    MODEL_ID: str = "Qwen/Qwen3-0.6B"  # Back to original model with optimized loading
    GGUF_PATH: str = "path/to/Qwen3-0.6B-Instruct.gguf"
    USE_CUDA: bool = True  # Enable GPU detection by default
    USE_GPU_QUANTIZATION: bool = True  # Use 8-bit quantization on GPU for memory efficiency
    EMB_MODEL: str = "BAAI/bge-small-en-v1.5"  # High-performance embedding model
    MAX_TOKENS: int = 400  # Increased base token limit for complex answers
    MIN_TOKENS: int = 100  # Minimum tokens for simple answers
    MAX_TOKENS_COMPLEX: int = 600  # Maximum tokens for complex technical questions
    TOKEN_COMPLETENESS_THRESHOLD: float = 0.8  # Threshold to detect incomplete answers
    MAX_CONTEXT_CHARS: int = 5000
    RETRIEVE_TOP_K: int = 3  # Focus on top 3 most relevant chunks
    LOG_LEVEL: str = "INFO"
    VSTORE_BACKEND: str = "chroma"  # legacy stub
    vstore_backend: str = "chroma"   # lowercase alias for compatibility
    
    # Enhanced Q&A settings
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    REPETITION_PENALTY: float = 1.2
    MIN_ANSWER_LENGTH: int = 100
    
    # Enhanced RAG Pipeline Configuration - OPTIMIZED FOR BETTER RETRIEVAL
    # Chunking settings optimized for ≥90% retrieval recall
    CHUNK_SIZE: int = 300  # Optimized for better relevance
    CHUNK_OVERLAP: int = 50  # Reduced overlap for cleaner chunks
    
    # Enhanced retrieval settings - OPTIMIZED FOR QUALITY
    ENHANCED_RETRIEVE_TOP_K: int = 3  # Consistent with main retrieval
    RAG_SIMILARITY_THRESHOLD: float = 0.4  # Higher threshold for better relevance
    MAX_CONTEXT_CHUNKS: int = 3  # Limit context to top 3 chunks
    
    # Citation and quality settings
    REQUIRE_CITATIONS: bool = True  # Enforce citation requirement
    MAX_RETRY_ATTEMPTS: int = 2  # Retry attempts for citation enforcement
    DEDUPLICATION_THRESHOLD: float = 90.0  # RapidFuzz threshold for duplication detection
    
    # Metadata indexing settings
    INDEX_SECTION_TITLES: bool = True
    INDEX_PAGE_NUMBERS: bool = True
    INDEX_TECHNICAL_TERMS: bool = True
    ENHANCED_METADATA: bool = True

    # Smart Token Management
    COMPLEX_QUESTION_KEYWORDS: List[str] = [
        "design", "how to", "process", "methodology", "steps", "procedure", 
        "implementation", "development", "construction", "analysis", "optimization"
    ]

    class Config:
        extra = "forbid"

try:
    settings = Settings()
except ValidationError as e:
    import sys, rich
    rich.print("[red]Config error:[/red]", e)
    sys.exit(1) 