from pydantic_settings import BaseSettings
from pydantic import ValidationError
from typing import List

class Settings(BaseSettings):
    CHROMA_PATH: str = "data/03_intermediate/vector_db"
    MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"  # Fixed: Use real Qwen model
    GGUF_PATH: str = "path/to/Qwen2.5-1.5B-Instruct.gguf"  # Updated to match real model
    USE_CUDA: bool = True  # Enable GPU detection by default
    USE_GPU_QUANTIZATION: bool = True  # Use 8-bit quantization on GPU for memory efficiency
    EMB_MODEL: str = "BAAI/bge-small-en-v1.5"  # High-performance embedding model
    MAX_TOKENS: int = 150  # Reduced base token limit for concise answers
    MIN_TOKENS: int = 50   # Minimum tokens for simple answers
    MAX_TOKENS_COMPLEX: int = 300  # Maximum tokens for complex technical questions
    TOKEN_COMPLETENESS_THRESHOLD: float = 0.8  # Threshold to detect incomplete answers
    MAX_CONTEXT_CHARS: int = 2000  # Reduced context size for more focused answers
    RETRIEVE_TOP_K: int = 3  # Focus on top 3 most relevant chunks
    LOG_LEVEL: str = "INFO"
    VSTORE_BACKEND: str = "chroma"  # legacy stub
    vstore_backend: str = "chroma"   # lowercase alias for compatibility
    
    # Enhanced Q&A settings
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    REPETITION_PENALTY: float = 1.2
    MIN_ANSWER_LENGTH: int = 50
    MAX_ANSWER_LENGTH: int = 500  # Reduced for more concise answers
    
    # Enhanced RAG Pipeline Configuration - OPTIMIZED FOR BETTER RETRIEVAL
    # Chunking settings optimized for ≥90% retrieval recall
    CHUNK_SIZE: int = 300  # Optimized for better relevance
    CHUNK_OVERLAP: int = 50  # Reduced overlap for cleaner chunks
    
    # Enhanced retrieval settings - OPTIMIZED FOR QUALITY
    ENHANCED_RETRIEVE_TOP_K: int = 3  # Consistent with main retrieval
    RAG_SIMILARITY_THRESHOLD: float = 0.05  # Much lower threshold for better recall
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
    
    # Enhanced Model Selection for Better Answer Quality
    # Priority order: Use best available model for improved responses
    PREFERRED_MODELS: List[str] = [
        "Qwen/Qwen2.5-7B-Instruct",    # Best for complex chemical engineering
        "Qwen/Qwen2.5-3B-Instruct",    # Good balance of quality and speed
        "Qwen/Qwen2.5-1.5B-Instruct",  # Decent for simpler questions
        "Qwen/Qwen2.5-1.5B-Instruct",  # Updated: Use real model as fallback
    ]
    
    # ChromaDB telemetry settings
    CHROMA_TELEMETRY_ENABLED: bool = False  # Disable telemetry to prevent errors
    
    # Quality-based answer improvement
    USE_QUALITY_MODEL_SELECTION: bool = True
    MIN_QUALITY_THRESHOLD_FOR_RETRY: float = 0.3
    ENABLE_ANSWER_IMPROVEMENT: bool = True

    class Config:
        extra = "forbid"
        env_file = ".env"
        env_file_encoding = "utf-8"

try:
    settings = Settings()
except ValidationError as e:
    import sys, rich
    rich.print("[red]Config error:[/red]", e)
    sys.exit(1) 