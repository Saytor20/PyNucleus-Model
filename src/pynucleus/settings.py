from pydantic_settings import BaseSettings
from pydantic import ValidationError
from typing import List

class Settings(BaseSettings):
    # Core database and model paths
    CHROMA_PATH: str = "data/03_intermediate/vector_db"
    MODEL_ID: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # Switched from Qwen after performance testing
    OPTIMIZED_MODEL_PATH: str = "models/optimized/SmolLM2-1.7B-Instruct.gguf"  # For quantized versions
    USE_CUDA: bool = True  # Auto-detect GPU, fallback to CPU works fine
    USE_GPU_QUANTIZATION: bool = True  # Saves memory on consumer GPUs
    EMB_MODEL: str = "BAAI/bge-small-en-v1.5"  # BGE models work better than sentence-transformers
    
    # Token limits - found these work best through testing
    MAX_TOKENS: int = 200  # Sweet spot for response quality vs speed
    MIN_TOKENS: int = 50   # Prevents overly brief answers
    MAX_TOKENS_COMPLEX: int = 250  # For detailed technical questions
    TOKEN_COMPLETENESS_THRESHOLD: float = 0.8  # When to retry for complete answers
    MAX_CONTEXT_CHARS: int = 2000  # Prevents context overflow
    RETRIEVE_TOP_K: int = 3  # More than 3 chunks adds noise
    LOG_LEVEL: str = "INFO"
    VSTORE_BACKEND: str = "chroma"  # Legacy compatibility
    vstore_backend: str = "chroma"   # Some modules expect lowercase
    
    # Language model parameters - tuned for chemical engineering Q&A
    TEMPERATURE: float = 0.2  # Low temp keeps answers factual
    TOP_P: float = 0.85  # Nucleus sampling for coherent responses
    REPETITION_PENALTY: float = 1.15  # Prevents model getting stuck in loops
    MIN_ANSWER_LENGTH: int = 50
    MAX_ANSWER_LENGTH: int = 400  # Long enough for technical explanations
    
    # RAG pipeline settings - these took some tuning to get right
    CHUNK_SIZE: int = 300  # Good balance of context vs granularity
    CHUNK_OVERLAP: int = 50  # Helps maintain context across chunks
    
    # Retrieval quality controls
    ENHANCED_RETRIEVE_TOP_K: int = 3  # More than this gets noisy
    RAG_SIMILARITY_THRESHOLD: float = 0.05  # Pretty lenient to catch relevant docs
    MAX_CONTEXT_CHUNKS: int = 3  # Keeps context focused
    
    # Quality assurance features
    REQUIRE_CITATIONS: bool = True  # Users need to know source documents
    MAX_RETRY_ATTEMPTS: int = 2  # Don't retry forever if model struggles
    DEDUPLICATION_THRESHOLD: float = 90.0  # Fuzzy matching for similar chunks
    
    # Document processing features
    INDEX_SECTION_TITLES: bool = True
    INDEX_PAGE_NUMBERS: bool = True
    INDEX_TECHNICAL_TERMS: bool = True
    ENHANCED_METADATA: bool = True

    # Detect complex questions that need more tokens
    COMPLEX_QUESTION_KEYWORDS: List[str] = [
        "design", "how to", "process", "methodology", "steps", "procedure", 
        "implementation", "development", "construction", "analysis", "optimization"
    ]
    
    # Model preference order - fallback chain if primary fails
    PREFERRED_MODELS: List[str] = [
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",    # Best performance for chemical engineering
        "Qwen/Qwen2.5-1.5B-Instruct",            # Good multilingual support
        "microsoft/Phi-3.5-mini-instruct",        # Microsoft's efficient model
        "HuggingFaceTB/SmolLM2-360M-Instruct",   # Emergency lightweight option
    ]
    
    # ChromaDB can be chatty, disable telemetry
    CHROMA_TELEMETRY_ENABLED: bool = False  
    
    # Experimental features for answer quality
    USE_QUALITY_MODEL_SELECTION: bool = True
    MIN_QUALITY_THRESHOLD_FOR_RETRY: float = 0.3  # Retry if confidence too low
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