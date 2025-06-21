from pydantic_settings import BaseSettings
from pydantic import ValidationError

class Settings(BaseSettings):
    CHROMA_PATH: str = "data/03_intermediate/vector_db"
    MODEL_ID: str = "Qwen/Qwen3-0.6B-Instruct"
    GGUF_PATH: str = "path/to/Qwen3-0.6B-Instruct.gguf"
    USE_CUDA: bool = False
    EMB_MODEL: str = "all-MiniLM-L6-v2"
    MAX_TOKENS: int = 300
    MAX_CONTEXT_CHARS: int = 1000
    RETRIEVE_TOP_K: int = 3
    LOG_LEVEL: str = "INFO"
    VSTORE_BACKEND: str = "chroma"  # legacy stub
    vstore_backend: str = "chroma"   # lowercase alias for compatibility

    class Config:
        extra = "forbid"

try:
    settings = Settings()
except ValidationError as e:
    import sys, rich
    rich.print("[red]Config error:[/red]", e)
    sys.exit(1) 