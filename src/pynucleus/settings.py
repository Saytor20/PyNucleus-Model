from pydantic_settings import BaseSettings
from pydantic import ValidationError

class Settings(BaseSettings):
    CHROMA_PATH: str = "data/03_intermediate/vector_db"
    MODEL_ID: str = "Qwen/Qwen1.5-0.5B-Chat"
    GGUF_PATH: str = "models/qwen-0.5b.Q4_K_M.gguf"
    USE_CUDA: bool = False
    EMB_MODEL: str = "all-MiniLM-L6-v2"
    MAX_TOKENS: int = 512
    RETRIEVE_TOP_K: int = 8
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