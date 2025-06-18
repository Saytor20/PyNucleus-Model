"""
PyNucleus Settings Module

Loads configuration from environment variables and .env files.
Provides centralized access to API keys, device settings, and other configuration.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_file = project_root / '.env'
if env_file.exists():
    load_dotenv(env_file)

# Load additional .env.local for local overrides
env_local_file = project_root / '.env.local'
if env_local_file.exists():
    load_dotenv(env_local_file, override=True)


class Settings:
    """Configuration settings for PyNucleus."""

    def __init__(self):
        # API Configuration
        self.api_key: Optional[str] = os.getenv('PYNUCLEUS_API_KEY')
        self.openai_api_key: Optional[str] = os.getenv('OPENAI_API_KEY')
        self.huggingface_api_key: Optional[str] = os.getenv('HUGGINGFACE_API_KEY')

        # Device Configuration
        self.device: str = os.getenv('PYNUCLEUS_DEVICE', 'cpu')
        self.cuda_device: str = os.getenv('CUDA_VISIBLE_DEVICES', '0')

        # Model Configuration
        self.default_model: str = os.getenv('PYNUCLEUS_DEFAULT_MODEL', 'microsoft/DialoGPT-large')
        self.embedding_model: str = os.getenv('PYNUCLEUS_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

        # Data Paths
        self.data_dir: str = os.getenv('PYNUCLEUS_DATA_DIR', 'data')
        self.index_dir: str = os.getenv('PYNUCLEUS_INDEX_DIR', 'data/04_models')

        # Server Configuration
        self.host: str = os.getenv('PYNUCLEUS_HOST', '0.0.0.0')
        self.port: int = int(os.getenv('PYNUCLEUS_PORT', '5000'))
        self.debug: bool = os.getenv('PYNUCLEUS_DEBUG', 'false').lower() == 'true'

        # RAG Configuration
        self.top_k: int = int(os.getenv('PYNUCLEUS_TOP_K', '5'))
        self.chunk_size: int = int(os.getenv('PYNUCLEUS_CHUNK_SIZE', '1000'))
        self.chunk_overlap: int = int(os.getenv('PYNUCLEUS_CHUNK_OVERLAP', '200'))

    def validate(self) -> bool:
        """Validate that required settings are present."""
        required_for_api = ['api_key'] if self.api_key else []
        missing = [key for key in required_for_api if not getattr(self, key)]

        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

        return True

    def get_device(self) -> str:
        """Get the appropriate device for PyTorch/transformers."""
        if self.device.lower() == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.device


# Global settings instance
settings = Settings()


# Convenience functions
def get_api_key() -> Optional[str]:
    """Get the PyNucleus API key."""
    return settings.api_key


def get_device() -> str:
    """Get the device configuration."""
    return settings.get_device()


def is_gpu_available() -> bool:
    """Check if GPU is available and configured."""
    try:
        import torch
        return torch.cuda.is_available() and settings.device.lower() in ['cuda', 'auto']
    except ImportError:
        return False
