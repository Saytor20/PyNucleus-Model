"""
PyNucleus Settings Module

Loads configuration from environment variables and .env files.
Provides centralized access to API keys, device settings, and other configuration.
"""

import os
from pathlib import Path
from typing import Optional, Literal, ClassVar, Dict, Any
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field
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


class Settings(BaseSettings):
    """Configuration settings for PyNucleus."""

    # API Configuration
    api_key: Optional[str] = Field(default=None, env='PYNUCLEUS_API_KEY')
    openai_api_key: Optional[str] = Field(default=None, env='OPENAI_API_KEY')
    huggingface_api_key: Optional[str] = Field(default=None, env='HUGGINGFACE_API_KEY')

    # Vector Store Configuration
    vstore_backend: Literal['faiss', 'qdrant'] = Field(default='faiss', env='VSTORE_BACKEND')
    
    # Device Configuration
    device: str = Field(default='cpu', env='PYNUCLEUS_DEVICE')
    device_preference: Literal['cpu', 'cuda'] = Field(default='cpu', env='DEVICE_PREFERENCE')
    cuda_device: str = Field(default='0', env='CUDA_VISIBLE_DEVICES')

    # Model Configuration
    default_model: str = Field(default='Qwen/Qwen2.5-0.5B-Instruct', env='PYNUCLEUS_DEFAULT_MODEL')
    embedding_model: str = Field(default='all-MiniLM-L6-v2', env='PYNUCLEUS_EMBEDDING_MODEL')
    
    # Model Selection Flags (disabled by default for resource-constrained environments)
    enable_larger_qwen: bool = Field(default=False, env='PYNUCLEUS_ENABLE_LARGER_QWEN')
    enable_deepseek: bool = Field(default=False, env='PYNUCLEUS_ENABLE_DEEPSEEK')
    enable_heavy_models: bool = Field(default=False, env='PYNUCLEUS_ENABLE_HEAVY_MODELS')

    # Data Paths
    data_dir: str = Field(default='data', env='PYNUCLEUS_DATA_DIR')
    index_dir: str = Field(default='data/04_models', env='PYNUCLEUS_INDEX_DIR')

    # Server Configuration
    host: str = Field(default='0.0.0.0', env='PYNUCLEUS_HOST')
    port: int = Field(default=5000, env='PYNUCLEUS_PORT')
    debug: bool = Field(default=False, env='PYNUCLEUS_DEBUG')

    # RAG Configuration
    top_k: int = Field(default=5, env='PYNUCLEUS_TOP_K')
    chunk_size: int = Field(default=1000, env='PYNUCLEUS_CHUNK_SIZE')
    chunk_overlap: int = Field(default=200, env='PYNUCLEUS_CHUNK_OVERLAP')

    # Model configurations
    DEFAULT_MODEL_CONFIG: ClassVar[Dict[str, Any]] = {
        'model_id': 'Qwen/Qwen2.5-0.5B-Instruct',  # Ultra-lightweight for resource-constrained devices
        'trust_remote_code': True,
        'torch_dtype': 'auto',
        'device_map': 'auto'
    }

    # Available models organized by resource requirements
    AVAILABLE_MODELS: ClassVar[Dict[str, Dict[str, Any]]] = {
        # Lightweight models (always available)
        'qwen_0_5b': {
            'model_id': 'Qwen/Qwen2.5-0.5B-Instruct',
            'display_name': 'Qwen 0.5B (Ultra-light)',
            'memory_requirement': '~1GB',
            'enabled': True,
            'category': 'lightweight'
        },
        
        # Medium models (requires enable_larger_qwen=True)
        'qwen_1_5b': {
            'model_id': 'Qwen/Qwen2.5-1.5B-Instruct',
            'display_name': 'Qwen 1.5B (Light)',
            'memory_requirement': '~3GB',
            'enabled': False,
            'category': 'medium',
            'requires_flag': 'enable_larger_qwen'
        },
        'qwen_3b': {
            'model_id': 'Qwen/Qwen2.5-3B-Instruct',
            'display_name': 'Qwen 3B (Balanced)',
            'memory_requirement': '~6GB',
            'enabled': False,
            'category': 'medium',
            'requires_flag': 'enable_larger_qwen'
        },
        
        # DeepSeek models (requires enable_deepseek=True)
        'deepseek_1_3b': {
            'model_id': 'deepseek-ai/DeepSeek-V2.5-1.3B-Chat',
            'display_name': 'DeepSeek 1.3B (Alternative)',
            'memory_requirement': '~3GB',
            'enabled': False,
            'category': 'alternative',
            'requires_flag': 'enable_deepseek'
        },
        
        # Heavy models (requires enable_heavy_models=True)
        'qwen_7b': {
            'model_id': 'Qwen/Qwen2.5-7B-Instruct',
            'display_name': 'Qwen 7B (High Performance)',
            'memory_requirement': '~14GB',
            'enabled': False,
            'category': 'heavy',
            'requires_flag': 'enable_heavy_models'
        },
        'deepseek_7b': {
            'model_id': 'deepseek-ai/DeepSeek-V2.5-7B-Chat',
            'display_name': 'DeepSeek 7B (High Performance)',
            'memory_requirement': '~14GB',
            'enabled': False,
            'category': 'heavy',
            'requires_flag': 'enable_heavy_models'
        }
    }

    # Legacy compatibility - keeping for backward compatibility
    RECOMMENDED_MODELS: ClassVar[Dict[str, str]] = {
        'tiny': 'Qwen/Qwen2.5-0.5B-Instruct',         # Ultra-lightweight (DEFAULT)
        'small_fast': 'Qwen/Qwen2.5-1.5B-Instruct',   # Light performance
        'balanced': 'Qwen/Qwen2.5-3B-Instruct',       # Balanced performance
        'deepseek': 'deepseek-ai/DeepSeek-V2.5-1.3B-Chat',  # DeepSeek alternative
        'heavy': 'Qwen/Qwen2.5-7B-Instruct',          # High performance
    }

    class Config:
        env_file = ['.env', '.env.local']
        env_file_encoding = 'utf-8'

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
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models based on current configuration flags."""
        available = {}
        
        for model_key, model_config in self.AVAILABLE_MODELS.items():
            # Always include lightweight models
            if model_config['category'] == 'lightweight':
                available[model_key] = model_config.copy()
                continue
            
            # Check if model requires a flag and if that flag is enabled
            requires_flag = model_config.get('requires_flag')
            if requires_flag:
                flag_enabled = getattr(self, requires_flag, False)
                if flag_enabled:
                    config_copy = model_config.copy()
                    config_copy['enabled'] = True
                    available[model_key] = config_copy
                else:
                    # Include in list but mark as disabled
                    config_copy = model_config.copy()
                    config_copy['enabled'] = False
                    available[model_key] = config_copy
            else:
                available[model_key] = model_config.copy()
        
        return available
    
    def get_active_model(self) -> str:
        """Get the currently active model ID."""
        return self.default_model


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
