"""
Device Manager for PyNucleus

Manages device selection and configuration for PyTorch/transformers models,
including dynamic batch size optimization based on available hardware.
"""

import logging
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from ..settings import settings

logger = logging.getLogger(__name__)


@dataclass
class DeviceConfig:
    """Device configuration parameters."""
    device: str
    device_name: str
    memory_gb: float
    compute_capability: str
    recommended_batch_size: int
    max_sequence_length: int


class DeviceManager:
    """Manages device selection and optimization for ML workloads."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._device_config = None
        self._torch_available = None
        
        # Initialize device configuration
        self._detect_devices()
    
    def _detect_devices(self):
        """Detect available devices and their capabilities."""
        try:
            import torch
            self._torch_available = True
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            
            if cuda_available and settings.device_preference == 'cuda':
                self._configure_cuda_device()
            else:
                self._configure_cpu_device()
                
        except ImportError:
            self.logger.warning("PyTorch not available - falling back to CPU configuration")
            self._torch_available = False
            self._configure_cpu_device()
    
    def _configure_cuda_device(self):
        """Configure CUDA device with optimal settings."""
        try:
            import torch
            
            device_id = int(settings.cuda_device)
            device_name = torch.cuda.get_device_name(device_id)
            memory_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
            
            # Get compute capability
            capability = torch.cuda.get_device_capability(device_id)
            compute_capability = f"{capability[0]}.{capability[1]}"
            
            # Calculate optimal batch sizes based on memory
            if memory_gb >= 24:  # High-end GPU (RTX 4090, A100, etc.)
                batch_size = 32
                max_seq_len = 2048
            elif memory_gb >= 12:  # Mid-range GPU (RTX 4070, RTX 3080, etc.)
                batch_size = 16
                max_seq_len = 1024
            elif memory_gb >= 8:   # Lower-end GPU (RTX 4060, GTX 1080, etc.)
                batch_size = 8
                max_seq_len = 512
            else:                  # Very limited GPU memory
                batch_size = 4
                max_seq_len = 256
            
            self._device_config = DeviceConfig(
                device=f"cuda:{device_id}",
                device_name=device_name,
                memory_gb=memory_gb,
                compute_capability=compute_capability,
                recommended_batch_size=batch_size,
                max_sequence_length=max_seq_len
            )
            
            self.logger.info(f"CUDA device configured: {device_name} ({memory_gb:.1f}GB)")
            
        except Exception as e:
            self.logger.warning(f"CUDA configuration failed: {e}. Falling back to CPU.")
            self._configure_cpu_device()
    
    def _configure_cpu_device(self):
        """Configure CPU device with conservative settings."""
        import psutil
        
        # Get system memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        # Conservative batch sizes for CPU
        if memory_gb >= 32:
            batch_size = 8
            max_seq_len = 1024
        elif memory_gb >= 16:
            batch_size = 4
            max_seq_len = 512
        else:
            batch_size = 2
            max_seq_len = 256
        
        self._device_config = DeviceConfig(
            device="cpu",
            device_name=f"CPU ({cpu_count} cores)",
            memory_gb=memory_gb,
            compute_capability="N/A",
            recommended_batch_size=batch_size,
            max_sequence_length=max_seq_len
        )
        
        self.logger.info(f"CPU device configured: {cpu_count} cores, {memory_gb:.1f}GB RAM")
    
    def get_device(self) -> str:
        """Get the optimal device string for PyTorch."""
        if self._device_config:
            return self._device_config.device
        return "cpu"
    
    def get_device_config(self) -> DeviceConfig:
        """Get complete device configuration."""
        if self._device_config is None:
            self._detect_devices()
        return self._device_config
    
    def get_optimal_batch_size(self, model_type: str = "default") -> int:
        """
        Get optimal batch size for the current device.
        
        Args:
            model_type: Type of model ("embedding", "llm", "default")
            
        Returns:
            Recommended batch size
        """
        if self._device_config is None:
            return 1
        
        base_batch_size = self._device_config.recommended_batch_size
        
        # Adjust based on model type
        if model_type == "embedding":
            # Embedding models can typically handle larger batches
            return min(base_batch_size * 2, 64)
        elif model_type == "llm":
            # LLMs are more memory-intensive
            return max(base_batch_size // 2, 1)
        else:
            return base_batch_size
    
    def get_max_sequence_length(self, model_type: str = "default") -> int:
        """Get optimal maximum sequence length for the current device."""
        if self._device_config is None:
            return 512
        
        return self._device_config.max_sequence_length
    
    def get_device_stats(self) -> Dict[str, Any]:
        """Get detailed device statistics."""
        if self._device_config is None:
            return {"error": "Device not configured"}
        
        stats = {
            "device": self._device_config.device,
            "device_name": self._device_config.device_name,
            "memory_gb": self._device_config.memory_gb,
            "compute_capability": self._device_config.compute_capability,
            "recommended_batch_size": self._device_config.recommended_batch_size,
            "max_sequence_length": self._device_config.max_sequence_length,
            "torch_available": self._torch_available
        }
        
        # Add CUDA-specific stats if available
        if self._torch_available and self._device_config.device.startswith("cuda"):
            try:
                import torch
                device_id = int(self._device_config.device.split(":")[1])
                stats.update({
                    "cuda_version": torch.version.cuda,
                    "memory_allocated_gb": torch.cuda.memory_allocated(device_id) / (1024**3),
                    "memory_reserved_gb": torch.cuda.memory_reserved(device_id) / (1024**3),
                    "memory_free_gb": (torch.cuda.get_device_properties(device_id).total_memory - 
                                     torch.cuda.memory_allocated(device_id)) / (1024**3)
                })
            except Exception as e:
                stats["cuda_error"] = str(e)
        
        return stats
    
    def optimize_for_inference(self) -> Dict[str, Any]:
        """Get optimization settings for inference workloads."""
        config = self.get_device_config()
        
        return {
            "device": config.device,
            "batch_size": self.get_optimal_batch_size("llm"),
            "max_length": config.max_sequence_length,
            "num_workers": 1 if config.device.startswith("cuda") else 2,
            "pin_memory": config.device.startswith("cuda"),
            "torch_compile": config.device.startswith("cuda") and self._torch_available
        }


# Global device manager instance
device_manager = DeviceManager()


def get_device() -> str:
    """Get the optimal device for the current system."""
    return device_manager.get_device()


def get_optimal_batch_size(model_type: str = "default") -> int:
    """Get optimal batch size for the current device and model type."""
    return device_manager.get_optimal_batch_size(model_type)


def get_device_stats() -> Dict[str, Any]:
    """Get comprehensive device statistics."""
    return device_manager.get_device_stats() 