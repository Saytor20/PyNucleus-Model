#!/usr/bin/env python3
"""
PyNucleus Model Manager

A simple CLI tool to manage and switch between different models
based on available hardware resources.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path to import pynucleus
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from pynucleus.settings import settings
    from pynucleus.llm.device_manager import DeviceManager
except ImportError as e:
    print(f"❌ Failed to import PyNucleus modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def get_system_info():
    """Get system information for model recommendations."""
    device_manager = DeviceManager()
    config = device_manager.get_device_config()
    
    return {
        'device': config.device,
        'device_name': config.device_name,
        'memory_gb': config.memory_gb,
        'recommended_batch_size': config.recommended_batch_size,
        'max_sequence_length': config.max_sequence_length
    }


def display_available_models():
    """Display all available models with their status."""
    print("\n🤖 Available Models:")
    print("=" * 80)
    
    available_models = settings.get_available_models()
    current_model = settings.get_active_model()
    
    categories = {
        'lightweight': '🟢 Lightweight Models (Always Available)',
        'medium': '🟡 Medium Models',
        'alternative': '🔵 Alternative Models',
        'heavy': '🔴 Heavy Models'
    }
    
    for category, title in categories.items():
        category_models = {k: v for k, v in available_models.items() if v['category'] == category}
        if not category_models:
            continue
            
        print(f"\n{title}")
        print("-" * 60)
        
        for model_key, model_config in category_models.items():
            status = "✅ ENABLED" if model_config['enabled'] else "❌ DISABLED"
            current = "⭐ CURRENT" if model_config['model_id'] == current_model else ""
            
            print(f"  {model_key:12} | {model_config['display_name']:25} | {status} {current}")
            print(f"               Memory: {model_config['memory_requirement']:8} | ID: {model_config['model_id']}")
            
            if not model_config['enabled'] and 'requires_flag' in model_config:
                flag = model_config['requires_flag']
                print(f"               To enable: Set {flag.upper()}=true in .env")
            print()


def display_system_recommendations():
    """Display model recommendations based on system capabilities."""
    print("\n💡 System-Based Recommendations:")
    print("=" * 80)
    
    system_info = get_system_info()
    memory_gb = system_info['memory_gb']
    device = system_info['device']
    
    print(f"System: {system_info['device_name']}")
    print(f"Memory: {memory_gb:.1f}GB")
    print(f"Device: {device}")
    print()
    
    if memory_gb >= 16 and device.startswith('cuda'):
        print("🚀 High-performance setup detected!")
        print("   Recommended: Enable heavy models (7B+)")
        print("   Command: Set PYNUCLEUS_ENABLE_HEAVY_MODELS=true")
    elif memory_gb >= 8:
        print("⚡ Medium-performance setup detected!")
        print("   Recommended: Enable larger Qwen models (1.5B-3B)")
        print("   Command: Set PYNUCLEUS_ENABLE_LARGER_QWEN=true")
        if memory_gb >= 12:
            print("   Optional: Enable DeepSeek models")
            print("   Command: Set PYNUCLEUS_ENABLE_DEEPSEEK=true")
    else:
        print("💾 Resource-constrained setup detected!")
        print("   Recommended: Keep using lightweight model (0.5B)")
        print("   Current setting is optimal for your system")


def display_current_config():
    """Display current model configuration."""
    print("\n⚙️  Current Configuration:")
    print("=" * 80)
    
    print(f"Active Model: {settings.get_active_model()}")
    print(f"Device Preference: {settings.device_preference}")
    print(f"Enable Larger Qwen: {settings.enable_larger_qwen}")
    print(f"Enable DeepSeek: {settings.enable_deepseek}")
    print(f"Enable Heavy Models: {settings.enable_heavy_models}")


def display_env_template():
    """Display environment variable template for easy copying."""
    print("\n📋 Environment Variables Template:")
    print("=" * 80)
    print("# Copy these to your .env file to configure models")
    print()
    print("# Default lightweight model (recommended for most setups)")
    print("PYNUCLEUS_DEFAULT_MODEL=Qwen/Qwen2.5-0.5B-Instruct")
    print()
    print("# Model flags - set to 'true' to enable when you have sufficient resources")
    print("PYNUCLEUS_ENABLE_LARGER_QWEN=false    # Enables 1.5B-3B models (~3-6GB)")
    print("PYNUCLEUS_ENABLE_DEEPSEEK=false       # Enables DeepSeek models (~3GB+)")
    print("PYNUCLEUS_ENABLE_HEAVY_MODELS=false   # Enables 7B+ models (~14GB+)")
    print()
    print("# Device configuration")
    print("DEVICE_PREFERENCE=cpu                  # Options: cpu, cuda")
    print("PYNUCLEUS_DEVICE=auto                 # Options: cpu, cuda, auto")


def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        print("🤖 PyNucleus Model Manager")
        print("=" * 40)
        print()
        print("Usage: python scripts/model_manager.py <command>")
        print()
        print("Commands:")
        print("  list      - Show all available models")
        print("  config    - Show current configuration")
        print("  recommend - Show recommendations for your system")
        print("  template  - Show .env template")
        print("  all       - Show everything")
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        display_available_models()
    elif command == "config":
        display_current_config()
    elif command == "recommend":
        display_system_recommendations()
    elif command == "template":
        display_env_template()
    elif command == "all":
        display_current_config()
        display_available_models()
        display_system_recommendations()
        display_env_template()
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: list, config, recommend, template, all")


if __name__ == "__main__":
    main() 