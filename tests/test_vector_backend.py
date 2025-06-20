"""
Test vector backend configuration.
"""

from pynucleus.settings import settings

def test_backend_stub():
    """Test that vector store backend is properly configured."""
    assert settings.VSTORE_BACKEND == "chroma"
    assert settings.vstore_backend == "chroma"  # Test lowercase alias too

def test_backend_consistency():
    """Test that both backend settings are consistent."""
    assert settings.VSTORE_BACKEND == settings.vstore_backend 