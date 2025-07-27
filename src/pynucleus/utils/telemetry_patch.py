"""
ChromaDB Telemetry Disabler

This module provides utilities to completely disable ChromaDB telemetry
by monkey-patching the telemetry functions to prevent any telemetry calls.
"""

import os
import sys
from typing import Any, Dict, List, Optional

def disable_chromadb_telemetry():
    """
    Completely disable ChromaDB telemetry by setting environment variables
    and monkey-patching telemetry functions.
    """
    # Set environment variables to disable telemetry
    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
    os.environ["CHROMA_TELEMETRY"] = "false"
    
    # More aggressive monkey-patching approach
    def create_noop_capture():
        """Create a no-op capture function that accepts any arguments."""
        def noop_capture(*args, **kwargs):
            """No-op function to replace telemetry capture - accepts any arguments."""
            # Silently ignore all telemetry calls
            pass
        return noop_capture
    
    noop_capture = create_noop_capture()
    
    # Patch all possible telemetry modules
    telemetry_modules = [
        "chromadb.telemetry.product.posthog",
        "chromadb.telemetry",
        "chromadb.telemetry.product",
        "chromadb.telemetry.product.telemetry",
        "chromadb.telemetry.telemetry"
    ]
    
    for module_name in telemetry_modules:
        try:
            module = __import__(module_name, fromlist=['capture'])
            if hasattr(module, 'capture'):
                module.capture = noop_capture
                print(f"Patched telemetry in {module_name}")
        except (ImportError, AttributeError):
            pass
    
    # Also patch any existing telemetry instances
    try:
        import chromadb.telemetry.product.posthog as posthog_module
        if hasattr(posthog_module, 'capture'):
            posthog_module.capture = noop_capture
    except ImportError:
        pass
    
    try:
        import chromadb.telemetry as telemetry_module
        if hasattr(telemetry_module, 'capture'):
            telemetry_module.capture = noop_capture
    except ImportError:
        pass
    
    try:
        import chromadb.telemetry.product as product_module
        if hasattr(product_module, 'capture'):
            product_module.capture = noop_capture
    except ImportError:
        pass
    
    # Patch PostHog client class methods if they exist
    try:
        import chromadb.telemetry.product.posthog as posthog_module
        # Look for PostHog client classes and patch their capture methods
        for attr_name in dir(posthog_module):
            attr = getattr(posthog_module, attr_name)
            if hasattr(attr, '__call__') and hasattr(attr, 'capture'):
                # PostHog client class
                attr.capture = noop_capture
    except ImportError:
        pass
    
    # Patch any PostHog client instances that might already exist
    try:
        import chromadb.telemetry.product.posthog as posthog_module
        # Find any existing PostHog client instances
        for attr_name in dir(posthog_module):
            attr = getattr(posthog_module, attr_name)
            if hasattr(attr, 'capture') and callable(attr.capture):
                # This might be a PostHog client instance
                attr.capture = noop_capture
    except ImportError:
        pass
    
    # Direct patch of PostHog client class if it exists
    try:
        import chromadb.telemetry.product.posthog as posthog_module
        # Look for PostHogClient class specifically
        if hasattr(posthog_module, 'PostHogClient'):
            PostHogClient = posthog_module.PostHogClient
            # Replace the capture method with our no-op version
            PostHogClient.capture = noop_capture
    except ImportError:
        pass
    
    # Also try to patch the posthog module directly
    try:
        import posthog
        if hasattr(posthog, 'capture'):
            posthog.capture = noop_capture
    except ImportError:
        pass

def apply_telemetry_patch():
    """
    Apply telemetry patch - this should be called before any ChromaDB imports.
    """
    disable_chromadb_telemetry()
    return True

# Auto-apply the patch when this module is imported
apply_telemetry_patch() 