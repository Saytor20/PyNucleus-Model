"""
DWSIM Bridge Module

This module has been reorganized. The enhanced DWSIM bridge with RAG integration
is now located in the dwsim_rag_integration package.

For backward compatibility, we import the main components here.
"""

# Import from the new organized structure
try:
    import sys
    import os
    
    # Add the dwsim_rag_integration path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    dwsim_rag_path = os.path.join(project_root, "dwsim_rag_integration")
    
    if dwsim_rag_path not in sys.path:
        sys.path.insert(0, dwsim_rag_path)
    
    from dwsim_rag_integration import (
        EnhancedDWSimBridge,
        ProcessConditions,
        EconomicParameters,
        OptimizationResult,
        SimulationResult,
        create_default_feed_conditions,
        create_default_economic_params,
        DWSimBridge
    )
    
    # Backward compatibility aliases
    DwsimBridge = EnhancedDWSimBridge
    
except ImportError as e:
    print(f"Warning: Could not import from dwsim_rag_integration: {e}")
    print("Please ensure the dwsim_rag_integration package is available.")
    
    # Fallback: Import from local dwsim_bridge if it exists
    try:
        from .dwsim_bridge import *
    except ImportError:
        print("No local dwsim_bridge module found either.")
        raise ImportError("Cannot import DWSIM bridge components. Please check your installation.")
