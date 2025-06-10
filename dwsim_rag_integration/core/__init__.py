"""Core simulation bridge components."""

from .enhanced_dwsim_bridge import (
    DWSimBridge,
    ProcessConditions,
    EconomicParameters,
    OptimizationResult,
    SimulationResult,
    create_default_feed_conditions,
    create_default_economic_params
)

# Alias for clarity
EnhancedDWSimBridge = DWSimBridge

__all__ = [
    "DWSimBridge",
    "EnhancedDWSimBridge", 
    "ProcessConditions",
    "EconomicParameters",
    "OptimizationResult",
    "SimulationResult",
    "create_default_feed_conditions",
    "create_default_economic_params"
] 