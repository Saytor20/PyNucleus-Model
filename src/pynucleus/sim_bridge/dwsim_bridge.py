#!/usr/bin/env python3
"""
DWSIM Bridge Module

Provides interface between PyNucleus and DWSIM simulation software.
"""

import os
import sys
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Try importing pythonnet
try:
    import clr
    PYTHONNET_AVAILABLE = True
except ImportError:
    PYTHONNET_AVAILABLE = False
    warnings.warn("pythonnet not available. DWSIM integration will be limited.")

class DWSIMBridge:
    """Bridge class for DWSIM integration."""
    
    def __init__(self, dll_path: Optional[str] = None):
        """Initialize DWSIM bridge with optional DLL path."""
        self.logger = logging.getLogger(__name__)
        
        # Set DLL path
        if dll_path:
            os.environ["DWSIM_DLL_PATH"] = dll_path
        self.dll_path = os.getenv("DWSIM_DLL_PATH")
        
        if not self.dll_path:
            raise RuntimeError("DWSIM_DLL_PATH environment variable not set")
            
        if not PYTHONNET_AVAILABLE:
            raise RuntimeError("pythonnet not available. Please install pythonnet package.")
            
        # Initialize DWSIM components
        self._initialize_dwsim()
        
    def _initialize_dwsim(self):
        """Initialize DWSIM components."""
        try:
            # Add DWSIM DLL references
            clr.AddReference(os.path.join(self.dll_path, "DWSIM.Thermodynamics.dll"))
            clr.AddReference(os.path.join(self.dll_path, "DWSIM.UnitOperations.dll"))
            clr.AddReference(os.path.join(self.dll_path, "DWSIM.Interfaces.dll"))
            
            # Import DWSIM types
            from DWSIM.Thermodynamics import Streams
            from DWSIM.UnitOperations import UnitOperations
            from DWSIM.Interfaces import IFlowsheet
            
            self.Streams = Streams
            self.UnitOperations = UnitOperations
            self.IFlowsheet = IFlowsheet
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DWSIM: {e}")
            raise RuntimeError(f"DWSIM initialization failed: {e}")
            
    def create_flowsheet(self) -> Any:
        """Create a new DWSIM flowsheet."""
        try:
            flowsheet = self.IFlowsheet()
            return flowsheet
        except Exception as e:
            self.logger.error(f"Failed to create flowsheet: {e}")
            raise RuntimeError(f"Flowsheet creation failed: {e}")
            
    def add_stream(self, flowsheet: Any, name: str) -> Any:
        """Add a stream to the flowsheet."""
        try:
            stream = self.Streams.MaterialStream(name, flowsheet)
            return stream
        except Exception as e:
            self.logger.error(f"Failed to add stream: {e}")
            raise RuntimeError(f"Stream addition failed: {e}")
            
    def add_unit_operation(self, flowsheet: Any, name: str, type: str) -> Any:
        """Add a unit operation to the flowsheet."""
        try:
            if type.lower() == "mixer":
                unit = self.UnitOperations.Mixer(name, flowsheet)
            elif type.lower() == "splitter":
                unit = self.UnitOperations.Splitter(name, flowsheet)
            else:
                raise ValueError(f"Unsupported unit operation type: {type}")
            return unit
        except Exception as e:
            self.logger.error(f"Failed to add unit operation: {e}")
            raise RuntimeError(f"Unit operation addition failed: {e}")
            
    def run_simulation(self, flowsheet: Any) -> Dict[str, Any]:
        """Run simulation on the flowsheet."""
        try:
            flowsheet.CalculateFlowsheet()
            return {
                "status": "success",
                "message": "Simulation completed successfully"
            }
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass  # Cleanup handled by Python's GC

def main():
    """Main function for testing."""
    if not PYTHONNET_AVAILABLE:
        print("⚠️ pythonnet not available. Please install pythonnet package.")
        return
        
    try:
        with DWSIMBridge() as bridge:
            # Create a new flowsheet
            flowsheet = bridge.create_flowsheet()
            print("✅ Created flowsheet")
            
            # Add a stream
            stream = bridge.add_stream(flowsheet, "TestStream")
            print("✅ Added stream")
            
            # Add a mixer
            mixer = bridge.add_unit_operation(flowsheet, "TestMixer", "mixer")
            print("✅ Added mixer")
            
            # Run simulation
            result = bridge.run_simulation(flowsheet)
            print(f"✅ Simulation result: {result['status']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
