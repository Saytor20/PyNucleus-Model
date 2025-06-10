"""
DWSIM Bridge Module

This module provides a Python interface to DWSIM simulation engine through
.NET interoperability using pythonnet.
"""

import os
import glob
import csv
from pathlib import Path
from typing import Optional, Any, List, Sequence

# Conditional imports for .NET/DWSIM dependencies
try:
    import clr
    _CLR_AVAILABLE = True
except ImportError:
    _CLR_AVAILABLE = False
    clr = None


class DWSIMBridge:
    """
    A bridge class for interfacing with DWSIM simulation engine.
    
    This class provides methods to create, load, run, and save DWSIM flowsheets
    through .NET interoperability.
    """
    
    def __init__(self) -> None:
        """
        Initialize the DWSIM bridge by loading required DLLs.
        
        Raises:
            RuntimeError: If DWSIM_DLL_PATH environment variable is not set
                         or if DLL loading fails.
        """
        if not _CLR_AVAILABLE:
            raise RuntimeError(
                "pythonnet (clr) is not available. Please install with: pip install pythonnet"
            )
        
        self._flowsheet: Optional[Any] = None
        self._dwsim_app: Optional[Any] = None
        self._load_dwsim_dlls()
        self._initialize_dwsim()
    
    def _load_dwsim_dlls(self) -> None:
        """
        Load all DWSIM DLL files from the specified path.
        
        Raises:
            RuntimeError: If DWSIM_DLL_PATH is not set or DLL loading fails.
        """
        dll_path = os.getenv('DWSIM_DLL_PATH')
        if not dll_path:
            raise RuntimeError(
                "DWSIM_DLL_PATH environment variable is not set. "
                "Please set it to the directory containing DWSIM DLL files."
            )
        
        dll_path = Path(dll_path)
        if not dll_path.exists():
            raise RuntimeError(f"DWSIM DLL path does not exist: {dll_path}")
        
        # Find all DLL files in the specified directory
        dll_files = list(dll_path.glob("*.dll"))
        if not dll_files:
            raise RuntimeError(f"No DLL files found in: {dll_path}")
        
        # Load each DLL
        loaded_dlls: List[str] = []
        failed_dlls: List[str] = []
        
        # Initialize CLR if needed
        if not hasattr(clr, 'AddReference'):
            import System
        
        for dll_file in dll_files:
            try:
                clr.AddReference(str(dll_file))
                loaded_dlls.append(dll_file.name)
            except Exception as e:
                failed_dlls.append(f"{dll_file.name}: {str(e)}")
        
        if not loaded_dlls:
            raise RuntimeError(
                f"Failed to load any DLL files. Errors: {failed_dlls}"
            )
        
        # Log successful and failed loads
        print(f"Successfully loaded {len(loaded_dlls)} DLLs: {loaded_dlls}")
        if failed_dlls:
            print(f"Failed to load {len(failed_dlls)} DLLs: {failed_dlls}")
    
    def _initialize_dwsim(self) -> None:
        """
        Initialize DWSIM application and required components.
        
        Raises:
            RuntimeError: If DWSIM initialization fails.
        """
        try:
            # Import DWSIM types after DLLs are loaded
            from DWSIM.Interfaces import IFlowsheet
            from DWSIM.Thermodynamics import PropertyPackages
            from DWSIM.UnitOperations import UnitOperations
            from DWSIM.Drawing.SkiaSharp import FlowsheetSurface_SkiaSharp
            
            # Store references to important types
            self._IFlowsheet = IFlowsheet
            self._PropertyPackages = PropertyPackages
            self._UnitOperations = UnitOperations
            self._FlowsheetSurface = FlowsheetSurface_SkiaSharp
            
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import DWSIM types. Ensure all required DLLs are loaded. Error: {e}"
            )
    
    def new_case(self) -> Any:
        """
        Create a new empty DWSIM flowsheet.
        
        Returns:
            Any: A new DWSIM flowsheet object.
            
        Raises:
            RuntimeError: If flowsheet creation fails.
        """
        try:
            # Create a new flowsheet instance
            self._flowsheet = self._FlowsheetSurface()
            
            # Initialize the flowsheet with basic settings
            if hasattr(self._flowsheet, 'Initialize'):
                self._flowsheet.Initialize()
            
            return self._flowsheet
            
        except Exception as e:
            raise RuntimeError(f"Failed to create new flowsheet: {e}")
    
    def load_case(self, file_path: str) -> Any:
        """
        Load a DWSIM case from file.
        
        Args:
            file_path (str): Path to the DWSIM file to load.
            
        Returns:
            Any: The loaded DWSIM flowsheet object.
            
        Raises:
            RuntimeError: If file loading fails.
            FileNotFoundError: If the specified file does not exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"DWSIM file not found: {file_path}")
        
        try:
            # Create new flowsheet if not exists
            if self._flowsheet is None:
                self.new_case()
            
            # Load the file
            self._flowsheet.LoadFromFile(str(file_path))
            
            return self._flowsheet
            
        except Exception as e:
            raise RuntimeError(f"Failed to load DWSIM file '{file_path}': {e}")
    
    def run(self) -> None:
        """
        Run the current flowsheet simulation.
        
        Raises:
            RuntimeError: If no flowsheet is loaded or simulation fails.
        """
        if self._flowsheet is None:
            raise RuntimeError(
                "No flowsheet loaded. Use new_case() or load_case() first."
            )
        
        try:
            # Run the simulation
            self._flowsheet.Solve()
            
        except Exception as e:
            raise RuntimeError(f"Simulation failed: {e}")
    
    def save(self, output_path: str) -> None:
        """
        Save the current flowsheet to file.
        
        Args:
            output_path (str): Path where the flowsheet should be saved.
            
        Raises:
            RuntimeError: If no flowsheet is loaded or saving fails.
        """
        if self._flowsheet is None:
            raise RuntimeError(
                "No flowsheet loaded. Use new_case() or load_case() first."
            )
        
        output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save the flowsheet
            self._flowsheet.SaveToFile(str(output_path))
            
        except Exception as e:
            raise RuntimeError(f"Failed to save flowsheet to '{output_path}': {e}")
    
    def export_stream_data(
        self,
        csv_path: str | Path,
        *,
        stream_names: Sequence[str] | None = None,
    ) -> Path:
        """
        Dump basic results (mass-flow, T, P) for each material stream
        into a CSV.  If *stream_names* is given, only those streams are written.

        Args:
            csv_path: Path where the CSV file will be saved.
            stream_names: Optional sequence of stream names to export. 
                         If None, all material streams are exported.

        Returns:
            Path: The resolved path of the CSV created.
            
        Raises:
            RuntimeError: If no flowsheet is loaded or no matching streams found.
        """
        if self._flowsheet is None:
            raise RuntimeError("No flowsheet is loaded – run new_case() or load_case() first.")

        csv_path = Path(csv_path).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # DWSIM stores all sim objects in a dictionary.  Filter by ClassId.
        rows = []
        for obj in self._flowsheet.SimulationObjects.Values:
            if obj.ClassId != "MaterialStream":
                continue
            if stream_names and obj.Name not in stream_names:
                continue

            # MassFlow is a nullable double → GetValueOrDefault()
            rows.append(
                dict(
                    stream=obj.Name,
                    mass_flow_kg_s=obj.MassFlow.GetValueOrDefault(),
                    temperature_K=obj.PhaseProperties[1].Temperature.GetValueOrDefault(),
                    pressure_kPa=obj.PhaseProperties[1].Pressure.GetValueOrDefault() / 1000.0,
                )
            )

        if not rows:
            raise RuntimeError("No matching material streams found.")

        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        return csv_path
    
    def get_flowsheet(self) -> Optional[Any]:
        """
        Get the current flowsheet object.
        
        Returns:
            Optional[Any]: The current flowsheet object or None if not loaded.
        """
        return self._flowsheet
    
    def close(self) -> None:
        """
        Clean up resources and close the DWSIM bridge.
        """
        if self._flowsheet is not None:
            try:
                # Clean up flowsheet resources if possible
                if hasattr(self._flowsheet, 'Close'):
                    self._flowsheet.Close()
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                self._flowsheet = None
    
    def __enter__(self) -> 'DWSIMBridge':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.close()


# Example usage
if __name__ == "__main__":
    # Set the DWSIM DLL path (example)
    # os.environ['DWSIM_DLL_PATH'] = '/path/to/dwsim/libs'
    
    try:
        with DWSIMBridge() as bridge:
            # Create a new case
            flowsheet = bridge.new_case()
            print("New flowsheet created successfully")
            
            # Run simulation (if flowsheet has components)
            # bridge.run()
            
            # Save the flowsheet
            # bridge.save("output/simulation.dwsim")
            
    except RuntimeError as e:
        print(f"Error: {e}") 