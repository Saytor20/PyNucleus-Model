#!/usr/bin/env python3
"""
Load a .dwsim case, run it, and dump all material-stream data to CSV.
"""
import os, sys
from pathlib import Path

# Local import; adjust if your PYTHONPATH differs.
sys.path.append("src")
from sim_bridge.dwsim_bridge import DWSIMBridge

# ---- config (hard-coded for the demo; parameterise later) ------------------
os.environ.setdefault(
    "DWSIM_DLL_PATH", "/Applications/DWSIM.app/Contents/MonoBundle"
)
CASE_FILE = Path("examples/simple_demo.dwsim")     # a small sample case
CSV_OUT   = Path("outputs/stream_results.csv")
# ----------------------------------------------------------------------------

def main():
    """Main function to load, run, and export DWSIM simulation data."""
    try:
        with DWSIMBridge() as br:
            print(f"Loading case: {CASE_FILE}")
            br.load_case(CASE_FILE)
            
            print("Running simulation...")
            br.run()
            
            print(f"Exporting stream data to: {CSV_OUT}")
            csv_path = br.export_stream_data(CSV_OUT)
            print(f"✔ results written to {csv_path}")
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure the DWSIM case file exists at the specified path.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ Runtime error: {e}")
        print("Check DWSIM_DLL_PATH environment variable and DLL files.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 