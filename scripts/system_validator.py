#!/usr/bin/env python3
"""
PyNucleus System Validator

DEPRECATED: This script is now a thin wrapper around the unified diagnostic runner.
Please use `python -m pynucleus.diagnostics.runner --quick` or `--full` instead.

This wrapper is maintained for backward compatibility but will be removed in a future version.
"""

import sys
import warnings
from pathlib import Path

# Add src directory to Python path
root_dir = Path(__file__).parent.parent
src_path = str(root_dir / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def main():
    """Main function - deprecated wrapper"""
    warnings.warn(
        "scripts/system_validator.py is deprecated. "
        "Use 'python -m pynucleus.diagnostics.runner --quick' or '--full' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    print("⚠️  DEPRECATION WARNING:")
    print("   This script is deprecated and will be removed in a future version.")
    print("   Please use the unified diagnostic runner instead:")
    print("   $ python -m pynucleus.diagnostics.runner --quick  # For quick validation")
    print("   $ python -m pynucleus.diagnostics.runner --full   # For comprehensive validation")
    print()
    
    try:
        from pynucleus.diagnostics.runner import DiagnosticRunner
        
        # Parse basic arguments to maintain compatibility
        import argparse
        parser = argparse.ArgumentParser(description="PyNucleus System Validator (deprecated)")
        parser.add_argument('--quick', action='store_true', help='Quick validation mode')
        parser.add_argument('--notebook', action='store_true', help='Include notebook testing (mapped to --full)')
        parser.add_argument('--quiet', action='store_true', help='Quiet mode')
        parser.add_argument('--validation', action='store_true', help='Validation tests (mapped to --full)')
        parser.add_argument('--citations', action='store_true', help='Citation tests (mapped to --full)')
        
        args = parser.parse_args()
        
        # Create runner and execute
        runner = DiagnosticRunner(quiet_mode=args.quiet)
        
        # Map old arguments to new runner modes
        if args.quick:
            runner.run_quick_diagnostic()
        elif args.notebook or args.validation or args.citations:
            runner.run_full_diagnostic()
        else:
            # Default to full validation to maintain original behavior
            runner.run_full_diagnostic()
        
        # Exit with appropriate code
        success_rate = runner.passed_checks / runner.total_checks if runner.total_checks > 0 else 0
        exit_code = 0 if success_rate >= 0.9 else 1
        sys.exit(exit_code)
        
    except ImportError as e:
        print(f"❌ Error importing unified diagnostic runner: {e}")
        print("   Falling back to basic environment check...")
        
        # Basic fallback check
        import importlib
        required_packages = ["numpy", "pandas", "requests", "tqdm", "jinja2"]
        missing = []
        
        for package in required_packages:
            try:
                importlib.import_module(package.replace("-", "_"))
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} (missing)")
                missing.append(package)
        
        if missing:
            print(f"\n❌ Missing required packages: {', '.join(missing)}")
            sys.exit(1)
        else:
            print("\n✅ Basic environment check passed")
            sys.exit(0)

if __name__ == "__main__":
    main() 