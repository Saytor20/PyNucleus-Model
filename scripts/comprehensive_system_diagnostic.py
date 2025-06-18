#!/usr/bin/env python3
"""
Comprehensive PyNucleus System Diagnostic & Testing Suite

DEPRECATED: This script is now a thin wrapper around the unified diagnostic runner.
Please use `python -m pynucleus.diagnostics.runner --full` instead.

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
        "scripts/comprehensive_system_diagnostic.py is deprecated. "
        "Use 'python -m pynucleus.diagnostics.runner --full' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    print("⚠️  DEPRECATION WARNING:")
    print("   This script is deprecated and will be removed in a future version.")
    print("   Please use the unified diagnostic runner instead:")
    print("   $ python -m pynucleus.diagnostics.runner --full")
    print()
    
    try:
        from pynucleus.diagnostics.runner import DiagnosticRunner
        
        # Parse basic arguments to maintain compatibility
        import argparse
        parser = argparse.ArgumentParser(description="PyNucleus System Diagnostic (deprecated)")
        parser.add_argument('--test', action='store_true', help='Test suite mode (mapped to --quick)')
        parser.add_argument('--quiet', action='store_true', help='Quiet mode')
        parser.add_argument('--mock', action='store_true', help='Mock testing (mapped to --full)')
        parser.add_argument('--validation', action='store_true', help='Validation tests (mapped to --full)')
        
        args = parser.parse_args()
        
        # Create runner and execute
        runner = DiagnosticRunner(quiet_mode=args.quiet)
        
        if args.test:
            runner.run_quick_diagnostic()
        else:
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