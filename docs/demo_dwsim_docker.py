#!/usr/bin/env python3
"""
DWSIM Docker Demo Script

Demonstrates how to use the Docker-based DWSIM workflow in PyNucleus-Model.
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pynucleus.sim_bridge.dwsim_workflow import (
    get_docker_workflow_status,
    run_dwsim_simulation_docker,
    quick_dwsim_docker_demo
)


def main():
    """Main demo function."""
    print("🐋 DWSIM Docker Integration Demo")
    print("=" * 50)
    
    # Show current status
    print("\n📊 Current Status:")
    status = get_docker_workflow_status()
    
    print(f"Service URL: {status['service_url']}")
    print(f"Service Available: {'✅' if status['service_available'] else '❌'}")
    print(f"API Healthy: {'✅' if status['api_healthy'] else '❌'}")
    print(f"Bridge Status: {status['bridge_status']}")
    print(f"Examples Available: {status['dwsim_files_count']} files")
    
    if not status['service_available']:
        print("\n❌ DWSIM Docker service is not available!")
        print("\n💡 To start the service:")
        print("   docker-compose up dwsim-service")
        print("\n   Wait for the service to show as 'healthy' before running simulations.")
        return 1
    
    if not status['api_healthy']:
        print("\n⚠️  DWSIM service is running but API is not healthy")
        details = status.get('dwsim_service_details', {})
        if details:
            print(f"   Reason: {details.get('message', 'Unknown')}")
            missing_dlls = details.get('missingDlls', [])
            if missing_dlls:
                print(f"   Missing DLLs: {missing_dlls}")
        return 1
    
    print("\n✅ DWSIM Docker service is ready!")
    
    # Run the demo
    print("\n🚀 Running Demo...")
    success = quick_dwsim_docker_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        
        # Show example usage
        print("\n📚 Example Usage:")
        print("```python")
        print("from dwsim_workflow import run_dwsim_simulation_docker")
        print("")
        print("# Run a simulation")
        print("result = run_dwsim_simulation_docker(")
        print("    'path/to/your/simulation.dwsim',")
        print("    'results/output.csv'")
        print(")")
        print("```")
        
        return 0
    else:
        print("\n❌ Demo failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 