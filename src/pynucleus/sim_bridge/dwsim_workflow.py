#!/usr/bin/env python3
"""
DWSIM Docker Workflow Module

Provides Docker-based DWSIM simulation workflow functionality.
This module handles Docker service status, simulation execution, and demo functions.
"""

import sys
import os
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional


def get_docker_workflow_status() -> Dict[str, Any]:
    """Get the status of the DWSIM Docker workflow service."""
    status = {
        'service_url': 'http://localhost:8080',
        'service_available': False,
        'api_healthy': False,
        'bridge_status': 'not_available',
        'dwsim_files_count': 0,
        'dwsim_service_details': {}
    }
    
    try:
        # Check if service is available
        response = requests.get(f"{status['service_url']}/health", timeout=5)
        if response.status_code == 200:
            status['service_available'] = True
            
            # Check API health
            health_response = requests.get(f"{status['service_url']}/api/health", timeout=5)
            if health_response.status_code == 200:
                status['api_healthy'] = True
                status['bridge_status'] = 'healthy'
                
                # Get service details
                try:
                    details_response = requests.get(f"{status['service_url']}/api/status", timeout=5)
                    if details_response.status_code == 200:
                        status['dwsim_service_details'] = details_response.json()
                except:
                    pass
            else:
                status['bridge_status'] = 'unhealthy'
                
    except requests.RequestException:
        status['bridge_status'] = 'connection_failed'
    
    # Count example DWSIM files
    dwsim_examples_dir = Path("examples/dwsim_files")
    if dwsim_examples_dir.exists():
        status['dwsim_files_count'] = len(list(dwsim_examples_dir.glob("*.dwsim")))
    
    return status


def run_dwsim_simulation_docker(dwsim_file_path: str, output_csv_path: str, 
                               service_url: str = "http://localhost:8080") -> Dict[str, Any]:
    """Run a DWSIM simulation using the Docker service."""
    result = {
        'success': False,
        'message': '',
        'output_file': output_csv_path,
        'simulation_data': {},
        'errors': []
    }
    
    try:
        # Check if DWSIM file exists
        if not Path(dwsim_file_path).exists():
            result['message'] = f"DWSIM file not found: {dwsim_file_path}"
            result['errors'].append(f"File not found: {dwsim_file_path}")
            return result
        
        # Prepare the request
        with open(dwsim_file_path, 'rb') as f:
            files = {'simulation_file': f}
            data = {'output_format': 'csv'}
            
            # Send simulation request
            response = requests.post(
                f"{service_url}/api/simulate",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout
            )
        
        if response.status_code == 200:
            # Save the CSV result
            with open(output_csv_path, 'w') as f:
                f.write(response.text)
            
            result['success'] = True
            result['message'] = f"Simulation completed successfully. Results saved to {output_csv_path}"
            
            # Try to parse simulation data
            try:
                import pandas as pd
                df = pd.read_csv(output_csv_path)
                result['simulation_data'] = {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'summary': df.describe().to_dict() if len(df) > 0 else {}
                }
            except Exception as e:
                result['errors'].append(f"Could not parse simulation results: {e}")
                
        else:
            result['message'] = f"Simulation failed with status code: {response.status_code}"
            result['errors'].append(f"HTTP {response.status_code}: {response.text}")
            
    except requests.RequestException as e:
        result['message'] = f"Network error: {e}"
        result['errors'].append(f"Network error: {e}")
    except Exception as e:
        result['message'] = f"Unexpected error: {e}"
        result['errors'].append(f"Unexpected error: {e}")
    
    return result


def quick_dwsim_docker_demo() -> bool:
    """Run a quick demo of the DWSIM Docker workflow."""
    print("🚀 Running DWSIM Docker Demo...")
    
    try:
        # Check service status first
        status = get_docker_workflow_status()
        
        if not status['service_available']:
            print("❌ DWSIM Docker service is not available for demo")
            return False
        
        if not status['api_healthy']:
            print("❌ DWSIM API is not healthy for demo")
            return False
        
        # Look for a demo DWSIM file
        demo_files = [
            "examples/dwsim_files/demo.dwsim",
            "examples/dwsim_files/simple_distillation.dwsim",
            "data/dwsim_examples/demo.dwsim"
        ]
        
        demo_file = None
        for file_path in demo_files:
            if Path(file_path).exists():
                demo_file = file_path
                break
        
        if not demo_file:
            print("⚠️  No demo DWSIM file found. Creating a minimal demo scenario...")
            # Create a simple demo output
            demo_output = "results/demo_simulation_output.csv"
            Path(demo_output).parent.mkdir(parents=True, exist_ok=True)
            
            # Create a simple CSV output for demo
            with open(demo_output, 'w') as f:
                f.write("Stream,Temperature_C,Pressure_kPa,Mass_Flow_kg_h\n")
                f.write("Feed,25.0,101.325,1000.0\n")
                f.write("Product,78.4,101.325,950.0\n")
                f.write("Waste,25.0,101.325,50.0\n")
            
            print(f"✅ Demo simulation completed (simulated). Results: {demo_output}")
            return True
        
        # Run the actual simulation
        output_file = "results/demo_docker_simulation.csv"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Running simulation: {demo_file}")
        result = run_dwsim_simulation_docker(demo_file, output_file)
        
        if result['success']:
            print(f"✅ Demo completed successfully!")
            print(f"📄 Results saved to: {result['output_file']}")
            
            if result['simulation_data']:
                sim_data = result['simulation_data']
                print(f"📊 Simulation data: {sim_data['rows']} rows, {len(sim_data['columns'])} columns")
            
            return True
        else:
            print(f"❌ Demo failed: {result['message']}")
            for error in result['errors']:
                print(f"   Error: {error}")
            return False
            
    except Exception as e:
        print(f"❌ Demo failed with exception: {e}")
        return False


# Utility functions for backward compatibility
def check_docker_status():
    """Alias for get_docker_workflow_status()"""
    return get_docker_workflow_status()


def run_simulation(dwsim_file, output_file):
    """Simplified simulation runner"""
    return run_dwsim_simulation_docker(dwsim_file, output_file)


if __name__ == "__main__":
    # Quick test when run directly
    print("Testing DWSIM Docker Workflow...")
    status = get_docker_workflow_status()
    print(f"Service status: {status}")
    
    if status['service_available'] and status['api_healthy']:
        print("Running demo...")
        success = quick_dwsim_docker_demo()
        print(f"Demo result: {'Success' if success else 'Failed'}")
    else:
        print("Service not ready for demo") 