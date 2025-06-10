#!/usr/bin/env python3
"""
System Diagnostic Script
Tests all major components of the PyNucleus system
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

def check_python_environment() -> Tuple[bool, List[str]]:
    """Check Python environment and dependencies."""
    print("\n🔍 Checking Python Environment...")
    issues = []
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"✅ Python version: {python_version}")
    
    # Check required packages
    required_packages = [
        "pythonnet",
        "numpy",
        "pandas",
        "sklearn",  # scikit-learn
        "faiss",
        "dotenv",   # python-dotenv
        "requests",
        "bs4",      # beautifulsoup4
        "tqdm"
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            issues.append(f"❌ {package} is not installed")
    
    return len(issues) == 0, issues

def check_dwsim_environment() -> Tuple[bool, List[str]]:
    """Check DWSIM environment setup."""
    print("\n🔍 Checking DWSIM Environment...")
    issues = []
    
    # Check DWSIM DLL path
    dwsim_path = os.getenv('DWSIM_DLL_PATH')
    if not dwsim_path:
        issues.append("❌ DWSIM_DLL_PATH not set")
    else:
        print(f"✅ DWSIM_DLL_PATH set to: {dwsim_path}")
        
        # Check DLL directory
        dll_path = Path(dwsim_path)
        if not dll_path.exists():
            issues.append(f"❌ DWSIM DLL directory does not exist: {dwsim_path}")
        else:
            # Count DLLs
            dlls = list(dll_path.glob("*.dll"))
            print(f"✅ Found {len(dlls)} DLL files:")
            for dll in dlls:
                print(f"   - {dll.name}")
            
            if not dlls:
                issues.append("❌ No DLL files found")
    
    return len(issues) == 0, issues

def check_rag_system() -> Tuple[bool, List[str]]:
    """Check RAG system components."""
    print("\n🔍 Checking RAG System...")
    issues = []
    
    # Check required directories
    required_dirs = [
        "inputs",
        "processed", 
        "vector_db",
        "reports",
        "src/rag",
        "tests/rag"
    ]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            issues.append(f"❌ Directory not found: {dir_path}")
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    # Check vector database
    vector_db = Path("vector_db")
    if not vector_db.exists():
        issues.append("❌ Vector database directory not found")
    else:
        print("✅ Vector database directory exists")
    
    return len(issues) == 0, issues

def check_docker_environment() -> Tuple[bool, List[str]]:
    """Check Docker environment."""
    print("\n🔍 Checking Docker Environment...")
    issues = []
    
    # Check Dockerfile
    if not Path("Dockerfile").exists():
        issues.append("❌ Dockerfile not found")
    else:
        print("✅ Dockerfile exists")
    
    # Check docker-compose.yml
    if not Path("docker-compose.yml").exists():
        issues.append("❌ docker-compose.yml not found")
    else:
        print("✅ docker-compose.yml exists")
    
    return len(issues) == 0, issues

def main() -> None:
    """Run all system diagnostics."""
    print("🚀 Starting System Diagnostics...")
    
    # Run all checks
    checks = [
        ("Python Environment", check_python_environment),
        ("DWSIM Environment", check_dwsim_environment),
        ("RAG System", check_rag_system),
        ("Docker Environment", check_docker_environment)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n📋 Running {name} Check...")
        try:
            success, issues = check_func()
            results.append((name, success, issues))
        except Exception as e:
            print(f"❌ Check failed with error: {e}")
            results.append((name, False, [str(e)]))
    
    # Print summary
    print("\n📊 Diagnostic Summary:")
    print("─" * 60)
    for name, success, issues in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:20} {status}")
        if issues:
            print("   Issues:")
            for issue in issues:
                print(f"   - {issue}")
    print("─" * 60)
    
    # Exit with appropriate code
    all_passed = all(success for _, success, _ in results)
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main() 