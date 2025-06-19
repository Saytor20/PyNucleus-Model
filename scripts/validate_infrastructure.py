#!/usr/bin/env python3
"""
PyNucleus Infrastructure Validator
Validates all requirements files, Docker configurations, and dependencies.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import yaml


class InfrastructureValidator:
    """Validates PyNucleus infrastructure components."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.errors = []
        self.warnings = []
        
    def print_status(self, message: str, status: str = "info"):
        """Print colored status messages."""
        colors = {
            "success": "\033[0;32m✅",  # Green
            "error": "\033[0;31m❌",    # Red  
            "warning": "\033[1;33m⚠️",  # Yellow
            "info": "\033[0;34mℹ️"     # Blue
        }
        reset = "\033[0m"
        print(f"{colors.get(status, colors['info'])} {message}{reset}")
        
    def validate_requirements_file(self, req_file: Path) -> bool:
        """Validate a requirements file for proper pip specifiers."""
        if not req_file.exists():
            self.errors.append(f"Requirements file not found: {req_file}")
            return False
            
        try:
            with open(req_file, 'r') as f:
                lines = f.readlines()
                
            invalid_lines = []
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Check for invalid .* specifiers
                    if '.*' in line and not ('==' in line or '!=' in line):
                        invalid_lines.append((i, line))
                        
            if invalid_lines:
                self.errors.append(
                    f"Invalid pip specifiers in {req_file}:\n" +
                    "\n".join([f"  Line {num}: {line}" for num, line in invalid_lines])
                )
                return False
                
            self.print_status(f"Requirements file valid: {req_file.name}", "success")
            return True
            
        except Exception as e:
            self.errors.append(f"Error reading {req_file}: {e}")
            return False
            
    def validate_docker_files(self) -> bool:
        """Validate Docker configuration files."""
        docker_dir = self.project_root / "docker"
        
        # Check Dockerfiles
        dockerfiles = [
            docker_dir / "Dockerfile",
            docker_dir / "Dockerfile.api", 
            docker_dir / "Dockerfile.model"
        ]
        
        dockerfile_valid = True
        for dockerfile in dockerfiles:
            if not dockerfile.exists():
                self.errors.append(f"Dockerfile not found: {dockerfile}")
                dockerfile_valid = False
                continue
                
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                    
                # Basic validation - check for FROM instruction (may have comments before)
                lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
                if not lines or not lines[0].startswith('FROM'):
                    self.errors.append(f"Invalid Dockerfile format: {dockerfile}")
                    dockerfile_valid = False
                else:
                    self.print_status(f"Dockerfile valid: {dockerfile.name}", "success")
                    
            except Exception as e:
                self.errors.append(f"Error reading {dockerfile}: {e}")
                dockerfile_valid = False
                
        # Check docker-compose.yml
        compose_file = docker_dir / "docker-compose.yml"
        if compose_file.exists():
            try:
                with open(compose_file, 'r') as f:
                    compose_data = yaml.safe_load(f)
                    
                # Basic validation
                if 'services' not in compose_data:
                    self.errors.append("docker-compose.yml missing 'services' section")
                    dockerfile_valid = False
                else:
                    self.print_status("Docker Compose configuration valid", "success")
                    
            except Exception as e:
                self.errors.append(f"Error parsing docker-compose.yml: {e}")
                dockerfile_valid = False
        else:
            self.warnings.append("docker-compose.yml not found")
            
        return dockerfile_valid
        
    def validate_python_imports(self) -> bool:
        """Test if core dependencies can be imported."""
        core_deps = [
            'torch',
            'transformers', 
            'sentence_transformers',
            'langchain',
            'numpy',
            'pandas',
            'pydantic'
        ]
        
        import_errors = []
        for dep in core_deps:
            try:
                __import__(dep)
                self.print_status(f"Can import {dep}", "success")
            except ImportError:
                import_errors.append(dep)
                
        if import_errors:
            self.warnings.append(
                f"Cannot import some dependencies: {', '.join(import_errors)}\n"
                "This is normal if dependencies aren't installed yet."
            )
            
        return len(import_errors) == 0
        
    def check_directory_structure(self) -> bool:
        """Validate project directory structure."""
        required_dirs = [
            "src/pynucleus",
            "data",
            "configs", 
            "docker",
            "scripts",
            "docs"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
            else:
                self.print_status(f"Directory exists: {dir_path}", "success")
                
        if missing_dirs:
            self.errors.append(f"Missing directories: {', '.join(missing_dirs)}")
            return False
            
        return True
        
    def validate_pyproject_toml(self) -> bool:
        """Validate pyproject.toml configuration."""
        pyproject_file = self.project_root / "pyproject.toml"
        
        if not pyproject_file.exists():
            self.warnings.append("pyproject.toml not found")
            return True  # Not critical
            
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                self.warnings.append("Cannot validate pyproject.toml (tomllib/tomli not available)")
                return True
                
        try:
            with open(pyproject_file, 'rb') as f:
                pyproject_data = tomllib.load(f)
                
            # Basic validation
            if 'project' not in pyproject_data:
                self.errors.append("pyproject.toml missing [project] section")
                return False
                
            project = pyproject_data['project']
            if 'dependencies' not in project:
                self.warnings.append("pyproject.toml missing dependencies")
                
            self.print_status("pyproject.toml is valid", "success")
            return True
            
        except Exception as e:
            self.errors.append(f"Error parsing pyproject.toml: {e}")
            return False
            
    def run_validation(self) -> bool:
        """Run complete infrastructure validation."""
        print("🔍 PyNucleus Infrastructure Validation")
        print("=" * 50)
        
        # Validate requirements files
        print("\n📦 Validating Requirements Files...")
        req_files = [
            self.project_root / "requirements.txt",
            self.project_root / "requirements-colab.txt", 
            self.project_root / "requirements-minimal.txt",
            self.project_root / "docker/requirements_simple.txt"
        ]
        
        requirements_valid = all(
            self.validate_requirements_file(req_file) 
            for req_file in req_files if req_file.exists()
        )
        
        # Validate Docker files
        print("\n🐳 Validating Docker Configuration...")
        docker_valid = self.validate_docker_files()
        
        # Check directory structure
        print("\n📁 Validating Directory Structure...")
        structure_valid = self.check_directory_structure()
        
        # Validate pyproject.toml
        print("\n⚙️ Validating Project Configuration...")
        pyproject_valid = self.validate_pyproject_toml()
        
        # Test Python imports
        print("\n🐍 Testing Python Imports...")
        imports_valid = self.validate_python_imports()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Validation Summary")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
                
        if self.warnings:
            print(f"\n⚠️ Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
                
        overall_valid = (
            requirements_valid and 
            docker_valid and 
            structure_valid and 
            pyproject_valid
        )
        
        if overall_valid:
            self.print_status("✅ Infrastructure validation passed!", "success")
            print("\n🚀 Ready for deployment!")
        else:
            self.print_status("❌ Infrastructure validation failed", "error")
            print("\n🔧 Please fix the errors above before proceeding")
            
        return overall_valid


def main():
    """Main validation function."""
    validator = InfrastructureValidator()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        print("🔧 Auto-fix mode not implemented yet")
        print("Please manually fix the errors reported below\n")
        
    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 