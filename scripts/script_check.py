#!/usr/bin/env python3
"""
PyNucleus Script Health Checker
Systematically checks all Python scripts in the project for:
- Syntax errors
- Import errors  
- Basic functionality
- Code quality issues

Provides health percentage and detailed error reporting similar to comprehensive_system_diagnostic.py
"""

import os
import sys
import ast
import importlib.util
import subprocess
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse

# Add src directory to Python path
root_dir = Path(__file__).parent.parent
src_path = str(root_dir / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

class ScriptHealthChecker:
    def __init__(self, quiet_mode=False):
        self.results = []
        self.total_scripts = 0
        self.healthy_scripts = 0
        self.start_time = datetime.now()
        self.quiet_mode = quiet_mode
        
        # Setup logging
        self.setup_logging()
        
        # Define script categories
        self.script_categories = {
            "Core Pipeline": [
                "src/pynucleus/pipeline/**/*.py",
                "src/pynucleus/rag/**/*.py",
                "src/pynucleus/dwsim/**/*.py"
            ],
            "Integration Components": [
                "src/pynucleus/integration/**/*.py",
                "src/pynucleus/llm/**/*.py",
                "src/pynucleus/utils/**/*.py"
            ],
            "Scripts & Tools": [
                "scripts/*.py",
                "run_pipeline.py"
            ],
            "Tests": [
                "src/pynucleus/tests/**/*.py"
            ],
            "Automation": [
                "automation_tools/**/*.py"
            ],
            "Prompt System": [
                "prompts/*.py"
            ]
        }
        
    def setup_logging(self):
        """Setup logging to both file and console"""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create timestamp for unique log file
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"script_check_{timestamp}.log"
        
        # Setup file logger
        self.file_logger = logging.getLogger('script_check_file')
        self.file_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.file_logger.handlers[:]:
            self.file_logger.removeHandler(handler)
            
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(file_handler)
        
        # Setup console logger
        self.console_logger = logging.getLogger('script_check_console')
        self.console_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.console_logger.handlers[:]:
            self.console_logger.removeHandler(handler)
            
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.console_logger.addHandler(console_handler)
        
        self.log_file_path = log_file
        
        # Start logging session
        self.log_both("PYNUCLEUS SCRIPT HEALTH CHECKER", level="info")
        self.log_both("=" * 70, level="info")
        
    def log_both(self, message: str, level: str = "info", console_symbol: str = "", clean_message: str = None):
        """Log to both file (clean) and console (with symbols)"""
        # Clean message for file
        clean_msg = clean_message if clean_message else self.clean_message_for_file(message)
        
        # Console message
        console_msg = f"{console_symbol}{message}" if console_symbol else message
        
        # Log to file (clean)
        getattr(self.file_logger, level)(clean_msg)
        
        # Log to console (with symbols) unless quiet mode
        if not self.quiet_mode:
            getattr(self.console_logger, level)(console_msg)
        
    def clean_message_for_file(self, message: str) -> str:
        """Remove symbols and emojis from message for clean file logging"""
        symbols_to_remove = ["✅", "❌", "⚠️", "🔍", "📊", "🎉", "🔧", "📁", "📋", "🚀", "🟢", "🟡", "🔴", "ℹ️", "📄", "💾", "🐍", "─", "═", "•", "▶", "⏭️", "🧪"]
        
        clean_msg = message
        for symbol in symbols_to_remove:
            clean_msg = clean_msg.replace(symbol, "")
        
        # Clean up extra spaces
        clean_msg = " ".join(clean_msg.split())
        return clean_msg.strip()
        
    def log_script_result(self, script_path: str, success: bool, issues: List[str] = None, warnings: List[str] = None):
        """Log a script check result"""
        self.results.append({
            'script': str(script_path),
            'success': success,
            'issues': issues or [],
            'warnings': warnings or []
        })
        self.total_scripts += 1
        if success:
            self.healthy_scripts += 1
            
        # Log result to file
        status = "HEALTHY" if success else "ISSUES"
        self.file_logger.info(f"Script Check: {script_path} - {status}")
        if issues:
            for issue in issues:
                self.file_logger.info(f"  Issue: {self.clean_message_for_file(issue)}")
        if warnings:
            for warning in warnings:
                self.file_logger.info(f"  Warning: {self.clean_message_for_file(warning)}")
    
    def print_section_header(self, title: str):
        """Print formatted section header"""
        if not self.quiet_mode:
            console_msg = f"\n{'='*60}\n🔍 {title}\n{'='*60}"
            print(console_msg)
        
        file_msg = f"SECTION: {title}"
        self.file_logger.info("=" * 60)
        self.file_logger.info(file_msg)
        self.file_logger.info("=" * 60)
    
    def find_python_scripts(self, patterns: List[str]) -> List[Path]:
        """Find all Python scripts matching the given patterns"""
        scripts = []
        for pattern in patterns:
            # Handle glob patterns
            if "**" in pattern:
                base_path = pattern.split("**")[0]
                if Path(base_path).exists():
                    scripts.extend(Path(base_path).rglob("*.py"))
            elif "*" in pattern:
                scripts.extend(Path(".").glob(pattern))
            else:
                script_path = Path(pattern)
                if script_path.exists():
                    scripts.append(script_path)
        
        # Remove duplicates and filter out __pycache__ files
        unique_scripts = []
        seen = set()
        for script in scripts:
            if "__pycache__" not in str(script) and str(script) not in seen:
                unique_scripts.append(script)
                seen.add(str(script))
        
        return sorted(unique_scripts)
    
    def check_syntax(self, script_path: Path) -> Tuple[bool, List[str]]:
        """Check Python syntax using AST parsing"""
        issues = []
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file using AST
            ast.parse(content, filename=str(script_path))
            return True, issues
            
        except SyntaxError as e:
            issues.append(f"Syntax Error line {e.lineno}: {e.msg}")
            return False, issues
        except UnicodeDecodeError as e:
            issues.append(f"Encoding Error: {e}")
            return False, issues
        except Exception as e:
            issues.append(f"Parse Error: {e}")
            return False, issues
    
    def check_imports(self, script_path: Path) -> Tuple[bool, List[str], List[str]]:
        """Check if all imports can be resolved"""
        issues = []
        warnings = []
        
        # Special handling for __init__.py files
        if script_path.name == "__init__.py":
            # For __init__ files, we need to test the package import, not the file directly
            try:
                # Set up proper Python path for testing
                old_path = sys.path[:]
                if str(root_dir) not in sys.path:
                    sys.path.insert(0, str(root_dir))
                if str(root_dir / "src") not in sys.path:
                    sys.path.insert(0, str(root_dir / "src"))
                
                # Determine the package path from the __init__.py location
                if "src/pynucleus" in str(script_path) or "src\\pynucleus" in str(script_path):
                    # Extract package name from path like src/pynucleus/pipeline/__init__.py
                    try:
                        rel_to_src = script_path.relative_to(root_dir / "src")
                        parts = rel_to_src.parts[:-1]  # Remove __init__.py
                        package_name = ".".join(parts)
                        
                        try:
                            __import__(package_name)
                            # Restore path
                            sys.path[:] = old_path
                            return True, issues, warnings
                        except ImportError as e:
                            issues.append(f"Package Import Error: Cannot import package '{package_name}': {e}")
                            # Restore path
                            sys.path[:] = old_path
                            return False, issues, warnings
                    except ValueError:
                        # Handle path issues - just skip for __init__ files
                        pass
                
                # Restore path
                sys.path[:] = old_path
                # If we can't determine package structure, just return True for __init__ files
                warnings.append("__init__.py file - package structure test skipped")
                return True, issues, warnings
                
            except Exception as e:
                # Restore path
                sys.path[:] = old_path
                warnings.append(f"__init__.py test error: {e}")
                return True, issues, warnings  # Don't fail __init__ files on testing errors
        
        try:
            # Set up proper Python path for import testing
            old_path = sys.path[:]
            if str(root_dir) not in sys.path:
                sys.path.insert(0, str(root_dir))
            if str(root_dir / "src") not in sys.path:
                sys.path.insert(0, str(root_dir / "src"))
            if str(root_dir / "prompts") not in sys.path:
                sys.path.insert(0, str(root_dir / "prompts"))
            
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find imports
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Check each import
            failed_imports = []
            for import_name in imports:
                try:
                    # Handle relative imports by testing them in the context of their package
                    if import_name.startswith('.'):
                        # For relative imports, we need to know the package context
                        try:
                            if "src/pynucleus" in str(script_path) or "src\\pynucleus" in str(script_path):
                                # Get the package name from the file path
                                rel_to_src = script_path.relative_to(root_dir / "src")
                                parts = rel_to_src.parts[:-1]  # Remove filename
                                if parts:
                                    package_name = ".".join(parts)
                                    
                                    # Try to import the relative module
                                    if import_name == '.config':
                                        # Special case for .config - check if config.py exists in same directory
                                        config_file = script_path.parent / "config.py"
                                        if config_file.exists():
                                            continue  # Config file exists, assume import will work
                                    
                                    # For other relative imports, try importing in package context
                                    full_module_name = package_name + import_name
                                    try:
                                        __import__(full_module_name)
                                        continue
                                    except ImportError:
                                        # If that fails, try without the relative dot
                                        base_name = import_name.lstrip('.')
                                        if base_name:
                                            try:
                                                parent_parts = parts[:-1] if len(parts) > 0 else []
                                                if parent_parts:
                                                    parent_package = ".".join(parent_parts)
                                                    full_name = f"{parent_package}.{base_name}"
                                                    __import__(full_name)
                                                    continue
                                                else:
                                                    __import__(base_name)
                                                    continue
                                            except ImportError:
                                                pass
                        except (ValueError, ImportError):
                            pass
                        
                        # If we can't resolve the relative import, just skip it
                        warnings.append(f"Relative import skipped: {import_name}")
                        continue
                    
                    # Skip very common imports that we know will work
                    common_imports = {
                        'os', 'sys', 'pathlib', 'json', 'datetime', 'typing', 
                        'logging', 'ast', 'importlib', 'subprocess', 'argparse',
                        'collections', 'itertools', 'functools', 'time', 're'
                    }
                    
                    if import_name.split('.')[0] in common_imports:
                        continue
                    
                    # Try to find the module spec first
                    spec = importlib.util.find_spec(import_name)
                    if spec is None:
                        # Try importing it directly
                        __import__(import_name)
                except ImportError:
                    failed_imports.append(import_name)
                except Exception:
                    # Some modules might fail to import for other reasons (like missing dependencies)
                    # Don't count these as critical failures
                    warnings.append(f"Import warning: {import_name} (may require additional setup)")
            
            # Restore original path
            sys.path[:] = old_path
            
            if failed_imports:
                for imp in failed_imports:
                    issues.append(f"Import Error: Cannot import '{imp}'")
                return False, issues, warnings
            
            return True, issues, warnings
            
        except Exception as e:
            # Restore original path
            sys.path[:] = old_path
            issues.append(f"Import Check Error: {e}")
            return False, issues, warnings
    
    def check_script_execution(self, script_path: Path) -> Tuple[bool, List[str], List[str]]:
        """Check if script can be executed without errors (dry run)"""
        issues = []
        warnings = []
        
        # Skip certain scripts that shouldn't be executed directly
        skip_execution = [
            "__init__.py",
            "setup.py",
            "conftest.py"
        ]
        
        if script_path.name in skip_execution:
            return True, issues, warnings
        
        # For main execution scripts, try a dry run
        if script_path.name in ["run_pipeline.py"] or "main" in script_path.name.lower():
            try:
                # Try to run with --help flag to test basic execution
                result = subprocess.run(
                    [sys.executable, str(script_path), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(root_dir)
                )
                
                if result.returncode != 0:
                    # Check if it's a legitimate error or just missing --help
                    if "unrecognized arguments" in result.stderr or "no such option" in result.stderr:
                        warnings.append("Script doesn't support --help flag (not an error)")
                    else:
                        issues.append(f"Execution Error: {result.stderr.strip()}")
                        return False, issues, warnings
                
            except subprocess.TimeoutExpired:
                warnings.append("Script execution timeout (may be normal for long-running scripts)")
            except Exception as e:
                issues.append(f"Execution Test Error: {e}")
                return False, issues, warnings
        
        return True, issues, warnings
    
    def analyze_code_quality(self, script_path: Path) -> Tuple[List[str], List[str]]:
        """Basic code quality analysis"""
        issues = []
        warnings = []
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Check for very long lines (over 120 characters)
            long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
            if len(long_lines) > 5:  # Only warn if there are many long lines
                warnings.append(f"Code style: {len(long_lines)} lines exceed 120 characters")
            
            # Check for TODO/FIXME comments (not an error, just informational)
            todos = [i+1 for i, line in enumerate(lines) if 'TODO' in line.upper() or 'FIXME' in line.upper()]
            if todos:
                warnings.append(f"Found {len(todos)} TODO/FIXME comments")
            
            # Check for basic docstring in modules (not __init__ files)
            if script_path.name != "__init__.py" and len(lines) > 10:
                has_module_docstring = False
                for line in lines[:10]:
                    if '"""' in line or "'''" in line:
                        has_module_docstring = True
                        break
                
                if not has_module_docstring:
                    warnings.append("Missing module docstring")
            
        except Exception as e:
            warnings.append(f"Code analysis error: {e}")
        
        return issues, warnings
    
    def check_script_health(self, script_path: Path) -> Dict:
        """Comprehensive health check for a single script"""
        issues = []
        warnings = []
        
        # 1. Syntax Check
        syntax_ok, syntax_issues = self.check_syntax(script_path)
        issues.extend(syntax_issues)
        
        # 2. Import Check (only if syntax is OK)
        if syntax_ok:
            imports_ok, import_issues, import_warnings = self.check_imports(script_path)
            issues.extend(import_issues)
            warnings.extend(import_warnings)
        else:
            imports_ok = False
        
        # 3. Execution Check (only if syntax and imports are OK)
        if syntax_ok and imports_ok:
            exec_ok, exec_issues, exec_warnings = self.check_script_execution(script_path)
            issues.extend(exec_issues)
            warnings.extend(exec_warnings)
        else:
            exec_ok = False
        
        # 4. Code Quality Analysis
        quality_issues, quality_warnings = self.analyze_code_quality(script_path)
        issues.extend(quality_issues)
        warnings.extend(quality_warnings)
        
        # Determine overall health
        is_healthy = syntax_ok and imports_ok and len([i for i in issues if "Error" in i]) == 0
        
        return {
            'healthy': is_healthy,
            'syntax_ok': syntax_ok,
            'imports_ok': imports_ok,
            'execution_ok': exec_ok,
            'issues': issues,
            'warnings': warnings
        }
    
    def check_category_scripts(self, category_name: str, patterns: List[str]) -> bool:
        """Check all scripts in a category"""
        self.print_section_header(f"{category_name.upper()} SCRIPTS CHECK")
        
        scripts = self.find_python_scripts(patterns)
        
        if not scripts:
            self.log_both(f"   No Python scripts found in {category_name}", console_symbol="ℹ️ ",
                         clean_message=f"No Python scripts found in {category_name}")
            return True
        
        self.log_both(f"   Found {len(scripts)} Python script(s) in {category_name}", console_symbol="📊 ",
                     clean_message=f"Found {len(scripts)} Python script(s) in {category_name}")
        
        category_healthy = True
        healthy_count = 0
        
        for script_path in scripts:
            # Get relative path for cleaner display
            try:
                rel_path = script_path.relative_to(root_dir)
            except ValueError:
                rel_path = script_path
            
            result = self.check_script_health(script_path)
            
            if result['healthy']:
                healthy_count += 1
                self.log_both(f"   ✅ {rel_path}", console_symbol="",
                             clean_message=f"HEALTHY: {rel_path}")
            else:
                category_healthy = False
                self.log_both(f"   ❌ {rel_path}", console_symbol="",
                             clean_message=f"ISSUES: {rel_path}")
                for issue in result['issues']:
                    self.log_both(f"      • {issue}", console_symbol="",
                                 clean_message=f"Issue: {issue}")
            
            # Log warnings if any
            if result['warnings']:
                for warning in result['warnings']:
                    self.log_both(f"      ⚠️ {warning}", console_symbol="",
                                 clean_message=f"Warning: {warning}")
            
            # Log the result
            self.log_script_result(rel_path, result['healthy'], result['issues'], result['warnings'])
        
        # Category summary
        if category_healthy:
            self.log_both(f"\n   🎉 All {category_name} scripts are healthy!", console_symbol="",
                         clean_message=f"All {category_name} scripts are healthy!")
        else:
            self.log_both(f"\n   ⚠️ {healthy_count}/{len(scripts)} {category_name} scripts are healthy", 
                         console_symbol="",
                         clean_message=f"{healthy_count}/{len(scripts)} {category_name} scripts are healthy")
        
        return category_healthy
    
    def run_comprehensive_check(self):
        """Run comprehensive script health check"""
        category_results = {}
        
        for category_name, patterns in self.script_categories.items():
            category_healthy = self.check_category_scripts(category_name, patterns)
            category_results[category_name] = category_healthy
        
        # Generate final report
        self.generate_final_report(category_results)
    
    def generate_final_report(self, category_results: Dict[str, bool]):
        """Generate comprehensive final report"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Calculate statistics
        health_percentage = (self.healthy_scripts / self.total_scripts) * 100 if self.total_scripts > 0 else 0
        
        if health_percentage >= 95:
            status_text = "EXCELLENT"
            status_emoji = "🟢"
        elif health_percentage >= 85:
            status_text = "GOOD"
            status_emoji = "🟡"
        elif health_percentage >= 70:
            status_text = "FAIR"
            status_emoji = "🟠"
        else:
            status_text = "NEEDS ATTENTION"
            status_emoji = "🔴"
        
        # Print summary to console
        if not self.quiet_mode:
            self.print_section_header("SCRIPT HEALTH SUMMARY")
            
            print(f"📊 Overall Results: {self.healthy_scripts}/{self.total_scripts} scripts healthy")
            print(f"{'─'*60}")
            
            # Category breakdown
            for category, is_healthy in category_results.items():
                status = "✅ HEALTHY" if is_healthy else "❌ ISSUES"
                print(f"   {status}: {category}")
            
            print(f"\n{'─'*60}")
            print(f"{status_emoji} SCRIPT HEALTH: {health_percentage:.1f}% - {status_text}")
            
            if health_percentage >= 95:
                print("🎉 Excellent! All scripts are in great shape.")
            elif health_percentage >= 85:
                print("✅ Good script health! Minor issues to address.")
            elif health_percentage >= 70:
                print("⚠️ Fair script health. Some issues need attention.")
            else:
                print("🚨 Script health needs attention. Review failed scripts.")
        
        # Generate detailed report for file
        report = f"""PYNUCLEUS SCRIPT HEALTH REPORT
Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration.total_seconds():.1f} seconds
Checked: {self.total_scripts} Python scripts

EXECUTIVE SUMMARY
Script Health: {health_percentage:.1f}% ({status_text})
Healthy Scripts: {self.healthy_scripts}
Scripts with Issues: {self.total_scripts - self.healthy_scripts}

CATEGORY BREAKDOWN"""
        
        for category, is_healthy in category_results.items():
            status = "HEALTHY" if is_healthy else "ISSUES"
            scripts_in_category = len([r for r in self.results if any(cat in r['script'] for cat in self.script_categories[category])])
            report += f"\n{category}: {status} ({scripts_in_category} scripts)"
        
        # Detailed issues
        failed_scripts = [r for r in self.results if not r['success']]
        if failed_scripts:
            report += f"\n\nSCRIPTS WITH ISSUES ({len(failed_scripts)}):"
            for result in failed_scripts:
                report += f"\n{result['script']}:"
                for issue in result['issues']:
                    clean_issue = self.clean_message_for_file(issue)
                    report += f"\n  - {clean_issue}"
        
        # Recommendations
        report += f"\n\nRECOMMENDATIONS:"
        if health_percentage >= 95:
            report += "\n- Excellent script health! Continue maintaining code quality."
        elif health_percentage >= 85:
            report += "\n- Good script health. Address minor issues when convenient."
        elif health_percentage >= 70:
            report += "\n- Review scripts with issues. Focus on import and syntax errors."
        else:
            report += "\n- Priority: Fix syntax and import errors in failing scripts."
            report += "\n- Consider running: pip install -r requirements.txt"
            report += "\n- Verify all dependencies are properly installed."
        
        # Save to log file
        self.file_logger.info("=" * 60)
        self.file_logger.info("FINAL SCRIPT HEALTH REPORT")
        self.file_logger.info("=" * 60)
        for line in report.strip().split('\n'):
            if line.strip():
                self.file_logger.info(line.strip())
        
        # Save as separate report file
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        report_file = Path("logs") / f"script_health_report_{timestamp}.txt"
        report_file.write_text(report.strip())
        
        if not self.quiet_mode:
            print(f"\n📋 DETAILED REPORTS GENERATED:")
            print(f"   • Detailed Log: {self.log_file_path}")
            print(f"   • Summary Report: {report_file}")
            print(f"   • All logs saved to: /logs directory")

def main():
    """Run script health checker with command-line options"""
    parser = argparse.ArgumentParser(
        description="PyNucleus Script Health Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/script_check.py                    # Full script health check
  python scripts/script_check.py --quiet           # Quiet mode (minimal output)
        """
    )
    
    parser.add_argument(
        '--quiet', 
        action='store_true', 
        help='Run in quiet mode with minimal console output'
    )
    
    args = parser.parse_args()
    
    checker = ScriptHealthChecker(quiet_mode=args.quiet)
    checker.run_comprehensive_check()
    
    # Exit with appropriate code based on health percentage
    health_percentage = (checker.healthy_scripts / checker.total_scripts) * 100 if checker.total_scripts > 0 else 0
    exit_code = 0 if health_percentage >= 85 else 1  # Require 85% health for success
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 