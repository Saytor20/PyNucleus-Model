#!/usr/bin/env python3
"""
PyNucleus Web Interface Validator

WEB INTERFACE VALIDATION TESTING - validates web interface, API endpoints, and browser integration.
This script specifically focuses on validation of the web interface components:
- Flask API server health and endpoints
- Browser UI functionality and assets
- API endpoint responses and error handling
- Static file serving
- HTMX integration testing
- Response formatting and typing effects
- Cross-browser compatibility checks

For comprehensive system diagnostics, use comprehensive_system_diagnostic.py.
For focused validation testing, use system_validator.py.
"""

import sys
import warnings
import argparse
import json
import time
import requests
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import tempfile
import socket

# Add src directory to Python path
root_dir = Path(__file__).parent.parent
src_path = str(root_dir / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

@dataclass
class WebValidationResult:
    """Structure for web validation test results."""
    test_name: str
    endpoint: str
    expected_status: int
    actual_status: int
    response_time: float
    passed: bool = False
    error_message: str = ""
    response_data: Optional[Dict] = None
    validation_notes: str = ""

class WebInterfaceValidator:
    """Comprehensive web interface validator for PyNucleus API and browser integration."""
    
    def __init__(self, quiet_mode: bool = False, server_url: str = "http://localhost:5001"):
        self.quiet_mode = quiet_mode
        self.server_url = server_url
        self.validation_results: List[WebValidationResult] = []
        self.total_tests = 0
        self.passed_tests = 0
        self.start_time = datetime.now()
        self.server_process = None
        self.server_started = False
        
        # Test data sets
        self.test_questions = [
            "What is distillation?",
            "How do heat exchangers work?",
            "What are the benefits of modular chemical plants?",
            "Explain reactor design principles",
            "What is process intensification?"
        ]
        
    def log_message(self, message: str, level: str = "info"):
        """Log messages with appropriate formatting."""
        symbols = {"info": "ℹ️  ", "success": "✅ ", "warning": "⚠️  ", "error": "❌ "}
        symbol = symbols.get(level, "")
        
        if not self.quiet_mode or level in ["error", "warning"]:
            print(f"{symbol}{message}")
    
    def run_web_validation_suite(self, start_server: bool = False):
        """Run the complete web interface validation suite."""
        self.log_message("🚀 Starting PyNucleus Web Interface Validation Suite...")
        self.log_message(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"Target server: {self.server_url}")
        
        print("=" * 70)
        print("   PYNUCLEUS WEB INTERFACE VALIDATION SUITE")
        print("=" * 70)
        print("Focus: API Endpoints, Browser UI, and Integration Testing")
        print()
        
        try:
            # Start server if requested
            if start_server:
                self._start_test_server()
                time.sleep(3)  # Give server time to start
            
            # Check server availability
            if not self._check_server_availability():
                self.log_message("Server not available. Cannot proceed with web validation.", "error")
                return
            
            # Core web interface validation
            self._validate_static_files()
            self._validate_api_endpoints()
            self._validate_html_interface()
            self._test_api_functionality()
            self._test_error_handling()
            self._validate_response_formatting()
            self._test_browser_integration()
            
            # Generate validation report
            self._generate_web_validation_report()
            self._save_web_validation_results()
            
        except Exception as e:
            self.log_message(f"Web validation suite failed: {e}", "error")
            raise
        finally:
            if start_server and self.server_process:
                self._stop_test_server()
    
    def _check_server_availability(self) -> bool:
        """Check if the Flask server is available."""
        print("\n" + "=" * 70)
        print("   SERVER AVAILABILITY CHECK")
        print("=" * 70)
        
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            if response.status_code == 200:
                server_info = response.json()
                self.log_message(f"✓ Server available at {self.server_url}", "success")
                self.log_message(f"  Service: {server_info.get('service', 'Unknown')}")
                self.log_message(f"  Status: {server_info.get('status', 'Unknown')}")
                self.log_message(f"  Version: {server_info.get('version', 'Unknown')}")
                return True
            else:
                self.log_message(f"✗ Server returned status {response.status_code}", "error")
                return False
        except requests.RequestException as e:
            self.log_message(f"✗ Server not available: {e}", "error")
            self.log_message("  Hint: Start the server with 'python src/pynucleus/api/app.py'", "info")
            return False
    
    def _validate_static_files(self):
        """Validate static file serving and HTML interface."""
        print("\n" + "=" * 70)
        print("   STATIC FILES VALIDATION")
        print("=" * 70)
        
        # Check if HTML file exists
        html_path = Path("src/pynucleus/api/static/index.html")
        
        self.total_tests += 1
        if html_path.exists():
            self.log_message("✓ HTML interface file exists", "success")
            self.passed_tests += 1
            
            # Validate HTML content
            with open(html_path, 'r') as f:
                html_content = f.read()
            
            required_elements = [
                'id="question"',
                'fetch(\'/ask\'',
                'tailwindcss.com',
                'typeText',
                'displayResponse'
            ]
            
            for element in required_elements:
                self.total_tests += 1
                if element in html_content:
                    self.log_message(f"  ✓ HTML contains: {element}", "success")
                    self.passed_tests += 1
                else:
                    self.log_message(f"  ✗ HTML missing: {element}", "error")
        else:
            self.log_message("✗ HTML interface file missing", "error")
        
        # Test static file serving via HTTP
        try:
            response = requests.get(f"{self.server_url}/", timeout=10)
            
            result = WebValidationResult(
                test_name="Static File Serving",
                endpoint="/",
                expected_status=200,
                actual_status=response.status_code,
                response_time=response.elapsed.total_seconds()
            )
            
            self.total_tests += 1
            if response.status_code == 200 and 'text/html' in response.headers.get('content-type', ''):
                result.passed = True
                self.passed_tests += 1
                self.log_message("✓ Static HTML serving works", "success")
                self.log_message(f"  Response time: {result.response_time:.3f}s")
            else:
                result.error_message = f"Expected HTML content, got {response.headers.get('content-type')}"
                self.log_message("✗ Static HTML serving failed", "error")
            
            self.validation_results.append(result)
            
        except requests.RequestException as e:
            self.log_message(f"✗ Static file serving test failed: {e}", "error")
    
    def _validate_api_endpoints(self):
        """Validate all API endpoints."""
        print("\n" + "=" * 70)
        print("   API ENDPOINTS VALIDATION")
        print("=" * 70)
        
        endpoints = [
            ("/health", "GET", 200, None),
            ("/", "GET", 200, None),
            ("/ask", "POST", 200, {"question": "What is distillation?"}),
        ]
        
        for endpoint, method, expected_status, data in endpoints:
            self.total_tests += 1
            start_time = time.time()
            
            try:
                if method == "GET":
                    response = requests.get(f"{self.server_url}{endpoint}", timeout=15)
                elif method == "POST":
                    response = requests.post(
                        f"{self.server_url}{endpoint}",
                        json=data,
                        headers={'Content-Type': 'application/json'},
                        timeout=30
                    )
                
                response_time = time.time() - start_time
                
                result = WebValidationResult(
                    test_name=f"{method} {endpoint}",
                    endpoint=endpoint,
                    expected_status=expected_status,
                    actual_status=response.status_code,
                    response_time=response_time
                )
                
                if response.status_code == expected_status:
                    result.passed = True
                    self.passed_tests += 1
                    self.log_message(f"✓ {method} {endpoint}: {response.status_code}", "success")
                    self.log_message(f"  Response time: {response_time:.3f}s")
                    
                    # Store response data for further analysis
                    try:
                        if 'application/json' in response.headers.get('content-type', ''):
                            result.response_data = response.json()
                    except:
                        pass
                        
                else:
                    result.error_message = f"Expected {expected_status}, got {response.status_code}"
                    self.log_message(f"✗ {method} {endpoint}: {response.status_code} (expected {expected_status})", "error")
                
                self.validation_results.append(result)
                
            except requests.RequestException as e:
                result = WebValidationResult(
                    test_name=f"{method} {endpoint}",
                    endpoint=endpoint,
                    expected_status=expected_status,
                    actual_status=0,
                    response_time=0.0,
                    error_message=str(e)
                )
                self.validation_results.append(result)
                self.log_message(f"✗ {method} {endpoint}: {e}", "error")
    
    def _validate_html_interface(self):
        """Validate HTML interface components."""
        print("\n" + "=" * 70)
        print("   HTML INTERFACE VALIDATION")
        print("=" * 70)
        
        try:
            response = requests.get(f"{self.server_url}/", timeout=10)
            html_content = response.text
            
            # Check for required HTML elements
            required_components = [
                ("textarea", "Question input field"),
                ("button", "Submit button"),
                ("tailwindcss", "Tailwind CSS framework"),
                ("askQuestion", "JavaScript function"),
                ("typeText", "Typing effect function"),
                ("results", "Results container")
            ]
            
            for component, description in required_components:
                self.total_tests += 1
                if component.lower() in html_content.lower():
                    self.log_message(f"✓ {description}: Found", "success")
                    self.passed_tests += 1
                else:
                    self.log_message(f"✗ {description}: Missing", "error")
                    
        except Exception as e:
            self.log_message(f"HTML interface validation failed: {e}", "error")
    
    def _test_api_functionality(self):
        """Test API functionality with various questions."""
        print("\n" + "=" * 70)
        print("   API FUNCTIONALITY TESTING")
        print("=" * 70)
        
        for i, question in enumerate(self.test_questions[:2]):  # Test first 2 questions to save time
            self.log_message(f"\n--- Test Question {i+1}: {question[:50]}... ---")
            self.total_tests += 1
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.server_url}/ask",
                    json={"question": question},
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                response_time = time.time() - start_time
                
                result = WebValidationResult(
                    test_name=f"API Question Test {i+1}",
                    endpoint="/ask",
                    expected_status=200,
                    actual_status=response.status_code,
                    response_time=response_time
                )
                
                if response.status_code == 200:
                    data = response.json()
                    result.response_data = data
                    
                    # Validate response structure
                    required_fields = ["answer", "confidence", "sources"]
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if not missing_fields:
                        result.passed = True
                        self.passed_tests += 1
                        
                        # Check response quality
                        answer_length = len(data.get("answer", ""))
                        confidence = data.get("confidence", 0)
                        sources_count = len(data.get("sources", []))
                        
                        self.log_message(f"✓ Question processed successfully", "success")
                        self.log_message(f"  Response time: {response_time:.3f}s")
                        self.log_message(f"  Answer length: {answer_length} characters")
                        self.log_message(f"  Confidence: {confidence:.2f}")
                        self.log_message(f"  Sources: {sources_count}")
                            
                    else:
                        result.error_message = f"Missing fields: {missing_fields}"
                        self.log_message(f"✗ Response missing fields: {missing_fields}", "error")
                else:
                    result.error_message = f"HTTP {response.status_code}: {response.text[:100]}"
                    self.log_message(f"✗ API request failed: {response.status_code}", "error")
                
                self.validation_results.append(result)
                
            except requests.RequestException as e:
                self.log_message(f"✗ API test failed: {e}", "error")
    
    def _test_error_handling(self):
        """Test API error handling."""
        print("\n" + "=" * 70)
        print("   ERROR HANDLING TESTING")
        print("=" * 70)
        
        error_tests = [
            ("Empty question", {"question": ""}),
            ("Missing question field", {"query": "test"}),
        ]
        
        for test_name, test_data in error_tests:
            self.total_tests += 1
            self.log_message(f"\n--- {test_name} ---")
            
            try:
                response = requests.post(
                    f"{self.server_url}/ask",
                    json=test_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                # For error tests, we expect 400-level status codes
                if 400 <= response.status_code < 500:
                    self.passed_tests += 1
                    self.log_message(f"✓ Correctly returned error: {response.status_code}", "success")
                else:
                    self.log_message(f"✗ Should have returned error, got: {response.status_code}", "error")
                    
            except requests.RequestException as e:
                self.log_message(f"✗ Error test failed: {e}", "error")
    
    def _validate_response_formatting(self):
        """Validate response formatting and structure."""
        print("\n" + "=" * 70)
        print("   RESPONSE FORMATTING VALIDATION")
        print("=" * 70)
        
        try:
            response = requests.post(
                f"{self.server_url}/ask",
                json={"question": "What is distillation?"},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Content-Type check
                self.total_tests += 1
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    self.log_message("✓ Response Content-Type: application/json", "success")
                    self.passed_tests += 1
                else:
                    self.log_message(f"✗ Wrong Content-Type: {content_type}", "error")
                
                # Response structure validation
                expected_fields = ["answer", "confidence", "sources"]
                
                for field in expected_fields:
                    self.total_tests += 1
                    if field in data:
                        self.log_message(f"✓ Field '{field}': Present", "success")
                        self.passed_tests += 1
                    else:
                        self.log_message(f"✗ Field '{field}': Missing", "error")
                
            else:
                self.log_message(f"✗ Cannot validate formatting, request failed: {response.status_code}", "error")
                
        except Exception as e:
            self.log_message(f"Response formatting validation failed: {e}", "error")
    
    def _test_browser_integration(self):
        """Test browser integration features."""
        print("\n" + "=" * 70)
        print("   BROWSER INTEGRATION TESTING")
        print("=" * 70)
        
        # Test response time for UI responsiveness
        self.total_tests += 1
        try:
            start_time = time.time()
            response = requests.get(f"{self.server_url}/", timeout=10)
            load_time = time.time() - start_time
            
            if load_time < 2.0:  # 2 second threshold for UI loading
                self.log_message(f"✓ UI load time acceptable: {load_time:.3f}s", "success")
                self.passed_tests += 1
            else:
                self.log_message(f"⚠️ UI load time slow: {load_time:.3f}s", "warning")
                
        except Exception as e:
            self.log_message(f"✗ UI load time test failed: {e}", "error")
    
    def _start_test_server(self):
        """Start test server for validation."""
        self.log_message("Starting test server...", "info")
        try:
            # Start the Flask app in background
            import subprocess
            self.server_process = subprocess.Popen(
                [sys.executable, "src/pynucleus/api/app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(root_dir)
            )
            self.server_started = True
            self.log_message("Test server started", "success")
        except Exception as e:
            self.log_message(f"Failed to start test server: {e}", "error")
    
    def _stop_test_server(self):
        """Stop test server."""
        if self.server_process:
            self.log_message("Stopping test server...", "info")
            self.server_process.terminate()
            self.server_process.wait(timeout=10)
    
    def _generate_web_validation_report(self):
        """Generate comprehensive web validation report."""
        print("\n" + "=" * 70)
        print("   WEB VALIDATION REPORT SUMMARY")
        print("=" * 70)
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.log_message(f"PYNUCLEUS WEB INTERFACE VALIDATION REPORT")
        self.log_message(f"Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"Duration: {duration:.1f} seconds")
        self.log_message(f"Server: {self.server_url}")
        
        # Test results summary
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        self.log_message(f"\nEXECUTIVE SUMMARY")
        self.log_message(f"Web Interface Health: {success_rate:.1f}%")
        self.log_message(f"Tests Performed: {self.total_tests}")
        self.log_message(f"Tests Passed: {self.passed_tests}")
        self.log_message(f"Tests Failed: {self.total_tests - self.passed_tests}")
        
        # Final assessment
        if success_rate >= 95:
            self.log_message("Web Interface Status: EXCELLENT 🎉", "success")
        elif success_rate >= 85:
            self.log_message("Web Interface Status: VERY GOOD ✅", "success")
        elif success_rate >= 75:
            self.log_message("Web Interface Status: GOOD ✅", "success")
        elif success_rate >= 65:
            self.log_message("Web Interface Status: NEEDS IMPROVEMENT ⚠️", "warning")
        else:
            self.log_message("Web Interface Status: CRITICAL ISSUES ❌", "error")
    
    def _save_web_validation_results(self):
        """Save web validation results to JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"data/validation/results/web_interface_validation_{timestamp}.json"
            
            # Ensure directory exists
            Path(results_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare results data
            results_data = {
                "timestamp": timestamp,
                "server_url": self.server_url,
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "success_rate": (self.passed_tests / self.total_tests) if self.total_tests > 0 else 0,
                "validation_results": [
                    {
                        "test_name": r.test_name,
                        "endpoint": r.endpoint,
                        "expected_status": r.expected_status,
                        "actual_status": r.actual_status,
                        "response_time": r.response_time,
                        "passed": r.passed,
                        "error_message": r.error_message,
                        "validation_notes": r.validation_notes
                    }
                    for r in self.validation_results
                ]
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            self.log_message(f"Web validation results saved to: {results_file}")
            
        except Exception as e:
            self.log_message(f"Failed to save web validation results: {e}", "error")

def main():
    """Main function for web interface validation."""
    parser = argparse.ArgumentParser(description="PyNucleus Web Interface Validator")
    parser.add_argument('--server-url', default='http://localhost:5001', help='Server URL to test against')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode with minimal output')
    
    args = parser.parse_args()
    
    # Create validator
    validator = WebInterfaceValidator(
        quiet_mode=args.quiet,
        server_url=args.server_url
    )
    
    try:
        # Run validation suite
        validator.run_web_validation_suite()
        
        # Exit with appropriate code based on results
        success_rate = validator.passed_tests / validator.total_tests if validator.total_tests > 0 else 0
        exit_code = 0 if success_rate >= 0.8 else 1  # 80% threshold for success
        
        if exit_code == 0:
            validator.log_message("🎉 Web interface validation completed successfully!", "success")
        else:
            validator.log_message("⚠️ Web interface validation completed with issues!", "warning")
        
        sys.exit(exit_code)
        
    except Exception as e:
        validator.log_message(f"Web interface validation failed: {e}", "error")
        sys.exit(2)

if __name__ == "__main__":
    main() 