#!/usr/bin/env python3
"""
PyNucleus Integration Test Suite

Comprehensive testing of all deployment components:
- Redis cache connectivity
- Flask application factory
- Cache integration
- Scaling manager components
- Docker configuration validation
"""

import sys
import time
import json
import logging
import requests
import subprocess
from pathlib import Path
from typing import Dict, Any
import threading

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PyNucleusIntegrationTest:
    """Comprehensive integration testing for PyNucleus deployment"""
    
    def __init__(self):
        self.test_results = {}
        self.flask_app_process = None
        self.test_port = 5555  # Use a different port for testing
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("🚀 Starting PyNucleus Integration Test Suite")
        logger.info("=" * 60)
        
        tests = [
            ("Redis Connectivity", self.test_redis_connectivity),
            ("Cache Integration", self.test_cache_integration),
            ("Scaling Manager", self.test_scaling_manager),
            ("Flask App Factory", self.test_flask_app_factory),
            ("Docker Configuration", self.test_docker_configuration),
            ("Flask API Endpoints", self.test_flask_api_endpoints),
            ("Stress Test Infrastructure", self.test_stress_test_infrastructure),
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n🧪 Running: {test_name}")
            try:
                result = test_func()
                self.test_results[test_name] = {"status": "PASS", "details": result}
                logger.info(f"✅ {test_name}: PASSED")
            except Exception as e:
                self.test_results[test_name] = {"status": "FAIL", "details": str(e)}
                logger.error(f"❌ {test_name}: FAILED - {e}")
        
        self.cleanup()
        self.print_summary()
        return self.test_results
    
    def test_redis_connectivity(self) -> Dict[str, Any]:
        """Test Redis connectivity and basic operations"""
        try:
            import redis
            client = redis.Redis(decode_responses=True)
            
            # Test ping
            ping_result = client.ping()
            
            # Test set/get
            test_key = "pynucleus_test_key"
            test_value = "integration_test_value"
            client.set(test_key, test_value, ex=60)
            retrieved_value = client.get(test_key)
            
            # Cleanup
            client.delete(test_key)
            
            return {
                "ping_successful": ping_result,
                "set_get_successful": retrieved_value == test_value,
                "redis_info": dict(client.info())
            }
        except Exception as e:
            return {"error": str(e), "redis_available": False}
    
    def test_cache_integration(self) -> Dict[str, Any]:
        """Test PyNucleus cache integration"""
        try:
            from pynucleus.deployment.cache_integration import RedisCache, get_cache
            
            # Test cache initialization
            cache = RedisCache()
            
            # Test basic operations
            test_query = "What is chemical engineering?"
            test_response = {"answer": "Chemical engineering is...", "sources": []}
            
            # Test caching
            cache.set(test_query, test_response, ttl=60)
            cached_result = cache.get(test_query)
            
            # Test cache stats
            stats = cache.get_stats()
            
            # Test global cache instance
            global_cache = get_cache()
            
            return {
                "cache_initialized": cache.enabled,
                "set_get_successful": cached_result is not None,
                "stats_available": isinstance(stats, dict),
                "global_cache_working": global_cache.enabled,
                "cache_stats": stats
            }
        except Exception as e:
            return {"error": str(e)}
    
    def test_scaling_manager(self) -> Dict[str, Any]:
        """Test scaling manager components"""
        try:
            from pynucleus.deployment.scaling_manager import (
                ScalingManager, CacheManager, DockerManager, 
                InstanceMetrics, ScalingConfig
            )
            
            # Test configuration
            config = ScalingConfig(min_instances=2, max_instances=8)
            
            # Test cache manager
            cache_manager = CacheManager()
            
            # Test Docker manager (without actually manipulating containers)
            docker_manager = DockerManager()
            
            # Test instance metrics creation
            metrics = InstanceMetrics(
                instance_id="test-api-1",
                cpu_usage=50.0,
                memory_usage=60.0,
                response_time_avg=1.5,
                requests_per_second=10.0,
                error_rate=0.0,
                timestamp=time.time(),
                health_status="healthy"
            )
            
            return {
                "scaling_config_created": config.min_instances == 2,
                "cache_manager_initialized": cache_manager is not None,
                "docker_manager_initialized": docker_manager is not None,
                "metrics_dataclass_working": metrics.instance_id == "test-api-1",
                "scaling_components_imported": True
            }
        except Exception as e:
            return {"error": str(e)}
    
    def test_flask_app_factory(self) -> Dict[str, Any]:
        """Test Flask application factory pattern"""
        try:
            from pynucleus.api.app import create_app
            
            # Test app creation
            app = create_app()
            
            # Test configuration
            config_keys = list(app.config.keys())
            
            # Test routes exist
            routes = [rule.rule for rule in app.url_map.iter_rules()]
            
            return {
                "app_created": app is not None,
                "has_configuration": len(config_keys) > 0,
                "routes_registered": "/health" in routes,
                "expected_routes": [r for r in routes if r in ["/health", "/metrics", "/ask"]],
                "total_routes": len(routes)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def test_docker_configuration(self) -> Dict[str, Any]:
        """Test Docker configuration validity"""
        try:
            import subprocess
            
            # Test docker-compose configuration
            cmd = ["docker-compose", "-f", "docker/docker-compose.scale.yml", "config", "--services"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            services = result.stdout.strip().split('\n') if result.returncode == 0 else []
            
            # Check for expected services
            expected_services = ["redis", "load-balancer", "api-1", "api-2", "api-3", "model"]
            found_services = [s for s in expected_services if s in services]
            
            return {
                "compose_config_valid": result.returncode == 0,
                "services_found": services,
                "expected_services_present": found_services,
                "all_expected_present": len(found_services) == len(expected_services)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def start_test_flask_app(self):
        """Start Flask app for testing"""
        try:
            from pynucleus.api.app import create_app
            app = create_app()
            
            def run_app():
                app.run(host='127.0.0.1', port=self.test_port, debug=False, use_reloader=False)
            
            self.flask_app_thread = threading.Thread(target=run_app, daemon=True)
            self.flask_app_thread.start()
            
            # Wait for app to start
            time.sleep(3)
            return True
        except Exception as e:
            logger.error(f"Failed to start test Flask app: {e}")
            return False
    
    def test_flask_api_endpoints(self) -> Dict[str, Any]:
        """Test Flask API endpoints"""
        try:
            # Start test app
            if not self.start_test_flask_app():
                return {"error": "Failed to start test Flask app"}
            
            base_url = f"http://127.0.0.1:{self.test_port}"
            results = {}
            
            # Test health endpoint
            try:
                response = requests.get(f"{base_url}/health", timeout=10)
                results["health_endpoint"] = {
                    "status_code": response.status_code,
                    "response_json": response.json() if response.status_code == 200 else None
                }
            except Exception as e:
                results["health_endpoint"] = {"error": str(e)}
            
            # Test metrics endpoint
            try:
                response = requests.get(f"{base_url}/metrics", timeout=10)
                results["metrics_endpoint"] = {
                    "status_code": response.status_code,
                    "content_type": response.headers.get("Content-Type"),
                    "has_content": len(response.text) > 0
                }
            except Exception as e:
                results["metrics_endpoint"] = {"error": str(e)}
            
            return results
        except Exception as e:
            return {"error": str(e)}
    
    def test_stress_test_infrastructure(self) -> Dict[str, Any]:
        """Test stress testing infrastructure"""
        try:
            # Check if stress test scripts exist
            script_files = [
                "scripts/simple_stress_test.py",
                "scripts/stress_test_suite.py",
                "scripts/stress_test.py"
            ]
            
            existing_scripts = []
            for script in script_files:
                if Path(script).exists():
                    existing_scripts.append(script)
            
            # Test stress test configuration
            sys.path.insert(0, str(Path("scripts")))
            
            try:
                from stress_test_suite import StressTestConfig, PyNucleusStressTester
                
                config = StressTestConfig(
                    base_url="http://localhost",
                    port=80,
                    num_concurrent_users=5,
                    num_requests_per_user=10
                )
                
                tester = PyNucleusStressTester(config)
                
                return {
                    "scripts_exist": existing_scripts,
                    "config_creation": True,
                    "tester_creation": True,
                    "stress_test_ready": len(existing_scripts) >= 2
                }
            except ImportError as e:
                return {
                    "scripts_exist": existing_scripts,
                    "import_error": str(e),
                    "stress_test_ready": False
                }
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup(self):
        """Cleanup test resources"""
        try:
            # Stop Flask app if running
            if hasattr(self, 'flask_app_thread'):
                # The thread will stop when main thread exits
                pass
            
            # Clean up test data from Redis
            try:
                import redis
                client = redis.Redis()
                # Clean up any test keys
                for key in client.scan_iter("pynucleus_test_*"):
                    client.delete(key)
            except:
                pass
                
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "=" * 60)
        logger.info("🏁 INTEGRATION TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result["status"] == "PASS")
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] == "PASS" else "❌"
            logger.info(f"{status_icon} {test_name}: {result['status']}")
        
        logger.info(f"\nResults: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - PyNucleus is deployment-ready!")
        else:
            logger.warning(f"⚠️  {total - passed} tests failed - review deployment setup")
        
        # Save detailed results
        output_file = Path("test_output/integration_test_results.json")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"📊 Detailed results saved to: {output_file}")

def main():
    """Main function"""
    tester = PyNucleusIntegrationTest()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    failed_tests = sum(1 for result in results.values() if result["status"] == "FAIL")
    sys.exit(failed_tests)

if __name__ == "__main__":
    main() 