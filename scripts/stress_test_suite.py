#!/usr/bin/env python3
"""
PyNucleus Stress Test Suite

Comprehensive performance and scalability testing for PyNucleus deployment.
Tests horizontal scaling, load balancing, and system stability under load.
"""

import asyncio
import aiohttp
import time
import json
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@dataclass
class StressTestConfig:
    """Configuration for stress testing"""
    base_url: str = "http://localhost"
    port: int = 80
    num_concurrent_users: int = 10
    num_requests_per_user: int = 100
    ramp_up_time: int = 30  # seconds
    test_duration: int = 300  # seconds
    request_timeout: int = 30
    think_time: float = 1.0  # seconds between requests
    
@dataclass
class RequestResult:
    """Result of a single request"""
    timestamp: datetime
    response_time: float
    status_code: int
    success: bool
    error_message: str = ""
    response_size: int = 0

@dataclass
class LoadTestResults:
    """Aggregated load test results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    error_rate: float
    test_duration: float
    concurrent_users: int

class PyNucleusStressTester:
    """Comprehensive stress tester for PyNucleus"""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.results: List[RequestResult] = []
        self.start_time: datetime = None
        self.end_time: datetime = None
        self._lock = threading.Lock()
        
        # Test scenarios
        self.test_scenarios = [
            {
                "name": "health_check",
                "endpoint": "/health",
                "method": "GET",
                "weight": 0.1  # 10% of requests
            },
            {
                "name": "simple_query",
                "endpoint": "/api/ask",
                "method": "POST",
                "payload": {"question": "What is chemical engineering?"},
                "weight": 0.4  # 40% of requests
            },
            {
                "name": "complex_query",
                "endpoint": "/api/ask",
                "method": "POST", 
                "payload": {"question": "How can I optimize a methanol synthesis reactor for maximum yield and energy efficiency?"},
                "weight": 0.3  # 30% of requests
            },
            {
                "name": "documentation",
                "endpoint": "/docs",
                "method": "GET",
                "weight": 0.1  # 10% of requests
            },
            {
                "name": "metrics",
                "endpoint": "/metrics",
                "method": "GET",
                "weight": 0.1  # 10% of requests
            }
        ]
    
    def _select_test_scenario(self) -> Dict[str, Any]:
        """Select a test scenario based on weights"""
        import random
        rand = random.random()
        cumulative_weight = 0
        
        for scenario in self.test_scenarios:
            cumulative_weight += scenario["weight"]
            if rand <= cumulative_weight:
                return scenario
        
        return self.test_scenarios[0]  # Fallback
    
    async def _make_async_request(self, session: aiohttp.ClientSession, scenario: Dict[str, Any]) -> RequestResult:
        """Make an asynchronous HTTP request"""
        url = f"{self.config.base_url}:{self.config.port}{scenario['endpoint']}"
        method = scenario.get("method", "GET")
        payload = scenario.get("payload")
        
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            if method == "POST" and payload:
                async with session.post(
                    url, 
                    json=payload, 
                    timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
                ) as response:
                    response_text = await response.text()
                    response_time = time.time() - start_time
                    
                    return RequestResult(
                        timestamp=timestamp,
                        response_time=response_time,
                        status_code=response.status,
                        success=200 <= response.status < 400,
                        response_size=len(response_text)
                    )
            else:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
                ) as response:
                    response_text = await response.text()
                    response_time = time.time() - start_time
                    
                    return RequestResult(
                        timestamp=timestamp,
                        response_time=response_time,
                        status_code=response.status,
                        success=200 <= response.status < 400,
                        response_size=len(response_text)
                    )
        
        except Exception as e:
            response_time = time.time() - start_time
            return RequestResult(
                timestamp=timestamp,
                response_time=response_time,
                status_code=0,
                success=False,
                error_message=str(e)
            )
    
    def _make_sync_request(self, scenario: Dict[str, Any]) -> RequestResult:
        """Make a synchronous HTTP request"""
        import requests
        
        url = f"{self.config.base_url}:{self.config.port}{scenario['endpoint']}"
        method = scenario.get("method", "GET")
        payload = scenario.get("payload")
        
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            if method == "POST" and payload:
                response = requests.post(
                    url, 
                    json=payload, 
                    timeout=self.config.request_timeout
                )
            else:
                response = requests.get(
                    url,
                    timeout=self.config.request_timeout
                )
            
            response_time = time.time() - start_time
            
            return RequestResult(
                timestamp=timestamp,
                response_time=response_time,
                status_code=response.status_code,
                success=200 <= response.status_code < 400,
                response_size=len(response.text)
            )
        
        except Exception as e:
            response_time = time.time() - start_time
            return RequestResult(
                timestamp=timestamp,
                response_time=response_time,
                status_code=0,
                success=False,
                error_message=str(e)
            )
    
    async def _async_user_simulation(self, user_id: int) -> List[RequestResult]:
        """Simulate a single user's requests asynchronously"""
        user_results = []
        
        # Stagger user start times for realistic ramp-up
        ramp_delay = (user_id / self.config.num_concurrent_users) * self.config.ramp_up_time
        await asyncio.sleep(ramp_delay)
        
        async with aiohttp.ClientSession() as session:
            for request_num in range(self.config.num_requests_per_user):
                scenario = self._select_test_scenario()
                result = await self._make_async_request(session, scenario)
                user_results.append(result)
                
                # Think time between requests
                if request_num < self.config.num_requests_per_user - 1:
                    await asyncio.sleep(self.config.think_time)
        
        return user_results
    
    def _sync_user_simulation(self, user_id: int) -> List[RequestResult]:
        """Simulate a single user's requests synchronously"""
        user_results = []
        
        # Stagger user start times
        ramp_delay = (user_id / self.config.num_concurrent_users) * self.config.ramp_up_time
        time.sleep(ramp_delay)
        
        for request_num in range(self.config.num_requests_per_user):
            scenario = self._select_test_scenario()
            result = self._make_sync_request(scenario)
            user_results.append(result)
            
            # Store result thread-safely
            with self._lock:
                self.results.append(result)
            
            # Think time between requests
            if request_num < self.config.num_requests_per_user - 1:
                time.sleep(self.config.think_time)
        
        return user_results
    
    async def run_async_load_test(self) -> LoadTestResults:
        """Run asynchronous load test"""
        print(f"🚀 Starting async load test with {self.config.num_concurrent_users} users")
        print(f"📊 Each user will make {self.config.num_requests_per_user} requests")
        print(f"🎯 Target: {self.config.base_url}:{self.config.port}")
        
        self.start_time = datetime.now()
        
        # Create user simulation tasks
        tasks = [
            self._async_user_simulation(user_id) 
            for user_id in range(self.config.num_concurrent_users)
        ]
        
        # Run all users concurrently
        all_results = await asyncio.gather(*tasks)
        
        # Flatten results
        self.results = [result for user_results in all_results for result in user_results]
        
        self.end_time = datetime.now()
        
        return self._calculate_results()
    
    def run_sync_load_test(self) -> LoadTestResults:
        """Run synchronous load test using ThreadPoolExecutor"""
        print(f"🚀 Starting sync load test with {self.config.num_concurrent_users} users")
        print(f"📊 Each user will make {self.config.num_requests_per_user} requests")
        print(f"🎯 Target: {self.config.base_url}:{self.config.port}")
        
        self.start_time = datetime.now()
        self.results = []
        
        with ThreadPoolExecutor(max_workers=self.config.num_concurrent_users) as executor:
            # Submit user simulation tasks
            future_to_user = {
                executor.submit(self._sync_user_simulation, user_id): user_id
                for user_id in range(self.config.num_concurrent_users)
            }
            
            # Monitor progress
            completed = 0
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    user_results = future.result()
                    completed += 1
                    print(f"✅ User {user_id} completed ({completed}/{self.config.num_concurrent_users})")
                except Exception as e:
                    print(f"❌ User {user_id} failed: {e}")
        
        self.end_time = datetime.now()
        
        return self._calculate_results()
    
    def _calculate_results(self) -> LoadTestResults:
        """Calculate aggregated test results"""
        if not self.results:
            raise ValueError("No results to calculate")
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        response_times = [r.response_time for r in self.results]
        test_duration = (self.end_time - self.start_time).total_seconds()
        
        return LoadTestResults(
            total_requests=len(self.results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            average_response_time=statistics.mean(response_times),
            median_response_time=statistics.median(response_times),
            p95_response_time=statistics.quantiles(response_times, n=20)[18] if response_times else 0,  # 95th percentile
            p99_response_time=statistics.quantiles(response_times, n=100)[98] if response_times else 0,  # 99th percentile
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            requests_per_second=len(self.results) / test_duration if test_duration > 0 else 0,
            error_rate=(len(failed_results) / len(self.results) * 100) if self.results else 0,
            test_duration=test_duration,
            concurrent_users=self.config.num_concurrent_users
        )
    
    def run_warmup_requests(self, num_requests: int = 10):
        """Run warmup requests to prepare the system"""
        print(f"🔥 Warming up system with {num_requests} requests...")
        
        import requests
        warmup_url = f"{self.config.base_url}:{self.config.port}/health"
        
        for i in range(num_requests):
            try:
                response = requests.get(warmup_url, timeout=10)
                print(f"   Warmup {i+1}/{num_requests}: {response.status_code}")
            except Exception as e:
                print(f"   Warmup {i+1}/{num_requests}: Failed - {e}")
            time.sleep(0.5)
        
        print("✅ Warmup completed")
    
    def save_results(self, results: LoadTestResults, output_file: str):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            "test_timestamp": timestamp,
            "test_config": asdict(self.config),
            "results": asdict(results),
            "detailed_results": [asdict(r) for r in self.results[-100:]]  # Last 100 for analysis
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Results saved to: {output_path}")
    
    def print_results(self, results: LoadTestResults):
        """Print formatted test results"""
        print("\n" + "="*60)
        print("📊 PYLAUNCH STRESS TEST RESULTS")
        print("="*60)
        print(f"🔢 Total Requests:        {results.total_requests:,}")
        print(f"✅ Successful:            {results.successful_requests:,} ({100-results.error_rate:.1f}%)")
        print(f"❌ Failed:                {results.failed_requests:,} ({results.error_rate:.1f}%)")
        print(f"👥 Concurrent Users:      {results.concurrent_users}")
        print(f"⏱️  Test Duration:         {results.test_duration:.1f}s")
        print(f"🚀 Requests/Second:       {results.requests_per_second:.1f}")
        print()
        print("📈 Response Time Analysis:")
        print(f"   Average:     {results.average_response_time:.3f}s")
        print(f"   Median:      {results.median_response_time:.3f}s")
        print(f"   95th %ile:   {results.p95_response_time:.3f}s")
        print(f"   99th %ile:   {results.p99_response_time:.3f}s")
        print(f"   Min:         {results.min_response_time:.3f}s")
        print(f"   Max:         {results.max_response_time:.3f}s")
        print()
        
        # Performance Assessment
        if results.error_rate < 1.0 and results.p95_response_time < 3.0:
            print("🎉 EXCELLENT: System handling load well!")
        elif results.error_rate < 5.0 and results.p95_response_time < 5.0:
            print("✅ GOOD: System performance is acceptable")
        elif results.error_rate < 10.0:
            print("⚠️  FAIR: System showing some stress")
        else:
            print("🚨 POOR: System struggling under load")
        
        print("="*60)


def run_scaling_validation_tests():
    """Run comprehensive scaling validation tests"""
    print("🏗️  PYNUCLEUS SCALING VALIDATION SUITE")
    print("="*50)
    
    test_configs = [
        {
            "name": "Light Load",
            "users": 5,
            "requests": 20,
            "ramp_up": 10
        },
        {
            "name": "Medium Load", 
            "users": 20,
            "requests": 50,
            "ramp_up": 30
        },
        {
            "name": "Heavy Load",
            "users": 50,
            "requests": 100,
            "ramp_up": 60
        },
        {
            "name": "Stress Test",
            "users": 100,
            "requests": 50,
            "ramp_up": 30
        }
    ]
    
    all_results = []
    
    for test_config in test_configs:
        print(f"\n🧪 Running {test_config['name']} Test...")
        
        config = StressTestConfig(
            num_concurrent_users=test_config["users"],
            num_requests_per_user=test_config["requests"],
            ramp_up_time=test_config["ramp_up"],
            think_time=0.5
        )
        
        tester = PyNucleusStressTester(config)
        tester.run_warmup_requests(5)
        
        try:
            results = tester.run_sync_load_test()
            tester.print_results(results)
            
            # Save individual test results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"test_output/stress_test_{test_config['name'].lower().replace(' ', '_')}_{timestamp}.json"
            tester.save_results(results, output_file)
            
            all_results.append({
                "test_name": test_config["name"],
                "results": results
            })
            
        except Exception as e:
            print(f"❌ {test_config['name']} test failed: {e}")
    
    # Generate summary report
    print("\n📋 SCALING VALIDATION SUMMARY")
    print("="*40)
    for test_result in all_results:
        results = test_result["results"]
        print(f"{test_result['test_name']:15} | "
              f"RPS: {results.requests_per_second:6.1f} | "
              f"Errors: {results.error_rate:5.1f}% | "
              f"P95: {results.p95_response_time:6.3f}s")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="PyNucleus Stress Testing Suite")
    parser.add_argument("--url", default="http://localhost", help="Base URL to test")
    parser.add_argument("--port", type=int, default=80, help="Port to test")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--requests", type=int, default=50, help="Requests per user")
    parser.add_argument("--ramp-up", type=int, default=30, help="Ramp-up time in seconds")
    parser.add_argument("--think-time", type=float, default=1.0, help="Think time between requests")
    parser.add_argument("--async", dest="async_mode", action="store_true", help="Use async testing")
    parser.add_argument("--validation", action="store_true", help="Run full scaling validation suite")
    parser.add_argument("--output", default="test_output/stress_test_results.json", help="Output file")
    
    args = parser.parse_args()
    
    if args.validation:
        run_scaling_validation_tests()
        return
    
    # Single test run
    config = StressTestConfig(
        base_url=args.url,
        port=args.port,
        num_concurrent_users=args.users,
        num_requests_per_user=args.requests,
        ramp_up_time=args.ramp_up,
        think_time=args.think_time
    )
    
    tester = PyNucleusStressTester(config)
    tester.run_warmup_requests()
    
    if args.async_mode:
        results = asyncio.run(tester.run_async_load_test())
    else:
        results = tester.run_sync_load_test()
    
    tester.print_results(results)
    tester.save_results(results, args.output)


if __name__ == "__main__":
    main()