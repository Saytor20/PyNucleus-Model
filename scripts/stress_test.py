#!/usr/bin/env python3
"""
PyNucleus Stress Testing Suite

Comprehensive load testing for PyNucleus API to evaluate:
- Horizontal scaling performance
- Cache effectiveness
- Response time under load
- Error rates and failure modes
- Resource utilization
"""

import asyncio
import aiohttp
import time
import json
import logging
import argparse
import random
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Individual test result"""
    timestamp: float
    response_time: float
    status_code: int
    success: bool
    cached: bool
    error: Optional[str] = None
    confidence: Optional[float] = None
    instance_id: Optional[str] = None

@dataclass
class StressTestConfig:
    """Configuration for stress test"""
    base_url: str = "http://localhost"
    total_requests: int = 1000
    concurrent_requests: int = 50
    ramp_up_time: int = 60  # seconds
    test_duration: int = 300  # seconds
    cache_test_ratio: float = 0.3  # 30% repeated queries
    timeout: int = 30  # seconds per request

class StressTestRunner:
    """Main stress test runner"""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.test_queries = self._load_test_queries()
        self.session: Optional[aiohttp.ClientSession] = None
        
    def _load_test_queries(self) -> List[str]:
        """Load test queries from various sources"""
        # Chemical engineering questions for testing
        base_queries = [
            "What is distillation and how does it work?",
            "Explain the concept of vapor-liquid equilibrium",
            "What are the main types of heat exchangers?",
            "How do you calculate the number of theoretical plates in a distillation column?",
            "What is the difference between absorption and adsorption?",
            "Explain the principles of mass transfer",
            "What are the applications of chemical reactors?",
            "How do you design a cooling tower?",
            "What is the role of catalysts in chemical reactions?",
            "Explain the concept of residence time distribution",
            "What are the safety considerations in chemical plant design?",
            "How do you optimize a chemical process?",
            "What is the difference between batch and continuous processes?",
            "Explain the principles of crystallization",
            "What are the factors affecting reaction kinetics?",
            "How do you calculate pressure drop in pipes?",
            "What is the purpose of separation processes?",
            "Explain the concept of thermodynamic equilibrium",
            "What are the applications of membrane separation?",
            "How do you design a packed column?",
        ]
        
        # Generate variations to create more realistic load
        variations = [
            "Can you explain {}?",
            "Tell me about {}",
            "What do you know about {}?",
            "Please describe {}",
            "I need information on {}"
        ]
        
        extended_queries = []
        for query in base_queries:
            extended_queries.append(query)
            # Add some variations
            for variation in random.sample(variations, 2):
                if "{}" in variation:
                    topic = query.split("?")[0].lower()
                    extended_queries.append(variation.format(topic))
                else:
                    extended_queries.append(f"{variation} {query}")
        
        return extended_queries
    
    async def make_request(self, query: str) -> TestResult:
        """Make a single API request"""
        start_time = time.time()
        
        try:
            payload = {
                "question": query,
                "use_cache": True
            }
            
            async with self.session.post(
                f"{self.config.base_url}/ask",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    return TestResult(
                        timestamp=start_time,
                        response_time=response_time,
                        status_code=response.status,
                        success=True,
                        cached=data.get("metadata", {}).get("cache_hit", False),
                        confidence=data.get("confidence"),
                        instance_id=data.get("metadata", {}).get("instance_id")
                    )
                else:
                    error_text = await response.text()
                    return TestResult(
                        timestamp=start_time,
                        response_time=response_time,
                        status_code=response.status,
                        success=False,
                        cached=False,
                        error=error_text[:200]
                    )
                    
        except asyncio.TimeoutError:
            return TestResult(
                timestamp=start_time,
                response_time=time.time() - start_time,
                status_code=0,
                success=False,
                cached=False,
                error="Request timeout"
            )
        except Exception as e:
            return TestResult(
                timestamp=start_time,
                response_time=time.time() - start_time,
                status_code=0,
                success=False,
                cached=False,
                error=str(e)[:200]
            )
    
    async def run_load_phase(self, phase_name: str, duration: int, concurrency: int):
        """Run a specific load testing phase"""
        logger.info(f"Starting {phase_name} phase: {duration}s duration, {concurrency} concurrent requests")
        
        phase_start = time.time()
        phase_results = []
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request():
            async with semaphore:
                # Select query (mix of new and repeated for cache testing)
                if random.random() < self.config.cache_test_ratio and len(phase_results) > 10:
                    # Use a previous query to test caching
                    query = random.choice(self.test_queries[:10])
                else:
                    # Use random query
                    query = random.choice(self.test_queries)
                
                result = await self.make_request(query)
                phase_results.append(result)
                return result
        
        # Generate requests for the duration
        tasks = []
        while time.time() - phase_start < duration:
            # Add new requests to maintain concurrency
            while len(tasks) < concurrency:
                task = asyncio.create_task(bounded_request())
                tasks.append(task)
            
            # Wait for some tasks to complete
            done, pending = await asyncio.wait(
                tasks, 
                timeout=1.0, 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Remove completed tasks
            for task in done:
                tasks.remove(task)
            
            # Short pause to prevent overwhelming the system
            await asyncio.sleep(0.1)
        
        # Wait for remaining tasks
        if tasks:
            await asyncio.wait(tasks)
        
        self.results.extend(phase_results)
        logger.info(f"Completed {phase_name} phase: {len(phase_results)} requests")
        return phase_results
    
    async def run_stress_test(self):
        """Run the complete stress test suite"""
        logger.info("Starting PyNucleus Stress Test")
        logger.info(f"Configuration: {self.config}")
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=self.config.concurrent_requests * 2)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        
        try:
            # Test phases with increasing load
            await self.run_load_phase("Warmup", 30, 5)
            await self.run_load_phase("Low Load", 60, 10)
            await self.run_load_phase("Medium Load", 120, 25)
            await self.run_load_phase("High Load", 120, self.config.concurrent_requests)
            await self.run_load_phase("Peak Load", 60, self.config.concurrent_requests * 2)
            await self.run_load_phase("Cooldown", 30, 5)
            
        finally:
            await self.session.close()
        
        logger.info(f"Stress test completed: {len(self.results)} total requests")
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and generate report"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Basic statistics
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        cached_results = [r for r in successful_results if r.cached]
        
        response_times = [r.response_time for r in successful_results]
        confidences = [r.confidence for r in successful_results if r.confidence is not None]
        
        # Instance distribution
        instance_counts = {}
        for result in successful_results:
            if result.instance_id:
                instance_counts[result.instance_id] = instance_counts.get(result.instance_id, 0) + 1
        
        # Time-based analysis
        test_start = min(r.timestamp for r in self.results)
        test_end = max(r.timestamp for r in self.results)
        test_duration = test_end - test_start
        
        # Error analysis
        error_types = {}
        for result in failed_results:
            error_type = result.error.split(":")[0] if result.error else "Unknown"
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        analysis = {
            "test_summary": {
                "total_requests": len(self.results),
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": len(successful_results) / len(self.results) * 100,
                "cache_hits": len(cached_results),
                "cache_hit_rate": len(cached_results) / len(successful_results) * 100 if successful_results else 0,
                "test_duration": test_duration,
                "requests_per_second": len(self.results) / test_duration if test_duration > 0 else 0
            },
            "performance_metrics": {
                "response_times": {
                    "mean": statistics.mean(response_times) if response_times else 0,
                    "median": statistics.median(response_times) if response_times else 0,
                    "p95": self._percentile(response_times, 95) if response_times else 0,
                    "p99": self._percentile(response_times, 99) if response_times else 0,
                    "min": min(response_times) if response_times else 0,
                    "max": max(response_times) if response_times else 0
                },
                "confidence_scores": {
                    "mean": statistics.mean(confidences) if confidences else 0,
                    "median": statistics.median(confidences) if confidences else 0,
                    "min": min(confidences) if confidences else 0,
                    "max": max(confidences) if confidences else 0
                }
            },
            "scaling_analysis": {
                "instance_distribution": instance_counts,
                "load_balancing_effectiveness": self._calculate_load_balance_score(instance_counts)
            },
            "error_analysis": {
                "error_types": error_types,
                "error_rate_by_time": self._calculate_error_rate_by_time()
            },
            "system_resources": self._get_system_resources()
        }
        
        return analysis
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _calculate_load_balance_score(self, instance_counts: Dict[str, int]) -> float:
        """Calculate how well load is balanced across instances"""
        if not instance_counts or len(instance_counts) <= 1:
            return 1.0
        
        values = list(instance_counts.values())
        mean_val = statistics.mean(values)
        variance = statistics.variance(values) if len(values) > 1 else 0
        
        # Perfect balance = score of 1.0, higher variance = lower score
        return max(0.0, 1.0 - (variance / (mean_val ** 2)) if mean_val > 0 else 0)
    
    def _calculate_error_rate_by_time(self) -> Dict[str, float]:
        """Calculate error rate over time windows"""
        if not self.results:
            return {}
        
        # Group results by 30-second windows
        window_size = 30
        start_time = min(r.timestamp for r in self.results)
        end_time = max(r.timestamp for r in self.results)
        
        error_rates = {}
        current_time = start_time
        
        while current_time < end_time:
            window_end = current_time + window_size
            window_results = [r for r in self.results if current_time <= r.timestamp < window_end]
            
            if window_results:
                error_count = sum(1 for r in window_results if not r.success)
                error_rate = error_count / len(window_results) * 100
                time_label = f"{int(current_time - start_time)}s-{int(window_end - start_time)}s"
                error_rates[time_label] = error_rate
            
            current_time = window_end
        
        return error_rates
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "network_connections": len(psutil.net_connections()),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except Exception as e:
            return {"error": f"Failed to get system resources: {e}"}
    
    def generate_report(self) -> str:
        """Generate detailed test report"""
        analysis = self.analyze_results()
        
        report = []
        report.append("=" * 80)
        report.append("PYNUCLEUS STRESS TEST REPORT")
        report.append("=" * 80)
        report.append(f"Test completed at: {datetime.now().isoformat()}")
        report.append(f"Configuration: {self.config}")
        report.append("")
        
        # Test Summary
        summary = analysis["test_summary"]
        report.append("TEST SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Requests:      {summary['total_requests']:,}")
        report.append(f"Successful:          {summary['successful_requests']:,} ({summary['success_rate']:.1f}%)")
        report.append(f"Failed:              {summary['failed_requests']:,}")
        report.append(f"Cache Hits:          {summary['cache_hits']:,} ({summary['cache_hit_rate']:.1f}%)")
        report.append(f"Test Duration:       {summary['test_duration']:.1f} seconds")
        report.append(f"Requests/Second:     {summary['requests_per_second']:.1f}")
        report.append("")
        
        # Performance Metrics
        perf = analysis["performance_metrics"]
        report.append("PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append("Response Times:")
        report.append(f"  Mean:              {perf['response_times']['mean']:.3f}s")
        report.append(f"  Median:            {perf['response_times']['median']:.3f}s")
        report.append(f"  95th Percentile:   {perf['response_times']['p95']:.3f}s")
        report.append(f"  99th Percentile:   {perf['response_times']['p99']:.3f}s")
        report.append(f"  Min:               {perf['response_times']['min']:.3f}s")
        report.append(f"  Max:               {perf['response_times']['max']:.3f}s")
        report.append("")
        
        # Scaling Analysis
        scaling = analysis["scaling_analysis"]
        report.append("SCALING ANALYSIS")
        report.append("-" * 40)
        report.append("Instance Distribution:")
        for instance, count in scaling["instance_distribution"].items():
            percentage = count / summary["successful_requests"] * 100
            report.append(f"  {instance}: {count:,} requests ({percentage:.1f}%)")
        report.append(f"Load Balance Score:  {scaling['load_balancing_effectiveness']:.3f}")
        report.append("")
        
        # Error Analysis
        errors = analysis["error_analysis"]
        if errors["error_types"]:
            report.append("ERROR ANALYSIS")
            report.append("-" * 40)
            for error_type, count in errors["error_types"].items():
                report.append(f"  {error_type}: {count}")
        
        return "\n".join(report)
    
    def save_results(self, output_dir: str = "test_output"):
        """Save test results and analysis to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = output_path / f"stress_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        # Save analysis
        analysis_file = output_path / f"stress_test_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(self.analyze_results(), f, indent=2)
        
        # Save report
        report_file = output_path / f"stress_test_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(self.generate_report())
        
        logger.info(f"Results saved to {output_dir}/")
        return results_file, analysis_file, report_file

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PyNucleus Stress Testing Suite")
    parser.add_argument("--url", default="http://localhost", help="Base URL for API")
    parser.add_argument("--requests", type=int, default=1000, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=50, help="Concurrent requests")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--output", default="test_output", help="Output directory")
    
    args = parser.parse_args()
    
    config = StressTestConfig(
        base_url=args.url,
        total_requests=args.requests,
        concurrent_requests=args.concurrency,
        test_duration=args.duration,
        timeout=args.timeout
    )
    
    runner = StressTestRunner(config)
    
    try:
        await runner.run_stress_test()
        print(runner.generate_report())
        runner.save_results(args.output)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 