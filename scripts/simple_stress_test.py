#!/usr/bin/env python3
"""
PyNucleus Simple Stress Test

Lightweight stress testing tool for evaluating:
- Response times under load
- Cache effectiveness
- Instance load balancing
- Error rates
"""

import asyncio
import aiohttp
import time
import json
import logging
import argparse
import random
import statistics
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    response_time: float
    status_code: int
    success: bool
    cached: bool
    instance_id: str = None

class StressTest:
    def __init__(self, base_url: str, concurrent: int = 50, duration: int = 300):
        self.base_url = base_url
        self.concurrent = concurrent
        self.duration = duration
        self.results = []
        
        self.test_queries = [
            "What is distillation and how does it work?",
            "Explain vapor-liquid equilibrium",
            "What are heat exchanger types?",
            "How to calculate theoretical plates?",
            "Difference between absorption and adsorption?",
            "Principles of mass transfer",
            "Chemical reactor applications",
            "How to design cooling towers?",
            "Role of catalysts in reactions",
            "Residence time distribution concept"
        ]
    
    async def make_request(self, session: aiohttp.ClientSession, query: str) -> TestResult:
        start_time = time.time()
        
        try:
            payload = {"question": query, "use_cache": True}
            async with session.post(f"{self.base_url}/ask", json=payload, timeout=30) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    return TestResult(
                        response_time=response_time,
                        status_code=response.status,
                        success=True,
                        cached=data.get("metadata", {}).get("cache_hit", False),
                        instance_id=data.get("metadata", {}).get("instance_id")
                    )
                else:
                    return TestResult(
                        response_time=response_time,
                        status_code=response.status,
                        success=False,
                        cached=False
                    )
        except Exception as e:
            return TestResult(
                response_time=time.time() - start_time,
                status_code=0,
                success=False,
                cached=False
            )
    
    async def run_test(self):
        logger.info(f"Starting stress test: {self.concurrent} concurrent, {self.duration}s duration")
        
        connector = aiohttp.TCPConnector(limit=100)
        async with aiohttp.ClientSession(connector=connector) as session:
            
            # Warmup
            warmup_tasks = [
                self.make_request(session, random.choice(self.test_queries))
                for _ in range(10)
            ]
            await asyncio.gather(*warmup_tasks)
            
            # Main test
            start_time = time.time()
            semaphore = asyncio.Semaphore(self.concurrent)
            
            async def bounded_request():
                async with semaphore:
                    # 30% chance of repeated query for cache testing
                    if random.random() < 0.3 and len(self.results) > 10:
                        query = random.choice(self.test_queries[:5])
                    else:
                        query = random.choice(self.test_queries)
                    
                    result = await self.make_request(session, query)
                    self.results.append(result)
                    return result
            
            tasks = []
            while time.time() - start_time < self.duration:
                if len(tasks) < self.concurrent:
                    task = asyncio.create_task(bounded_request())
                    tasks.append(task)
                
                done, pending = await asyncio.wait(
                    tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    tasks.remove(task)
            
            # Wait for remaining tasks
            if tasks:
                await asyncio.wait(tasks)
        
        logger.info(f"Test completed: {len(self.results)} requests")
    
    def analyze_results(self) -> Dict[str, Any]:
        if not self.results:
            return {"error": "No results"}
        
        successful = [r for r in self.results if r.success]
        cached = [r for r in successful if r.cached]
        response_times = [r.response_time for r in successful]
        
        # Instance distribution
        instance_counts = {}
        for result in successful:
            if result.instance_id:
                instance_counts[result.instance_id] = instance_counts.get(result.instance_id, 0) + 1
        
        return {
            "summary": {
                "total_requests": len(self.results),
                "successful": len(successful),
                "failed": len(self.results) - len(successful),
                "success_rate": len(successful) / len(self.results) * 100,
                "cache_hits": len(cached),
                "cache_hit_rate": len(cached) / len(successful) * 100 if successful else 0
            },
            "performance": {
                "mean_response_time": statistics.mean(response_times) if response_times else 0,
                "median_response_time": statistics.median(response_times) if response_times else 0,
                "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0
            },
            "scaling": {
                "instance_distribution": instance_counts,
                "instances_used": len(instance_counts)
            }
        }
    
    def print_report(self):
        analysis = self.analyze_results()
        
        print("\n" + "="*60)
        print("PYNUCLEUS STRESS TEST REPORT")
        print("="*60)
        
        summary = analysis["summary"]
        print(f"Total Requests:     {summary['total_requests']:,}")
        print(f"Successful:         {summary['successful']:,} ({summary['success_rate']:.1f}%)")
        print(f"Failed:             {summary['failed']:,}")
        print(f"Cache Hits:         {summary['cache_hits']:,} ({summary['cache_hit_rate']:.1f}%)")
        
        perf = analysis["performance"]
        print(f"\nResponse Times:")
        print(f"  Mean:             {perf['mean_response_time']:.3f}s")
        print(f"  Median:           {perf['median_response_time']:.3f}s")
        print(f"  95th Percentile:  {perf['p95_response_time']:.3f}s")
        print(f"  Min/Max:          {perf['min_response_time']:.3f}s / {perf['max_response_time']:.3f}s")
        
        scaling = analysis["scaling"]
        print(f"\nScaling Analysis:")
        print(f"  Instances Used:   {scaling['instances_used']}")
        for instance, count in scaling["instance_distribution"].items():
            pct = count / summary["successful"] * 100
            print(f"  {instance}: {count:,} requests ({pct:.1f}%)")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"stress_test_{timestamp}.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nResults saved to: stress_test_{timestamp}.json")

async def main():
    parser = argparse.ArgumentParser(description="PyNucleus Simple Stress Test")
    parser.add_argument("--url", default="http://localhost", help="API base URL")
    parser.add_argument("--concurrent", type=int, default=50, help="Concurrent requests")
    parser.add_argument("--duration", type=int, default=300, help="Test duration (seconds)")
    
    args = parser.parse_args()
    
    test = StressTest(args.url, args.concurrent, args.duration)
    await test.run_test()
    test.print_report()

if __name__ == "__main__":
    asyncio.run(main()) 