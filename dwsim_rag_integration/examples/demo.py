#!/usr/bin/env python3
"""
Simple Enhanced DWSIM Demo with RAG Integration

This script demonstrates the key enhanced DWSIM capabilities:
- RAG-based intelligent process queries
- Process optimization for modular plant design  
- Economic analysis and design recommendations

Usage:
    python examples/simple_enhanced_demo.py
"""

import requests
import json
import time
from urllib.parse import quote


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def test_service_health():
    """Test the enhanced DWSIM service health."""
    print_section("1. ENHANCED DWSIM SERVICE HEALTH CHECK")
    
    base_url = "http://localhost:8080"
    
    try:
        # Basic health check
        response = requests.get(f"{base_url}/health", timeout=10)
        basic_health = response.json()
        print(f"🏥 Basic Service: {basic_health.get('service', 'Unknown')}")
        print(f"📊 Mode: {basic_health.get('mode', 'Unknown')}")
        
        # Enhanced API health check
        response = requests.get(f"{base_url}/api/simulation/health", timeout=10)
        api_health = response.json()
        print(f"✅ Enhanced Service: {'Healthy' if api_health.get('isHealthy') else 'Unhealthy'}")
        
        # Display enhanced features
        features = api_health.get('enhancedFeatures', {})
        print(f"\n🔧 Enhanced Features Available:")
        for feature, available in features.items():
            status = "✅" if available else "❌"
            print(f"   {status} {feature.replace('_', ' ').title()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Service health check failed: {str(e)}")
        return False


def demonstrate_rag_queries():
    """Demonstrate RAG-based intelligent process queries."""
    print_section("2. RAG-BASED INTELLIGENT PROCESS QUERIES")
    
    base_url = "http://localhost:8080"
    
    # Define various process design queries
    queries = [
        ("Optimal Conditions", "What are the optimal operating conditions for methanol synthesis?"),
        ("Profitability", "How can I improve the profitability of a methanol plant?"),
        ("Modular Design", "What are the key design considerations for modular methanol plants?"),
        ("Energy Efficiency", "How can I minimize energy consumption in the process?"),
        ("Sensitivity Analysis", "What sensitivity analysis should I consider for process optimization?")
    ]
    
    for query_name, query_text in queries:
        print_subsection(f"Query: {query_name}")
        print(f"🤖 Question: {query_text}")
        
        try:
            # URL encode the query
            encoded_query = quote(query_text)
            
            # Make RAG query
            response = requests.get(
                f"{base_url}/api/simulation/rag/query",
                params={
                    "query": query_text,
                    "process_type": "methanol_synthesis"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                rag_response = response.json()
                
                print(f"💡 RAG Response: {rag_response.get('response', 'No response')}")
                
                # Check if simulation was triggered
                if rag_response.get('requiresSimulation'):
                    print("🔄 Simulation triggered for detailed analysis")
                    
                    # Display simulation results if available
                    sim_results = rag_response.get('simulationResults')
                    if sim_results:
                        # Economic analysis
                        econ = sim_results.get('economic_analysis', {})
                        print(f"💰 Annual Profit: ${econ.get('annual_profit', 0):,.0f}")
                        print(f"📊 Profit Margin: {econ.get('profit_margin', 0):.1%}")
                        
                        # Production analysis
                        products = sim_results.get('product_streams', {})
                        methanol = products.get('methanol_product', {})
                        print(f"🍶 Methanol Production: {methanol.get('mass_flow', 0):.0f} kg/hr")
                
                # Display design recommendations
                design_recs = rag_response.get('designRecommendations', [])
                if design_recs:
                    print("🏗️  Design Recommendations:")
                    for i, rec in enumerate(design_recs, 1):
                        print(f"   {i}. {rec}")
                
                # Display economic feasibility
                econ_feasibility = rag_response.get('economicFeasibility', {})
                if econ_feasibility:
                    print("💰 Economic Feasibility:")
                    for key, value in econ_feasibility.items():
                        print(f"   {key.replace('_', ' ').title()}: {value}")
            
            else:
                print(f"❌ Query failed: {response.text}")
                
        except Exception as e:
            print(f"❌ RAG query error: {str(e)}")
        
        time.sleep(1)  # Brief pause between queries


def demonstrate_process_optimization():
    """Demonstrate process optimization capabilities."""
    print_section("3. PROCESS OPTIMIZATION FOR MODULAR PLANTS")
    
    base_url = "http://localhost:8080"
    
    # Define optimization scenarios
    optimization_scenarios = [
        ("Profit Maximization", "profit_maximization", ["safety", "environmental"]),
        ("Energy Efficiency", "energy_efficiency", ["safety", "economic"]),
        ("Cost Minimization", "cost_minimization", ["safety", "environmental", "quality"])
    ]
    
    for scenario_name, objective, constraints in optimization_scenarios:
        print_subsection(f"Optimization: {scenario_name}")
        print(f"🎯 Objective: {objective.replace('_', ' ').title()}")
        print(f"🔒 Constraints: {', '.join(constraints)}")
        
        try:
            # Run optimization
            response = requests.get(
                f"{base_url}/api/simulation/optimize",
                params={
                    "process_type": "methanol_synthesis",
                    "objective": objective,
                    "constraints": ",".join(constraints)
                },
                timeout=30
            )
            
            if response.status_code == 200:
                opt_result = response.json()
                
                print(f"✅ Optimization completed successfully!")
                print(f"🆔 Optimization ID: {opt_result.get('optimizationId')}")
                
                # Display optimal parameters
                optimal_params = opt_result.get('optimalParameters', {})
                print(f"\n🎯 Optimal Operating Conditions:")
                print(f"   🌡️  Temperature: {optimal_params.get('reactor_temperature', 0):.1f} K")
                print(f"   📈 Pressure: {optimal_params.get('reactor_pressure', 0):.0f} kPa")
                print(f"   🧪 H2/CO Ratio: {optimal_params.get('h2_co_ratio', 0):.2f}")
                print(f"   ⚡ Feed Flow: {optimal_params.get('feed_flow', 0):.0f} kg/hr")
                
                # Economic metrics
                objective_value = opt_result.get('objectiveValue', 0)
                econ_analysis = opt_result.get('economicAnalysis', {})
                
                print(f"\n💰 Economic Performance:")
                print(f"   🎯 Objective Value: ${objective_value:,.0f}")
                print(f"   💲 Annual Profit: ${econ_analysis.get('annual_profit', 0):,.0f}")
                print(f"   💸 Production Cost: ${econ_analysis.get('production_cost', 0):.0f}/tonne")
                print(f"   ⚡ Energy Efficiency: {econ_analysis.get('energy_efficiency', 0):.1f}%")
                
                # Design recommendations
                design_recs = opt_result.get('designRecommendations', [])
                if design_recs:
                    print(f"\n🏗️  Design Recommendations:")
                    for i, rec in enumerate(design_recs, 1):
                        print(f"   {i}. {rec}")
            
            else:
                print(f"❌ Optimization failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Optimization error: {str(e)}")
        
        time.sleep(1)


def demonstrate_modular_design_insights():
    """Demonstrate modular plant design insights."""
    print_section("4. MODULAR PLANT DESIGN INSIGHTS")
    
    base_url = "http://localhost:8080"
    
    # Modular design specific queries
    modular_queries = [
        "What are the best practices for modular plant transportation?",
        "How do I optimize modular plant assembly time?", 
        "What are the key considerations for ISO container compatibility?",
        "How can I minimize the footprint of a modular methanol plant?"
    ]
    
    for i, query in enumerate(modular_queries, 1):
        print_subsection(f"Modular Design Query {i}")
        print(f"🏗️  Question: {query}")
        
        try:
            response = requests.get(
                f"{base_url}/api/simulation/rag/query",
                params={
                    "query": query,
                    "process_type": "methanol_synthesis"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                rag_response = response.json()
                
                print(f"🤖 Response: {rag_response.get('response', 'No response')}")
                
                # Design recommendations
                design_recs = rag_response.get('designRecommendations', [])
                if design_recs:
                    print("💡 Recommendations:")
                    for j, rec in enumerate(design_recs, 1):
                        print(f"   {j}. {rec}")
                
                # Economic feasibility
                econ_feasibility = rag_response.get('economicFeasibility', {})
                if econ_feasibility:
                    print("💰 Economic Impact:")
                    for key, value in econ_feasibility.items():
                        print(f"   {key.replace('_', ' ').title()}: {value}")
            
            else:
                print(f"❌ Query failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Query error: {str(e)}")


def main():
    """Main demonstration function."""
    print("🚀 Enhanced DWSIM with RAG Integration - Simple Demo")
    print("=" * 60)
    print("Showcasing intelligent chemical process design capabilities")
    print("for modular plant applications.")
    
    try:
        # 1. Service health check
        if not test_service_health():
            print("\n❌ Service not available. Please start the enhanced DWSIM service:")
            print("   cd docker_config && docker-compose up -d dwsim-service")
            return
        
        # 2. RAG-based queries
        demonstrate_rag_queries()
        
        # 3. Process optimization
        demonstrate_process_optimization()
        
        # 4. Modular design insights
        demonstrate_modular_design_insights()
        
        # Summary
        print_section("DEMO SUMMARY")
        print("✅ Enhanced DWSIM demo completed successfully!")
        print("\n🌟 Key Capabilities Demonstrated:")
        print("   🤖 RAG-based intelligent process queries")
        print("   🎯 Multi-objective process optimization")
        print("   🏗️  Modular plant design recommendations")
        print("   💰 Economic feasibility analysis")
        print("   ⚡ Real-time simulation integration")
        print("\n🔗 This system enables intelligent design and optimization")
        print("   of modular chemical plants with AI-driven insights.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 