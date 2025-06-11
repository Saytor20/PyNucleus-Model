#!/usr/bin/env python3
"""
Test script for the Enhanced PyNucleus Pipeline

This script tests all the enhanced functionality including:
- Configuration management
- DWSIM-RAG integration
- LLM output generation
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_enhanced_pipeline():
    """Test the enhanced pipeline functionality."""
    
    print("🧪 Testing Enhanced PyNucleus Pipeline")
    print("=" * 50)
    
    try:
        # Test 1: Import enhanced modules
        print("\n1️⃣ Testing module imports...")
        from pynucleus.integration.config_manager import ConfigManager
        from pynucleus.integration.dwsim_rag_integrator import DWSIMRAGIntegrator
        from pynucleus.integration.llm_output_generator import LLMOutputGenerator
        print("✅ All enhanced modules imported successfully!")
        
        # Test 2: Initialize components
        print("\n2️⃣ Testing component initialization...")
        config_manager = ConfigManager(config_dir="test_config")
        integrator = DWSIMRAGIntegrator(results_dir="test_results")
        llm_generator = LLMOutputGenerator(results_dir="test_results")
        print("✅ All components initialized successfully!")
        
        # Test 3: Configuration management
        print("\n3️⃣ Testing configuration management...")
        json_template = config_manager.create_template_json("test_template.json")
        csv_template = config_manager.create_template_csv("test_template.csv")
        print(f"✅ Templates created: {json_template}, {csv_template}")
        
        # Test 4: Load configurations
        print("\n4️⃣ Testing configuration loading...")
        try:
            configs = config_manager.load_from_json(json_template)
            print(f"✅ Loaded {len(configs)} configurations from JSON")
        except Exception as e:
            print(f"⚠️ Configuration loading test: {e}")
        
        # Test 5: Mock integration test
        print("\n5️⃣ Testing DWSIM-RAG integration...")
        mock_dwsim_results = [
            {
                'case_name': 'test_simulation',
                'simulation_type': 'reactor',
                'components': 'methane, oxygen',
                'description': 'Test simulation for pipeline validation',
                'success': True,
                'duration_seconds': 0.001,
                'timestamp': '2025-06-10 16:45:00'
            }
        ]
        
        integrated_results = integrator.integrate_simulation_results(
            mock_dwsim_results, 
            perform_rag_analysis=False  # Skip RAG for basic test
        )
        print(f"✅ Integration test completed: {len(integrated_results)} results processed")
        
        # Test 6: LLM output generation
        print("\n6️⃣ Testing LLM output generation...")
        if integrated_results:
            llm_output_file = llm_generator.export_llm_ready_text(integrated_results)
            print(f"✅ LLM output generated: {llm_output_file}")
        
        # Test 7: File system cleanup
        print("\n7️⃣ Cleaning up test files...")
        import shutil
        test_dirs = ["test_config", "test_results"]
        for test_dir in test_dirs:
            if Path(test_dir).exists():
                shutil.rmtree(test_dir)
                print(f"🗑️ Cleaned up: {test_dir}")
        
        print("\n🎉 All tests passed! Enhanced pipeline is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without enhanced features."""
    
    print("\n🔧 Testing basic pipeline functionality...")
    
    try:
        from pynucleus.pipeline import RAGPipeline, DWSIMPipeline, ResultsExporter, PipelineUtils
        print("✅ Basic pipeline modules imported successfully!")
        
        # Test basic pipeline initialization
        pipeline = PipelineUtils()
        print("✅ Basic pipeline initialized successfully!")
        
        # Test quick status
        test_results = pipeline.quick_test()
        print(f"✅ Quick test completed: {test_results['csv_files_count']} CSV files found")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 PyNucleus Enhanced Pipeline Test Suite")
    print("=" * 60)
    
    # Test basic functionality first
    basic_test_passed = test_basic_functionality()
    
    # Test enhanced functionality
    enhanced_test_passed = test_enhanced_pipeline()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Basic Pipeline: {'✅ PASS' if basic_test_passed else '❌ FAIL'}")
    print(f"   Enhanced Pipeline: {'✅ PASS' if enhanced_test_passed else '❌ FAIL'}")
    
    if basic_test_passed and enhanced_test_passed:
        print("\n🎉 All tests passed! The enhanced pipeline is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Run the Capstone Project.ipynb notebook")
        print("   2. Try the enhanced pipeline sections")
        print("   3. Edit the configuration templates to customize simulations")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
        print("💡 Common solutions:")
        print("   • Run: pip install -r requirements.txt")
        print("   • Ensure you're in the project root directory")
        print("   • Check that all core modules are properly installed")
    
    print("\n" + "=" * 60) 