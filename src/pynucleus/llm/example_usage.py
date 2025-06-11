"""
Example usage of the LLMRunner class.

This script demonstrates how to use the LLMRunner for various text generation tasks.
Note: This requires a model download on first run and may take some time.
"""

import time
import sys
import os

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.pynucleus.llm import LLMRunner


def basic_example():
    """Demonstrate basic usage of LLMRunner."""
    print("=" * 50)
    print("Basic LLMRunner Example")
    print("=" * 50)
    
    # Initialize with default model (gpt2)
    print("\n1. Initializing LLMRunner with default model...")
    runner = LLMRunner()
    
    # Get model information
    print("\n2. Model Information:")
    info = runner.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Generate some text
    print("\n3. Text Generation Examples:")
    
    prompts = [
        "The future of artificial intelligence",
        "Climate change and renewable energy",
        "The importance of scientific research"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n   Example {i}: '{prompt}'")
        try:
            response = runner.ask(
                prompt, 
                max_length=50,  # Keep it short for demo
                temperature=0.7,
                do_sample=True
            )
            print(f"   Response: {response}")
        except Exception as e:
            print(f"   Error: {e}")


def advanced_example():
    """Demonstrate advanced features of LLMRunner."""
    print("=" * 50)
    print("Advanced LLMRunner Example")
    print("=" * 50)
    
    # Initialize with a different model
    print("\n1. Initializing with custom model...")
    try:
        # Try to use a slightly larger model
        runner = LLMRunner(model_id="gpt2-medium", device="cpu")
        print("   Using gpt2-medium model")
    except Exception as e:
        print(f"   Fallback to gpt2: {e}")
        runner = LLMRunner(model_id="gpt2", device="cpu")
    
    # Multiple response generation
    print("\n2. Generating multiple responses:")
    prompt = "Once upon a time, in a land far away,"
    print(f"   Prompt: '{prompt}'")
    
    responses = runner.ask(
        prompt, 
        max_length=60,
        num_return_sequences=3,
        temperature=0.8,
        do_sample=True
    )
    
    for i, response in enumerate(responses, 1):
        print(f"   Response {i}: '{response}'")
    
    # Greedy decoding (deterministic)
    print("\n3. Greedy decoding (deterministic):")
    prompt = "To solve this problem, we need to"
    print(f"   Prompt: '{prompt}'")
    
    response = runner.ask(
        prompt,
        max_length=50,
        do_sample=False  # Greedy decoding
    )
    print(f"   Response: '{response}'")
    
    # Creative generation with high temperature
    print("\n4. Creative generation (high temperature):")
    prompt = "In a world where robots can think,"
    print(f"   Prompt: '{prompt}'")
    
    response = runner.ask(
        prompt,
        max_length=70,
        temperature=1.0,
        top_p=0.9,
        do_sample=True
    )
    print(f"   Response: '{response}'")


def different_models_example():
    """Demonstrate using different models."""
    print("\n" + "=" * 50)
    print("Different Models Example")
    print("=" * 50)
    
    models_to_try = [
        "gpt2",
        "distilgpt2",
        "microsoft/DialoGPT-small"
    ]
    
    for model_id in models_to_try:
        print(f"\n--- Testing model: {model_id} ---")
        try:
            runner = LLMRunner(model_id=model_id, device="cpu")
            
            # Test simple generation
            prompt = "Hello, how are you?"
            response = runner.ask(prompt, max_length=40, temperature=0.7)
            
            print(f"   Model: {model_id}")
            print(f"   Prompt: '{prompt}'")
            print(f"   Response: '{response}'")
            
            # Show model info
            info = runner.get_model_info()
            print(f"   Parameters: {info.get('parameters_human', 'Unknown')}")
            
        except Exception as e:
            print(f"   Failed to load {model_id}: {e}")
            continue


def error_handling_example():
    """Demonstrate error handling."""
    print("\n" + "=" * 50)
    print("Error Handling Example")
    print("=" * 50)
    
    try:
        runner = LLMRunner()
    except Exception as e:
        print(f"Failed to initialize runner: {e}")
        return
    
    # Test empty prompt
    print("\n1. Testing empty prompt:")
    try:
        runner.ask("")
    except ValueError as e:
        print(f"   Expected error: {e}")
    
    # Test invalid parameters
    print("\n2. Testing invalid max_length:")
    try:
        runner.ask("Hello", max_length=0)
    except ValueError as e:
        print(f"   Expected error: {e}")
    
    # Test invalid device (during initialization)
    print("\n3. Testing invalid device:")
    try:
        bad_runner = LLMRunner(device="invalid")
    except ValueError as e:
        print(f"   Expected error: {e}")
    
    print("\n   All error handling tests passed!")


def performance_comparison():
    """Compare performance with different parameters."""
    print("\n" + "=" * 50)
    print("Performance Comparison")
    print("=" * 50)
    
    try:
        runner = LLMRunner()
    except Exception as e:
        print(f"Failed to initialize runner: {e}")
        return
    
    prompt = "The benefits of renewable energy include"
    configurations = [
        {"name": "Greedy (fastest)", "do_sample": False, "max_length": 50},
        {"name": "Low temp sampling", "do_sample": True, "temperature": 0.3, "max_length": 50},
        {"name": "High temp sampling", "do_sample": True, "temperature": 0.9, "max_length": 50},
    ]
    
    print(f"\nPrompt: '{prompt}'\n")
    
    for config in configurations:
        name = config.pop("name")
        print(f"--- {name} ---")
        
        start_time = time.time()
        response = runner.ask(prompt, **config)
        end_time = time.time()
        
        print(f"Response: '{response}'")
        print(f"Time: {end_time - start_time:.2f} seconds\n")


def main():
    """Run all examples."""
    print("LLMRunner Example Usage")
    print("This may take some time on first run as models need to be downloaded.")
    print("Press Ctrl+C to interrupt if needed.\n")
    
    try:
        # Run examples
        basic_example()
        advanced_example()
        error_handling_example()
        performance_comparison()
        
        # Skip different models example by default to avoid long download times
        # Uncomment the line below to test multiple models
        # different_models_example()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 