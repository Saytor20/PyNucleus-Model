#!/usr/bin/env python3
"""
Notebook Integration for PyNucleus Prompt System
================================================
Simple integration script for Jupyter notebooks.

Usage in notebook:
    exec(open('prompts/notebook_integration.py').read())
    
    # Then use:
    prompt = create_prompt("What is process optimization?")
    demo_prompts()
    validate_prompts()
"""

import sys
from pathlib import Path

# Add prompts directory to path
prompts_dir = Path("prompts").resolve()
if str(prompts_dir) not in sys.path:
    sys.path.insert(0, str(prompts_dir))

# Initialize prompt system
ps = None

try:
    try:
        from prompts.prompt_system import PromptSystem
    except ImportError:
        from prompt_system import PromptSystem
    
    # Initialize global prompt system
    ps = PromptSystem()
    
    print("✅ Prompt system loaded successfully!")
    
except ImportError as e:
    print(f"❌ Error importing prompt system: {e}")
    print("Make sure Jinja2 is installed: pip install jinja2")
    
except Exception as e:
    print(f"❌ Error initializing prompt system: {e}")
    print("Check that prompts/qwen_prompt.j2 exists")

# Define functions (always available, even if ps is None)
def create_prompt(question, system_msg=None, context=None, **kwargs):
    """Quick prompt generation function."""
    if ps is None:
        return f"Error: Prompt system not initialized. Question was: {question}"
    
    return ps.generate_prompt(
        question=question,
        system_msg=system_msg,
        context=context,
        **kwargs
    )

def demo_prompts():
    """Show demonstration prompts."""
    if ps is None:
        print("❌ Error: Prompt system not initialized")
        return
        
    print("🧪 PyNucleus Prompt System Demo")
    print("=" * 50)
    
    demos = ps.create_demo_prompts()
    
    for i, (name, prompt) in enumerate(demos.items(), 1):
        print(f"\n📋 Example {i}: {name.replace('_', ' ').title()}")
        print("─" * 40)
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
        print("\n" + "═" * 60)

def validate_prompts():
    """Validate prompt system functionality."""
    if ps is None:
        print("❌ Error: Prompt system not initialized")
        return False
        
    return ps.validate_template()

def integrate_pynucleus_results(results):
    """Generate prompts from PyNucleus simulation results."""
    if ps is None:
        print("❌ Error: Prompt system not initialized")
        return None
        
    return ps.integrate_with_pynucleus(results)

def save_prompt(prompt, filename):
    """Save prompt to file."""
    if ps is None:
        print("❌ Error: Prompt system not initialized")
        return False
        
    return ps.save_prompt_to_file(prompt, filename)

# Show available functions
if ps is not None:
    print("📚 Available functions:")
    print("   • create_prompt(question, system_msg=None, context=None, **kwargs)")
    print("   • demo_prompts() - Show example prompts")
    print("   • validate_prompts() - Test system functionality")
    print("   • integrate_pynucleus_results(results) - Generate from simulation data")
    print("   • save_prompt(prompt, filename) - Save to file") 