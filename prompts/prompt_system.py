#!/usr/bin/env python3
"""
PyNucleus Prompt System
=======================
Jinja2-based prompt template system for standardized LLM interactions.

Usage:
    from prompts.prompt_system import PromptSystem
    
    ps = PromptSystem()
    prompt = ps.generate_prompt(
        system_msg="You are an expert chemical engineer",
        context="Process data shows declining efficiency",
        question="What optimization strategies do you recommend?"
    )
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    print("❌ Jinja2 not found. Install with: pip install jinja2")
    sys.exit(1)


class PromptSystem:
    """Jinja2-based prompt template system for LLM interactions."""
    
    def __init__(self, template_dir: str = None):
        """
        Initialize the prompt system.
        
        Args:
            template_dir: Directory containing Jinja2 templates (default: current directory)
        """
        self.template_dir = template_dir or str(Path(__file__).parent)
        self.env = None
        self.template = None
        
        # Initialize Jinja2 environment
        self._setup_environment()
        
    def _setup_environment(self):
        """Setup Jinja2 environment and load template."""
        try:
            self.env = Environment(loader=FileSystemLoader(self.template_dir))
            self.template = self.env.get_template('qwen_prompt.j2')
            print(f"✅ Prompt system initialized - Template loaded from: {self.template_dir}")
        except Exception as e:
            print(f"❌ Error loading template: {e}")
            print(f"Template directory: {self.template_dir}")
            
    def generate_prompt(self, 
                       system_msg: Optional[str] = None,
                       context: Optional[str] = None, 
                       question: str = None,
                       constraints: Optional[str] = None,
                       format_instructions: Optional[str] = None,
                       **kwargs) -> str:
        """
        Generate a standardized LLM prompt using the Jinja2 template.
        
        Args:
            system_msg: System instructions for the AI
            context: Background information or data
            question: The main question or task (required)
            constraints: Optional constraints or requirements
            format_instructions: Optional output format specifications
            **kwargs: Additional template variables
            
        Returns:
            str: Rendered prompt ready for LLM
        """
        if not self.template:
            return "❌ Template not available"
            
        if not question:
            return "❌ Question parameter is required"
            
        try:
            rendered = self.template.render(
                system_message=system_msg,
                context=context,
                question=question,
                constraints=constraints,
                format_instructions=format_instructions,
                **kwargs
            )
            return rendered
        except Exception as e:
            return f"❌ Error rendering template: {e}"
    
    def validate_template(self) -> bool:
        """
        Validate template functionality with test cases.
        
        Returns:
            bool: True if all tests pass
        """
        print("🔍 Running Template Validation Tests...")
        print("=" * 50)
        
        if not self.template:
            print("❌ Template not loaded")
            return False
        
        tests = [
            {
                'name': 'Minimal Input',
                'params': {'question': 'What is process optimization?'},
                'check': lambda result: '<QUESTION>' in result and '<ANSWER>' in result
            },
            {
                'name': 'Complete Input',
                'params': {
                    'system_msg': 'Expert advisor',
                    'context': 'Industrial process data',
                    'question': 'Optimize this process',
                    'constraints': 'Safety first',
                    'format_instructions': 'Bullet points'
                },
                'check': lambda result: all(tag in result for tag in ['<SYSTEM>', '<CONTEXT>', '<QUESTION>', '<ANSWER>'])
            },
            {
                'name': 'Section Structure',
                'params': {'question': 'Test'},
                'check': lambda result: all(section in result for section in [
                    '<SYSTEM>', '</SYSTEM>', '<CONTEXT>', '</CONTEXT>', 
                    '<QUESTION>', '</QUESTION>', '<ANSWER>', '</ANSWER>'
                ])
            }
        ]
        
        all_passed = True
        for i, test in enumerate(tests, 1):
            try:
                result = self.generate_prompt(**test['params'])
                passed = test['check'](result)
                status = "✅" if passed else "❌"
                print(f"{status} Test {i} ({test['name']}): {'Passed' if passed else 'Failed'}")
                
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ Test {i} ({test['name']}): Error - {e}")
                all_passed = False
        
        print("=" * 50)
        if all_passed:
            print("🎉 All validation tests passed!")
        else:
            print("❌ Some tests failed!")
            
        return all_passed
    
    def create_demo_prompts(self) -> Dict[str, str]:
        """
        Create demonstration prompts for chemical engineering use cases.
        
        Returns:
            Dict[str, str]: Dictionary of demo prompts
        """
        demos = {}
        
        # Demo 1: Process Troubleshooting
        demos['process_troubleshooting'] = self.generate_prompt(
            system_msg="You are an expert chemical process engineer specializing in distillation optimization.",
            context="A methanol-water distillation column is showing declining separation efficiency. Recent data indicates reduced purity in the distillate and increased energy consumption.",
            question="What are the most likely causes of this efficiency decline and what systematic troubleshooting approach would you recommend?",
            constraints="Consider safety protocols, environmental regulations, and cost-effectiveness in all recommendations.",
            format_instructions="Provide response as: 1) Potential causes (ranked by probability), 2) Diagnostic steps, 3) Corrective actions with expected outcomes."
        )
        
        # Demo 2: Safety Analysis
        demos['safety_analysis'] = self.generate_prompt(
            system_msg="You are a chemical safety engineer with expertise in process hazard analysis.",
            context="A new reactor design is being proposed for acetylene production. The process involves high-pressure hydrogen and elevated temperatures.",
            question="What are the critical safety considerations and risk mitigation strategies for this reactor design?",
            constraints="Follow OSHA standards and chemical industry best practices.",
            format_instructions="Structure as: Risk Assessment, Safety Systems, Emergency Procedures, Compliance Requirements."
        )
        
        # Demo 3: Process Optimization
        demos['process_optimization'] = self.generate_prompt(
            system_msg="You are a process optimization consultant specializing in energy efficiency and cost reduction.",
            context="A chemical plant wants to reduce energy consumption in their distillation operations while maintaining product quality.",
            question="What optimization strategies would you recommend to achieve 15% energy reduction without compromising separation efficiency?",
            format_instructions="Present as: Current Analysis, Optimization Opportunities, Implementation Plan, Expected ROI."
        )
        
        return demos
    
    def integrate_with_pynucleus(self, simulation_results: List[Dict]) -> List[str]:
        """
        Generate prompts based on PyNucleus simulation results.
        
        Args:
            simulation_results: List of simulation result dictionaries
            
        Returns:
            List[str]: Generated prompts for each simulation
        """
        prompts = []
        
        for result in simulation_results:
            try:
                # Handle different data types
                if isinstance(result, dict):
                    case_name = result.get('case_name', 'Unknown')
                    components = result.get('components', [])
                else:
                    # If result is not a dict, convert to string and use as case_name
                    case_name = str(result)
                    components = []
                
                # Build context string
                context_parts = [f"Simulation: {case_name}"]
                
                if components:
                    context_parts.append(f"Components: {', '.join(components)}")
                
                # Add performance metrics if available (only if result is dict)
                if isinstance(result, dict) and 'performance_metrics' in result:
                    metrics = result['performance_metrics']
                    if 'overall_performance' in metrics:
                        context_parts.append(f"Performance: {metrics['overall_performance']}")
                    if 'efficiency_rating' in metrics:
                        context_parts.append(f"Efficiency: {metrics['efficiency_rating']}")
                
                # Add conversion/yield data if available (only if result is dict)
                if isinstance(result, dict):
                    if 'conversion' in result:
                        context_parts.append(f"Conversion: {result['conversion']:.1%}")
                    if 'selectivity' in result:
                        context_parts.append(f"Selectivity: {result['selectivity']:.1%}")
                
                context = " | ".join(context_parts)
                
                prompt = self.generate_prompt(
                    system_msg="You are a chemical engineering consultant analyzing simulation results for process optimization.",
                    context=context,
                    question="Based on these simulation results, what process improvements would you recommend to enhance efficiency and reduce costs?",
                    format_instructions="Structure response with: Executive Summary, Technical Analysis, Recommendations, Expected ROI"
                )
                
                prompts.append(prompt)
                
            except Exception as e:
                # Handle error message properly for both dict and non-dict types
                if isinstance(result, dict):
                    result_name = result.get('case_name', 'unknown')
                else:
                    result_name = str(result)[:50]  # First 50 chars if string
                    
                error_prompt = f"❌ Error generating prompt for {result_name}: {e}"
                prompts.append(error_prompt)
        
        return prompts
    
    def save_prompt_to_file(self, prompt: str, filename: str, output_dir: str = "outputs") -> str:
        """
        Save generated prompt to a file.
        
        Args:
            prompt: The generated prompt text
            filename: Output filename
            output_dir: Output directory (created if doesn't exist)
            
        Returns:
            str: Path to saved file
        """
        output_path = Path(self.template_dir) / output_dir
        output_path.mkdir(exist_ok=True)
        
        file_path = output_path / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            return str(file_path)
        except Exception as e:
            return f"❌ Error saving file: {e}"


def main():
    """Command-line interface for the prompt system."""
    print("🎯 PyNucleus Prompt System")
    print("=" * 50)
    
    # Initialize system
    ps = PromptSystem()
    
    # Run validation
    if ps.validate_template():
        print("\n🧪 Demo Prompts:")
        print("=" * 50)
        
        demos = ps.create_demo_prompts()
        
        for name, prompt in demos.items():
            print(f"\n📋 {name.replace('_', ' ').title()}:")
            print("─" * 40)
            print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
            print()
        
        print("✅ Prompt system ready for use!")
        print("\n💡 Usage:")
        print("  from prompts.prompt_system import PromptSystem")
        print("  ps = PromptSystem()")
        print("  prompt = ps.generate_prompt(question='Your question here')")
        
    else:
        print("❌ System validation failed!")


if __name__ == "__main__":
    main() 