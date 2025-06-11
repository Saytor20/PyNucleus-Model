# PyNucleus Prompt System

A Jinja2-based prompt template system for standardized LLM interactions in chemical engineering contexts.

## 📁 Files Structure

```
prompts/
├── qwen_prompt.j2           # Main Jinja2 template with SYSTEM, CONTEXT, QUESTION, ANSWER sections
├── prompt_system.py         # Core PromptSystem class with all functionality
├── notebook_integration.py  # Simple integration for Jupyter notebooks
├── outputs/                 # Generated prompts will be saved here
└── README.md               # This file
```

## 🚀 Quick Start

### Option 1: Jupyter Notebook Integration (Recommended)

Add this to a new cell in your `Capstone Project.ipynb`:

```python
# Load the prompt system
exec(open('prompts/notebook_integration.py').read())

# Create a simple prompt
prompt = create_prompt(
    question="What are the key factors in distillation column optimization?",
    system_msg="You are an expert chemical engineer",
    context="We're analyzing a methanol-water separation process"
)
print(prompt)

# Run demos and validation
demo_prompts()
validate_prompts()
```

### Option 2: Direct Python Usage

```python
from prompts.prompt_system import PromptSystem

ps = PromptSystem()
prompt = ps.generate_prompt(
    system_msg="You are an expert chemical engineer specializing in process optimization.",
    context="A distillation column is showing declining efficiency.",
    question="What troubleshooting steps do you recommend?",
    constraints="Consider safety and environmental regulations.",
    format_instructions="Provide numbered steps with expected outcomes."
)
```

### Option 3: Command Line

```bash
cd prompts
python prompt_system.py
```

## 🧪 Integration with PyNucleus

The prompt system can automatically generate prompts from your PyNucleus simulation results:

```python
# After running your PyNucleus pipeline
if 'integrated_results' in globals():
    prompts = integrate_pynucleus_results(integrated_results)
    for i, prompt in enumerate(prompts):
        save_prompt(prompt, f"simulation_analysis_{i+1}.txt")
```

## 📝 Template Structure

The `qwen_prompt.j2` template includes:

- **SYSTEM**: AI role and behavior instructions
- **CONTEXT**: Background information and data
- **QUESTION**: Main query or task
- **ANSWER**: Placeholder for LLM response

### Template Variables

- `system_message`: AI instructions and role
- `context`: Background information
- `question`: Main question (required)
- `constraints`: Optional limitations
- `format_instructions`: Output format specifications

## 🔧 Available Functions

### Notebook Integration Functions

- `create_prompt(question, system_msg=None, context=None, **kwargs)` - Quick prompt generation
- `demo_prompts()` - Show chemical engineering examples
- `validate_prompts()` - Test system functionality
- `integrate_pynucleus_results(results)` - Generate from simulation data
- `save_prompt(prompt, filename)` - Save to outputs/ directory

### PromptSystem Class Methods

- `generate_prompt()` - Main prompt generation
- `validate_template()` - Run validation tests
- `create_demo_prompts()` - Get demo prompt dictionary
- `integrate_with_pynucleus()` - Process simulation results
- `save_prompt_to_file()` - File saving functionality

## 📊 Chemical Engineering Examples

The system includes pre-built templates for:

1. **Process Troubleshooting** - Diagnosing efficiency issues
2. **Safety Analysis** - Risk assessment and mitigation
3. **Process Optimization** - Energy efficiency and cost reduction

## 🔍 Validation

The system includes comprehensive validation:
- Template loading verification
- Section structure checks
- Variable substitution tests
- Error handling validation

## 💡 Usage Tips

1. **Always provide a question** - It's the only required parameter
2. **Use descriptive system messages** - They improve LLM performance
3. **Include relevant context** - More context = better responses
4. **Specify output format** - Get structured responses
5. **Save important prompts** - Use `save_prompt()` for reuse

## 🛠️ Requirements

- Python 3.7+
- Jinja2 (`pip install jinja2`)
- PyNucleus pipeline (for integration features)

## 📈 Example Output

```
<SYSTEM>
You are an expert chemical engineer specializing in distillation optimization.

CONSTRAINTS:
Consider safety protocols and environmental regulations.

OUTPUT FORMAT:
Provide numbered steps with expected outcomes.
</SYSTEM>

<CONTEXT>
A methanol-water distillation column is showing declining efficiency.
</CONTEXT>

<QUESTION>
What troubleshooting steps do you recommend?
</QUESTION>

<ANSWER>

</ANSWER> 