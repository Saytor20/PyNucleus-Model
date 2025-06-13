# PyNucleus Prompt System

A **production-ready** Jinja2-based prompt template system for standardized LLM interactions in chemical engineering contexts. Fully integrated with the PyNucleus pipeline for **DWSIM simulation analysis** and **RAG-enhanced knowledge retrieval**.

## 📁 **Current System Structure**

```
prompts/
├── qwen_prompt.j2           # Main Jinja2 template with SYSTEM, CONTEXT, QUESTION, ANSWER sections
├── prompt_system.py         # Core PromptSystem class with full functionality
├── notebook_integration.py  # Simple integration for Jupyter notebooks
├── outputs/                 # Generated prompts saved here
└── README.md               # This documentation
```

**Integrated with PyNucleus Pipeline:**
```
src/pynucleus/
├── llm/query_llm.py        # LLM query manager with Jinja2 support
├── integration/            # Enhanced DWSIM-RAG integration
└── pipeline/               # Complete pipeline with prompt generation
```

---

## 🚀 **Quick Start - Production Usage**

### **Option 1: User-Friendly Notebook Integration (Recommended)**

**For Standard Users** - Add this to `Capstone Project.ipynb`:

```python
# Load the integrated prompt system (included in enhanced cells)
exec(open('prompts/notebook_integration.py').read())

# Create chemical engineering prompts with PyNucleus data
prompt = create_prompt(
    question="What optimization strategies would improve this distillation process?",
    system_msg="You are an expert chemical engineer specializing in process optimization",
    context="Analyzing ethanol-water separation with 82% efficiency and 6.5% ROI"
)

# Run system validation and demos
demo_prompts()
validate_prompts()
```

**For Advanced Users** - Available in `Developer_Notebook.ipynb`:
- **Section 4**: LLM Development & Testing (Cells 10-12)
- **Advanced prompt engineering** with multi-scenario testing
- **Custom template development** and validation
- **Integration with PyNucleus results** for enhanced context

### **Option 2: Integration with PyNucleus Pipeline**

```python
from src.pynucleus.llm.query_llm import LLMQueryManager

# Initialize with template directory
manager = LLMQueryManager(max_tokens=2048)

# Create prompts from simulation results
prompt = manager.render_prompt(
    user_query="Analyze the performance of this DWSIM simulation",
    system_message="You are a chemical process engineer",
    context=simulation_results_summary
)

# Query LLM with rendered prompt
response = manager.ask_llm(prompt)
```

### **Option 3: Direct Python Usage**

```python
from prompts.prompt_system import PromptSystem

ps = PromptSystem()
prompt = ps.generate_prompt(
    system_msg="You are an expert chemical engineer specializing in process optimization.",
    context="A distillation column showing 82% efficiency with potential improvements.",
    question="What troubleshooting and optimization steps do you recommend?",
    constraints="Consider safety protocols, environmental regulations, and cost-effectiveness.",
    format_instructions="Provide numbered steps with expected outcomes and ROI impact."
)
```

---

## 🔗 **PyNucleus Integration Features**

### **Automatic Simulation Analysis**

The prompt system automatically integrates with PyNucleus simulation results:

```python
# After running enhanced pipeline (Cells 10-14 in notebook)
if 'integrated_results' in globals():
    prompts = integrate_pynucleus_results(integrated_results)
    for i, prompt in enumerate(prompts):
        save_prompt(prompt, f"simulation_analysis_{i+1}.txt")
        
# Example generated prompt includes:
# - Process type and components
# - Feed conditions with mole fractions
# - Performance metrics (conversion, selectivity, yield)
# - Financial analysis (ROI, profit projections)
# - Optimization recommendations
```

### **Enhanced Context Generation**

Prompts automatically include:
- **DWSIM Simulation Data**: Operating conditions, performance metrics
- **RAG Knowledge**: Literature insights and best practices  
- **Financial Analysis**: ROI calculations and profit projections
- **Process Conditions**: Detailed feed conditions, temperatures, pressures

---

## 📝 **Template Structure & Variables**

### **Jinja2 Template (`qwen_prompt.j2`)**

```jinja2
<SYSTEM>
{{ system_message }}
{% if constraints %}

CONSTRAINTS:
{{ constraints }}
{% endif %}
{% if format_instructions %}

OUTPUT FORMAT:
{{ format_instructions }}
{% endif %}
</SYSTEM>

<CONTEXT>
{{ context }}
</CONTEXT>

<QUESTION>
{{ question }}
</QUESTION>

<ANSWER>

</ANSWER> 
```

### **Available Template Variables**

- **`system_message`**: AI role and behavior instructions
- **`context`**: Background information and simulation data
- **`question`**: Main query or analysis task (required)
- **`constraints`**: Optional safety/operational limitations
- **`format_instructions`**: Desired output structure

---

## 🧪 **Chemical Engineering Examples**

### **Process Optimization Example**

```python
prompt = create_prompt(
    question="How can we optimize this distillation process for better ROI?",
    system_msg="You are a chemical process optimization expert",
    context="""
    Process: Ethanol-Water Distillation
    Current Performance:
    - Efficiency: 82%
    - Recovery Rate: 85%
    - Daily Revenue: $148,500
    - Daily Profit: $58,500
    - ROI: 6.5%
    
    Operating Conditions:
    - Temperature: 78.4°C
    - Pressure: 101.325 kPa
    - Reflux Ratio: 2.5
    """,
    constraints="Maintain safety standards and environmental compliance",
    format_instructions="Provide recommendations with expected ROI improvement"
)
```

### **Financial Analysis Example**

```python
prompt = create_prompt(
    question="What is the financial impact of proposed process improvements?",
    system_msg="You are a chemical engineering economics specialist",
    context=f"""
    Current Financial Metrics:
    - Recovery Rate: 82.5%
    - Daily Revenue: $148,500
    - Operating Costs: $90,000/day
    - Net Profit: $58,500/day
    - ROI: 6.5%
    
    Proposed Improvements:
    - Increase reflux ratio to 3.0
    - Install heat integration
    - Optimize feed preheating
    """,
    format_instructions="Calculate ROI improvement and payback period"
)
```

---

## 🔧 **Available Functions**

### **Notebook Integration Functions**

- **`create_prompt(question, system_msg=None, context=None, **kwargs)`** - Quick prompt generation
- **`demo_prompts()`** - Show chemical engineering examples
- **`validate_prompts()`** - Test system functionality with 3 validation tests
- **`integrate_pynucleus_results(results)`** - Generate prompts from simulation data
- **`save_prompt(prompt, filename)`** - Save to `prompts/outputs/` directory

### **PromptSystem Class Methods**

- **`generate_prompt()`** - Main prompt generation with full parameter control
- **`validate_template()`** - Run comprehensive validation tests
- **`create_demo_prompts()`** - Get dictionary of demo prompts
- **`integrate_with_pynucleus()`** - Process PyNucleus simulation results
- **`save_prompt_to_file()`** - File saving with automatic output directory

### **LLM Integration Functions**

- **`manager.render_prompt()`** - Render prompts with template system
- **`manager.ask_llm()`** - Query HuggingFace models with prompts
- **`quick_ask_llm()`** - Single-function prompt generation and LLM query

---

## 📊 **System Validation & Health**

### **Comprehensive Validation**

The prompt system includes automated validation:
```python
validate_prompts()

# Runs 3 comprehensive tests:
# ✅ Test 1 (Minimal Input): Basic functionality
# ✅ Test 2 (Complete Input): Full feature testing  
# ✅ Test 3 (Section Structure): Template structure validation
```

### **Integration with System Health Monitoring**

- **System Diagnostic**: Included in `comprehensive_system_diagnostic.py`
- **Script Validation**: Part of `system_validator.py`
- **Health Status**: ✅ 100% healthy (2/2 prompt system scripts)

---

## 💡 **Production Usage Tips**

### **Best Practices**

1. **Always provide a question** - It's the only required parameter
2. **Use domain-specific system messages** - Improves LLM performance for chemical engineering
3. **Include simulation context** - Use PyNucleus integration for automatic context
4. **Specify output format** - Get structured, actionable responses
5. **Save important prompts** - Use `save_prompt()` for reusability

### **Performance Optimization**

- **Token Management**: Integrated with `src/pynucleus/utils/token_utils.py`
- **Template Caching**: Templates loaded once and reused
- **Context Optimization**: Automatic summarization for large simulation datasets

---

## 🔄 **Workflow Integration**

### **Standard PyNucleus Workflow**

```python
# 1. Run PyNucleus pipeline
from pynucleus.pipeline import PipelineUtils
pipeline = PipelineUtils(results_dir="data/05_output/results")
results = pipeline.run_complete_pipeline()

# 2. Generate analysis prompts
exec(open('prompts/notebook_integration.py').read())
analysis_prompts = integrate_pynucleus_results(results)

# 3. Save for LLM analysis
for i, prompt in enumerate(analysis_prompts):
    save_prompt(prompt, f"process_analysis_{i+1}.txt")
```

### **Enhanced Pipeline Workflow**

```python
# 1. Run enhanced pipeline with financial analysis
enhanced_results = integrator.integrate_simulation_results(dwsim_results)
llm_files = llm_generator.export_llm_ready_text(enhanced_results)

# 2. Create specialized prompts
optimization_prompt = create_prompt(
    question="What are the top 3 optimization opportunities?",
    context=open(llm_files[0]).read(),
    system_msg="You are a process optimization consultant"
)

# 3. Query LLM for insights
from src.pynucleus.llm.query_llm import quick_ask_llm
response = quick_ask_llm(optimization_prompt)
```

---

## 🛠️ **System Requirements**

### **Dependencies**
- **Python 3.9+** (tested with 3.13.1)
- **Jinja2** (`pip install jinja2`) - Template engine
- **PyNucleus Pipeline** - For simulation integration
- **HuggingFace Transformers** - For LLM integration (optional)

### **System Health Verification**
```bash
# Verify prompt system health
python scripts/comprehensive_system_diagnostic.py

# Expected: ✅ Jinja2 Prompts System: HEALTHY
```

---

## 📈 **Example Output Structures**

### **Process Troubleshooting Prompt**

```
<SYSTEM>
You are an expert chemical engineer specializing in distillation optimization.

CONSTRAINTS:
Consider safety protocols, environmental regulations, and cost-effectiveness.

OUTPUT FORMAT:
Provide numbered steps with expected outcomes and ROI impact.
</SYSTEM>

<CONTEXT>
Process: Ethanol-Water Distillation
Current Efficiency: 82%
Issues: Declining performance, higher energy costs
Operating Conditions: 78.4°C, 101.325 kPa, Reflux Ratio 2.5
</CONTEXT>

<QUESTION>
What troubleshooting and optimization steps do you recommend?
</QUESTION>

<ANSWER>

</ANSWER> 
```

### **Financial Analysis Prompt**

```
<SYSTEM>
You are a chemical engineering economics specialist.

OUTPUT FORMAT:
Calculate ROI improvement and payback period for each recommendation.
</SYSTEM>

<CONTEXT>
Current Financial Performance:
- Daily Revenue: $148,500
- Operating Costs: $90,000/day  
- Net Profit: $58,500/day
- ROI: 6.5%

Proposed Improvements: Heat integration, process optimization
</CONTEXT>

<QUESTION>
What is the financial impact of these process improvements?
</QUESTION>

<ANSWER>

</ANSWER> 
```

---

## 🎯 **Production Deployment Status**

### **✅ Ready for Production**
- **System Health**: 100% operational (verified)
- **Template Validation**: All tests passing
- **PyNucleus Integration**: Complete integration with pipeline
- **Documentation**: Comprehensive user guides and examples

### **✅ Enterprise Features**
- **Error Handling**: Graceful fallbacks for missing dependencies
- **Performance Monitoring**: Integrated with system diagnostics
- **Flexible Configuration**: Template-based customization
- **Scalable Architecture**: Supports multiple concurrent prompt generations

---

## 📚 **Additional Resources**

- **PyNucleus Main Documentation**: `README.md`
- **System Architecture**: `docs/project_info/PROJECT_STRUCTURE.md`
- **Enhanced Pipeline Guide**: `docs/ENHANCED_PIPELINE_SUMMARY.md`
- **API Documentation**: In-code docstrings and type hints

---

**Ready for production use with 100% system health and comprehensive PyNucleus integration!**

*Last Updated: 2025-06-11 - Integrated with PyNucleus v2.0 Production System* 