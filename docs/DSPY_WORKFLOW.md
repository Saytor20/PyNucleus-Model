# **DSPy Workflow Guide for PyNucleus** 🧠

## **Overview**

This guide explains how to use DSPy (Declarative Self-improving Language Programs) with PyNucleus for enhanced chemical engineering query processing. DSPy enables structured prompting and automatic optimization of language model interactions.

### **What DSPy Provides:**
- **Structured Prompting**: Replace manual prompt engineering with declarative programs
- **Automatic Optimization**: Compile programs using development examples
- **Fallback Support**: Graceful degradation when compiled programs aren't available
- **Performance Validation**: Built-in testing and validation workflows

---

## **Quick Start** ⚡

### **1. Create Sample Development Dataset**
```bash
# Create sample dataset for testing
pynucleus dspy-compile --create-sample
```

### **2. Compile DSPy Program**
```bash
# Compile using the sample dataset
pynucleus dspy-compile
```

### **3. Test Enhanced Queries**
```bash
# Now queries will use compiled DSPy program (no WARN line)
pynucleus ask --question "What is the optimal reactor temperature?"
```

---

## **Development Workflow** 🔄

### **Step 1: Manage Development Examples**

The development dataset is stored in `docs/devset/dspy_examples.csv` with the following format:

```csv
question,context,expected_answer,domain,difficulty
"What is the optimal temperature for ethanol distillation?","Chemical engineering process optimization","The optimal temperature for ethanol distillation is typically between 78-85°C...",distillation,medium
```

**Required Columns:**
- `question`: The input question
- `context`: Background context for the question
- `expected_answer`: The ideal response
- `domain`: Chemical engineering domain (optional)
- `difficulty`: Question difficulty level (optional)

### **Step 2: Add New Examples**

To improve DSPy performance, add new examples to the CSV:

```bash
# Edit the development dataset
nano docs/devset/dspy_examples.csv

# Add your new examples following the format
```

**Example Addition:**
```csv
"How do you calculate reactor sizing for continuous flow?","Reactor design for chemical processes","Reactor sizing for continuous flow uses V = Q*τ where V is volume, Q is flow rate, and τ is residence time.",reactor_design,medium
```

### **Step 3: Recompile Program**

After adding examples, recompile the DSPy program:

```bash
# Recompile with updated examples
pynucleus dspy-compile --verbose

# Check compilation success
ls -la data/dspy_artifacts/
```

### **Step 4: Validate Performance**

Test the compiled program with your new examples:

```bash
# Test individual questions
pynucleus ask --question "Your new question here"

# The response should show "🧠 DSPY ENHANCED RESPONSE" instead of warnings
```

---

## **CI/CD Integration** 🔄

### **Continuous Integration Workflow**

The DSPy compilation is integrated into the CI pipeline:

```yaml
# In .github/workflows/ci.yml
- name: Test DSPy Compilation
  run: |
    python -m pynucleus.cli dspy-compile --ci
    # This creates artifacts but doesn't commit them
```

### **CI Mode Behavior**

When running `pynucleus dspy-compile --ci`:
- ✅ Compiles DSPy program using development examples
- ✅ Validates compilation success
- ✅ Creates artifacts in `data/dspy_artifacts/`
- ❌ **Does NOT commit artifacts to repository**
- ⚠️ Fails CI if compilation errors occur

### **Local vs CI Compilation**

| Mode | Command | Artifacts Saved | Purpose |
|------|---------|----------------|---------|
| **Local** | `pynucleus dspy-compile` | ✅ Yes | Development & production use |
| **CI** | `pynucleus dspy-compile --ci` | ✅ Yes (not committed) | Validation only |

---

## **Fallback Mechanism** 🛟

### **How Fallback Works**

PyNucleus automatically handles missing compiled programs:

```python
# When you run: pynucleus ask --question "..."

1. ✅ Check for compiled DSPy program in data/dspy_artifacts/
2. If found:    → Use compiled program (enhanced response)
3. If missing:  → Fall back to uncompiled RAG system
4. Display appropriate response type indicator
```

### **Response Type Indicators**

| Indicator | Program Type | Performance |
|-----------|-------------|-------------|
| `🧠 DSPY ENHANCED RESPONSE` | Compiled DSPy | ⚡ Optimized |
| `📋 RAG SYSTEM RESPONSE` | Fallback RAG | 📊 Standard |

### **When Fallback Occurs**

- No compiled artifacts in `data/dspy_artifacts/`
- Compilation artifacts are corrupted
- DSPy dependencies not installed
- Compilation failed during development

---

## **Advanced Usage** 🚀

### **Custom Development Datasets**

Create domain-specific datasets for specialized compilation:

```bash
# Create custom dataset for reactor design
pynucleus dspy-compile \
  --csv-path docs/devset/reactor_design_examples.csv \
  --output-dir data/dspy_artifacts/reactor_specialized
```

### **Multiple Domain Support**

Organize examples by chemical engineering domains:

```
docs/devset/
├── dspy_examples.csv          # General examples
├── distillation_examples.csv  # Distillation-specific
├── reactor_examples.csv       # Reactor design
└── safety_examples.csv        # Process safety
```

### **Validation and Testing**

The DSPy compiler includes built-in validation:

```bash
# Verbose compilation with validation details
pynucleus dspy-compile --verbose

# Check validation results
cat logs/pynucleus_*.log | grep "Validation"
```

### **Performance Monitoring**

Monitor DSPy performance vs fallback:

```bash
# Test same question with both systems
pynucleus ask --question "Your test question"

# Compare response quality and generation time
```

---

## **Troubleshooting** 🔧

### **Common Issues**

**1. Compilation Fails**
```bash
# Check logs for detailed error information
tail -f logs/pynucleus_*.log

# Verify CSV format
python -c "import pandas as pd; print(pd.read_csv('docs/devset/dspy_examples.csv').head())"
```

**2. No DSPy Enhancement**
```bash
# Check if artifacts exist
ls -la data/dspy_artifacts/

# Recompile if missing
pynucleus dspy-compile
```

**3. Import Errors**
```bash
# Verify DSPy installation
pip install dspy-ai

# Check import
python -c "import dspy; print('DSPy available')"
```

### **Debugging Commands**

```bash
# Create fresh sample dataset
pynucleus dspy-compile --create-sample

# Force recompilation
rm -rf data/dspy_artifacts/
pynucleus dspy-compile --verbose

# Test fallback behavior
mv data/dspy_artifacts data/dspy_artifacts_backup
pynucleus ask --question "Test question"
mv data/dspy_artifacts_backup data/dspy_artifacts
```

---

## **Best Practices** 📚

### **Development Examples Quality**

1. **Diverse Questions**: Cover multiple chemical engineering domains
2. **Clear Context**: Provide sufficient background information
3. **Accurate Answers**: Ensure expected answers are technically correct
4. **Balanced Difficulty**: Mix easy, medium, and hard questions
5. **Realistic Scenarios**: Use actual process engineering situations

### **Example Quality Checklist**

- [ ] Question is clear and specific
- [ ] Context provides necessary background
- [ ] Expected answer is technically accurate
- [ ] Answer length is appropriate (50-200 words)
- [ ] Covers important chemical engineering concepts

### **Maintenance Schedule**

- **Weekly**: Review and update development examples
- **Monthly**: Analyze query performance and add new domains
- **Quarterly**: Full compilation validation and optimization
- **Before releases**: Ensure CI compilation passes

---

## **Integration with Notebooks** 📓

### **Jupyter Notebook Usage**

The compiled DSPy programs work seamlessly in notebooks:

```python
# In Capstone Project.ipynb or Developer_Notebook.ipynb
from src.pynucleus.llm.dspy_compile import DSPyCompiler

# Check compilation status
compiler = DSPyCompiler()
program = compiler.get_program()  # Gets compiled or fallback

# Use in notebook workflows
result = program.answer_general("context", "question")
```

### **Notebook-Specific Features**

- Automatic compilation status checks
- Visual indicators for DSPy vs fallback
- Interactive compilation from notebook cells
- Performance comparison tools

---

## **Getting Help** 💬

### **Resources**

- **PyNucleus Documentation**: `docs/README.md`
- **DSPy Official Documentation**: [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- **Development Logs**: `logs/pynucleus_*.log`
- **System Diagnostics**: `python -m pynucleus.diagnostics.runner --full`

### **Support Commands**

```bash
# Check system health
python scripts/comprehensive_system_diagnostic.py

# Validate DSPy integration
python -m pytest tests/test_dspy_flow.py

# Generate detailed logs
pynucleus dspy-compile --verbose 2>&1 | tee dspy_debug.log
```

---

**🎉 You're now ready to use DSPy with PyNucleus for enhanced chemical engineering analysis!** 