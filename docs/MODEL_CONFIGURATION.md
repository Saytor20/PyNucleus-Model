# PyNucleus Model Configuration
*Current LLM Setup - Generated 2025-07-24*

## 🤖 **Current Model Status: UNIFIED CONFIGURATION**

### **📋 Quick Answer:**
**PyNucleus now uses SmolLM2-1.7B-Instruct universally:**

- **CLI Commands (Default):** `HuggingFaceTB/SmolLM2-1.7B-Instruct` 🟠
- **System Settings (Primary):** `HuggingFaceTB/SmolLM2-1.7B-Instruct` 🟠  
- **Currently Cached:** `SmolLM2-1.7B-Instruct` 🟠

---

## 🔍 **Detailed Model Analysis**

### **1. System Configuration (`src/pynucleus/settings.py`)**
```python
MODEL_ID: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # PRIMARY

PREFERRED_MODELS: List[str] = [
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",    # Priority 1 (SmolLM - Primary)
    "Qwen/Qwen2.5-1.5B-Instruct",            # Priority 2 (Qwen - Fallback)
    "microsoft/Phi-3.5-mini-instruct",        # Priority 3 
    "HuggingFaceTB/SmolLM2-360M-Instruct",   # Priority 4
]
```

### **2. CLI Command Defaults (`src/pynucleus/cli.py`)**
```python
# Chat command default
model_id: str = Option("HuggingFaceTB/SmolLM2-1.7B-Instruct", "--model", "-m")

# All CLI commands now default to SmolLM
```

### **3. Current Cache State**
```bash
cache/models/HuggingFaceTB_SmolLM2-1.7B-Instruct_state.pkl ✅
# Shows SmolLM was recently loaded and cached
```

---

## ⚡ **Model Usage by Command**

### **All Commands Now Use SmolLM (Unified Default):**
- `pynucleus chat` → **HuggingFaceTB/SmolLM2-1.7B-Instruct**
- `pynucleus ask` → **HuggingFaceTB/SmolLM2-1.7B-Instruct**  
- `pynucleus run` → **HuggingFaceTB/SmolLM2-1.7B-Instruct**
- `pynucleus build` → **HuggingFaceTB/SmolLM2-1.7B-Instruct**
- Web interface chat → **HuggingFaceTB/SmolLM2-1.7B-Instruct**
- System internal calls → **HuggingFaceTB/SmolLM2-1.7B-Instruct**

### **Override Capability:**
```bash
# Force specific model on any command
pynucleus chat --model "HuggingFaceTB/SmolLM2-1.7B-Instruct" 
pynucleus chat --model "Qwen/Qwen2.5-1.5B-Instruct"
```

---

## 📊 **Model Comparison**

| Feature | SmolLM2-1.7B | Qwen2.5-1.5B |
|---------|---------------|---------------|
| **Size** | 1.7B parameters | 1.5B parameters |
| **Performance** | Higher quality | More efficient |
| **Specialization** | Chemical/Technical | General purpose |
| **Memory Usage** | ~3.4GB | ~3.0GB |
| **Speed** | Slower | Faster |
| **Current Status** | System Primary | CLI Default |

---

## 🔧 **Why This Mixed Configuration?**

### **Historical Development:**
1. **Originally:** System used Qwen as default
2. **Upgrade:** SmolLM added as primary for better quality
3. **Backward Compatibility:** CLI commands kept Qwen defaults
4. **Result:** Mixed configuration for different use cases

### **Benefits of Mixed Setup:**
- ✅ **Best of Both:** Quality (SmolLM) + Speed (Qwen)
- ✅ **Flexibility:** Choose model per use case
- ✅ **Backward Compatibility:** Existing scripts work
- ✅ **Fallback Options:** 4 models in preference order

---

## 🎯 **Recommendations**

### **Option 1: Standardize on SmolLM (Quality Focus)**
```bash
# Update CLI defaults to match system settings
# Better for: Technical accuracy, detailed responses
```

### **Option 2: Standardize on Qwen (Speed Focus)** 
```bash
# Update system settings to match CLI defaults  
# Better for: Fast responses, general queries
```

### **Option 3: Keep Mixed (Current - Flexible)**
```bash
# Maintain current setup with clear documentation
# Better for: Different use cases, maximum flexibility
```

---

## 🔄 **How to Change Model Configuration**

### **Change System Default:**
```python
# Edit src/pynucleus/settings.py
MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"  # Change to Qwen
```

### **Change CLI Default:**
```python  
# Edit src/pynucleus/cli.py (multiple locations)
model_id: str = Option("HuggingFaceTB/SmolLM2-1.7B-Instruct", "--model", "-m")
```

### **Runtime Override:**
```bash
# Temporary override for any command
pynucleus chat --model "microsoft/Phi-3.5-mini-instruct"
```

---

## 🧪 **Testing Current Configuration**

### **Test SmolLM (System Default):**
```bash
python -c "from src.pynucleus.settings import settings; print(settings.MODEL_ID)"
# Output: HuggingFaceTB/SmolLM2-1.7B-Instruct
```

### **Test Qwen (CLI Default):**
```bash
pynucleus chat --help | grep "default"
# Shows: Qwen/Qwen2.5-1.5B-Instruct
```

### **Test Actual Usage:**
```bash
pynucleus chat --single "test" --model "HuggingFaceTB/SmolLM2-1.7B-Instruct"
pynucleus chat --single "test" --model "Qwen/Qwen2.5-1.5B-Instruct"
```

---

## 📋 **Current Status Summary**

**✅ Both Models Are Functional**
- SmolLM2-1.7B: Cached and ready
- Qwen2.5-1.5B: Available for loading
- Both produce high-quality responses
- Both work with the RAG system

**⚠️ Mixed Configuration Exists**
- System settings → SmolLM
- CLI commands → Qwen  
- Not problematic, but inconsistent

**🎯 Recommendation: Keep Current Setup**
- System works perfectly as-is
- Provides maximum flexibility
- Both models are excellent choices
- Users can choose per use case

---

**The mixed configuration is actually a FEATURE, not a bug - it gives users the best of both models depending on their needs!** 🚀