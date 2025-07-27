# PyNucleus Troubleshooting Guide

This guide addresses common issues and provides permanent solutions for PyNucleus system problems.

## Quick Fixes

### 🔧 Automated Fixes
Run these scripts to automatically resolve most issues:

```bash
# Complete environment setup and dependency check
./setup_environment.sh

# Comprehensive system diagnostics
python diagnose_system.py
```

## Common Issues and Solutions

### 1. Bitsandbytes Quantization Error
**Error**: `Using bitsandbytes 8-bit quantization requires the latest version of bitsandbytes`

**Cause**: Outdated bitsandbytes or version mismatch in requirements.txt

**Solution**:
```bash
pip install -U bitsandbytes
# Or run: ./setup_environment.sh
```

**Note**: On macOS, you may see warnings about GPU support - this is normal and expected.

### 2. RAG System Returns 0 Documents
**Error**: `Enhanced retrieval: 0 documents from 0 unique sources`

**Causes**: 
- Empty ChromaDB database
- Documents not ingested
- ChromaDB connection issues

**Solutions**:
1. **Check document count**:
   ```bash
   python diagnose_system.py
   ```

2. **Re-ingest documents**:
   ```bash
   pynucleus
   # Choose option 5: Ingest documents
   ```

3. **Manual check**:
   ```bash
   python -c "
   import sys; sys.path.append('src')
   from pynucleus.rag.collector import ingest
   result = ingest('data/raw/source_documents')
   print(f'Ingestion result: {result}')
   "
   ```

### 3. Vector Database Gets Reset
**Problem**: Documents disappear between sessions

**Permanent Prevention**:
1. **Never delete**: `data/03_intermediate/vector_db/`
2. **Automatic backups**: Created in `data/backups/chromadb_*/`
3. **Run setup script**: `./setup_environment.sh` creates backups
4. **Check .gitignore**: Ensure vector DB isn't being ignored

**Recovery**: Use backups from `data/backups/` directory

### 4. Model Loading Issues After Reopening
**Problem**: Model fails to load or uses wrong model

**Solutions**:
1. **Clear model cache**:
   ```bash
   rm -rf cache/models/*_state.pkl
   ```

2. **Verify model configuration**:
   ```python
   from pynucleus.settings import settings
   print(f"Configured model: {settings.MODEL_ID}")
   ```

3. **Force model reload**:
   ```bash
   pynucleus
   # Choose option 2: Chat with LLM (forces model reload)
   ```

### 5. Virtual Environment Issues
**Problem**: Dependencies missing after closing/reopening

**Solution**:
```bash
# Always activate virtual environment first
source .venv/bin/activate

# Then run setup script
./setup_environment.sh

# Verify environment
python diagnose_system.py
```

## Preventive Maintenance

### Weekly Maintenance
```bash
# Run environment setup to update dependencies
./setup_environment.sh

# Verify system health
python diagnose_system.py
```

### Before Each Session
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Quick health check (optional)
python diagnose_system.py | grep -E "(SUMMARY|❌|⚠️)"

# 3. Start PyNucleus
pynucleus
```

## File Structure Protection

### Critical Directories (Never Delete)
- `data/03_intermediate/vector_db/` - Main ChromaDB storage
- `.venv/` - Virtual environment
- `cache/models/` - Model cache (delete only to force reload)

### Backup Locations
- `data/backups/chromadb_*/` - Automatic ChromaDB backups
- Keep at least 5 most recent backups

## Environment Variables

Add to your `.bashrc` or `.zshrc` for convenience:
```bash
# PyNucleus aliases
alias pn-setup='cd /path/to/PyNucleus-Model && source .venv/bin/activate && ./setup_environment.sh'
alias pn-diagnose='cd /path/to/PyNucleus-Model && source .venv/bin/activate && python diagnose_system.py'
alias pn-start='cd /path/to/PyNucleus-Model && source .venv/bin/activate && pynucleus'
```

## Advanced Diagnostics

### Manual ChromaDB Check
```python
import sys; sys.path.append('src')
from pynucleus.utils.telemetry_patch import apply_telemetry_patch
apply_telemetry_patch()
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path='data/03_intermediate/vector_db',
    settings=Settings(anonymized_telemetry=False, allow_reset=True)
)

for collection in client.list_collections():
    print(f"Collection: {collection.name}, Documents: {collection.count()}")
```

### Model Information
```python
import sys; sys.path.append('src')
from pynucleus.llm.model_loader import get_model_info
print(get_model_info())
```

## Emergency Recovery

### Complete Reset (Last Resort)
```bash
# 1. Backup important data
cp -r data/03_intermediate/vector_db data/vector_db_backup

# 2. Clean installation
pip install -r requirements.txt --force-reinstall

# 3. Re-ingest documents
pynucleus  # Choose option 5

# 4. Verify recovery
python diagnose_system.py
```

## Getting Help

1. **First**: Run `python diagnose_system.py` and share the output
2. **Include**: Error messages, system info, and steps to reproduce
3. **Check**: This troubleshooting guide first
4. **Contact**: Include diagnostic output when reporting issues

---

*This guide is automatically updated. For the latest version, run:*
```bash
git pull origin main
```