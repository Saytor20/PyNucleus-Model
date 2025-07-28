#!/bin/bash

# PyNucleus Environment Setup Script
# This script ensures all dependencies are properly installed and the vector database is preserved

set -e  # Exit on any error

echo "🚀 PyNucleus Environment Setup"
echo "================================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Please activate your virtual environment first:"
    echo "   source .venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment detected: $VIRTUAL_ENV"

# Function to check if a Python package is installed
check_package() {
    local package=$1
    local min_version=$2
    
    if python -c "import $package" 2>/dev/null; then
        local version=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
        echo "✅ $package: $version"
        return 0
    else
        echo "❌ $package: Not installed"
        return 1
    fi
}

# Function to test bitsandbytes functionality
test_bitsandbytes() {
    echo "🔧 Testing bitsandbytes functionality..."
    
    python -c "
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
print('✅ bitsandbytes: Import successful')

try:
    config = BitsAndBytesConfig(load_in_8bit=True)
    print('✅ bitsandbytes: 8-bit quantization config created')
except Exception as e:
    print(f'⚠️  bitsandbytes: 8-bit quantization unavailable - {e}')
    print('   This is expected on macOS without GPU support')
" || echo "❌ bitsandbytes test failed"
}

# Function to check ChromaDB and document count
check_chromadb() {
    echo "🗄️  Checking ChromaDB status..."
    
    python -c "
import sys
sys.path.append('src')

try:
    from pynucleus.utils.telemetry_patch import apply_telemetry_patch
    apply_telemetry_patch()
    
    import chromadb
    from chromadb.config import Settings
    from pathlib import Path
    
    # Check main ChromaDB instance
    client = chromadb.PersistentClient(
        path='data/03_intermediate/vector_db',
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    collections = client.list_collections()
    total_docs = 0
    
    if not collections:
        print('⚠️  ChromaDB: No collections found')
        print('   Run option 5 (Ingest documents) from the main menu to populate the database')
    else:
        for collection in collections:
            count = collection.count()
            total_docs += count
            print(f'✅ ChromaDB: Collection \"{collection.name}\" has {count} documents')
    
    if total_docs == 0:
        print('⚠️  ChromaDB: Database is empty')
        print('   This might be why you see \"0 documents from 0 unique sources\"')
        print('   Run option 5 (Ingest documents) from the main menu to fix this')
    else:
        print(f'✅ ChromaDB: Total of {total_docs} documents available')

except Exception as e:
    print(f'❌ ChromaDB: Error - {e}')
    print('   This might indicate a configuration issue')
" || echo "❌ ChromaDB check failed"
}

# Function to backup ChromaDB
backup_chromadb() {
    local backup_dir="data/backups/chromadb_$(date +%Y%m%d_%H%M%S)"
    
    if [[ -d "data/03_intermediate/vector_db" ]] && [[ $(ls -A data/03_intermediate/vector_db 2>/dev/null | wc -l) -gt 0 ]]; then
        echo "💾 Creating ChromaDB backup..."
        mkdir -p "$backup_dir"
        cp -r data/03_intermediate/vector_db/* "$backup_dir/"
        echo "✅ ChromaDB backed up to: $backup_dir"
        
        # Keep only the 5 most recent backups
        echo "🧹 Cleaning old backups (keeping 5 most recent)..."
        find data/backups -maxdepth 1 -name "chromadb_*" -type d | sort -r | tail -n +6 | xargs rm -rf
    else
        echo "ℹ️  No ChromaDB data to backup"
    fi
}

# Main execution
echo ""
echo "📦 Checking core dependencies..."

# Check Python version
python_version=$(python --version 2>&1)
echo "✅ Python: $python_version"

# Check critical packages
critical_packages=(
    "torch"
    "transformers" 
    "chromadb"
    "bitsandbytes"
    "sentence_transformers"
)

missing_packages=()

for package in "${critical_packages[@]}"; do
    if ! check_package "$package"; then
        missing_packages+=("$package")
    fi
done

# Install missing packages if any
if [[ ${#missing_packages[@]} -gt 0 ]]; then
    echo ""
    echo "📥 Installing missing packages..."
    pip install "${missing_packages[@]}"
    echo "✅ Package installation completed"
fi

# Update packages to ensure compatibility
echo ""
echo "🔄 Updating packages to ensure compatibility..."
pip install --upgrade bitsandbytes transformers torch sentence-transformers

echo ""
test_bitsandbytes

echo ""
backup_chromadb

echo ""
check_chromadb

echo ""
echo "🎯 Environment Setup Complete!"
echo "================================"
echo ""
echo "📋 Summary:"
echo "   • All dependencies are installed and up-to-date"
echo "   • ChromaDB status checked and backed up"
echo "   • bitsandbytes tested for compatibility"
echo ""
echo "💡 To prevent losing your vector database:"
echo "   1. Never delete data/03_intermediate/vector_db/"
echo "   2. Backups are automatically created in data/backups/"
echo "   3. If you see '0 documents', run: pynucleus -> option 5 (Ingest documents)"
echo ""
echo "🚀 You can now run: pynucleus"