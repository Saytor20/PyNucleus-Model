#!/usr/bin/env python3
"""
PyNucleus Directory Reorganization Script
========================================

This script reorganizes the PyNucleus directory structure for better organization
and eliminates duplicate storage locations.
"""

import shutil
import json
from pathlib import Path
from datetime import datetime

class DirectoryReorganizer:
    def __init__(self, base_path=None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.backup_created = False
        
    def create_backup(self):
        """Create a backup of the current structure."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.base_path / f"backup_before_reorganization_{timestamp}"
        
        print(f"📦 Creating backup at: {backup_path}")
        
        # Items to backup
        backup_items = [
            "chroma_db",
            "data/03_processed/chromadb", 
            "cache/models"
        ]
        
        backup_path.mkdir(parents=True, exist_ok=True)
        
        for item in backup_items:
            src = self.base_path / item
            if src.exists():
                dst = backup_path / item
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
                print(f"  ✅ Backed up: {item}")
        
        self.backup_created = True
        return backup_path
    
    def create_new_structure(self):
        """Create the new directory structure."""
        print("🏗️  Creating new directory structure...")
        
        new_dirs = [
            "config",
            "models/cache",
            "models/trained", 
            "models/embeddings",
            "storage/vector_db",
            "storage/metadata",
            "outputs/pipeline",
            "outputs/validation",
            "outputs/exports",
            "docs"
        ]
        
        for dir_path in new_dirs:
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created: {dir_path}")
    
    def migrate_data(self):
        """Migrate data to new structure."""
        print("📦 Migrating data to new structure...")
        
        migrations = [
            # Configuration files
            ("configs", "config"),
            
            # Model files
            ("cache/models", "models/cache"),
            
            # Vector database - consolidate to single location
            ("data/03_intermediate/vector_db", "storage/vector_db"),
            
            # Validation results
            ("data/validation", "outputs/validation"),
            
            # Pipeline outputs
            ("data/05_output", "outputs/pipeline"),
        ]
        
        for src, dst in migrations:
            src_path = self.base_path / src
            dst_path = self.base_path / dst
            
            if src_path.exists():
                if dst_path.exists():
                    # Merge directories
                    self._merge_directories(src_path, dst_path)
                else:
                    # Move directory
                    shutil.move(str(src_path), str(dst_path))
                print(f"  ✅ Migrated: {src} → {dst}")
            else:
                print(f"  ℹ️  Skipped: {src} (doesn't exist)")
    
    def _merge_directories(self, src, dst):
        """Merge source directory into destination."""
        for item in src.rglob('*'):
            if item.is_file():
                relative_path = item.relative_to(src)
                dst_file = dst / relative_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst_file)
    
    def cleanup_duplicates(self):
        """Clean up duplicate locations."""
        print("🧹 Cleaning up duplicate locations...")
        
        cleanup_items = [
            "chroma_db",  # Root level ChromaDB
            "data/03_processed/chromadb",  # Duplicate ChromaDB
        ]
        
        for item in cleanup_items:
            item_path = self.base_path / item
            if item_path.exists():
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                else:
                    item_path.unlink()
                print(f"  ✅ Removed: {item}")
    
    def update_settings(self):
        """Update settings to reflect new paths."""
        print("⚙️  Updating settings for new paths...")
        
        settings_file = self.base_path / "src/pynucleus/settings.py"
        if settings_file.exists():
            content = settings_file.read_text()
            
            # Update ChromaDB path
            old_path = 'CHROMA_PATH: str = "data/03_intermediate/vector_db"'
            new_path = 'CHROMA_PATH: str = "storage/vector_db"'
            
            if old_path in content:
                content = content.replace(old_path, new_path)
                settings_file.write_text(content)
                print("  ✅ Updated ChromaDB path in settings.py")
            else:
                print("  ℹ️  ChromaDB path setting not found or already updated")
    
    def create_directory_map(self):
        """Create a directory structure map."""
        print("📋 Creating directory structure documentation...")
        
        structure = {
            "reorganization_date": datetime.now().isoformat(),
            "new_structure": {
                "config/": "All configuration files (production, development, logging)",
                "data/": "Data pipeline following DVC structure (01_raw, 02_processed, etc.)",
                "models/": {
                    "cache/": "Cached model files and states",
                    "trained/": "Custom trained models", 
                    "embeddings/": "Embedding model files"
                },
                "storage/": {
                    "vector_db/": "ChromaDB vector database (consolidated)",
                    "metadata/": "System metadata and indexes"
                },
                "outputs/": {
                    "pipeline/": "Pipeline execution results",
                    "validation/": "System validation reports",
                    "exports/": "Data exports and reports"
                },
                "logs/": "Application and system logs",
                "scripts/": "Utility and maintenance scripts", 
                "src/": "Source code (unchanged)",
                "docs/": "Documentation and system overviews"
            },
            "migration_log": {
                "configs → config": "Configuration consolidation",
                "cache/models → models/cache": "Model cache centralization",
                "data/03_intermediate/vector_db → storage/vector_db": "Vector DB consolidation",
                "data/validation → outputs/validation": "Validation results organization",
                "data/05_output → outputs/pipeline": "Pipeline output organization"
            }
        }
        
        docs_dir = self.base_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        map_file = docs_dir / "directory_structure.json"
        with open(map_file, 'w') as f:
            json.dump(structure, f, indent=2)
        
        print(f"  ✅ Created directory map: {map_file}")
    
    def reorganize(self, create_backup=True):
        """Execute the complete reorganization."""
        print("🔄 PyNucleus Directory Reorganization")
        print("=" * 40)
        
        if create_backup:
            backup_path = self.create_backup()
            print(f"📦 Backup created at: {backup_path}")
        
        try:
            self.create_new_structure()
            self.migrate_data()
            self.update_settings()
            self.cleanup_duplicates()
            self.create_directory_map()
            
            print("\n✅ Reorganization completed successfully!")
            print("📋 Summary:")
            print("  • Consolidated ChromaDB storage")
            print("  • Centralized model cache")
            print("  • Organized outputs by type")
            print("  • Updated configuration paths")
            print("  • Created documentation")
            
            if self.backup_created:
                print(f"  • Backup available at: backup_before_reorganization_*")
            
            print("\n💡 Next steps:")
            print("  1. Test system functionality: python -m src.pynucleus.cli version")
            print("  2. Run health check: python scripts/comprehensive_health_check.py")
            print("  3. Remove backup if everything works: rm -rf backup_before_reorganization_*")
            
        except Exception as e:
            print(f"\n❌ Error during reorganization: {e}")
            if self.backup_created:
                print("💡 Restore from backup if needed")
            raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reorganize PyNucleus directory structure")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    parser.add_argument("--path", help="Base path (default: current directory)")
    
    args = parser.parse_args()
    
    reorganizer = DirectoryReorganizer(args.path)
    reorganizer.reorganize(create_backup=not args.no_backup)