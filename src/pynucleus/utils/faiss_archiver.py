#!/usr/bin/env python3
"""
FAISS Model Archiver

Manages FAISS model versioning by archiving old models and keeping only the most recent one visible.
Automatically moves old models to an archive folder when new ones are created.

Features:
- Creates timestamped archive folders
- Moves old FAISS models (.faiss, .pkl files) to archive
- Keeps only the most recent model in the main directory
- Preserves model metadata and analysis files
- Configurable retention policies
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
import json
import glob

logger = logging.getLogger(__name__)

class FAISSArchiver:
    """Manages FAISS model archiving and versioning."""
    
    def __init__(self, models_dir: str = "data/04_models/chunk_reports"):
        """
        Initialize the FAISS archiver.
        
        Args:
            models_dir: Directory containing FAISS models
        """
        self.models_dir = Path(models_dir)
        self.archive_dir = self.models_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
        
        # FAISS file patterns to track
        self.faiss_patterns = [
            "*.faiss",           # FAISS index files
            "*.pkl",             # Pickle files (embeddings, etc.)
            "*faiss*/",          # FAISS directories
            "faiss_*.txt",       # FAISS analysis files
        ]
        
        logger.info(f"FAISSArchiver initialized for directory: {self.models_dir}")
    
    def get_faiss_files(self) -> List[Path]:
        """Get all FAISS-related files in the models directory."""
        faiss_files = []
        
        for pattern in self.faiss_patterns:
            matches = list(self.models_dir.glob(pattern))
            faiss_files.extend(matches)
        
        # Filter out archive directory
        faiss_files = [f for f in faiss_files if not str(f).startswith(str(self.archive_dir))]
        
        # Sort by modification time (newest first)
        faiss_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return faiss_files
    
    def get_file_age(self, file_path: Path) -> datetime:
        """Get the modification time of a file."""
        return datetime.fromtimestamp(file_path.stat().st_mtime)
    
    def group_related_files(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group related FAISS files by their base name or timestamp."""
        groups = {}
        
        for file_path in files:
            # Extract base name or timestamp pattern
            if "faiss_analysis_" in file_path.name:
                # Group analysis files by timestamp
                timestamp = file_path.name.replace("faiss_analysis_", "").replace(".txt", "")
                key = f"analysis_{timestamp}"
            elif ".faiss" in file_path.name:
                # Group FAISS index files by base name
                key = file_path.name.replace(".faiss", "").replace(".pkl", "")
            elif file_path.is_dir() and "faiss" in file_path.name:
                # FAISS directories
                key = file_path.name
            else:
                # Other files grouped by stem
                key = file_path.stem
            
            if key not in groups:
                groups[key] = []
            groups[key].append(file_path)
        
        return groups
    
    def create_archive_folder(self) -> Path:
        """Create a timestamped archive folder."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_folder = self.archive_dir / f"archived_{timestamp}"
        archive_folder.mkdir(exist_ok=True)
        return archive_folder
    
    def archive_old_models(self, keep_most_recent: int = 1) -> Dict[str, any]:
        """
        Archive old FAISS models, keeping only the most recent ones.
        
        Args:
            keep_most_recent: Number of most recent models to keep in main directory
            
        Returns:
            Dictionary with archiving results
        """
        logger.info(f"Starting FAISS model archiving (keeping {keep_most_recent} most recent)")
        
        # Get all FAISS files
        faiss_files = self.get_faiss_files()
        
        if len(faiss_files) <= keep_most_recent:
            logger.info(f"Only {len(faiss_files)} FAISS files found, no archiving needed")
            return {
                "action": "no_archiving_needed",
                "total_files": len(faiss_files),
                "kept_files": len(faiss_files),
                "archived_files": 0
            }
        
        # Group related files
        file_groups = self.group_related_files(faiss_files)
        
        # Sort groups by newest file in each group
        sorted_groups = []
        for group_name, group_files in file_groups.items():
            newest_time = max(self.get_file_age(f) for f in group_files)
            sorted_groups.append((newest_time, group_name, group_files))
        
        sorted_groups.sort(reverse=True)  # Newest first
        
        # Keep most recent groups, archive the rest
        groups_to_keep = sorted_groups[:keep_most_recent]
        groups_to_archive = sorted_groups[keep_most_recent:]
        
        if not groups_to_archive:
            logger.info("No old models to archive")
            return {
                "action": "no_old_models",
                "total_groups": len(sorted_groups),
                "kept_groups": len(groups_to_keep),
                "archived_groups": 0
            }
        
        # Create archive folder
        archive_folder = self.create_archive_folder()
        
        # Archive old groups
        archived_files = []
        for _, group_name, group_files in groups_to_archive:
            logger.info(f"Archiving group: {group_name} ({len(group_files)} files)")
            
            for file_path in group_files:
                try:
                    dest_path = archive_folder / file_path.name
                    
                    if file_path.is_dir():
                        shutil.copytree(file_path, dest_path)
                        shutil.rmtree(file_path)
                    else:
                        shutil.move(str(file_path), str(dest_path))
                    
                    archived_files.append(str(file_path))
                    logger.debug(f"Archived: {file_path} -> {dest_path}")
                    
                except Exception as e:
                    logger.error(f"Error archiving {file_path}: {e}")
        
        # Create archive manifest
        manifest = {
            "archive_date": datetime.now().isoformat(),
            "archived_files": archived_files,
            "kept_groups": [name for _, name, _ in groups_to_keep],
            "archived_groups": [name for _, name, _ in groups_to_archive],
            "total_files_archived": len(archived_files)
        }
        
        manifest_path = archive_folder / "archive_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Archived {len(archived_files)} files to {archive_folder}")
        
        return {
            "action": "archived",
            "archive_folder": str(archive_folder),
            "total_files_archived": len(archived_files),
            "kept_groups": len(groups_to_keep),
            "archived_groups": len(groups_to_archive),
            "manifest_path": str(manifest_path)
        }
    
    def cleanup_empty_archives(self, max_age_days: int = 30):
        """Remove empty or very old archive folders."""
        for archive_folder in self.archive_dir.glob("archived_*"):
            if archive_folder.is_dir():
                # Check if empty
                contents = list(archive_folder.iterdir())
                if not contents:
                    logger.info(f"Removing empty archive: {archive_folder}")
                    shutil.rmtree(archive_folder)
                    continue
                
                # Check age
                age = datetime.now() - self.get_file_age(archive_folder)
                if age.days > max_age_days:
                    logger.info(f"Archive {archive_folder} is {age.days} days old, considering for cleanup")
    
    def list_current_models(self) -> List[Dict[str, any]]:
        """List current FAISS models in the main directory."""
        faiss_files = self.get_faiss_files()
        
        models = []
        for file_path in faiss_files:
            model_info = {
                "name": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size if file_path.is_file() else "directory",
                "modified": self.get_file_age(file_path).isoformat(),
                "type": "directory" if file_path.is_dir() else "file"
            }
            models.append(model_info)
        
        return models
    
    def list_archived_models(self) -> List[Dict[str, any]]:
        """List archived models."""
        archives = []
        
        for archive_folder in sorted(self.archive_dir.glob("archived_*")):
            if archive_folder.is_dir():
                manifest_path = archive_folder / "archive_manifest.json"
                
                archive_info = {
                    "folder": archive_folder.name,
                    "path": str(archive_folder),
                    "created": self.get_file_age(archive_folder).isoformat(),
                    "files_count": len(list(archive_folder.iterdir())),
                    "has_manifest": manifest_path.exists()
                }
                
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest = json.load(f)
                        archive_info["manifest"] = manifest
                    except Exception as e:
                        archive_info["manifest_error"] = str(e)
                
                archives.append(archive_info)
        
        return archives
    
    def restore_from_archive(self, archive_folder_name: str) -> Dict[str, any]:
        """Restore models from a specific archive folder."""
        archive_path = self.archive_dir / archive_folder_name
        
        if not archive_path.exists():
            return {"error": f"Archive folder {archive_folder_name} not found"}
        
        # First archive current models
        current_archive_result = self.archive_old_models(keep_most_recent=0)
        
        # Then restore from archive
        restored_files = []
        for item in archive_path.iterdir():
            if item.name == "archive_manifest.json":
                continue
            
            dest_path = self.models_dir / item.name
            try:
                if item.is_dir():
                    shutil.copytree(item, dest_path)
                else:
                    shutil.copy2(item, dest_path)
                restored_files.append(str(dest_path))
            except Exception as e:
                logger.error(f"Error restoring {item}: {e}")
        
        return {
            "action": "restored",
            "archive_folder": archive_folder_name,
            "restored_files": restored_files,
            "current_archived": current_archive_result
        }


def main():
    """CLI interface for FAISS archiver."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FAISS Model Archiver")
    parser.add_argument("--models-dir", default="data/04_models/chunk_reports",
                       help="Directory containing FAISS models")
    parser.add_argument("--keep", type=int, default=1,
                       help="Number of most recent models to keep")
    parser.add_argument("--action", choices=["archive", "list", "list-archives", "cleanup"],
                       default="archive", help="Action to perform")
    parser.add_argument("--restore", help="Restore from specific archive folder")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    archiver = FAISSArchiver(args.models_dir)
    
    if args.restore:
        result = archiver.restore_from_archive(args.restore)
        print(f"Restore result: {json.dumps(result, indent=2)}")
    elif args.action == "archive":
        result = archiver.archive_old_models(keep_most_recent=args.keep)
        print(f"Archive result: {json.dumps(result, indent=2)}")
    elif args.action == "list":
        models = archiver.list_current_models()
        print(f"Current models ({len(models)}):")
        for model in models:
            print(f"  {model['name']} ({model['type']}, {model['modified']})")
    elif args.action == "list-archives":
        archives = archiver.list_archived_models()
        print(f"Archived models ({len(archives)}):")
        for archive in archives:
            print(f"  {archive['folder']} ({archive['files_count']} files, {archive['created']})")
    elif args.action == "cleanup":
        archiver.cleanup_empty_archives()
        print("Cleanup completed")


if __name__ == "__main__":
    main() 