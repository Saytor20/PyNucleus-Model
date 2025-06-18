"""
Results exporter for PyNucleus system.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

class ResultsExporter:
    """Export pipeline results in various formats."""
    
    def __init__(self, output_dir: str = "data/05_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def export_json(self, data: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """
        Export data as JSON file.
        
        Args:
            data: Data to export
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
            
        output_file = self.output_dir / "results" / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Results exported to JSON: {output_file}")
        return output_file
    
    def export_csv(self, data: List[Dict[str, Any]], filename: Optional[str] = None) -> Path:
        """
        Export data as CSV file.
        
        Args:
            data: List of dictionaries to export
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.csv"
            
        output_file = self.output_dir / "results" / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not data:
            self.logger.warning("No data to export to CSV")
            return output_file
            
        # Get fieldnames from first record
        fieldnames = list(data[0].keys())
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
            
        self.logger.info(f"Results exported to CSV: {output_file}")
        return output_file
    
    def export_llm_report(self, data: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """
        Export data as LLM-ready report.
        
        Args:
            data: Data to export
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_report_{timestamp}.md"
            
        output_file = self.output_dir / "llm_reports" / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate markdown report
        report_content = self._generate_markdown_report(data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        self.logger.info(f"LLM report exported: {output_file}")
        return output_file
    
    def _generate_markdown_report(self, data: Dict[str, Any]) -> str:
        """Generate markdown report from data."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# PyNucleus Analysis Report

Generated: {timestamp}

## Summary
{data.get('summary', 'Analysis completed successfully')}

## Results
```json
{json.dumps(data, indent=2)}
```

## Metadata
- Timestamp: {timestamp}
- Status: {data.get('status', 'completed')}
- Output Directory: {self.output_dir}
"""
        return report 