"""
Results Exporter Module for PyNucleus Pipeline

Handles exporting pipeline results to CSV files with focus on DWSIM simulation data.
"""

import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class ResultsExporter:
    """Handles exporting pipeline results to CSV files."""
    
    def __init__(self, results_dir="results"):
        """Initialize Results Exporter with output directory."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        print(f"📁 Results directory: {self.results_dir}")
    
    def export_dwsim_results(self, dwsim_data, filename="dwsim_simulation_results.csv"):
        """Export DWSIM simulation results to a clean, readable CSV."""
        if not dwsim_data:
            print("⚠️ No DWSIM simulation results to export")
            return None
        
        # Clean and format the data
        formatted_data = []
        for result in dwsim_data:
            formatted_result = {
                'Case Name': result.get('case_name', 'Unknown'),
                'Simulation Type': result.get('simulation_type', 'Unknown'),
                'Components': result.get('components', 'N/A'),
                'Description': result.get('description', 'N/A'),
                'Status': result.get('status', 'Unknown'),
                'Duration (s)': result.get('duration_seconds', 0),
                'Conversion': self._format_value(result.get('conversion')),
                'Selectivity': self._format_value(result.get('selectivity')),
                'Yield': self._format_value(result.get('yield')),
                'Temperature (°C)': self._format_value(result.get('temperature')),
                'Pressure (atm)': self._format_value(result.get('pressure')),
                'Success': 'Yes' if result.get('success', False) else 'No',
                'Timestamp': result.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            }
            formatted_data.append(formatted_result)
        
        # Export to CSV with clean formatting
        df = pd.DataFrame(formatted_data)
        filepath = self.results_dir / filename
        df.to_csv(filepath, index=False)
        
        print(f"✅ DWSIM results exported: {filepath}")
        print(f"   📊 {len(dwsim_data)} simulation results exported")
        print(f"   📋 Columns: Case Name, Type, Components, Status, Performance Metrics")
        return filepath
    
    def _format_value(self, value):
        """Format a value for clean CSV display."""
        if value is None or value == 'N/A':
            return 'N/A'
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                return round(value, 3)
            return value
        return str(value)
    
    def export_dwsim_summary(self, dwsim_data, filename="dwsim_summary.csv"):
        """Export a concise summary of DWSIM simulation results."""
        if not dwsim_data:
            print("⚠️ No DWSIM data for summary")
            return None
        
        # Calculate summary statistics
        total_sims = len(dwsim_data)
        successful_sims = sum(1 for r in dwsim_data if r.get('success', False))
        failed_sims = total_sims - successful_sims
        avg_duration = sum(r.get('duration_seconds', 0) for r in dwsim_data) / total_sims if total_sims > 0 else 0
        
        # Get simulation types
        sim_types = {}
        for result in dwsim_data:
            sim_type = result.get('simulation_type', 'Unknown')
            if sim_type not in sim_types:
                sim_types[sim_type] = {'count': 0, 'success': 0}
            sim_types[sim_type]['count'] += 1
            if result.get('success', False):
                sim_types[sim_type]['success'] += 1
        
        summary_data = [{
            'Metric': 'Total Simulations',
            'Value': total_sims,
            'Unit': 'count'
        }, {
            'Metric': 'Successful Simulations', 
            'Value': successful_sims,
            'Unit': 'count'
        }, {
            'Metric': 'Failed Simulations',
            'Value': failed_sims, 
            'Unit': 'count'
        }, {
            'Metric': 'Success Rate',
            'Value': round((successful_sims / total_sims * 100), 1) if total_sims > 0 else 0,
            'Unit': '%'
        }, {
            'Metric': 'Average Duration',
            'Value': round(avg_duration, 3),
            'Unit': 'seconds'
        }]
        
        # Add simulation type breakdown
        for sim_type, stats in sim_types.items():
            summary_data.append({
                'Metric': f'{sim_type.title()} Simulations',
                'Value': f"{stats['success']}/{stats['count']}",
                'Unit': 'success/total'
            })
        
        df = pd.DataFrame(summary_data)
        filepath = self.results_dir / filename
        df.to_csv(filepath, index=False)
        
        print(f"✅ DWSIM summary exported: {filepath}")
        return filepath
    
    def export_simulation_only(self, dwsim_data):
        """Export only DWSIM simulation data (simplified version)."""
        print("\n💾 Exporting DWSIM simulation results...")
        
        exported_files = []
        
        # Export main results
        if dwsim_data:
            results_file = self.export_dwsim_results(dwsim_data)
            if results_file:
                exported_files.append(results_file)
            
            # Export summary
            summary_file = self.export_dwsim_summary(dwsim_data)
            if summary_file:
                exported_files.append(summary_file)
        
        # Display final summary
        self._display_export_summary(exported_files)
        return exported_files
    
    def _display_export_summary(self, exported_files):
        """Display summary of exported files."""
        print(f"\n🎉 Export completed successfully!")
        print(f"📁 All results saved in: {self.results_dir}")
        print(f"📈 Files created:")
        
        for filepath in exported_files:
            file_size = filepath.stat().st_size
            print(f"   • {filepath.name} ({file_size:,} bytes)")
    
    def view_dwsim_results(self):
        """View DWSIM simulation results in a readable format."""
        print(f"\n📋 DWSIM Simulation Results:")
        
        try:
            results_path = self.results_dir / 'dwsim_simulation_results.csv'
            if results_path.exists():
                df = pd.read_csv(results_path)
                print(df.to_string(index=False, max_rows=10))
                if len(df) > 10:
                    print(f"... and {len(df) - 10} more rows")
            else:
                print("No DWSIM results file found.")
                
        except Exception as e:
            print(f"Error reading results: {e}")
    
    def quick_test(self):
        """Quick test to verify export functionality."""
        print("⚡ Testing DWSIM export functionality...")
        
        # Test data
        test_dwsim_data = [
            {
                'case_name': 'test_case', 
                'simulation_type': 'reactor',
                'components': 'test, components',
                'description': 'Test simulation',
                'success': True, 
                'duration_seconds': 1.0, 
                'conversion': 0.85,
                'selectivity': 0.92,
                'yield': 0.78,
                'temperature': 150.5,
                'pressure': 2.5,
                'status': 'Test Completed',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
        
        # Export test data
        self.export_dwsim_results(test_dwsim_data, "test_dwsim_results.csv")
        
        print("✅ Export test completed!")
        
        # Clean up test files
        (self.results_dir / "test_dwsim_results.csv").unlink(missing_ok=True)
        print("🗑️ Test files cleaned up.")
    
    def clean_results(self):
        """Clean all CSV files in results directory."""
        csv_files = list(self.results_dir.glob('*.csv'))
        for file in csv_files:
            file.unlink()
        print(f"🗑️ Cleaned {len(csv_files)} CSV files from {self.results_dir}")
    
    def get_latest_files(self):
        """Get list of latest CSV files with metadata."""
        csv_files = list(self.results_dir.glob('*.csv'))
        file_info = []
        
        for file in csv_files:
            stat = file.stat()
            file_info.append({
                'filename': file.name,
                'size_bytes': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return sorted(file_info, key=lambda x: x['modified'], reverse=True)
    
    # Legacy methods for backward compatibility
    def export_rag_results(self, rag_data, filename="rag_query_results.csv"):
        """Export RAG query results to CSV (legacy method)."""
        if not rag_data:
            return None
        
        df = pd.DataFrame(rag_data)
        filepath = self.results_dir / filename
        df.to_csv(filepath, index=False)
        return filepath
    
    def export_statistics(self, stats_data, filename="statistics.csv"):
        """Export statistics to CSV (legacy method)."""
        if not stats_data:
            return None
        
        # Convert single dict to list if needed
        if isinstance(stats_data, dict):
            stats_data = [stats_data]
        
        df = pd.DataFrame(stats_data)
        filepath = self.results_dir / filename
        df.to_csv(filepath, index=False)
        return filepath
    
    def view_summary(self):
        """View summary of results (legacy method)."""
        self.view_dwsim_results()
    
    def export_all_results(self, rag_data=None, dwsim_data=None, rag_stats=None, dwsim_stats=None):
        """Export all results - now focuses primarily on DWSIM data."""
        return self.export_simulation_only(dwsim_data) 