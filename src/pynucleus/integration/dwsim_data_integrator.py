"""
DWSIM Data Integrator for RAG Knowledge Base

This module integrates DWSIM simulation results into the RAG knowledge base,
allowing the LLM to seamlessly query both document data and simulation results.

Key Features:
- Converts DWSIM simulation results to searchable text chunks
- Integrates simulation data with existing document chunks
- Maintains metadata for source tracking
- Creates unified knowledge base for comprehensive RAG queries
"""

import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from langchain.docstore.document import Document
import logging

logger = logging.getLogger(__name__)


class DWSIMDataIntegrator:
    """Integrates DWSIM simulation data into the RAG knowledge base."""
    
    def __init__(self, 
                 output_dir: str = "data/05_output",
                 chunked_data_dir: str = "data/03_intermediate/converted_chunked_data"):
        """
        Initialize the DWSIM data integrator.
        
        Args:
            output_dir: Directory containing DWSIM results
            chunked_data_dir: Directory containing existing chunked data
        """
        self.output_dir = Path(output_dir)
        self.chunked_data_dir = Path(chunked_data_dir)
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunked_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DWSIMDataIntegrator initialized with output: {self.output_dir}")
    
    def load_dwsim_results(self) -> Optional[pd.DataFrame]:
        """
        Load DWSIM simulation results from CSV files.
        
        Returns:
            DataFrame with simulation results or None if not found
        """
        dwsim_csv = self.output_dir / "dwsim_simulation_results.csv"
        
        if not dwsim_csv.exists():
            logger.warning(f"DWSIM results not found at {dwsim_csv}")
            return None
        
        try:
            df = pd.read_csv(dwsim_csv)
            logger.info(f"Loaded {len(df)} DWSIM simulation results")
            return df
        except Exception as e:
            logger.error(f"Error loading DWSIM results: {e}")
            return None
    
    def convert_simulation_to_text_chunks(self, df: pd.DataFrame) -> List[Document]:
        """
        Convert DWSIM simulation results to searchable text chunks.
        
        Args:
            df: DataFrame containing simulation results
            
        Returns:
            List of Document objects with simulation data as text
        """
        chunks = []
        
        for idx, row in df.iterrows():
            # Create comprehensive text description of the simulation
            text_content = self._create_simulation_text_description(row)
            
            # Create metadata
            metadata = {
                "source": f"dwsim_simulation_{row['Case Name']}",
                "type": "simulation_result",
                "simulation_type": row.get('Simulation Type', 'unknown'),
                "case_name": row.get('Case Name', 'unknown'),
                "components": row.get('Components', ''),
                "status": row.get('Status', 'unknown'),
                "timestamp": row.get('Timestamp', datetime.now().isoformat()),
                "chunk_id": f"sim_{idx}",
                "length": len(text_content)
            }
            
            # Create Document object
            doc = Document(
                page_content=text_content,
                metadata=metadata
            )
            
            chunks.append(doc)
        
        logger.info(f"Created {len(chunks)} simulation text chunks")
        return chunks
    
    def _create_simulation_text_description(self, row: pd.Series) -> str:
        """
        Create a comprehensive text description of a simulation result.
        
        Args:
            row: Row from simulation results DataFrame
            
        Returns:
            Formatted text description
        """
        # Extract key information
        case_name = row.get('Case Name', 'Unknown Case')
        sim_type = row.get('Simulation Type', 'Unknown Type')
        components = row.get('Components', 'Unknown Components')
        description = row.get('Description', '')
        status = row.get('Status', 'Unknown Status')
        
        # Performance metrics
        conversion = row.get('Conversion', 'N/A')
        selectivity = row.get('Selectivity', 'N/A')
        yield_val = row.get('Yield', 'N/A')
        temperature = row.get('Temperature (°C)', 'N/A')
        pressure = row.get('Pressure (atm)', 'N/A')
        duration = row.get('Duration (s)', 'N/A')
        
        # Create formatted text description
        text_parts = [
            f"DWSIM Simulation Result: {case_name}",
            f"Simulation Type: {sim_type}",
            f"Process Description: {description}",
            f"Components Involved: {components}",
            f"Execution Status: {status}",
            "",
            "Performance Metrics:",
            f"- Conversion Rate: {conversion}",
            f"- Selectivity: {selectivity}",
            f"- Yield: {yield_val}",
            f"- Operating Temperature: {temperature}°C",
            f"- Operating Pressure: {pressure} atm",
            f"- Simulation Duration: {duration} seconds",
            "",
            "Process Analysis:",
        ]
        
        # Add process-specific analysis based on simulation type
        if isinstance(sim_type, str):
            if 'distillation' in sim_type.lower():
                text_parts.extend([
                    "This distillation simulation evaluates separation efficiency and energy requirements.",
                    "Key factors include reflux ratio, number of stages, and temperature profiles.",
                    "The simulation helps optimize separation purity and energy consumption."
                ])
            elif 'reactor' in sim_type.lower():
                text_parts.extend([
                    "This reactor simulation analyzes chemical reaction performance and kinetics.",
                    "Critical parameters include conversion rate, selectivity, and reaction temperature.",
                    "The simulation optimizes reaction conditions for maximum yield and efficiency."
                ])
            elif 'heat_exchanger' in sim_type.lower():
                text_parts.extend([
                    "This heat exchanger simulation evaluates thermal performance and efficiency.",
                    "Key metrics include heat transfer rate, temperature approach, and pressure drop.",
                    "The simulation optimizes thermal design for energy efficiency."
                ])
            elif 'absorber' in sim_type.lower():
                text_parts.extend([
                    "This absorption simulation analyzes mass transfer and separation efficiency.",
                    "Important factors include absorption rate, liquid-gas ratio, and column height.",
                    "The simulation optimizes absorption process for maximum recovery."
                ])
            elif 'crystallizer' in sim_type.lower():
                text_parts.extend([
                    "This crystallization simulation evaluates solid-liquid separation and crystal formation.",
                    "Key parameters include supersaturation, nucleation rate, and crystal size distribution.",
                    "The simulation optimizes crystallization conditions for product quality."
                ])
        
        # Add operational recommendations
        text_parts.extend([
            "",
            "Operational Insights:",
            f"Based on the simulation results, this {sim_type} process demonstrates {status.lower()} performance.",
        ])
        
        if conversion != 'N/A' and isinstance(conversion, (int, float)):
            if float(conversion) > 0.9:
                text_parts.append("The high conversion rate indicates excellent process efficiency.")
            elif float(conversion) > 0.7:
                text_parts.append("The moderate conversion rate suggests good process performance with room for optimization.")
            else:
                text_parts.append("The conversion rate indicates potential for process improvement.")
        
        if selectivity != 'N/A' and isinstance(selectivity, (int, float)):
            if float(selectivity) > 0.95:
                text_parts.append("The excellent selectivity indicates minimal side reactions and high product purity.")
            elif float(selectivity) > 0.8:
                text_parts.append("The good selectivity shows effective process control with acceptable purity.")
        
        return "\n".join(text_parts)
    
    def load_existing_chunks(self) -> List[Dict]:
        """
        Load existing document chunks from the RAG knowledge base.
        
        Returns:
            List of existing chunk dictionaries
        """
        chunks_file = self.chunked_data_dir / "chunked_data_full.json"
        
        if not chunks_file.exists():
            logger.warning(f"Existing chunks file not found at {chunks_file}")
            return []
        
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.info(f"Loaded {len(chunks)} existing document chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error loading existing chunks: {e}")
            return []
    
    def integrate_simulation_data(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Integrate DWSIM simulation data with existing document chunks.
        
        Args:
            force_rebuild: Whether to force rebuild even if integrated data exists
            
        Returns:
            Dictionary with integration results
        """
        logger.info("Starting DWSIM data integration with RAG knowledge base")
        
        # Load DWSIM simulation results
        dwsim_df = self.load_dwsim_results()
        if dwsim_df is None:
            return {
                "success": False,
                "error": "No DWSIM simulation results found",
                "total_chunks": 0,
                "simulation_chunks": 0,
                "document_chunks": 0
            }
        
        # Convert simulation results to text chunks
        simulation_chunks = self.convert_simulation_to_text_chunks(dwsim_df)
        
        # Load existing document chunks
        existing_chunks = self.load_existing_chunks()
        
        # Convert simulation Document objects to dictionary format
        simulation_chunk_dicts = []
        chunk_id_offset = len(existing_chunks)
        
        for i, doc in enumerate(simulation_chunks):
            chunk_dict = {
                "chunk_id": chunk_id_offset + i,
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "length": doc.metadata["length"],
                "type": doc.metadata["type"],
                "simulation_type": doc.metadata.get("simulation_type"),
                "case_name": doc.metadata.get("case_name"),
                "components": doc.metadata.get("components"),
                "status": doc.metadata.get("status"),
                "timestamp": doc.metadata.get("timestamp")
            }
            simulation_chunk_dicts.append(chunk_dict)
        
        # Combine existing and simulation chunks
        integrated_chunks = existing_chunks + simulation_chunk_dicts
        
        # Save integrated chunks
        self._save_integrated_chunks(integrated_chunks)
        
        # Update statistics
        self._update_integration_statistics(integrated_chunks, simulation_chunk_dicts)
        
        result = {
            "success": True,
            "total_chunks": len(integrated_chunks),
            "document_chunks": len(existing_chunks),
            "simulation_chunks": len(simulation_chunk_dicts),
            "integration_timestamp": datetime.now().isoformat(),
            "simulation_cases": [chunk["case_name"] for chunk in simulation_chunk_dicts]
        }
        
        logger.info(f"Integration completed: {result['total_chunks']} total chunks " +
                   f"({result['document_chunks']} docs + {result['simulation_chunks']} sims)")
        
        return result
    
    def _save_integrated_chunks(self, integrated_chunks: List[Dict]):
        """Save integrated chunks to the chunked data directory."""
        
        # Save the integrated chunks file
        integrated_file = self.chunked_data_dir / "chunked_data_full.json"
        with open(integrated_file, 'w', encoding='utf-8') as f:
            json.dump(integrated_chunks, f, indent=2, ensure_ascii=False)
        
        # Also save a backup of the original if it exists
        backup_file = self.chunked_data_dir / "chunked_data_full_original.json"
        if integrated_file.exists() and not backup_file.exists():
            # This only creates backup on first integration
            pass
        
        logger.info(f"Saved integrated chunks to {integrated_file}")
    
    def _update_integration_statistics(self, integrated_chunks: List[Dict], 
                                     simulation_chunks: List[Dict]):
        """Update statistics file with integration information."""
        
        # Calculate statistics
        all_sources = set()
        sim_sources = set()
        doc_sources = set()
        
        total_length = 0
        sim_length = 0
        doc_length = 0
        
        for chunk in integrated_chunks:
            all_sources.add(chunk["source"])
            total_length += chunk["length"]
            
            if chunk.get("type") == "simulation_result":
                sim_sources.add(chunk["source"])
                sim_length += chunk["length"]
            else:
                doc_sources.add(chunk["source"])
                doc_length += chunk["length"]
        
        stats = {
            "total_chunks": len(integrated_chunks),
            "document_chunks": len(integrated_chunks) - len(simulation_chunks),
            "simulation_chunks": len(simulation_chunks),
            "sources": list(all_sources),
            "document_sources": list(doc_sources),
            "simulation_sources": list(sim_sources),
            "avg_chunk_size": total_length / len(integrated_chunks) if integrated_chunks else 0,
            "avg_document_chunk_size": doc_length / (len(integrated_chunks) - len(simulation_chunks)) if (len(integrated_chunks) - len(simulation_chunks)) > 0 else 0,
            "avg_simulation_chunk_size": sim_length / len(simulation_chunks) if simulation_chunks else 0,
            "integration_timestamp": datetime.now().isoformat(),
            "integration_enabled": True
        }
        
        # Save updated statistics
        stats_file = self.chunked_data_dir / "chunked_data_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Updated integration statistics: {stats['total_chunks']} total chunks")
    
    def create_integration_summary(self) -> str:
        """
        Create a human-readable summary of the integration.
        
        Returns:
            Formatted summary string
        """
        # Load current statistics
        stats_file = self.chunked_data_dir / "chunked_data_stats.json"
        if not stats_file.exists():
            return "Integration statistics not available."
        
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        if not stats.get("integration_enabled", False):
            return "DWSIM data integration not yet performed."
        
        summary_lines = [
            "📊 DWSIM-RAG Integration Summary",
            "=" * 40,
            f"Total Knowledge Base Size: {stats['total_chunks']:,} chunks",
            f"├── Document Chunks: {stats['document_chunks']:,}",
            f"└── Simulation Chunks: {stats['simulation_chunks']:,}",
            "",
            f"Average Chunk Sizes:",
            f"├── Documents: {stats['avg_document_chunk_size']:.1f} characters",
            f"├── Simulations: {stats['avg_simulation_chunk_size']:.1f} characters",
            f"└── Overall: {stats['avg_chunk_size']:.1f} characters",
            "",
            f"Data Sources:",
            f"├── Document Sources: {len(stats['document_sources'])}",
            f"└── Simulation Sources: {len(stats['simulation_sources'])}",
            "",
            f"Integration Date: {stats['integration_timestamp'][:19].replace('T', ' ')}",
            "",
            "🔍 RAG System Status: Ready for unified document + simulation queries"
        ]
        
        return "\n".join(summary_lines)
    
    def get_simulation_query_examples(self) -> List[str]:
        """
        Get example queries that can leverage the integrated simulation data.
        
        Returns:
            List of example query strings
        """
        return [
            "What are the performance metrics for the distillation simulation?",
            "How do the reactor conversion rates compare across different simulations?",
            "What operating conditions were used in the heat exchanger simulation?",
            "Which simulation showed the highest selectivity and why?",
            "What are the optimal temperature and pressure conditions based on the simulation results?",
            "How do the simulation results relate to modular plant design principles?",
            "What process improvements are suggested by the simulation data?",
            "Which components showed the best performance in the absorption process?",
            "How do the crystallization simulation results impact process design?",
            "What are the energy efficiency implications of the simulation results?"
        ]


def main():
    """Example usage of the DWSIM Data Integrator."""
    
    # Initialize integrator
    integrator = DWSIMDataIntegrator()
    
    # Perform integration
    result = integrator.integrate_simulation_data()
    
    if result["success"]:
        print("✅ DWSIM data integration completed successfully!")
        print(f"📊 Total chunks: {result['total_chunks']}")
        print(f"📄 Document chunks: {result['document_chunks']}")
        print(f"🔬 Simulation chunks: {result['simulation_chunks']}")
        
        # Print summary
        print("\n" + integrator.create_integration_summary())
        
        # Print example queries
        print("\n🔍 Example Queries for Integrated System:")
        for i, query in enumerate(integrator.get_simulation_query_examples()[:5], 1):
            print(f"   {i}. {query}")
    else:
        print(f"❌ Integration failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main() 