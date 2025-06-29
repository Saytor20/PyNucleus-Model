"""
Mock Data Manager for PyNucleus

Provides unified access to plant templates, simulation data, and technical documents
for both chat and build functions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)

class MockDataManager:
    """Manages access to unified mock data for PyNucleus system."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the MockDataManager.
        
        Args:
            config_path: Path to mock data configuration file
        """
        if config_path is None:
            config_path = Path("configs/mock_data_config.json")
        
        self.config_path = config_path
        self._data = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load mock data from configuration file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Mock data config not found: {self.config_path}")
                self._data = self._get_default_data()
                return
            
            with open(self.config_path, 'r') as f:
                self._data = json.load(f)
            
            logger.info(f"Loaded mock data from {self.config_path}")
            logger.info(f"Available: {len(self._data.get('plant_templates', []))} plant templates, "
                       f"{len(self._data.get('technical_documents', []))} technical documents")
            
        except Exception as e:
            logger.error(f"Failed to load mock data: {e}")
            self._data = self._get_default_data()
    
    def _get_default_data(self) -> Dict[str, Any]:
        """Provide default data structure if config file is not available."""
        return {
            "metadata": {
                "version": "1.0",
                "description": "Default mock data",
                "data_sources": ["plant_templates", "simulation_cases", "technical_documents"]
            },
            "plant_templates": [],
            "technical_documents": [],
            "simulation_parameters": {}
        }
    
    def get_plant_template(self, template_id: int) -> Optional[Dict[str, Any]]:
        """
        Get plant template by ID.
        
        Args:
            template_id: Template ID (1-5)
            
        Returns:
            Plant template dictionary or None if not found
        """
        templates = self._data.get('plant_templates', [])
        for template in templates:
            if template.get('id') == template_id:
                return template
        return None
    
    def get_all_plant_templates(self) -> List[Dict[str, Any]]:
        """
        Get all available plant templates.
        
        Returns:
            List of all plant templates
        """
        return self._data.get('plant_templates', [])
    
    def get_plant_template_names(self) -> List[str]:
        """
        Get list of all plant template names.
        
        Returns:
            List of template names
        """
        templates = self.get_all_plant_templates()
        return [template.get('name', f"Template {template.get('id', 'Unknown')}") 
                for template in templates]
    
    def get_simulation_cases(self, template_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get simulation cases for a specific template or all cases.
        
        Args:
            template_id: Optional template ID to filter cases
            
        Returns:
            List of simulation cases
        """
        if template_id is not None:
            template = self.get_plant_template(template_id)
            if template:
                return template.get('simulation_cases', [])
            return []
        
        # Return all simulation cases from all templates
        all_cases = []
        templates = self.get_all_plant_templates()
        for template in templates:
            cases = template.get('simulation_cases', [])
            for case in cases:
                case['template_id'] = template.get('id')
                case['template_name'] = template.get('name')
            all_cases.extend(cases)
        
        return all_cases
    
    def get_technical_documents(self, category: Optional[str] = None, 
                               tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get technical documents with optional filtering.
        
        Args:
            category: Optional category filter
            tags: Optional list of tags to filter by
            
        Returns:
            List of technical documents
        """
        documents = self._data.get('technical_documents', [])
        
        if category:
            documents = [doc for doc in documents if doc.get('category') == category]
        
        if tags:
            filtered_docs = []
            for doc in documents:
                doc_tags = doc.get('tags', [])
                if any(tag in doc_tags for tag in tags):
                    filtered_docs.append(doc)
            documents = filtered_docs
        
        return documents
    
    def search_technical_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Search technical documents by content.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching documents
        """
        documents = self._data.get('technical_documents', [])
        query_lower = query.lower()
        
        matching_docs = []
        for doc in documents:
            title = doc.get('title', '').lower()
            content = doc.get('content', '').lower()
            tags = [tag.lower() for tag in doc.get('tags', [])]
            
            if (query_lower in title or 
                query_lower in content or 
                any(query_lower in tag for tag in tags)):
                matching_docs.append(doc)
        
        return matching_docs
    
    def get_simulation_parameters(self) -> Dict[str, Any]:
        """
        Get default simulation parameters.
        
        Returns:
            Dictionary of simulation parameters
        """
        return self._data.get('simulation_parameters', {})
    
    def get_available_categories(self) -> List[str]:
        """
        Get list of available document categories.
        
        Returns:
            List of unique categories
        """
        documents = self._data.get('technical_documents', [])
        categories = set(doc.get('category') for doc in documents if doc.get('category'))
        return list(categories)
    
    def get_available_tags(self) -> List[str]:
        """
        Get list of all available tags.
        
        Returns:
            List of unique tags
        """
        documents = self._data.get('technical_documents', [])
        all_tags = set()
        for doc in documents:
            tags = doc.get('tags', [])
            all_tags.update(tags)
        return list(all_tags)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of available data.
        
        Returns:
            Dictionary with data summary
        """
        templates = self.get_all_plant_templates()
        documents = self._data.get('technical_documents', [])
        simulation_cases = self.get_simulation_cases()
        
        return {
            "total_plant_templates": len(templates),
            "total_technical_documents": len(documents),
            "total_simulation_cases": len(simulation_cases),
            "available_categories": self.get_available_categories(),
            "available_tags": self.get_available_tags(),
            "template_names": self.get_plant_template_names(),
            "metadata": self._data.get('metadata', {}),
            "last_updated": datetime.now().isoformat()
        }
    
    def export_plant_data_for_chat(self) -> List[Dict[str, Any]]:
        """
        Export plant data in format suitable for chat/RAG system.
        
        Returns:
            List of documents formatted for RAG
        """
        chat_documents = []
        
        # Add plant templates as documents
        templates = self.get_all_plant_templates()
        for template in templates:
            doc = {
                "id": f"plant_template_{template['id']}",
                "title": template['name'],
                "content": f"{template['description']} Technology: {template['technology']}. "
                          f"Default parameters: Feedstock: {template['default_parameters']['feedstock']}, "
                          f"Production capacity: {template['default_parameters']['production_capacity']} tons/year, "
                          f"Capital cost: ${template['default_parameters']['capital_cost']:,.0f}, "
                          f"Operating cost: ${template['default_parameters']['operating_cost']:,.0f}/year, "
                          f"Product price: ${template['default_parameters']['product_price']}/ton. "
                          f"Valid feedstock options: {', '.join(template['feedstock_options'])}. "
                          f"Location factors: {template['location_factors']}",
                "category": "plant_template",
                "tags": ["plant_design", "chemical_process", template['technology'].lower().replace(' ', '_')],
                "metadata": {
                    "template_id": template['id'],
                    "technology": template['technology'],
                    "default_capacity": template['default_parameters']['production_capacity']
                }
            }
            chat_documents.append(doc)
        
        # Add simulation cases as documents
        simulation_cases = self.get_simulation_cases()
        for case in simulation_cases:
            doc = {
                "id": f"simulation_case_{case['case_name']}",
                "title": f"Simulation: {case['case_name']}",
                "content": f"Simulation case for {case.get('template_name', 'Unknown plant')}. "
                          f"Process type: {case['process_type']}, Temperature: {case['temperature']}°C, "
                          f"Pressure: {case['pressure']} bar, Feed rate: {case['feed_rate']} tons/hour, "
                          f"Catalyst: {case['catalyst_type']}, Conversion efficiency: {case['conversion_efficiency']:.2%}",
                "category": "simulation_case",
                "tags": ["simulation", case['process_type'], "process_optimization"],
                "metadata": {
                    "template_id": case.get('template_id'),
                    "process_type": case['process_type'],
                    "temperature": case['temperature'],
                    "pressure": case['pressure']
                }
            }
            chat_documents.append(doc)
        
        # Add technical documents
        tech_docs = self._data.get('technical_documents', [])
        for doc in tech_docs:
            chat_doc = {
                "id": doc['id'],
                "title": doc['title'],
                "content": doc['content'],
                "category": doc['category'],
                "tags": doc['tags'],
                "metadata": {
                    "category": doc['category'],
                    "tags": doc['tags']
                }
            }
            chat_documents.append(chat_doc)
        
        return chat_documents
    
    def reload_data(self) -> None:
        """Reload data from configuration file."""
        self._load_data()
    
    def is_data_loaded(self) -> bool:
        """
        Check if data is successfully loaded.
        
        Returns:
            True if data is loaded, False otherwise
        """
        return self._data is not None and len(self._data.get('plant_templates', [])) > 0


# Global instance for easy access
_mock_data_manager = None

def get_mock_data_manager() -> MockDataManager:
    """
    Get global mock data manager instance.
    
    Returns:
        MockDataManager instance
    """
    global _mock_data_manager
    if _mock_data_manager is None:
        _mock_data_manager = MockDataManager()
    return _mock_data_manager 