#!/usr/bin/env python3
"""
Test cases for Developer Console functionality.
"""

import pytest
import json
from pathlib import Path
import sys

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pynucleus.api.app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestDevConsole:
    """Test cases for developer console endpoints."""
    
    def test_dev_console_page_loads(self, client):
        """Test that the developer console page loads successfully."""
        response = client.get('/dev')
        assert response.status_code == 200
        assert b'PyNucleus' in response.data
        assert b'Developer Console' in response.data
        assert b'Ask the Model' in response.data
        assert b'System Diagnostics' in response.data
    
    def test_ask_endpoint_json(self, client):
        """Test the /ask endpoint with JSON data."""
        response = client.post('/ask', 
                             json={'question': 'What is distillation?'},
                             content_type='application/json')
        
        # Should return 200 or 500 (depending on system state)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'answer' in data
        else:
            # Error response should be JSON
            data = response.get_json()
            assert 'error' in data
    
    def test_ask_endpoint_form_data(self, client):
        """Test the /ask endpoint with form data (HTMX compatibility)."""
        response = client.post('/ask', 
                             data={'question': 'What is chemical engineering?'},
                             content_type='application/x-www-form-urlencoded')
        
        # Should return 200 or 500 (depending on system state)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'answer' in data
        else:
            # Error response should be JSON
            data = response.get_json()
            assert 'error' in data
    
    def test_ask_endpoint_missing_question(self, client):
        """Test the /ask endpoint with missing question."""
        response = client.post('/ask', 
                             json={},
                             content_type='application/json')
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'Missing' in data['error']
    
    def test_ask_endpoint_empty_question(self, client):
        """Test the /ask endpoint with empty question."""
        response = client.post('/ask', 
                             json={'question': ''},
                             content_type='application/json')
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'empty' in data['error']
    
    def test_system_diagnostic_endpoint(self, client):
        """Test the /system_diagnostic endpoint."""
        response = client.get('/system_diagnostic')
        
        # Should always return 200 (errors handled internally)
        assert response.status_code == 200
        
        data = response.get_json()
        assert data is not None
        
        # Should have either structured diagnostic data or error info
        expected_keys = ['status', 'total_tests', 'passed_tests', 'return_code', 'error']
        assert any(key in data for key in expected_keys)
    
    def test_system_diagnostic_json_structure(self, client):
        """Test that system diagnostic returns proper JSON structure."""
        response = client.get('/system_diagnostic')
        assert response.status_code == 200
        
        data = response.get_json()
        assert isinstance(data, dict)
        
        # If successful validation, should have test results
        if 'total_tests' in data:
            assert isinstance(data['total_tests'], int)
            assert data['total_tests'] >= 0
            
        if 'passed_tests' in data:
            assert isinstance(data['passed_tests'], int)
            assert data['passed_tests'] >= 0
    
    def test_dev_console_html_content(self, client):
        """Test developer console HTML contains expected elements."""
        response = client.get('/dev')
        assert response.status_code == 200
        
        html_content = response.data.decode('utf-8')
        
        # Check for essential HTML elements
        assert 'hx-post="/ask"' in html_content
        assert 'hx-get="/system_diagnostic"' in html_content
        assert 'id="question"' in html_content
        assert 'id="answer"' in html_content
        assert 'id="diag"' in html_content
        
        # Check for CRT styling
        assert 'crt' in html_content
        assert '#33ff00' in html_content  # Green color
        assert 'Courier New' in html_content
    
    def test_dev_console_htmx_integration(self, client):
        """Test that developer console includes HTMX."""
        response = client.get('/dev')
        assert response.status_code == 200
        
        html_content = response.data.decode('utf-8')
        assert 'htmx.org' in html_content
        assert 'hx-' in html_content  # HTMX attributes 