"""
Tests for PyNucleus Flask API health endpoint.

This module tests the basic functionality of the health check endpoint.
"""

import pytest
import json
from pynucleus.api.app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test the health endpoint returns expected response."""
    response = client.get('/health')
    
    # Check status code
    assert response.status_code == 200
    
    # Check content type
    assert response.content_type == 'application/json'
    
    # Parse JSON response
    data = json.loads(response.data)
    
    # Check required fields
    assert 'status' in data
    assert 'version' in data
    assert 'service' in data
    
    # Check values
    assert data['status'] == 'healthy'
    assert data['service'] == 'PyNucleus API'
    assert isinstance(data['version'], str)
    assert len(data['version']) > 0


def test_health_endpoint_method_not_allowed(client):
    """Test that POST to health endpoint returns 405."""
    response = client.post('/health')
    assert response.status_code == 405
    
    data = json.loads(response.data)
    assert 'error' in data


def test_nonexistent_endpoint(client):
    """Test that nonexistent endpoints return 404."""
    response = client.get('/nonexistent')
    assert response.status_code == 404
    
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'Endpoint not found' 