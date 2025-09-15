import json
import pytest
from main import app

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_healthz(client):
    rv = client.get('/healthz')
    assert rv.status_code == 200
    data = rv.get_json()
    assert 'status' in data

def test_chat_no_model(client):
    rv = client.post('/chat', json={'message':'hello'})
    assert rv.status_code in (200,503)
