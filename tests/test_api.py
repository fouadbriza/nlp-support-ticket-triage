import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture(scope="module")
def client():
    # Context manager triggers the lifespan (loading models)
    with TestClient(app) as test_client:
        yield test_client

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["best_model"] == "logistic_regression"
    assert "/docs" in data["docs_url"]

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "logistic_regression" in data["available_models"]
    assert "xgboost" in data["available_models"]

def test_predict_billing(client):
    payload = {"text": "can you tell me the due date for the water bill"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "billing"
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_damaged_card_refund(client):
    payload = {"text": "tell me how to go about reporting a damaged card"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "refund"
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_empty_text(client):
    payload = {"text": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_payload(client):
    payload = {"wrong_key": "some text"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

@pytest.mark.parametrize("model_name", [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost"
])
def test_experiment_valid_models(client, model_name):
    payload = {
        "text": "please cancel my reservation at red lobster for 6",
        "model": model_name
    }
    response = client.post("/experiment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence" in data
    assert data["model_used"] == model_name
    assert 0.0 <= data["confidence"] <= 1.0

def test_experiment_invalid_model(client):
    payload = {
        "text": "tell me how to go about reporting a damaged card",
        "model": "deep_learning_transformer"
    }
    response = client.post("/experiment", json=payload)
    assert response.status_code == 422
