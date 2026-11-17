import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict():
    data = {"data": [0.0] * 784}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()

def test_predict_wrong_size():
    data = {"data": [0.0] * 100}
    response = client.post("/predict", json=data)
    assert response.status_code == 400

def test_predict_batch():
    data = {"data": [[0.0] * 784, [0.0] * 784]}
    response = client.post("/predict/batch", json=data)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 2

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()