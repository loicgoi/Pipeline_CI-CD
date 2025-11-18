from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import MagicMock

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict(monkeypatch):
    # Mock le modèle
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.9, 0.1, 0.0]]

    # Remplace app.main.model par le mock
    monkeypatch.setattr("app.main.model", mock_model)

    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert data["prediction"] == 0

    assert "species" in data

    assert "probabilities" in data
    assert isinstance(data["probabilities"], list)
    assert len(data["probabilities"]) == 3
