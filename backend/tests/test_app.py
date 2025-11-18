"""
Tests unitaires et d'intégration pour l'API FastAPI de prédiction Iris (fichier app/main.py).

Ce module contient :
- Un test du health check endpoint (/)
- Un test de l'endpoint /predict avec un modèle mocké via monkeypatch
"""

from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import MagicMock

client = TestClient(app)


def test_health_check():
    """
    Vérifie que l'endpoint de health check renvoie un statut 200
    et que le champ 'status' vaut 'ok'.
    Le champ 'model_loaded' est également présent mais sa valeur
    dépend de la disponibilité réelle du modèle au moment du test.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict(monkeypatch):
    """
    Teste l'endpoint POST /predict avec un modèle entièrement mocké.

    Le modèle est remplacé par un MagicMock qui retourne :
    - predict → [0] (classe setosa)
    - predict_proba → [[0.9, 0.1, 0.0]]

    Vérifie :
    - Code HTTP 200
    - Présence et type des champs dans la réponse JSON
    - Valeur correcte de la prédiction (0)
    """
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
    assert isinstance(data["probabilities"], dict)
    assert len(data["probabilities"]) == 3
