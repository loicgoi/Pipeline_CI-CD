import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

from model_loader import load_model

# Configuration logs
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

app = FastAPI(title="Iris Prediction API", version="1.0.0")

# Chargement du modèle
try:
    model = load_model()
    LOG.info("Modèle chargé avec succès")
except Exception as e:
    LOG.error(f"Chargement du modèle échoué: {e}")
    model = None

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    species: str
    probabilities: List[float]

@app.get("/")
def health_check():
    """Health Check Endpoint"""
    return {"status": "ok", "model_loader": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Prédiction des espèces selon les caractéristiques"""
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    try:
        # Validation de l'input
        if len(request.features) != 4:
            raise HTTPException(status_code=400, detail="Exactement 4 fonctionnalités requises")
        
        features_array = np.array([request.features])
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0].tolist()

        species_names = ["setosa", "versicolor", "virginica"]
        species = species_names[prediction]

        LOG.info(f"Prédiction faite: {species} (index {prediction}) avec {probabilities} probabilité")

        return PredictionResponse(
            prediction=int(prediction),
            species=species,
            probabilities=probabilities
        )
    except Exception as e:
        LOG.error(f"Erreur de Prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Prédiction échouée: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", 8000)))