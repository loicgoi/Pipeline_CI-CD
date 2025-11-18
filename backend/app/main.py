import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv

from model_loader import load_model

load_dotenv()

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
    probabilities: Dict[str, float]

@app.get("/")
def health_check():
    """Health Check Endpoint"""
    return {"status": "ok", "model_loader": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    try:
        if len(request.features) != 4:
            raise HTTPException(status_code=400, detail="Exactement 4 éléments requis")
        
        features = [request.features]

        pred_raw = model.predict(features)[0]
        proba_raw = model.predict_proba(features)[0]

        prediction = int(pred_raw)
        species_names = ["setosa", "versicolor", "virginica"]
        species = species_names[prediction]

        probs_np = proba_raw.flatten() if hasattr(proba_raw, 'flatten') else proba_raw
        probs_list = [float(p) for p in probs_np]
        probabilities = dict(zip(species_names, [round(p, 4) for p in probs_list]))

        LOG.info(f"Prédiction: {species} → {probabilities}")

        return PredictionResponse(
            prediction=prediction,
            species=species,
            probabilities=probabilities
        )

    except Exception as e:
        LOG.error(f"Erreur de Prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Prédiction échouée: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run(app, host=str(os.getenv("API_URL", "127.0.0.1")), port=int(os.getenv("API_PORT", 8100)))