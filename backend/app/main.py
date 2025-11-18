import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv

from app.model_loader import load_model

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

# Modèles Pydantic
class PredictionRequest(BaseModel):
    """Requête pour effectuer une prédiction.

    Attributes:
        features (List[float]): Les 4 mesures de la fleur dans l'ordre suivant :
            1. sepal length (cm)
            2. sepal width (cm)
            3. petal length (cm)
            4. petal width (cm)
    """
    features: List[float]

class PredictionResponse(BaseModel):
    """Réponse renvoyée par l'endpoint de prédiction.

    Attributes:
        prediction (int): Classe prédite (0 = setosa, 1 = versicolor, 2 = virginica).
        species (str): Nom lisible de l'espèce prédite.
        probabilities (Dict[str, float]): Probabilités pour chaque classe (arrondies à 4 décimales).
    """
    prediction: int
    species: str
    probabilities: Dict[str, float]

@app.get("/")
def health_check():
    """Retourne le statut de l'API et indique si le modèle est chargé."""
    return {"status": "ok", "model_loader": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Effectue la prédiction de l'espèce d'iris à partir des caractéristiques fournies.

    Args:
        request (PredictionRequest): Les 4 mesures de la fleur.

    Returns:
        PredictionResponse: Classe prédite, nom de l'espèce et probabilités.

    Raises:
        HTTPException 400: Si le nombre de caractéristiques n'est pas exactement 4.
        HTTPException 500: Si le modèle n'est pas chargé ou en cas d'erreur lors de la prédiction.
    """
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    try:
        if len(request.features) != 4:
            raise HTTPException(status_code=400, detail="Exactement 4 caractéristiques requises")
        
        # Préparation des données pour le modèle (batch de taille 1)
        features = [request.features]

        # Prédiction
        pred_raw = model.predict(features)[0]
        proba_raw = model.predict_proba(features)[0]

        prediction = int(pred_raw)
        species_names = ["setosa", "versicolor", "virginica"]
        species = species_names[prediction]

        # Normalisation et arrondi des probabilités
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

# Lancement du serveur  
if __name__ == "__main__":
    host = str(os.getenv("API_HOST", "127.0.0.1"))
    port = int(os.getenv("API_PORT", 8100))
    LOG.info(f"Démarrage du serveur sur http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)