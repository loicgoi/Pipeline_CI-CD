import os
from pathlib import Path
import pickle
import logging

LOG = logging.getLogger(__name__)

def load_model():
    """Charge le modèle entrainé depuis la var env MODEL_DIR"""
    model_dir = Path(os.getenv("MODEL_DIR", "../model/model.pkl"))
    model_path = model_dir / "model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Fichier du modèle non trouvé dans {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        LOG.info(f"Modèle chargé avec succès depuis {model_path}")
        return model
    except Exception as e:
        LOG.error(f"Erreur de chargement du modèle: {e}")
        raise