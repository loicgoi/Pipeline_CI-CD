"""
Module de chargement du modèle de classification Iris.

Ce module contient une unique fonction ``load_model()`` qui charge le modèle
pré-entraîné au format pickle depuis l'emplacement indiqué par la variable
d'environnement ``MODEL_DIR`` (chemin par défaut : ../model/model.pkl).
"""

import os
from pathlib import Path
import pickle
import logging

LOG = logging.getLogger(__name__)

def load_model():
    """Charge le modèle entrainé depuis la var env MODEL_DIR.

    Le chemin du modèle est déterminé de la manière suivante :
    - Si la variable d'environnement ``MODEL_DIR`` est définie → ``<MODEL_DIR>/model.pkl``
    - Sinon → ``../model/model.pkl`` (chemin relatif par défaut)

    Returns:
        L'objet modèle désérialisé (généralement un classifieur scikit-learn).

    Raises:
        FileNotFoundError: Si le fichier ``model.pkl`` n'existe pas au chemin indiqué.
        Exception: Toute autre erreur lors du chargement (fichier corrompu, incompatibilité pickle, etc.).
    """
    model_dir = Path(os.getenv("MODEL_DIR", "model"))
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