"""
Application Streamlit de démonstration pour la prédiction d'espèce d'Iris.

Cette interface web permet à l'utilisateur d'ajuster interactivement les 4 caractéristiques
morphologiques d'une fleur d'Iris (longueur/largeur des sépales et pétales) via des sliders
et des champs numériques synchronisés, puis d'appeler l'API FastAPI de prédiction
( endpoint `/predict` ) pour obtenir l'espèce prédite et les probabilités associées.

Variables d'environnement utilisées :
- ``API_URL`` : adresse du serveur backend (défaut : 127.0.0.1)
- ``API_PORT`` : port du serveur backend (défaut : 8100)
"""

import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.title("Prédiction d'espèce Iris")
st.markdown(
    "Ajustez les caractéristiques morphologiques et cliquez sur **Prédire l'espèce**"
)

# URL du backend
backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8100")


# Fonction slider + input synchronisés
def synced_input(
    label: str,
    min_val: float,
    max_val: float,
    default: float,
    step: float = 0.1,
    key: str = None,
):
    """Crée un slider et un champ numérique synchronisés dans Streamlit.

    Les deux widgets partagent la même valeur via ``st.session_state``.
    Modifier l'un met instantanément à jour l'autre.

    Args:
        label (str): Libellé affiché pour le slider (le number_input est masqué).
        min_val (float): Valeur minimale autorisée.
        max_val (float): Valeur maximale autorisée.
        default (float): Valeur initiale.
        step (float): Pas d'incrémentation (défaut = 0.1).
        key (str | None): Préfixe unique pour les clés dans ``session_state``.

    Returns:
        float: La valeur courante (commune aux deux widgets).
    """
    slider_key = f"{key}_slider"
    input_key = f"{key}_input"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = default
    if input_key not in st.session_state:
        st.session_state[input_key] = default

    def from_slider():
        st.session_state[input_key] = st.session_state[slider_key]

    def from_input():
        try:
            value = float(st.session_state[input_key])
            value = max(min_val, min(max_val, value))
            st.session_state[slider_key] = value
            st.session_state[input_key] = value
        except:
            st.session_state[input_key] = st.session_state[slider_key]

    st.slider(label, min_val, max_val, step=step, key=slider_key, on_change=from_slider)
    st.number_input(
        label,
        min_val,
        max_val,
        step=step,
        key=input_key,
        label_visibility="collapsed",
        on_change=from_input,
    )

    return st.session_state[slider_key]


# Inputs utilisateur
col1, col2 = st.columns(2)
with col1:
    sepal_length = synced_input("Longueur du sépale (cm)", 0.0, 10.0, 5.8, 0.1, "sl")
    sepal_width = synced_input("Largeur du sépale (cm)", 0.0, 10.0, 3.5, 0.1, "sw")
with col2:
    petal_length = synced_input("Longueur du pétale (cm)", 0.0, 10.0, 4.0, 0.1, "pl")
    petal_width = synced_input("Largeur du pétale (cm)", 0.0, 10.0, 1.3, 0.1, "pw")

# Bouton de prédiction
if st.button("Prédire l'espèce", type="primary", use_container_width=True):
    payload = {"features": [sepal_length, sepal_width, petal_length, petal_width]}

    with st.spinner("Interrogation du modèle..."):
        try:
            response = requests.post(f"{backend_url}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            species = result["species"]
            probs = result["probabilities"]

            st.balloons()
            st.success(f"**Espèce prédite : {species.capitalize()}**")

            st.subheader("Probabilités par espèce")
            for specie, proba in probs.items():
                percentage = proba * 100
                st.progress(proba)
                st.write(f"**{specie.capitalize()}** → **{percentage:.2f}%**")

        except requests.exceptions.ConnectionError:
            st.error("🔌 Impossible de contacter le serveur backend.")
        except requests.exceptions.HTTPError as e:
            st.error(f"Erreur API : {e}")
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")
