import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.title("Prédiction d'espèce Iris")
st.markdown("Ajustez les caractéristiques morphologiques et cliquez sur **Prédire l'espèce**")

# URL du backend
api_url = os.getenv("API_URL", "127.0.0.1")
api_port = os.getenv("API_PORT", "8100")
backend_url = f"http://{api_url}:{api_port}"

# Fonction slider + input synchronisés
def synced_input(label: str, min_val: float, max_val: float, default: float, step: float = 0.1, key: str = None):
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
    st.number_input(label, min_val, max_val, step=step, key=input_key,
                    label_visibility="collapsed", on_change=from_input)

    return st.session_state[slider_key]

# Inputs utilisateur
col1, col2 = st.columns(2)
with col1:
    sepal_length = synced_input("Longueur du sépale (cm)", 0.0, 10.0, 5.8, 0.1, "sl")
    sepal_width  = synced_input("Largeur du sépale (cm)",  0.0, 10.0, 3.5, 0.1, "sw")
with col2:
    petal_length = synced_input("Longueur du pétale (cm)", 0.0, 10.0, 4.0, 0.1, "pl")
    petal_width  = synced_input("Largeur du pétale (cm)",  0.0, 10.0, 1.3, 0.1, "pw")

# Bouton de prédiction
if st.button("Prédire l'espèce", type="primary", use_container_width=True):
    payload = {
        "features": [sepal_length, sepal_width, petal_length, petal_width]
    }

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