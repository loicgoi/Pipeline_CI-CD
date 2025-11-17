import os
from pathlib import Path
import logging
import pickle
import pandas as pd

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_model_dir():
    # Fallback : env MODEL_DIR else ./model
    return Path(os.getenv("MODEL_DIR", "./model"))

def main():
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        LOG.info(f"URI de suivi MLflow défini sur {tracking_uri}")
    else:
        LOG.info("MLFLOW_TRACKING_URI non défini — utilisation du stockage de fichiers local par défaut")

    mlflow.set_experiment("iris-demo")

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42 # type: ignore
    )

    X_test_df = pd.DataFrame(X_test, columns=iris.feature_names) # type: ignore

    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, oob_score=True, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        LOG.info(f"Test accuracy: {acc:.4f}")

        mlflow.log_metric("test_accuracy", float(acc))
        input_example = X_test_df.iloc[:1]
        mlflow.sklearn.log_model( # type: ignore
            sk_model=clf,
            name="model",
            input_example=input_example
        )

        out_path = model_dir / "model.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(clf, f)
        LOG.info(f"Modèle sauvegardé dans {out_path}")

if __name__ == "__main__":
    main() 