# API FastAPI

## Endpoints

- `GET /` → Statut de l'API
- `POST /predict` → Prédiction de l'espèce d'Iris

## Exemple de requête

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

## Exemple de réponse

```json

{
  "prediction": 0,
  "species": "setosa",
  "probabilities": {
    "setosa": 0.99,
    "versicolor": 0.01,
    "virginica": 0.0
  }
}
```