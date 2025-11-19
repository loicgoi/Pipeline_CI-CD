# Pipeline CI/CD avec FastAPI, Streamlit, Docker et Azure

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121.2-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-red.svg)
![Docker](https://img.shields.io/badge/Docker-4.49-cyan.svg)
![Azure](https://img.shields.io/badge/Azure-Cloud-0078D4.svg)

## Table des matières

- [Pipeline CI/CD avec FastAPI, Streamlit, Docker et Azure](#pipeline-cicd-avec-fastapi-streamlit-docker-et-azure)
  - [Table des matières](#table-des-matières)
  - [Description](#description)
    - [Fonctionnalités](#fonctionnalités)
  - [Technologies utilisées](#technologies-utilisées)
  - [Architecture du projet](#architecture-du-projet)
  - [Installation et utilisation](#installation-et-utilisation)
    - [Prérequis](#prérequis)
    - [Lancer avec Docker Compose (Recommandé)](#lancer-avec-docker-compose-recommandé)
    - [Lancer en mode développement](#lancer-en-mode-développement)
  - [Tests](#tests)
  - [Dockerisation](#dockerisation)
    - [Lancer en local](#lancer-en-local)
  - [CI/CD](#cicd)
  - [Déploiement](#déploiement)
  - [Documentation](#documentation)
  - [Bonnes pratiques \& conventions](#bonnes-pratiques--conventions)
  - [Sécurité](#sécurité)
  - [Observabilité \& Monitoring](#observabilité--monitoring)
  - [Performances](#performances)
  - [Roadmap](#roadmap)
  - [Contributeurs](#contributeurs)
  - [Licence](#licence)

---

## Description

Ce projet propose un pipeline complet de **Machine Learning**, **API**, **interface utilisateur**, **suivi d’expériences**, **conteneurisation**, **tests automatisés** et **déploiement cloud**.  
Il constitue un exemple pédagogique complet pour apprendre à construire, versionner, tester et déployer une application ML moderne.

### Fonctionnalités

- Entraînement d’un modèle ML (`scikit-learn`)
- API REST avec FastAPI (`/predict`)
- Interface web via Streamlit
- Tracking des expérimentations avec MLflow
- Build & orchestration avec Docker Compose
- Pipeline CI/CD complet via GitHub Actions
- Déploiement cloud sur Azure App Service
- Documentation automatisée avec MkDocs
- Tests automatisés avec pytest

---

## Technologies utilisées

| Catégorie | Technologies |
|-----------|--------------|
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Données | Dataset Iris |
| Machine Learning | scikit-learn, MLflow, Joblib |
| Conteneurisation | Docker, Docker Compose |
| CI/CD | GitHub Actions, Azure App Service |
| Documentation | MkDocs, Material |
| Tests | pytest, HTTPX |

---

## Architecture du projet

```
Pipeline_CI-CD
├──backend
│   ├──app
│   │   ├──__init__.py
│   │   ├──main.py
│   │   └──model_loader.py
│   ├──ml
│   │   └──train.py
│   ├──model
│   │   └──model.pkl
│   ├──tests
│   │   ├──__init__.py
│   │   └──test_app.py
│   ├──Dockerfile
│   └──requirements.txt
├──docs
│   ├──docs
│   │   ├──api.md
│   │   ├──azure.md
│   │   ├──cicd.md
│   │   ├──docker.md
│   │   ├──frontend.md
│   │   └──index.md
│   └──mkdocs.yml
├──frontend
│   ├──app.py
│   ├──Dockerfile
│   └──requirements.txt
├──docker-compose.yml
├──README.md
└──.gitignore
``` 

---

## Installation et utilisation

### Prérequis
- Python 3.12+
- Docker & Docker Compose
- Compte Azure (pour le déploiement)

---

### Lancer avec Docker Compose (Recommandé)

```bash

git clone https://github.com/loicgoi/Pipeline_CI-CD.git  

cd Pipeline_CI-CD  

docker-compose up --build

# Mode détaché
docker-compose up -d --build

```

Services locaux :

- Streamlit : http://localhost:8501  
- FastAPI : http://localhost:8100/docs  
- MLflow : http://localhost:5000  

---

### Lancer en mode développement

```bash
# Backend
cd backend  

pip install -r requirements.txt  

uvicorn app.main:app --reload --port 8100

# Frontend
cd frontend  

pip install -r requirements.txt  

streamlit run app.py

```

---

## Tests

- Tests unitaires avec pytest  
- Tests d’intégration API  
- Test du chargement du modèle  
- Test de cohérence des prédictions  

---

## Dockerisation

```bash
# Backend
docker build -t iris-backend ./backend  

# Frontend
docker build -t iris-frontend ./frontend

```

### Lancer en local

```bash

docker-compose up --build

```

- Frontend : http://127.0.0.1:8501  
- Backend : http://127.0.0.1:8100/docs 


Fichiers associés :

- `backend/Dockerfile`
- `frontend/Dockerfile`
- `docker-compose.yml`

---

## CI/CD

Résumé rapide :
```yaml
name: CI/CD Pipeline
on: [push, pull_request]
jobs:
  test:
    - Tests automatiques avec pytest
  build:
    - Build des images Docker
    - Push vers Azure Container Registry
  deploy:
    - Déploiement sur Azure App Service
  docs:
    - Déploiement documentation sur GitHub Pages
```

---

## Déploiement

| Service | URL | Statut |
|--------|-----|--------|
| Frontend | [Lien Azure] | 🟢 Live |
| Backend | [Lien Azure] | 🟢 Live |

---

## Documentation

Documentation en ligne : https://loicgoi.github.io/Pipeline_CI-CD/

---

## Bonnes pratiques & conventions

- Respect du typage Python (type hints)
- Arborescence claire : `app/`, `models/`, `services/`
- Formatage & Linting : Ruff
- Validation stricte des données avec Pydantic

---

## Sécurité

- Variables d'environnement non commitables (`.env`)
- Secrets stockés dans GitHub Secrets ou Azure Key Vault
- Validation des schémas via Pydantic
- Images Docker basées sur des versions slim
- Désactivation du mode debug en production
- HTTPS activé côté Azure

---

## Observabilité & Monitoring

- Logs structurés JSON
- Statistiques API via FastAPI + middleware
- Journaux applicatifs intégrés à Azure App Service

---

## Performances

- Modèle chargé en mémoire (pas de rechargement à chaque requête)
- API asynchrone via FastAPI
- Caching possible des prédictions ou du modèle
- Docker multi-stage build pour images plus légères

---

## Roadmap
 
- [ ] Utilisation de Azure ML Tracking Server  

---

## Contributeurs

- **Loïc** — Développeur IA

---

## Licence

Ce projet n'est pas sous licence open-source.
Il a été développé dans le cadre d’un projet scolaire et est destiné à un usage éducatif uniquement.
Toute réutilisation ou diffusion du code nécessite l’accord préalable de l’auteur.

---