# CI/CD avec GitHub Actions

## Workflow

À chaque push sur `main`, le pipeline :

- Lance les tests
- Déploie la documentation
- Entraîne le modèle
- Construit et pousse les images Docker

## Fichier

`.github/workflows/ci-cd.yml`