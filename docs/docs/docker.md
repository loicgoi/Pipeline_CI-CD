# Dockerisation

## Images

- `iris-backend`: API FastAPI
- `iris-frontend`: Interface Streamlit

## Ports

- Backend: 8100
- Frontend: 8501

## Docker Compose

```yaml
services:
  backend:
    image: <username>/iris-backend
    ports:
      - "8100:8100"
  frontend:
    image: <username>/iris-frontend
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8100
```