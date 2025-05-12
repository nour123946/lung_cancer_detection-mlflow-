# Variables
PYTHON = python
PIP = pip
APP = app.py
API_URL = http://127.0.0.1:8000/predict  # URL de l'endpoint FastAPI
DATA = '{"data": [5.1, 3.5, 1.4, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}'

# Cibles principales
install:
	$(PIP) install -r requirements.txt

run:
	uvicorn app:app --reload

predict:
	curl -X POST $(API_URL) \
	-H "Content-Type: application/json" \
	-d $(DATA)

# Nouvelle cible pour lancer le serveur MLflow
mlflow-ui:
	mlflow ui --host 127.0.0.1 --port 5000

# Nouvelle cible pour lancer un script de sauvegarde de modèle (si nécessaire)
train:
	$(PYTHON) save_model.py

help:
	@echo "Utilisez l'une des cibles suivantes :"
	@echo "  make install    - Installe les dépendances nécessaires"
	@echo "  make run        - Lance l'application FastAPI"
	@echo "  make predict    - Envoie une requête de prédiction à l'API FastAPI"
	@echo "  make mlflow-ui  - Lance l'interface MLflow"
	@echo "  make train      - Entraîne et sauvegarde le modèle"