from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import mlflow

# Définir le modèle de données pour les requêtes
class PredictRequest(BaseModel):
    data: list

# Charger le modèle KNN pré-entraîné
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        return model
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle : {str(e)}")

# Initialiser l'application FastAPI
app = FastAPI()

# Charger le modèle au démarrage
model = load_model()

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # Convertir les données en tableau NumPy
        input_data = np.array(request.data).reshape(1, -1)

        # Effectuer une prédiction
        prediction = model.predict(input_data)

        # Convertir la prédiction en format lisible
        prediction_label = "Has Cancer" if prediction[0] == 1 else "No Cancer"

        # Enregistrer les paramètres et résultats dans MLflow
        with mlflow.start_run():
            mlflow.log_param("input_data", request.data)  # Enregistrer les données d'entrée
            mlflow.log_metric("prediction", int(prediction[0]))  # Loguer la prédiction

        return {"prediction": prediction_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain():
    try:
        # Exemple simple de réentraînement
        # Vous pouvez ajouter votre logique ici pour charger de nouvelles données et réentraîner le modèle
        # Puis sauvegarder le modèle mis à jour
        with mlflow.start_run():
            mlflow.log_param("retrain_example", "Réentraînement appelé")
            mlflow.log_metric("status", 1)  # 1 pour succès, 0 pour échec

        return {"message": "Réentraînement du modèle terminé avec succès (exemple)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))