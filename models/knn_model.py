import joblib

# Charger le modèle KNN pré-entraîné
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        return model
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle : {str(e)}")