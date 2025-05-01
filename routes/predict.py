from flask import Blueprint, request, jsonify
from models.knn_model import load_model
from utils.preprocess import preprocess_input

# Créer un blueprint pour les routes de prédiction
predict_blueprint = Blueprint('predict', __name__)

# Charger le modèle KNN
model = load_model()

@predict_blueprint.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON de la requête
        input_data = request.get_json()

        # Prétraiter les données
        preprocessed_data, error = preprocess_input(input_data)
        if error:
            return jsonify({"error": error}), 400

        # Effectuer une prédiction
        prediction = model.predict(preprocessed_data)

        # Convertir la prédiction en format lisible
        prediction_label = "Has Cancer" if prediction[0] == 1 else "No Cancer"

        return jsonify({"prediction": prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500