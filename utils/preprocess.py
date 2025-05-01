import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_input(data):
    """
    Prétraiter les données d'entrée avant la prédiction.
    Args:
        data (dict): Entrée JSON contenant les valeurs des caractéristiques.
    Returns:
        np.array: Données prétraitées prêtes pour le modèle.
        str: Message d'erreur si une caractéristique est manquante.
    """
    # Colonnes attendues dans le même ordre que les données d'entraînement
    expected_columns = [
        "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE",
        "FATIGUE ", "ALLERGY ", "WHEEZING", "ALCOHOL CONSUMING",
        "COUGHING", "SWALLOWING DIFFICULTY", "CHEST PAIN", "ANXYELFIN"
    ]

    # Convertir les données en DataFrame
    input_df = pd.DataFrame([data])

    # Vérifier la présence de toutes les colonnes attendues
    for col in expected_columns:
        if col not in input_df.columns:
            return None, f"Caractéristique manquante : {col}"

    # Extraire uniquement les colonnes attendues
    input_df = input_df[expected_columns]

    # Appliquer le scaling
    scaler = StandardScaler()
    preprocessed_data = scaler.fit_transform(input_df)

    return preprocessed_data, None