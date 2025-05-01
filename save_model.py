import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Charger les données
data = pd.read_csv("survey lung cancer.csv")

# Vérifiez les colonnes non numériques
print("Colonnes disponibles :", data.columns)

# Convertir la colonne 'GENDER' en données numériques
# Par exemple, 'M' -> 0 et 'F' -> 1
data['GENDER'] = data['GENDER'].map({'M': 0, 'F': 1})

# Supprimer les espaces dans les noms de colonnes pour éviter les erreurs
data.columns = data.columns.str.strip()

# Séparer les caractéristiques (X) et la cible (y)
X = data.drop("LUNG_CANCER", axis=1)  # Caractéristiques
y = data["LUNG_CANCER"]               # Cible

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner un modèle KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Évaluer la précision du modèle
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")

# Sauvegarder le modèle dans un fichier
joblib.dump(knn, "best_model.pkl")
print("Modèle sauvegardé sous le nom 'best_model.pkl'.")