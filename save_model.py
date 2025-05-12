import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train a KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save the model locally
joblib.dump(model, "best_model.pkl")

# Log parameters, metrics, and the model with MLflow
with mlflow.start_run():
    mlflow.log_param("n_neighbors", 3)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    print(f"Model logged with accuracy: {accuracy}")