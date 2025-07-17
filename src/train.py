import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.preprocess import load_dataset, handle_missing_values, drop_unnecessary_columns

def train_model(data_path: str, target_column: str = 'track_genre') -> None:

    """
    Train a logistic regression model on the preprocessed dataset.

    Parameters:
        data_path (str): Path to the dataset CSV file.
        target_column (str): Name of the target column to predict.
    """

    # Load and preprocess the dataset
    print("Loading and preprocessing the dataset...")
    df = load_dataset(data_path)
    df = handle_missing_values(df)
    df = drop_unnecessary_columns(df)

    X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[target_column], errors='ignore')
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # fit encoder on training labels only
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    # Train the model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train_encoded)

    y_pred = model.predict(X_test_scaled)

    print("Model Accuracy:", accuracy_score(y_test_encoded, y_pred))
    print("Classification Report:\n", classification_report(y_test_encoded, y_pred))

    # Save the model, scaler, and encoder
    save_model(model, 'models/logistic_regression_model.pkl')
    save_model(scaler, 'models/scaler.pkl')
    save_model(encoder, 'models/label_encoder.pkl')
    print("Model, scaler, and encoder saved successfully.")

def save_model(obj, filename: str) -> None:
    """
    Save a model or any object to a file using joblib.

    Parameters:
        obj: The object to save (e.g., model, scaler, encoder).
        filename (str): The name of the file to save the object to.
    """
    joblib.dump(obj, filename)
    print(f"{filename} saved successfully.")