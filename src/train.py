from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocess import load_dataset, handle_missing_values, drop_unnecessary_columns, encode_explicit

def train_model(data_path: str, target_column: str = 'track_genre') -> None:
    """
    Train a logistic regression model on the preprocessed dataset.

    Parameters:
        data_path (str): Path to the dataset CSV file.
        target_column (str): Name of the target column to predict.
    """

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    if not Path(data_path).exists():
        raise FileNotFoundError(f"The dataset file {data_path} does not exist.")
    
    print(f"Training model on dataset: {data_path} with target column: {target_column}")
    print("Loading and preprocessing the dataset...")

    # Load and preprocess the dataset
    df = load_dataset(data_path)
    df = handle_missing_values(df)
    df = encode_explicit(df)
    df = drop_unnecessary_columns(df)

    X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[target_column], errors='ignore')
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit encoder on training labels only
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    # Train the model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train_encoded)

    y_pred = model.predict(X_test_scaled)

    print("Model Accuracy:", accuracy_score(y_test_encoded, y_pred))
    print("Classification Report:\n", classification_report(y_test_encoded, y_pred, zero_division=0))

    #Train Random Foresst model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train_encoded)

    # Predict and evaluate Random Forest model
    rf_y_pred = rf_model.predict(X_test_scaled)
    print("\n--- Random Forest Model ---")
    print("Model Accuracy:", accuracy_score(y_test_encoded, rf_y_pred))
    print("Classification Report:\n", classification_report(y_test_encoded, rf_y_pred, zero_division=0))

    # Print the feature importances
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

    print("\nFeature Importances from Random Forest Model:")
    print(feature_importances.sort_values(ascending=False))

    # Print the results of the Random Forest model
    print("\n--- Random Forest Model ---")
    print("Model Accuracy:", accuracy_score(y_test_encoded, rf_y_pred))
    print("Classification Report:\n", classification_report(y_test_encoded, rf_y_pred, zero_division=0))

    # Train XGBoost model
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbose=1, random_state=42)
    xgb_model.fit(X_train_scaled, y_train_encoded)

    # Predict and evaluate XGBoost model
    xgb_y_pred = xgb_model.predict(X_test_scaled)
    print("\n--- XGBoost Model ---")
    print("Model Accuracy:", accuracy_score(y_test_encoded, xgb_y_pred))
    print("Classification Report:\n", classification_report(y_test_encoded, xgb_y_pred, zero_division=0))

    # K-Nearest Neighbors model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train_encoded)
    knn_y_pred = knn_model.predict(X_test_scaled)
    print("\n--- K-Nearest Neighbors Model ---")
    print("Model Accuracy:", accuracy_score(y_test_encoded, knn_y_pred))
    print("Classification Report:\n", classification_report(y_test_encoded, knn_y_pred, zero_division=0))
          
    print("Preprocessing complete.")
    
    # Collect F1 scores for each model
    f1_scores = {
        'Logistic Regression': f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0),
        'Random Forest': f1_score(y_test_encoded, rf_y_pred, average='weighted', zero_division=0),
        'XGBoost': f1_score(y_test_encoded, xgb_y_pred, average='weighted', zero_division=0),
        'K-Nearest Neighbors': f1_score(y_test_encoded, knn_y_pred, average='weighted', zero_division=0)
    }

    # Print F1 scores
    print("\nF1 Scores:")
    for model_name, score in f1_scores.items():
        print(f"{model_name}: {score:.4f}")
    
    # Determine the best model based on F1 score
    best_model_name = max(f1_scores, key=f1_scores.get)
    best_f1_score = f1_scores[best_model_name]
    print(f"\nBest model based on F1 score: {best_model_name} with F1 score of {best_f1_score:.4f}")

    # Save the best model and preprocessing artifacts
    print("Saving the trained model and preprocessing artifacts...")
    if best_model_name == 'Logistic Regression':
        best_model = model
    elif best_model_name == 'Random Forest':
        best_model = rf_model
    elif best_model_name == 'XGBoost':
        best_model = xgb_model
    elif best_model_name == 'K-Nearest Neighbors':
        best_model = knn_model
    else:
        raise ValueError("Unknown best model name.")
    
    save_model(best_model, models_dir / "best_model.pkl")

    # Save the scaler and encoder
    print("Saving scaler and label encoder...")
    save_model(scaler, models_dir / "scaler.pkl")
    save_model(encoder, models_dir / "label_encoder.pkl")

def save_model(obj, filename: Path) -> None:
    """
    Save a model or any object to a file using joblib.

    Parameters:
        obj: The object to save (e.g., model, scaler, encoder).
        filename (Path): The file path to save the object to.
    """
    joblib.dump(obj, filename)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a logistic regression model on the music genre dataset.")
    parser.add_argument('data_path', type=str, help='Path to the dataset CSV file.')
    parser.add_argument('--target_column', type=str, default='track_genre', help='Name of the target column (default: track_genre).')

    args = parser.parse_args()

    train_model(args.data_path, args.target_column)
