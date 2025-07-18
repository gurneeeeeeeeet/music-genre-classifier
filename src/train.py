from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier

from src.preprocess import (
    load_dataset,
    handle_missing_values,
    drop_unnecessary_columns,
    encode_explicit,
)
from src.evaluate import evaluate_model


def train_model(data_path: str, target_column: str = "track_genre") -> None:
    """Train and evaluate multiple ML models on the music genre dataset with tuning."""

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    if not Path(data_path).exists():
        raise FileNotFoundError(f"The dataset file {data_path} does not exist.")

    print(f"Training model on dataset: {data_path} with target column: {target_column}")
    print("Loading and preprocessing the dataset...")

    df = load_dataset(data_path)
    df = handle_missing_values(df)
    df = encode_explicit(df)
    df = drop_unnecessary_columns(df)

    X = df.select_dtypes(include=["float64", "int64"]).drop(
        columns=[target_column], errors="ignore"
    )
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    # --- Logistic Regression (Commented Out) ---
    # logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    # logistic_model.fit(X_train_scaled, y_train_encoded)
    # logistic_result = evaluate_model(logistic_model, X_test_scaled, y_test_encoded, encoder, "Logistic Regression")

    # Log data stats
    print(f"Training set size: {len(X_train)}, Unique classes: {len(set(y_train))}")

    # Use StratifiedKFold to preserve label distribution across CV folds
    stratified_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # --- Random Forest with GridSearch ---
    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True],
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        scoring="f1_macro",
        cv=stratified_cv,
        n_jobs=1,
        verbose=1,
    )
    rf_grid.fit(X_train_scaled, y_train_encoded)
    print("Best RF params:", rf_grid.best_params_)
    rf_result = evaluate_model(
        rf_grid.best_estimator_, X_test_scaled, y_test_encoded, encoder, "Random Forest"
    )

    # --- XGBoost with GridSearch ---
    xgb_params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "learning_rate": [0.01, 0.05],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "gamma": [0, 1],
    }

    xgb_grid = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
        xgb_params,
        scoring="f1_macro",
        cv=stratified_cv,
        n_jobs=1,
        verbose=1,
    )
    xgb_grid.fit(X_train_scaled, y_train_encoded)
    print("Best XGB params:", xgb_grid.best_params_)
    xgb_result = evaluate_model(
        xgb_grid.best_estimator_, X_test_scaled, y_test_encoded, encoder, "XGBoost"
    )

    # --- KNN (Commented Out) ---
    # knn_model = KNeighborsClassifier(n_neighbors=5)
    # knn_model.fit(X_train_scaled, y_train_encoded)
    # knn_result = evaluate_model(knn_model, X_test_scaled, y_test_encoded, encoder, "K-Nearest Neighbors")

    # F1 comparison
    results = [rf_result, xgb_result]  # Add back logistic_result, knn_result if needed
    best_result = max(results, key=lambda r: r["f1_score"])

    print(
        f"\nBest model based on F1 score: {best_result['name']} with F1 score of {best_result['f1_score']:.4f}"
    )
    print("Saving the trained model and preprocessing artifacts...")

    save_model(best_result["model"], models_dir / "best_model.pkl")
    save_model(scaler, models_dir / "scaler.pkl")
    save_model(encoder, models_dir / "label_encoder.pkl")


def save_model(obj, filename: Path) -> None:
    """Save a model or preprocessing object to a file."""
    joblib.dump(obj, filename)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and evaluate models on the music genre dataset."
    )
    parser.add_argument("data_path", type=str, help="Path to the dataset CSV file.")
    parser.add_argument(
        "--target_column",
        type=str,
        default="track_genre",
        help="Name of the target column (default: track_genre).",
    )

    args = parser.parse_args()
    train_model(args.data_path, args.target_column)
