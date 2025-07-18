import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score


def plot_confusion(y_true, y_pred, model_name: str, encoder, output_dir: str = "plots", normalize: bool = False) -> None:
    """
    Generate and save the confusion matrix plot for a model.

    Parameters:
        y_true (array-like): True encoded labels.
        y_pred (array-like): Predicted encoded labels.
        model_name (str): Name of the model.
        encoder (LabelEncoder): Fitted label encoder.
        output_dir (str): Folder to save the plots.
        normalize (bool): Whether to normalize the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    labels = encoder.classes_[np.argsort(encoder.transform(encoder.classes_))]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(25, 20))
    sns.heatmap(cm, annot=False, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}{" (Normalized)" if normalize else ""}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    filename = f"{output_dir}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved confusion matrix for {model_name} to {filename}")

def evaluate_model(model, X_test, y_test, encoder, model_name, save_plots=True):
    """
    Evaluate the trained model and generate performance metrics.

    Parameters:
        model: Trained model to evaluate.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.
        encoder (LabelEncoder): Fitted label encoder for decoding labels.
        model_name (str): Name of the model for reporting.
        save_plots (bool): Whether to save confusion matrix plots.
    """
    print(f"\n--- {model_name} ---")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", report)

    if save_plots:
        plot_confusion(y_test, y_pred, model_name, encoder)

    return {
        'name': model_name,
        'model': model,
        'f1_score': f1,
        'accuracy': accuracy,
        'report': report,
        'y_pred': y_pred
    }

