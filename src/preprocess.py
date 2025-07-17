import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters:
        path (str): File path to the CSV.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(path)
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that contain any missing values.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with no missing values.
    """
    df = df.dropna()
    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that are not useful for training the model.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame without unnecessary columns.
    """
    columns_to_drop = [
        'track_id', 'artists', 'album_name', 'track_name',
        'popularity', 'duration_ms', 'key', 'mode', 'time_signature'
    ]
    return df.copy().drop(columns=columns_to_drop, errors='ignore')

def normalize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize numeric features using StandardScaler.

    Parameters:
        df (pd.DataFrame): Input DataFrame with numerical features.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame with scaled numeric features.
            - StandardScaler: Fitted scaler for later use.
    """
    features = df.select_dtypes(include=['float64', 'int64']).columns
    features = features.drop('track_genre', errors='ignore')  # Prevent target leakage
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode the target genre column into integer labels.

    Parameters:
        df (pd.DataFrame): DataFrame containing the target column.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame with encoded genre labels.
            - LabelEncoder: Fitted label encoder.
    """
    encoder = LabelEncoder()
    df['track_genre'] = encoder.fit_transform(df['track_genre'])
    return df, encoder

def preprocess(path: str) -> tuple[pd.DataFrame, LabelEncoder, StandardScaler]:
    """
    Run the full preprocessing pipeline:
    - Load dataset
    - Handle missing values
    - Drop irrelevant columns
    - Normalize features
    - Encode target labels

    Parameters:
        path (str): File path to the CSV dataset.

    Returns:
        tuple:
            - pd.DataFrame: Fully preprocessed dataset.
            - LabelEncoder: Fitted label encoder for genre.
            - StandardScaler: Fitted scaler for numeric features.
    """
    df = load_dataset(path)
    df = handle_missing_values(df)
    df = drop_unnecessary_columns(df)
    df, scaler = normalize_features(df)
    df, encoder = encode_labels(df)
    return df, encoder, scaler
