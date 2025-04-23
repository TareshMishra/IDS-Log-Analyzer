import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging
import joblib
from pathlib import Path
from .config import config, model_config
from .utils import timing

logger = logging.getLogger(__name__)


def load_scaler() -> MinMaxScaler:
    """Load the saved scaler from disk"""
    try:
        scaler_path = Path(model_config.MODEL_DIR) / model_config.SCALER_NAME
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        return joblib.load(scaler_path)
    except Exception as e:
        logger.error(f"Error loading scaler: {str(e)}")
        raise


@timing
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset"""
    logger.info("Starting data cleaning process")

    # Handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop columns with too many missing values
    threshold = len(df) * 0.5
    df.dropna(thresh=threshold, axis=1, inplace=True)

    # Fill remaining missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Convert object types to categorical
    for col in df.select_dtypes(include=['object']).columns:
        if col != config.LABEL_COLUMN:
            df[col] = df[col].astype('category').cat.codes

    logger.info(f"Data cleaned. New shape: {df.shape}")
    return df


@timing
def extract_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Extract labels from the dataset"""
    if config.LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Label column '{config.LABEL_COLUMN}' not found in DataFrame")

    y = df[config.LABEL_COLUMN]
    X = df.drop(columns=[config.LABEL_COLUMN])

    # Encode labels (BENIGN=0, attacks=1,2,...)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder


@timing
def normalize_features(X: pd.DataFrame, scaler=None, fit: bool = False) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Normalize features using MinMax scaling"""
    if scaler is None:
        scaler = MinMaxScaler()

    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled_df, scaler


@timing
def prepare_data(df: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, np.ndarray, any, LabelEncoder]:
    """Prepare data for training or prediction"""
    df_clean = clean_data(df)
    X, y, label_encoder = extract_labels(df_clean)

    if is_train:
        X_scaled, scaler = normalize_features(X, fit=True)
        return X_scaled, y, scaler, label_encoder
    else:
        scaler = load_scaler()
        X_scaled, _ = normalize_features(X, scaler=scaler, fit=False)
        return X_scaled, y, label_encoder
