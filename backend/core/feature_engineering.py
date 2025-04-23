import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from typing import Tuple
import logging
from .config import config, model_config
from .utils import timing

logger = logging.getLogger(__name__)


@timing
def select_features(X: pd.DataFrame, y: pd.Series, k: int = 20) -> Tuple[pd.DataFrame, SelectKBest]:
    """Select top k features using ANOVA F-value"""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    selected_cols = X.columns[selector.get_support()]
    X_selected_df = pd.DataFrame(
        X_selected, columns=selected_cols, index=X.index)

    logger.info(f"Selected top {k} features: {list(selected_cols)}")
    return X_selected_df, selector


@timing
def apply_pca(X: pd.DataFrame, n_components: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
    """Apply PCA for dimensionality reduction"""
    pca = PCA(n_components=n_components,
              random_state=model_config.RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    logger.info(
        f"Reduced dimensions from {X.shape[1]} to {X_pca.shape[1]} with PCA")
    return pd.DataFrame(X_pca), pca


@timing
def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from timestamp if available"""
    if 'Timestamp' in df.columns:
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Hour'] = df['Timestamp'].dt.hour
            df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
            df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
            logger.info("Extracted temporal features from Timestamp")
        except Exception as e:
            logger.warning(f"Could not extract temporal features: {str(e)}")
    return df
