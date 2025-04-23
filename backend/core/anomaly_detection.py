import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from typing import Tuple
import logging
import joblib
from pathlib import Path
from .config import model_config
from .utils import timing

logger = logging.getLogger(__name__)


class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None

    @timing
    def train_isolation_forest(self, X: pd.DataFrame, contamination: float = 0.01) -> None:
        """Train Isolation Forest model for anomaly detection"""
        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=model_config.RANDOM_STATE,
            verbose=1
        )
        self.model.fit(X)
        logger.info("Isolation Forest trained successfully")

    @timing
    def train_lof(self, X: pd.DataFrame, n_neighbors: int = 20) -> None:
        """Train Local Outlier Factor model"""
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination='auto',
            novelty=True
        )
        self.model.fit(X)
        logger.info("Local Outlier Factor model trained successfully")

    @timing
    def detect_anomalies(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Detect anomalies in the data"""
        if self.model is None:
            raise ValueError(
                "Model not trained. Call train_isolation_forest() or train_lof() first")

        anomaly_scores = self.model.decision_function(X)
        predictions = (anomaly_scores <
                       model_config.ANOMALY_THRESHOLD).astype(int)

        results = pd.DataFrame({
            'anomaly_score': anomaly_scores,
            'is_anomaly': predictions
        })

        return results, anomaly_scores

    @timing
    def save_model(self, filename: str = None) -> None:
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save")

        filename = filename or model_config.UNSUPERVISED_MODEL_NAME
        model_path = Path(model_config.MODEL_DIR) / filename
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

    @timing
    def load_model(self, filename: str = None) -> None:
        """Load a trained model from disk"""
        filename = filename or model_config.UNSUPERVISED_MODEL_NAME
        model_path = Path(model_config.MODEL_DIR) / filename
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
