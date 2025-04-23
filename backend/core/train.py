import logging
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from .load_data import get_train_test_data
from .preprocess import prepare_data
from .feature_engineering import select_features, apply_pca
from .supervised_model import AttackClassifier
from .anomaly_detection import AnomalyDetector
from .config import config, model_config
from .utils import timing

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self):
        self.supervised_model = AttackClassifier()
        self.anomaly_detector = AnomalyDetector()
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.label_encoder = None

    @timing
    def run(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        try:
            logger.info("Starting training pipeline")

            # 1. Load data
            train_df, _ = get_train_test_data()
            if train_df.empty:
                raise ValueError("Training data is empty")

            # 2. Preprocess data
            X_train, y_train, self.scaler, self.label_encoder = prepare_data(
                train_df, is_train=True)
            benign_index = list(self.label_encoder.classes_).index(
                model_config.BENIGN_LABEL)
            logger.info(f"Benign label encoded as: {benign_index}")

            # 3. Feature engineering
            X_selected, self.feature_selector = select_features(
                X_train, y_train, k=20)
            X_pca, self.pca = apply_pca(X_selected)

            # 4. Train supervised model
            self.supervised_model.train_random_forest(X_pca, y_train)

            # 5. Train anomaly detection model (on benign data only)
            benign_mask = (y_train == 0)
            if sum(benign_mask) == 0:
                raise ValueError(
                    "No benign samples found for anomaly detection training")

            X_benign = X_pca[benign_mask]
            self.anomaly_detector.train_isolation_forest(X_benign)

            # 6. Save all models and artifacts
            self.save_artifacts()

            logger.info("Training pipeline completed successfully")
            return {
                'status': 'success',
                'message': 'Models trained successfully',
                'model_path': str(Path(model_config.MODEL_DIR).absolute()),
                'model_info': {
                    'supervised': type(self.supervised_model.model).__name__,
                    'unsupervised': type(self.anomaly_detector.model).__name__,
                    'feature_count': X_pca.shape[1],
                    'train_samples': len(X_pca)
                }
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }

    @timing
    def save_artifacts(self) -> None:
        """Save all trained models and preprocessing artifacts"""
        model_dir = Path(model_config.MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        self.supervised_model.save_model()
        self.anomaly_detector.save_model()

        # Save preprocessing artifacts
        artifacts = {
            'scaler': model_config.SCALER_NAME,
            'feature_selector': "feature_selector.pkl",
            'pca': "pca.pkl",
            'label_encoder': "label_encoder.pkl"
        }

        for name, filename in artifacts.items():
            path = model_dir / filename
            joblib.dump(getattr(self, name), path)
            logger.info(f"Saved {name} to {path}")

        logger.info(f"All artifacts saved to {model_dir.absolute()}")
