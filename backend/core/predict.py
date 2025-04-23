import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from .config import model_config
from .load_data import load_dataset
from .preprocess import prepare_data
from .utils import timing

logger = logging.getLogger(__name__)


class PredictionPipeline:
    def __init__(self):
        self.supervised_model = None
        self.anomaly_detector = None
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.label_encoder = None

    @timing
    def load_artifacts(self) -> bool:
        """Load all trained models and preprocessing artifacts"""
        try:
            model_dir = Path(model_config.MODEL_DIR)

            # Verify model directory exists
            if not model_dir.exists():
                logger.error(f"Model directory not found: {model_dir}")
                return False

            # Load models and artifacts
            artifacts = {
                'supervised_model': model_config.SUPERVISED_MODEL_NAME,
                'anomaly_detector': model_config.UNSUPERVISED_MODEL_NAME,
                'scaler': model_config.SCALER_NAME,
                'feature_selector': "feature_selector.pkl",
                'pca': "pca.pkl",
                'label_encoder': "label_encoder.pkl"
            }

            for attr, filename in artifacts.items():
                path = model_dir / filename
                if not path.exists():
                    logger.error(f"Artifact not found: {path}")
                    return False
                setattr(self, attr, joblib.load(path))

            logger.info("All artifacts loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}", exc_info=True)
            return False

    @timing
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions on new data"""
        if not self.load_artifacts():
            raise ValueError(
                "Could not load required artifacts for prediction")

        try:
            # Preprocess the data
            X, y, label_encoder = prepare_data(df, is_train=False)

            # Feature selection
            X_selected = self.feature_selector.transform(X)
            selected_cols = self.feature_selector.get_feature_names_out()
            X_selected = pd.DataFrame(
                X_selected, columns=selected_cols, index=X.index)

            # Dimensionality reduction
            X_pca = self.pca.transform(X_selected)
            X_pca = pd.DataFrame(X_pca, index=X.index)

            # Supervised predictions
            supervised_preds = self.supervised_model.predict(X_pca)
            attack_labels = self.label_encoder.inverse_transform(
                supervised_preds)

            # Anomaly detection
            anomaly_scores = self.anomaly_detector.decision_function(X_pca)
            anomaly_preds = (
                anomaly_scores < model_config.ANOMALY_THRESHOLD).astype(int)

            # Combine results
            results = pd.DataFrame({
                'is_anomaly': anomaly_preds,
                'anomaly_score': anomaly_scores,
                'predicted_attack': attack_labels,
                'is_malicious': (attack_labels != model_config.BENIGN_LABEL).astype(int)
            })

            # Add true labels if available
            if y is not None:
                results['true_label'] = label_encoder.inverse_transform(y)
                results['is_correct'] = (
                    results['predicted_attack'] == results['true_label'])

            return {
                'results': results,
                'features': X_pca,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_metadata': {
                    'supervised_model': type(self.supervised_model).__name__,
                    'anomaly_detector': type(self.anomaly_detector).__name__,
                    'feature_count': X_pca.shape[1]
                }
            }

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise ValueError(f"Prediction error: {str(e)}")

    @timing
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a complete file"""
        try:
            logger.info(f"Analyzing file: {file_path}")

            # Load and validate data
            df = load_dataset(file_path)
            if df.empty:
                return {'error': "Loaded empty dataset"}

            if len(df) > 100000:  # Sample if too large for demo
                df = df.sample(n=100000, random_state=42)
                logger.info(f"Sampled 100,000 records from large dataset")

            # Perform prediction
            return self.predict(df)

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return {'error': f"Analysis failed: {str(e)}"}

    @timing
    def generate_attack_stats(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistics about detected attacks"""
        stats = {}
        try:
            if not isinstance(results, pd.DataFrame):
                results = pd.DataFrame(results)

            stats['total_records'] = len(results)

            # Anomaly stats
            if all(col in results.columns for col in ['is_anomaly', 'anomaly_score']):
                stats['anomaly_rate'] = results['is_anomaly'].mean() * 100
                stats['avg_anomaly_score'] = results['anomaly_score'].mean()

            # Attack distribution
            if 'predicted_attack' in results.columns:
                attack_counts = results['predicted_attack'].value_counts(
                ).to_dict()
                stats['attack_distribution'] = {
                    # Ensure JSON-serializable
                    k: int(v) for k, v in attack_counts.items()
                }

                # Accuracy if ground truth available
                if 'true_label' in results.columns:
                    stats['accuracy'] = (
                        results['predicted_attack'] == results['true_label']
                    ).mean() * 100

        except Exception as e:
            logger.error(f"Stats generation failed: {str(e)}", exc_info=True)
            stats['error'] = str(e)

        return stats
