import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
from pathlib import Path
import logging
from .config import model_config
from .utils import timing

logger = logging.getLogger(__name__)


class AttackClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None

    @timing
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train Random Forest classifier"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=model_config.RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )

        logger.info("Training Random Forest model...")
        self.model.fit(X, y)
        logger.info("Random Forest training completed")

    @timing
    def train_svm(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train Support Vector Machine classifier"""
        self.model = SVC(
            kernel='rbf',
            gamma='scale',
            C=1.0,
            random_state=model_config.RANDOM_STATE,
            verbose=True,
            probability=True
        )

        logger.info("Training SVM model...")
        self.model.fit(X, y)
        logger.info("SVM training completed")

    @timing
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Perform hyperparameter tuning using GridSearchCV"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(
                random_state=model_config.RANDOM_STATE),
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=2
        )

        logger.info("Starting hyperparameter tuning...")
        grid_search.fit(X, y)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_}")

        self.model = grid_search.best_estimator_

    @timing
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError(
                "Model not trained. Call train_random_forest() or train_svm() first")

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n" +
                    classification_report(y_test, y_pred))

        return {
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred
        }

    @timing
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError(
                "Model not trained. Call train_random_forest() or train_svm() first")

        return self.model.predict(X)

    @timing
    def save_model(self, filename: str = None) -> None:
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save")

        filename = filename or model_config.SUPERVISED_MODEL_NAME
        model_path = Path(model_config.MODEL_DIR) / filename
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

    @timing
    def load_model(self, filename: str = None) -> None:
        """Load a trained model from disk"""
        filename = filename or model_config.SUPERVISED_MODEL_NAME
        model_path = Path(model_config.MODEL_DIR) / filename
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
