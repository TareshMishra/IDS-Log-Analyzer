import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from backend.core.anomaly_detection import AnomalyDetector
from backend.core.preprocess import clean_data
from backend.core.config import config


class TestAnomalyDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small test dataset
        cls.test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            config.LABEL_COLUMN: ['BENIGN'] * 95 + ['Attack'] * 5
        })

        # Add some anomalies
        cls.test_data.loc[95:100, 'feature1'] = 10

    def test_isolation_forest(self):
        detector = AnomalyDetector()
        X = clean_data(self.test_data).drop(columns=[config.LABEL_COLUMN])

        detector.train_isolation_forest(X)
        results, scores = detector.detect_anomalies(X)

        self.assertEqual(len(results), len(X))
        # Should detect some anomalies
        self.assertTrue(any(results['is_anomaly'] == 1))

    def test_lof(self):
        detector = AnomalyDetector()
        X = clean_data(self.test_data).drop(columns=[config.LABEL_COLUMN])

        detector.train_lof(X)
        results, scores = detector.detect_anomalies(X)

        self.assertEqual(len(results), len(X))
        self.assertTrue(any(results['is_anomaly'] == 1))


if __name__ == '__main__':
    unittest.main()
