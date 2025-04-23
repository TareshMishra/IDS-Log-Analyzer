import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from backend.core.supervised_model import AttackClassifier
from backend.core.preprocess import clean_data, extract_labels
from backend.core.config import config


class TestSupervisedModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small test dataset
        cls.test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            config.LABEL_COLUMN: ['BENIGN'] * 70 +
            ['Attack1'] * 15 + ['Attack2'] * 15
        })

        cls.X, cls.y, _ = extract_labels(clean_data(cls.test_data))

    def test_random_forest(self):
        classifier = AttackClassifier()
        classifier.train_random_forest(self.X, self.y)

        predictions = classifier.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

        # Should at least predict some attacks
        self.assertTrue(any(predictions != 0))

    def test_evaluation(self):
        classifier = AttackClassifier()
        classifier.train_random_forest(self.X, self.y)

        eval_results = classifier.evaluate(self.X, self.y)
        self.assertIn('accuracy', eval_results)
        # Should be better than random
        self.assertGreater(eval_results['accuracy'], 0.5)


if __name__ == '__main__':
    unittest.main()
