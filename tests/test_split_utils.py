import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from split_utils import split_and_scale

class TestChronologicalSplit(unittest.TestCase):
    def setUp(self):
        self.X = np.column_stack((np.arange(20), np.arange(20) ** 2)).astype(float)
        self.y = np.arange(20, dtype=float)

    def test_training_and_holdout_are_chronological(self):
        tr, te, ytr, yte, scaler = split_and_scale(self.X, self.y)
        self.assertEqual(len(ytr), 16)
        self.assertEqual(len(yte), 4)
        np.testing.assert_array_equal(ytr, np.arange(16))
        np.testing.assert_array_equal(yte, np.arange(16,20))
        np.testing.assert_allclose(tr.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(scaler.mean_, self.X[:16].mean(axis=0))
        self.assertFalse(np.allclose(scaler.mean_, self.X.mean(axis=0)))
        np.testing.assert_allclose(te, scaler.transform(self.X[16:]))

    def test_rejects_short_data(self):
        with self.assertRaises(ValueError):
            split_and_scale(self.X[:5], self.y[:5])

    def test_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            split_and_scale(self.X, self.y, test_fraction=1)

    def test_rejects_nonfinite(self):
        bad = self.X.copy(); bad[1,0] = np.nan
        with self.assertRaises(ValueError):
            split_and_scale(bad, self.y)

if __name__ == '__main__':
    unittest.main()
