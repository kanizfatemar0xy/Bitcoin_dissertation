"""Time-ordered train/test scaling to avoid fitting transforms on future data."""
import numpy as np
from sklearn.preprocessing import StandardScaler


def split_and_scale(features, target, test_fraction=0.2):
    X = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError('features must be 2D and target must be 1D with same rows')
    if not 0 < test_fraction < 1:
        raise ValueError('test_fraction must be between 0 and 1')
    if len(X) < 10:
        raise ValueError('need at least ten time-ordered observations')
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('features and target must be finite')
    cut = int(len(X) * (1 - test_fraction))
    if cut < 2 or cut >= len(X):
        raise ValueError('split leaves an empty train or test window')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:cut])
    X_test = scaler.transform(X[cut:])
    return X_train, X_test, y[:cut], y[cut:], scaler
