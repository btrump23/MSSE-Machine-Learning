import numpy as np

class LinearModelWrapper:
    """
    Wraps your trained weights/bias and exposes sklearn-like predict / predict_proba.
    Must be importable on Render for pickle loading.
    """

    def __init__(self, weights, bias=0.0, threshold=0.5):
        self.weights = np.asarray(weights, dtype=float).reshape(-1)
        self.bias = float(bias)
        self.threshold = float(threshold)

    def _to_numpy(self, X):
        # Accept pandas DataFrame/Series OR numpy array/list
        if hasattr(X, "to_numpy"):          # pandas DataFrame/Series
            Xn = X.to_numpy()
        else:
            Xn = np.asarray(X)

        Xn = np.asarray(Xn, dtype=float)

        # Ensure 2D (n_samples, n_features)
        if Xn.ndim == 1:
            Xn = Xn.reshape(1, -1)

        return Xn

    def decision_function(self, X):
        Xn = self._to_numpy(X)
        return Xn @ self.weights + self.bias

    def predict_proba(self, X):
        scores = self.decision_function(X)
        probs1 = 1.0 / (1.0 + np.exp(-scores))
        probs0 = 1.0 - probs1
        return np.column_stack([probs0, probs1])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)
