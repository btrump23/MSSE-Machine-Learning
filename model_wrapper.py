import numpy as np

class MalwareModelWrapper:
    """
    Simple linear/logistic style wrapper:
    - decision_function: X @ weights + bias
    - predict_proba: sigmoid(decision)
    - predict: threshold on proba
    """

    def __init__(self, weights, bias=0.0, threshold=0.5):
        # Store as numpy floats (critical fix)
        self.weights = np.asarray(weights, dtype=float).reshape(-1)
        self.bias = float(bias)
        self.threshold = float(threshold)

    def _to_2d_float_array(self, X):
        """
        Force X into a real 2D numpy float array.
        This prevents: "can't multiply sequence by non-int of type 'float'"
        """
        Xn = np.asarray(X, dtype=float)

        # If a single row (1D), make it (1, n_features)
        if Xn.ndim == 1:
            Xn = Xn.reshape(1, -1)

        return Xn

    def decision_function(self, X):
        Xn = self._to_2d_float_array(X)

        if Xn.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Feature mismatch: X has {Xn.shape[1]} columns but weights has {self.weights.shape[0]}."
            )

        return (Xn @ self.weights) + self.bias

    def predict_proba(self, X):
        scores = self.decision_function(X)
        probs_pos = 1.0 / (1.0 + np.exp(-scores))  # sigmoid

        # Return shape (n_samples, 2) like sklearn: [P(class0), P(class1)]
        probs_pos = probs_pos.reshape(-1, 1)
        probs_neg = 1.0 - probs_pos
        return np.hstack([probs_neg, probs_pos])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)
