import numpy as np

class LinearModelWrapper:
    """
    Wrapper class required for unpickling model.pkl.

    HARD GUARANTEE:
    - X is ALWAYS a numpy float array before math
    """

    def __init__(self, weights, bias=0.0, threshold=0.5):
        self.weights = np.asarray(weights, dtype=float).reshape(-1)
        self.bias = float(bias)
        self.threshold = float(threshold)

    def _to_2d_float_array(self, X):
        # 🔥 FINAL SAFETY NET — force numpy float array
        Xn = np.asarray(X, dtype=float)

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
        probs_pos = probs_pos.reshape(-1, 1)
        probs_neg = 1.0 - probs_pos
        return np.hstack([probs_neg, probs_pos])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)


# Optional alias (safe)
MalwareModelWrapper = LinearModelWrapper
