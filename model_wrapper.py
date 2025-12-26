import numpy as np

class LinearModelWrapper:
    """
    Must exist for model.pkl unpickling: model_wrapper.LinearModelWrapper
    """

    def __init__(self, weights, bias=0.0, threshold=0.5):
        # Force numeric
        self.weights = np.asarray(weights, dtype=float).reshape(-1)
        self.bias = float(bias)
        self.threshold = float(threshold)

    def decision_function(self, X):
        # 🔥 HARD FORCE conversion RIGHT HERE (no chance for object arrays)
        Xn = np.asarray(X, dtype=float)

        # Ensure 2D
        if Xn.ndim == 1:
            Xn = Xn.reshape(1, -1)

        # Sanity check
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


# harmless alias
MalwareModelWrapper = LinearModelWrapper
