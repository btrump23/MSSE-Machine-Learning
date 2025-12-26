import numpy as np

class LinearModelWrapper:
    """
    Must exist for model.pkl unpickling: model_wrapper.LinearModelWrapper
    """

    def __init__(self, weights, bias=0.0, threshold=0.5):
        self.weights = np.asarray(weights, dtype=float).reshape(-1)
        self.bias = float(bias)
        self.threshold = float(threshold)

    def decision_function(self, X):
        # HARD FORCE float + 2D, and fix object/sequence arrays
        Xn = np.asarray(X)

        if Xn.dtype == object:
            # common case: array of sequences -> stack them
            Xn = np.vstack(Xn)

        Xn = np.asarray(Xn, dtype=float)

        if Xn.ndim == 1:
            Xn = Xn.reshape(1, -1)

        if Xn.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Feature mismatch: X has {Xn.shape[1]} columns but weights has {self.weights.shape[0]}."
            )

        return (Xn @ self.weights) + self.bias

    def predict_proba(self, X):
        scores = self.decision_function(X)
        probs_pos = 1.0 / (1.0 + np.exp(-scores))
        probs_pos = probs_pos.reshape(-1, 1)
        probs_neg = 1.0 - probs_pos
        return np.hstack([probs_neg, probs_pos])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

MalwareModelWrapper = LinearModelWrapper
