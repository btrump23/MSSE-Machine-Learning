import numpy as np

class LinearModelWrapper:
    """
    Must exist for model.pkl unpickling: model_wrapper.LinearModelWrapper
    Robust against object arrays / list-of-lists / nested sequences.
    """

    def __init__(self, weights, bias=0.0, threshold=0.5):
        self.weights = np.asarray(weights, dtype=float).reshape(-1)
        self.bias = float(bias)
        self.threshold = float(threshold)

    def _ensure_2d_float(self, X):
        """
        Convert X into a strict 2D float ndarray.
        Handles:
          - list of lists
          - np.ndarray dtype=object (array of sequences)
          - 1D arrays
        """
        Xn = np.asarray(X)

        # If it's an object array (common cause of "sequence * float" errors)
        if Xn.dtype == object:
            # Typical failure mode: shape (n,) where each item is a list/array of length m
            try:
                Xn = np.vstack(Xn)
            except Exception:
                # Fallback: build row by row
                Xn = np.array([np.asarray(row).ravel() for row in Xn], dtype=object)
                Xn = np.vstack(Xn)

        # Now force float
        Xn = np.asarray(Xn, dtype=float)

        # Ensure 2D
        if Xn.ndim == 1:
            Xn = Xn.reshape(1, -1)

        return Xn

    def decision_function(self, X):
        Xn = self._ensure_2d_float(X)

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


# Safe alias
MalwareModelWrapper = LinearModelWrapper
