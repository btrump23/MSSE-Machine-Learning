import numpy as np

class LinearModelWrapper:
    def __init__(self, weights):
        self.weights = weights

    def predict(self, X):
        scores = X.values @ self.weights
        return (scores > 0).astype(int)

    def predict_proba(self, X):
        scores = X.values @ self.weights
        probs = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - probs, probs])
