"""
Pipeline.py

Compatibility shim for unpickling the saved model in production (Render/Linux)
and locally (Windows). Your pickle references a module named `Pipeline`
(capital P) and expects certain symbols (e.g., `dtype`) to exist at module scope.

Keep this file in the SAME folder as predict.py and model.pkl.
"""

import numpy as np
from sklearn.pipeline import Pipeline as SklearnPipeline

# Some pickles reference Pipeline.Pipeline
Pipeline = SklearnPipeline

# Your pickle is trying to access Pipeline.dtype
# Expose numpy's dtype factory at module scope.
dtype = np.dtype
