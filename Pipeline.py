"""
Compatibility shim for unpickling legacy models.

Your saved model references a module named `Pipeline` (capital P) and expects
certain symbols on that module during pickle.load().
"""

import numpy as np
from sklearn.pipeline import Pipeline as SklearnPipeline

# Re-export so pickles that expect Pipeline.Pipeline can resolve it
Pipeline = SklearnPipeline

# Pickle is trying to access Pipeline.dtype
# Provide numpy dtype factory to satisfy it.
dtype = np.dtype

