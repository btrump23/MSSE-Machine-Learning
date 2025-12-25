"""
Compatibility shim for old pickled models.

Some saved model.pkl files reference symbols in a module named `Pipeline`,
e.g. `Pipeline.dtype` or `Pipeline.Pipeline`. This file exists only to support unpickling.
"""

from sklearn.pipeline import Pipeline as Pipeline
import numpy as np

dtype = np.dtype
