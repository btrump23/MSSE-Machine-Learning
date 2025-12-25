"""
Pipeline compatibility shim for sklearn pickle loading
"""

import numpy as np
from sklearn.pipeline import Pipeline as SklearnPipeline

# Expose exactly what the pickle expects
Pipeline = SklearnPipeline
dtype = np.dtype
