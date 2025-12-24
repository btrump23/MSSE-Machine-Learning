# src/Pipeline.py
"""
Compatibility shim so older pickles that reference a module named `Pipeline`
can be loaded on Render.

If the original pickle referenced Pipeline.Pipeline, this class will exist and
delegate to sklearn's Pipeline implementation.
"""

from sklearn.pipeline import Pipeline as _SklearnPipeline

class Pipeline(_SklearnPipeline):
    pass
