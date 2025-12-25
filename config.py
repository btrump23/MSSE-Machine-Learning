# config.py
from __future__ import annotations

import os

# ✅ Single source of truth for the model download URL
MODEL_ZIP_URL: str = os.getenv(
    "MODEL_ZIP_URL",
    "https://github.com/btrump23/MSSE-Machine-Learning/releases/latest/download/model.zip",
)

# Where to cache downloads (works on local + cloud)
CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", ".cache")
MODEL_ZIP_NAME: str = os.getenv("MODEL_ZIP_NAME", "model.zip")
MODEL_DIR_NAME: str = os.getenv("MODEL_DIR_NAME", "model")
