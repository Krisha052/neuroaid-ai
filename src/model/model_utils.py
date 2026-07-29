from functools import lru_cache
from pathlib import Path

import joblib


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

@lru_cache(maxsize=4)
def load_model(path: Path):
    """Cached so the API doesn't hit disk/deserialize on every request."""
    return joblib.load(path)
