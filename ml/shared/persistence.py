import pathlib

import joblib


def save_model(model, path):
    """Save model to the given path using joblib."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path):
    """Load model from the given path using joblib."""
    return joblib.load(path)
