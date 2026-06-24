import os
import joblib
from typing import Any
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

def get_model_dir(model_id: str) -> Path:
    return ARTIFACTS_DIR / model_id

def get_model_path(model_id: str) -> Path:
    return get_model_dir(model_id) / "model.pkl"

def get_preprocessor_path(model_id: str) -> Path:
    return get_model_dir(model_id) / "preprocessor.pkl"

def get_schema_path(model_id: str) -> Path:
    return get_model_dir(model_id) / "schema.json"

def get_metadata_path(model_id: str) -> Path:
    return get_model_dir(model_id) / "metadata.json"


def save_model(model: Any, path: str):
    """
    Saves the trained model to disk using joblib.

    Args:
        model: The trained model.
        path (str): The path to save the model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str) -> Any:
    """
    Loads a trained model from disk using joblib.

    Args:
        path (str): The path to load the model from.

    Returns:
        Any: The loaded model.
    """
    return joblib.load(path)

def save_preprocessor(preprocessor: Any, path: str):
    """
    Saves the fitted preprocessor pipeline to disk using joblib.

    Args:
        preprocessor: The fitted preprocessor pipeline.
        path (str): The path to save the preprocessor.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)

def load_preprocessor(path: str) -> Any:
    """
    Loads a fitted preprocessor pipeline from disk using joblib.

    Args:
        path (str): The path to load the preprocessor from.

    Returns:
        Any: The loaded preprocessor pipeline.
    """
    return joblib.load(path)
