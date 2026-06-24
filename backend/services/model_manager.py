import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
from backend.services.model_registry import (
    ARTIFACTS_DIR,
    get_model_dir,
    get_model_path,
    get_preprocessor_path,
    get_schema_path,
    get_metadata_path
)

def list_models() -> List[Dict[str, Any]]:
    """
    Scans the artifacts directory and lists metadata for all registered models.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing model metadata.
    """
    if not ARTIFACTS_DIR.exists():
        return []
    
    models = []
    # Each sub-directory under ARTIFACTS_DIR represents a model_id
    for path in ARTIFACTS_DIR.iterdir():
        if path.is_dir():
            metadata_file = get_metadata_path(path.name)
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        meta = json.load(f)
                        models.append(meta)
                except Exception:
                    # Ignore malformed metadata files
                    pass
    
    # Sort models by creation time descending (newest first)
    models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return models

def get_model(model_id: str) -> Dict[str, Any]:
    """
    Retrieves detailed information for a specific model, including its metadata,
    schema, file sizes, and evaluation metrics.
    
    Args:
        model_id (str): The unique ID of the model.
        
    Returns:
        Dict[str, Any]: Unified dictionary containing model details.
        
    Raises:
        FileNotFoundError: If the model artifacts do not exist.
    """
    model_dir = get_model_dir(model_id)
    metadata_file = get_metadata_path(model_id)
    schema_file = get_schema_path(model_id)
    model_file = get_model_path(model_id)
    prep_file = get_preprocessor_path(model_id)
    
    if not model_dir.exists() or not metadata_file.exists():
        raise FileNotFoundError(f"Model with ID '{model_id}' not found.")
    
    # Load metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
        
    # Load schema
    schema = None
    if schema_file.exists():
        try:
            with open(schema_file, "r") as f:
                schema = json.load(f)
        except Exception:
            pass
            
    # Calculate file size info
    model_size = model_file.stat().st_size if model_file.exists() else 0
    prep_size = prep_file.stat().st_size if prep_file.exists() else 0
    
    model_info = {
        "model_file_size_bytes": model_size,
        "preprocessor_file_size_bytes": prep_size,
        "model_path": str(model_file),
        "preprocessor_path": str(prep_file)
    }
    
    return {
        "metadata": metadata,
        "schema": schema,
        "model_info": model_info,
        "evaluation_summary": metadata.get("metrics", {})
    }

def delete_model(model_id: str) -> None:
    """
    Deletes all artifacts associated with the specified model_id.
    
    Args:
        model_id (str): The unique ID of the model.
        
    Raises:
        FileNotFoundError: If the model directory does not exist.
    """
    model_dir = get_model_dir(model_id)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model with ID '{model_id}' not found.")
        
    shutil.rmtree(model_dir)
