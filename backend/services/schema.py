import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any

def build_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build schema definition from a training dataframe.
    Records feature names and numeric vs non-numeric traits.

    Args:
        df (pd.DataFrame): Training feature dataframe.

    Returns:
        Dict[str, Any]: Schema dictionary mapping columns to expectations.
    """
    schema = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        is_numeric = bool(pd.api.types.is_numeric_dtype(df[col]))
        schema[col] = {
            "dtype": dtype,
            "is_numeric": is_numeric
        }
    return schema

def validate_payload(payload: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate inference payload dataframe against the pre-built training schema.
    Detects:
    - Missing columns
    - Extra columns
    - Datatype mismatches (numeric vs non-numeric)

    Args:
        payload (pd.DataFrame): Input dataframe for inference.
        schema (Dict[str, Any]): Loaded training schema.

    Returns:
        Dict[str, Any]: Dict with validation status and error list.
    """
    errors = []
    
    # Missing columns (in schema but not in payload)
    for col in schema.keys():
        if col not in payload.columns:
            errors.append(f"Missing column: '{col}'")
            
    # Extra columns (in payload but not in schema)
    for col in payload.columns:
        if col not in schema:
            errors.append(f"Extra column: '{col}'")
            
    # Datatype mismatches
    for col in payload.columns:
        if col in schema:
            is_numeric_expected = schema[col]["is_numeric"]
            is_numeric_actual = bool(pd.api.types.is_numeric_dtype(payload[col]))
            
            if is_numeric_expected and not is_numeric_actual:
                errors.append(
                    f"Datatype mismatch in column '{col}': Expected numeric type, got '{payload[col].dtype}'"
                )
            elif not is_numeric_expected and is_numeric_actual:
                errors.append(
                    f"Datatype mismatch in column '{col}': Expected categorical/non-numeric type, got '{payload[col].dtype}'"
                )
                
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def save_schema(schema: Dict[str, Any], path: str):
    """
    Saves the schema as a JSON file to disk.

    Args:
        schema (Dict[str, Any]): The schema to save.
        path (str): The path to save the schema to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)

def load_schema(path: str) -> Dict[str, Any]:
    """
    Loads the schema from a JSON file.
    
    Args:
        path (str): The path to load the schema from.
    
    Returns:
        Dict[str, Any]: The loaded schema.
    """
    with open(path, "r") as f:
        return json.load(f)
