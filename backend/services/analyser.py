import numpy as np
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import re


def make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def analyse_dataset(file_path: Path)-> Dict[str, Any]:
    """
    Returns a descriptive JSON report of the dataset

    Args:
        file_path (Path): The path to the dataset file.

    Returns:
        Dict[str, Any]: A dictionary containing the analysis results.
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

    row_count = max(len(df), 1)

    # Identifier and high-cardinality detection

    identifier_columns = []
    high_cardinality_columns = []

    ID_KEYWORDS = {
        "id",
        "uuid",
        "guid",
        "email",
        "phone",
        "mobile",
        "account",
        "customer",
        "member",
        "transaction",
        "user"
    }

    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)
        ratio = unique_count / row_count

        col_lower = col.lower()

        has_id_keyword = any(
            keyword in col_lower
            for keyword in ID_KEYWORDS
        )

        is_object = pd.api.types.is_object_dtype(df[col])

        # Likely identifier
        if (
            (has_id_keyword and ratio > 0.50)
            or
            (is_object and ratio > 0.95)
        ):
            identifier_columns.append(col)

        # High-cardinality categorical
        elif (
            is_object
            and unique_count > 20
        ):
            high_cardinality_columns.append(col)

    suggested_targets = [col for col in df.columns if df[col].nunique() < 20]
    
    # Choose default target (prioritising target names or last column)
    default_target = None
    for name in ["churn", "target", "label", "y", "class"]:
        matching = [c for c in df.columns if name in c.lower()]
        if matching:
            default_target = matching[0]
            break
    if default_target is None and len(df.columns) > 0:
        default_target = df.columns[-1]
        
    recommended_task = "classification"
    if default_target is not None:
        y = df[default_target]
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            recommended_task = "regression"

    target_distributions = {}
    for col in suggested_targets:
        dist = df[col].value_counts(normalize=True).round(4)
        target_distributions[col] = {
            str(k): float(v) for k, v in dist.items()
        }
    
    analysis = {
        "columns": list(df.columns),
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(v) for col, v in df.isnull().sum().items()},
        "missing_percentage": {col: float(v) for col, v in (df.isnull().sum() / row_count * 100).items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_percentage": float(df.duplicated().sum() / row_count * 100),
        "numerical_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "possible_target_columns": suggested_targets,
        "identifier_columns": identifier_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "suggested_targets": suggested_targets,
        "recommended_task": recommended_task,
        "target_distributions": target_distributions,
        "preview": df.head(5).to_dict(orient="records")
    }
    
    return make_serializable(analysis)