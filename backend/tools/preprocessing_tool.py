from langchain_core.tools import tool

from backend.storage.session_store import SessionStore
from backend.services.preprocessor import preprocess_dataset
import pandas as pd
from backend.services.schema import build_schema
from pathlib import Path

@tool
def preprocess_dataset_tool(
    session_id: str,
    target_column: str
):
    """
    Preprocess the dataset using scikit-learn pipelines for classification. 

    Args:
        session_id (str): Session ID
        target_column (str): Target column name

    Returns:
        Dict[str, str]: Dictionary with status "preprocessing completed"
    """
    session_store = SessionStore()
    file_path = session_store.get_value(
        session_id,
        "file_path"
    )

    X_train, X_test, y_train, y_test, preprocessor = preprocess_dataset(
        file_path=file_path,
        target_column=target_column
    )

    session_store.set(
        session_id,
        "preprocessor",
        preprocessor
    )

    session_store.set(
        session_id,
        "X_train",
        X_train
    )

    session_store.set(
        session_id,
        "X_test",
        X_test
    )

    session_store.set(
        session_id,
        "y_train",
        y_train
    )

    session_store.set(
        session_id,
        "y_test",
        y_test
    )

    # Build schema from raw inputs (excluding target column)
    
    file_path_path = Path(file_path)
    if file_path_path.suffix.lower() == ".csv":
        df_raw = pd.read_csv(file_path_path)
    elif file_path_path.suffix.lower() == ".xlsx":
        df_raw = pd.read_excel(file_path_path)
    else:
        raise ValueError("Unsupported file format for schema building")
        
    df_raw = df_raw.drop_duplicates()
    df_raw = df_raw.dropna(subset=[target_column])
    X_raw = df_raw.drop(columns=[target_column])
    schema = build_schema(X_raw)
    session_store.set(session_id, "schema", schema)

    # Save fitted preprocessor to model-specific directory
    from backend.services.model_registry import save_preprocessor, get_preprocessor_path
    model_id = session_store.get_value(session_id, "model_id")
    prep_path = get_preprocessor_path(model_id)
    save_preprocessor(preprocessor, str(prep_path))

    return {
        "status": "preprocessing completed"
    }