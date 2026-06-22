from langchain_core.tools import tool

from backend.storage.session_store import SessionStore
from backend.services.preprocessor import preprocess_dataset

@tool
def preprocess_dataset_tool(
    session_id: str,
    target_column: str
):
    """
    Preprocess the dataset using scikit-learn pipelines for classification. 
    """
    session_store = SessionStore()
    file_path = session_store.get_value(
        session_id,
        "file_path"
    )

    X_train, X_test, y_train, y_test = preprocess_dataset(
        file_path=file_path,
        target_column=target_column
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

    return {
        "status": "preprocessing completed"
    }