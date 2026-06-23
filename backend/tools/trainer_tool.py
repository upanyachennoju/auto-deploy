from langchain_core.tools import tool
from backend.storage.session_store import SessionStore
from backend.services.trainer import train_automl

@tool
def train_model_tool(
    session_id: str,
    task_type: str
):
    """
    Train the best machine learning model using Flaml AutoML.

    Args:
        session_id (str): Session ID
        task_type (str): Task type (e.g., 'classification', 'regression')

    Returns:
        Dict[str, Any]: Dictionary containing training results.
    """
    session_store = SessionStore()
    X_train = session_store.get_value(
        session_id,
        "X_train"
    )

    y_train = session_store.get_value(
        session_id,
        "y_train"
    )

    training_results = train_automl(
        X_train,
        y_train,
        task_type
    )

    session_store.set(
        session_id,
        "training_results",
        training_results
    )

    return {
        "best_estimator":
            training_results["best_estimator"]
    }