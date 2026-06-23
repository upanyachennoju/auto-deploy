from langchain_core.tools import tool

from backend.storage.session_store import SessionStore
from backend.services.evaluator import evaluate_model


@tool
def evaluate_model_tool(
    session_id: str,
    task_type: str
):
    """
    Evaluate the trained model.
    """
    session_store = SessionStore()
    training_results = session_store.get_value(
        session_id,
        "training_results"
    )

    model = training_results["model"]

    X_test = session_store.get_value(
        session_id,
        "X_test"
    )

    y_test = session_store.get_value(
        session_id,
        "y_test"
    )

    evaluation_results = evaluate_model(
        model,
        X_test,
        y_test,
        task_type
    )

    session_store.set(
        session_id,
        "evaluation_results",
        evaluation_results
    )

    return evaluation_results