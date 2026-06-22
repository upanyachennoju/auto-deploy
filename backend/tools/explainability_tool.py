from langchain_core.tools import tool

from backend.storage.session_store import SessionStore
from backend.services.explainability import get_top_features


@tool
def explainability_tool(
    session_id: str,
    top_n: int = 10
):
    """
    Generate feature importance report.
    """
    session_store = SessionStore()
    training_results = session_store.get_value(
        session_id,
        "training_results"
    )

    model = training_results["model"]

    X_train = session_store.get_value(
        session_id,
        "X_train"
    )

    feature_importance = get_top_features(
        model=model,
        feature_names=X_train.columns.tolist(),
        top_n=top_n
    )

    session_store.set(
        session_id,
        "feature_importance",
        feature_importance
    )

    return feature_importance