from langchain_core.tools import tool

from backend.storage.session_store import SessionStore
from backend.services.report_generator import generate_report


@tool
def report_tool(
    session_id: str
):
    """
    Generate final AutoML report.
    """
    session_store = SessionStore()
    dataset_analysis = session_store.get_value(
        session_id,
        "dataset_analysis"
    )

    training_results = session_store.get_value(
        session_id,
        "training_results"
    )

    evaluation_results = session_store.get_value(
        session_id,
        "evaluation_results"
    )

    feature_importance = session_store.get_value(
        session_id,
        "feature_importance"
    )

    report = generate_report(
        dataset_analysis,
        training_results,
        evaluation_results,
        feature_importance
    )

    session_store.set(
        session_id,
        "report",
        report
    )

    return report