from langchain_core.tools import tool

from backend.storage.session_store import SessionStore
from backend.services.analyser import analyse_dataset


@tool
def analyze_dataset_tool(
    session_id: str
):
    """
    Analyze uploaded dataset.
    """

    file_path = SessionStore().get_value(
        session_id,
        "file_path"
    )

    analysis = analyse_dataset(file_path)

    SessionStore().set(
        session_id,
        "dataset_analysis",
        analysis
    )

    return analysis