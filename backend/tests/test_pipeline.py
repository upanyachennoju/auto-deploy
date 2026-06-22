from backend.storage.session_store import SessionStore
from backend.workflows.automl_pipeline import run_pipeline

session_store = SessionStore()
session_id = session_store.create_session()

session_store.set(
    session_id,
    "file_path",
    "backend/tests/churn.csv"
)

report = run_pipeline(
    session_id=session_id,
    target_column="Churn",
    task_type="classification"
)

print(report)