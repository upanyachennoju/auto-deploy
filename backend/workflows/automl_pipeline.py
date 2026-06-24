import uuid
import datetime
import json
import os
from backend.storage.session_store import SessionStore
from backend.services.model_registry import (
    get_metadata_path,
    get_dataset_summary_path,
    get_feature_importance_path,
    get_model_summary_path
)
from backend.tools.dataset_tool import analyze_dataset_tool
from backend.tools.preprocessing_tool import preprocess_dataset_tool
from backend.tools.trainer_tool import train_model_tool
from backend.tools.evaluation_tool import evaluate_model_tool
from backend.tools.explainability_tool import explainability_tool
from backend.tools.report_tool import report_tool

def run_pipeline(
    session_id: str,
    target_column: str,
    task_type: str
) -> str:
    session_store = SessionStore()
    model_id = str(uuid.uuid4())
    session_store.set(session_id, "model_id", model_id)

    analyze_dataset_tool.invoke(
        {"session_id": session_id}
    )

    preprocess_dataset_tool.invoke(
        {
            "session_id": session_id,
            "target_column": target_column
        }
    )

    train_model_tool.invoke(
        {
            "session_id": session_id,
            "task_type": task_type
        }
    )

    evaluate_model_tool.invoke(
        {
            "session_id": session_id,
            "task_type": task_type
        }
    )

    explainability_tool.invoke(
        {
            "session_id": session_id
        }
    )

    report_tool.invoke(
        {
            "session_id": session_id
        }
    )

    # Persistence of metadata
    training_results = session_store.get_value(session_id, "training_results")
    evaluation_results = session_store.get_value(session_id, "evaluation_results")
    
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "model_id": model_id,
        "created_at": created_at,
        "target_column": target_column,
        "task_type": task_type,
        "best_estimator": training_results.get("best_estimator"),
        "metrics": evaluation_results
    }
    
    meta_path = get_metadata_path(model_id)
    os.makedirs(meta_path.parent, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save dataset summary, feature importance, and model summary to disk
    report = session_store.get_value(session_id, "report")
    
    dataset_summary_path = get_dataset_summary_path(model_id)
    feature_importance_path = get_feature_importance_path(model_id)
    model_summary_path = get_model_summary_path(model_id)
    
    with open(dataset_summary_path, "w") as f:
        json.dump(report.get("dataset_summary", {}), f, indent=2)
        
    with open(feature_importance_path, "w") as f:
        json.dump(report.get("feature_importance", {}), f, indent=2)
        
    with open(model_summary_path, "w") as f:
        json.dump(report.get("model_summary", {}), f, indent=2)

    return model_id
