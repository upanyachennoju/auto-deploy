import json
import os
from typing import Dict, Any, Optional
from backend.services.model_registry import get_ai_report_path
from backend.agents.report_agent import generate_report_agent

def generate_ai_report(
    dataset_summary: Dict[str, Any],
    model_summary: Dict[str, Any],
    evaluation_summary: Dict[str, Any],
    feature_importance: Dict[str, float],
    task_type: str,
    model_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Coordinates AI report generation with artifact caching.
    
    If model_id is provided and report.json already exists under the model's directory,
    it is loaded from disk. Otherwise, it calls the report agent to generate a new report
    and persists it if model_id is provided.
    
    Args:
        dataset_summary (Dict[str, Any]): Dataset profiling summary.
        model_summary (Dict[str, Any]): AutoML model details.
        evaluation_summary (Dict[str, Any]): Model evaluation metrics.
        feature_importance (Dict[str, float]): Map of features to importances.
        task_type (str): Task type ('classification' or 'regression').
        model_id (Optional[str]): Unique model ID for caching.
        
    Returns:
        Dict[str, Any]: AI-generated report dictionary matching the structured schema.
    """
    if model_id:
        report_path = get_ai_report_path(model_id)
        if report_path.exists():
            try:
                with open(report_path, "r") as f:
                    return json.load(f)
            except Exception:
                # If cached report is malformed, fall back to regeneration
                pass

    # Call Groq/Fallback agent
    report = generate_report_agent(
        dataset_summary=dataset_summary,
        model_summary=model_summary,
        evaluation_summary=evaluation_summary,
        feature_importance=feature_importance,
        task_type=task_type
    )

    # Persist if model_id is given
    if model_id:
        report_path = get_ai_report_path(model_id)
        os.makedirs(report_path.parent, exist_ok=True)
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass

    return report
