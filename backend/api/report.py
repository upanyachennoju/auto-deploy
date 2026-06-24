from fastapi import APIRouter, HTTPException, status
import json
from backend.services.model_registry import (
    get_metadata_path,
    get_dataset_summary_path,
    get_feature_importance_path,
    get_model_summary_path
)
from backend.services.ai_report_generator import generate_ai_report

router = APIRouter()

@router.get("/report/{model_id}", status_code=status.HTTP_200_OK)
async def get_report_endpoint(model_id: str):
    """
    Retrieves the AI-generated insight report for a given model ID.
    Loads and runs the report generator using the model's persisted metadata and analysis results.
    """
    meta_path = get_metadata_path(model_id)
    if not meta_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model metadata for ID '{model_id}' not found."
        )

    try:
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
        task_type = metadata.get("task_type", "classification")
        evaluation_summary = metadata.get("metrics", {})
        
        # Load dataset summary
        dataset_summary_path = get_dataset_summary_path(model_id)
        if dataset_summary_path.exists():
            with open(dataset_summary_path, "r") as f:
                dataset_summary = json.load(f)
        else:
            dataset_summary = {
                "shape": {"rows": "N/A", "columns": "N/A"},
                "numerical_columns": [],
                "categorical_columns": [],
                "missing_values": {},
                "duplicate_rows": 0,
                "possible_target_columns": [metadata.get("target_column")] if metadata.get("target_column") else []
            }

        # Load feature importance
        feature_importance_path = get_feature_importance_path(model_id)
        if feature_importance_path.exists():
            with open(feature_importance_path, "r") as f:
                feature_importance = json.load(f)
        else:
            feature_importance = {}

        # Load model summary
        model_summary_path = get_model_summary_path(model_id)
        if model_summary_path.exists():
            with open(model_summary_path, "r") as f:
                model_summary = json.load(f)
        else:
            model_summary = {
                "best_estimator": metadata.get("best_estimator", "N/A"),
                "best_loss": 0.0
            }

        # Generate report
        report = generate_ai_report(
            dataset_summary=dataset_summary,
            model_summary=model_summary,
            evaluation_summary=evaluation_summary,
            feature_importance=feature_importance,
            task_type=task_type,
            model_id=model_id
        )
        
        return report

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate AI report: {str(e)}"
        )
