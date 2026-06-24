from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from pathlib import Path
from backend.storage.session_store import SessionStore
from backend.services.analyser import analyse_dataset
from backend.workflows.automl_pipeline import run_pipeline

router = APIRouter()

class TrainRequest(BaseModel):
    session_id: str
    target_column: str

@router.post("/train", status_code=status.HTTP_200_OK)
async def train_model_endpoint(req: TrainRequest):
    """
    Triggers model training pipeline for an active session.
    Automatically profiles the dataset to detect target task,
    runs preprocessing and AutoML training, and returns summaries.
    """
    session_store = SessionStore()
    
    # 1. Validate session and retrieve file_path
    try:
        file_path_str = session_store.get_value(req.session_id, "file_path")
    except (ValueError, KeyError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Active session '{req.session_id}' not found or has no uploaded file."
        )
    
    file_path = Path(file_path_str)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Uploaded file does not exist on disk: {file_path_str}"
        )
    
    try:
        # 2. Analyze dataset to identify task type
        analysis = analyse_dataset(file_path)
        session_store.set(req.session_id, "dataset_analysis", analysis)
        task_type = analysis.get("recommended_task", "classification")
        
        # 3. Run the AutoML pipeline
        model_id = run_pipeline(
            session_id=req.session_id,
            target_column=req.target_column,
            task_type=task_type
        )
        
        # 4. Fetch the report from session
        report = session_store.get_value(req.session_id, "report")
        
        # 5. Build and return the response
        return {
            "session_id": req.session_id,
            "model_id": model_id,
            "dataset_summary": report.get("dataset_summary"),
            "model_summary": report.get("model_summary"),
            "evaluation_summary": report.get("evaluation_summary")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training pipeline execution failed: {str(e)}"
        )
