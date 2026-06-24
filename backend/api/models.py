from fastapi import APIRouter, HTTPException, status
from backend.services.model_manager import list_models, get_model, delete_model

router = APIRouter()

@router.get("/models", status_code=status.HTTP_200_OK)
async def list_models_endpoint():
    """
    Retrieves all registered models and their core metadata.
    """
    try:
        return list_models()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )

@router.get("/models/{model_id}", status_code=status.HTTP_200_OK)
async def get_model_endpoint(model_id: str):
    """
    Retrieves detailed information (schema, file size, metrics) for a specific model.
    """
    try:
        return get_model(model_id)
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model details: {str(e)}"
        )

@router.delete("/models/{model_id}", status_code=status.HTTP_200_OK)
async def delete_model_endpoint(model_id: str):
    """
    Deletes all registered artifacts associated with a model ID.
    """
    try:
        delete_model(model_id)
        return {"message": f"Model '{model_id}' deleted successfully."}
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {str(e)}"
        )
