from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, List, Union
import pandas as pd
from backend.services.model_registry import (
    get_model_path,
    get_preprocessor_path,
    get_schema_path
)
from backend.services.schema import load_schema, validate_payload
from backend.services.predictor import predict

router = APIRouter()

class PredictRequest(BaseModel):
    model_id: str
    data: Union[Dict[str, Any], List[Dict[str, Any]]]

@router.post("/predict", status_code=status.HTTP_200_OK)
async def predict_endpoint(req: PredictRequest):
    """
    Executes inference for a single record or a batch of records.
    Validates data structure and types against model schema before execution.
    """
    model_id = req.model_id
    
    # 1. Resolve paths
    schema_path = get_schema_path(model_id)
    model_path = get_model_path(model_id)
    preprocessor_path = get_preprocessor_path(model_id)
    
    if not schema_path.exists() or not model_path.exists() or not preprocessor_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model artifacts for ID '{model_id}' not found."
        )
    
    try:
        # 2. Load schema
        schema = load_schema(str(schema_path))
        
        # 3. Format input to DataFrame
        is_batch = isinstance(req.data, list)
        if is_batch:
            df_input = pd.DataFrame(req.data)
        else:
            df_input = pd.DataFrame([req.data])
        
        # 4. Schema validation
        validation = validate_payload(df_input, schema)
        if not validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"errors": validation["errors"]}
            )
        
        # 5. Execute model inference
        pred_res = predict(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
            input_data=df_input
        )
        
        predictions = pred_res.get("predictions", [])
        probabilities = pred_res.get("probabilities")
        
        # 6. Format response structure depending on input type (single vs batch)
        if is_batch:
            return {
                "predictions": predictions,
                "probabilities": probabilities
            }
        else:
            prediction = predictions[0] if len(predictions) > 0 else None
            probability = probabilities[0] if probabilities and len(probabilities) > 0 else None
            return {
                "prediction": prediction,
                "probability": probability
            }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )
