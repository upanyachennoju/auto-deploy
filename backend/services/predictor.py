import pandas as pd
from typing import Dict, Any, Union, List
from backend.services.model_registry import load_model, load_preprocessor

def predict(
    model_path: str,
    preprocessor_path: str,
    input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Loads model and preprocessor, transforms raw incoming input data, 
    and outputs predictions and probability estimates (where applicable).

    Args:
        model_path (str): Path to the saved model file.
        preprocessor_path (str): Path to the saved preprocessor pipeline file.
        input_data (Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]): Raw input record(s).

    Returns:
        Dict[str, Any]: Predictions and probability lists.
    """
    # Load model and preprocessor
    model = load_model(model_path)
    preprocessor = load_preprocessor(preprocessor_path)
    
    # Ensure input data is formatted as a pandas DataFrame
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df_input = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise ValueError("Unsupported input data format. Must be a DataFrame, dict, or list of dicts.")

    # Transform the raw payload using the fitted ColumnTransformer pipeline
    X_processed = preprocessor.transform(df_input)
    
    # Run model prediction
    predictions = model.predict(X_processed)
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    else:
        predictions = list(predictions)
        
    # Run probability prediction (if classification and supported)
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            prob_arr = model.predict_proba(X_processed)
            if hasattr(prob_arr, "tolist"):
                probabilities = prob_arr.tolist()
            else:
                probabilities = list(prob_arr)
        except Exception:
            pass
            
    return {
        "predictions": predictions,
        "probabilities": probabilities
    }
