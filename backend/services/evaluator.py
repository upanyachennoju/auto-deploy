import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

def evaluate_model(model, X_test, y_test, task_type="classification"):
    """
    Evaluates the trained model

    Args:
        model: The trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test targets.
        task_type (str, optional): Type of task (classification or regression). Defaults to "classification".

    Returns:
        Dict[str, Any]: Dictionary containing the evaluation metrics.
    """
     
    predictions = model.predict(X_test)

    if task_type == "regression" or task_type == "regressor":
        return {
            "r2": float(r2_score(y_test, predictions)),
            "mse": float(mean_squared_error(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions)))
        }
    else:
        return {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )),
            "recall": float(recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )),
            "f1": float(f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ))
        }