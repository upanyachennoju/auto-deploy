import numpy as np
import pandas as pd
import shap
from typing import Dict, Any
from flaml import AutoML

def _resolve_model_and_features(model, X_sample: pd.DataFrame):
    if isinstance(model, AutoML):
        if hasattr(model, "feature_transformer") and model.feature_transformer is not None:
            try:
                X_sample = model.feature_transformer.transform(X_sample)
            except Exception:
                pass
        estimator = model.model
    else:
        estimator = model
    
    if hasattr(estimator, "estimator") and estimator.estimator is not None:
        underlying_model = estimator.estimator
    else:
        underlying_model = estimator

    return underlying_model, X_sample

def get_feature_importance(
    model,
    feature_names
) -> Dict[str, float]:
    """
    Extract feature importance from supported models.

    Args:
        model: Trained model
        feature_names: List of feature names

    Returns:
        Dictionary of feature importances sorted descending.
    """
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        feature_names = list(model.feature_names_in_)
        
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        estimator = getattr(model, "model", None)
        if estimator is not None and hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif estimator is not None and hasattr(estimator, "estimator"):
            underlying = estimator.estimator
            if hasattr(underlying, "feature_importances_"):
                importances = underlying.feature_importances_
            elif hasattr(underlying, "coef_"):
                importances = np.abs(underlying.coef_).mean(axis=0) if len(underlying.coef_.shape) > 1 else np.abs(underlying.coef_)
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)

    if importances is None:
        importances = np.zeros(len(feature_names))

    importance_dict = {
        str(feature): float(importance)
        for feature, importance in zip(
            feature_names,
            importances
        )
    }

    importance_dict = dict(
        sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    return importance_dict


def get_top_features(
    model,
    feature_names,
    top_n: int = 10
):
    """
    Return top N most important features.
    """
    importance_dict = get_feature_importance(
        model,
        feature_names
    )

    return dict(
        list(importance_dict.items())[:top_n]
    )


def generate_shap_explanation(
    model,
    X_sample: pd.DataFrame
):
    """
    Generate SHAP values for a sample dataset.
    """
    underlying_model, X_transformed = _resolve_model_and_features(model, X_sample)
    
    try:
        explainer = shap.TreeExplainer(underlying_model)
        shap_values = explainer.shap_values(X_transformed)
    except Exception:
        try:
            explainer = shap.Explainer(underlying_model, X_transformed)
            shap_values = explainer(X_transformed)
            if hasattr(shap_values, "values"):
                shap_values = shap_values.values
        except Exception:
            try:
                explainer = shap.LinearExplainer(underlying_model, X_transformed)
                shap_values = explainer.shap_values(X_transformed)
            except Exception:
                shap_values = np.zeros(X_transformed.shape)

    if isinstance(shap_values, list):
        shap_values_serialized = [sv.tolist() if isinstance(sv, np.ndarray) else sv for sv in shap_values]
    elif isinstance(shap_values, np.ndarray):
        shap_values_serialized = shap_values.tolist()
    else:
        if hasattr(shap_values, "tolist"):
            shap_values_serialized = shap_values.tolist()
        else:
            shap_values_serialized = list(shap_values)

    return {
        "shap_values": shap_values_serialized
    }


def get_shap_feature_importance(
    model,
    X_sample: pd.DataFrame
):
    underlying_model, X_transformed = _resolve_model_and_features(model, X_sample)
    
    explanation = generate_shap_explanation(model, X_sample)
    shap_values = explanation["shap_values"]

    if isinstance(shap_values, list):
        if len(shap_values) > 0 and isinstance(shap_values[0], list):
            shap_values_arr = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            shap_values_arr = np.array(shap_values)
    else:
        shap_values_arr = np.array(shap_values)

    importance = np.abs(shap_values_arr).mean(axis=0)

    importance_dict = {
        str(col): float(val)
        for col, val in zip(X_transformed.columns, importance)
    }

    return dict(
        sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )