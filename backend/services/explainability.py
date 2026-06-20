from typing import Dict
import pandas as pd
import shap


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

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            f"{type(model).__name__} does not support feature importances."
        )

    

    importance_dict = {
        feature: float(importance)
        for feature, importance in zip(
            feature_names,
            model.feature_importances_
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

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    return {
        "shap_values": shap_values
    }

import numpy as np

def get_shap_feature_importance(
    model,
    X_sample: pd.DataFrame
):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    importance = np.abs(shap_values).mean(axis=0)

    return dict(
        sorted(
            zip(X_sample.columns, importance),
            key=lambda x: x[1],
            reverse=True
        )
    )