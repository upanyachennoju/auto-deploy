from typing import Dict, Any


def generate_report(
    dataset_analysis: Dict[str, Any],
    training_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    feature_importance: Dict[str, float]
) -> Dict[str, Any]:
    """
    Generate a unified report from all pipeline stages.
    """

    report = {
        "dataset_summary": {
            "shape": dataset_analysis.get("shape"),
            "columns": dataset_analysis.get("columns"),
            "numerical_columns": dataset_analysis.get(
                "numerical_columns"
            ),
            "categorical_columns": dataset_analysis.get(
                "categorical_columns"
            ),
            "missing_values": dataset_analysis.get(
                "missing_values"
            ),
            "duplicate_rows": dataset_analysis.get(
                "duplicate_rows"
            ),
            "possible_target_columns": dataset_analysis.get(
                "possible_target_columns"
            ),
        },

        "model_summary": {
            "best_estimator": training_results.get(
                "best_estimator"
            ),
            "best_config": training_results.get(
                "best_config"
            ),
            "best_loss": training_results.get(
                "best_loss"
            ),
        },

        "evaluation_summary": evaluation_results,

        "feature_importance": feature_importance,
    }

    return report