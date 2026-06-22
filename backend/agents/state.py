from typing import TypedDict, Any


class AutoMLState(TypedDict):
    user_query: str

    file_path: str

    dataset_analysis: dict

    X_train: Any
    X_test: Any

    y_train: Any
    y_test: Any

    training_results: dict

    evaluation_results: dict

    feature_importance: dict

    report: dict

    response: str