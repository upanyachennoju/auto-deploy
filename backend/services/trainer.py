from flaml import AutoML

def train_automl(X_train, y_train, task_type, time_budget=60):
    """
    Trains an AutoML model using FLAML

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        task_type (str): Type of task (classification or regression).
        time_budget (int, optional): Time budget in seconds. Defaults to 60.

    Returns:
        Dict[str, Any]: Dictionary containing the trained model and related information.
    """

    automl = AutoML()

    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task=task_type,
        time_budget=time_budget,
        verbose=0,
    )

    return {
        "model": automl,
        "best_estimator": automl.best_estimator,
        "best_config": automl.best_config,
        "best_loss": automl.best_loss
    }