from flaml import AutoML

def train_automl(X_train, y_train, task_type, time_budget=60):

    automl = AutoML()

    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task=task_type,
        time_budget=time_budget,
        verbose=0
    )

    return {
        "model": automl.model,
        "best_estimator": automl.best_estimator,
        "best_config": automl.best_config,
        "best_loss": automl.best_loss
    }

