from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_model(model, X_test, y_test):
     
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            average="weighted"
        ),
        "recall": recall_score(
            y_test,
            predictions,
            average="weighted"
        ),
        "f1": f1_score(
            y_test,
            predictions,
            average="weighted"
        )
    }