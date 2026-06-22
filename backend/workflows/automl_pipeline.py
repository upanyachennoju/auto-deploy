from backend.tools.dataset_tool import analyze_dataset_tool
from backend.tools.preprocessing_tool import preprocess_dataset_tool
from backend.tools.trainer_tool import train_model_tool
from backend.tools.evaluation_tool import evaluate_model_tool
from backend.tools.explainability_tool import explainability_tool
from backend.tools.report_tool import report_tool

def run_pipeline(
    session_id,
    target_column,
    task_type
):
    analyze_dataset_tool.invoke(
        {"session_id": session_id}
    )

    preprocess_dataset_tool.invoke(
        {
            "session_id": session_id,
            "target_column": target_column
        }
    )

    train_model_tool.invoke(
        {
            "session_id": session_id,
            "task_type": task_type
        }
    )

    evaluate_model_tool.invoke(
        {
            "session_id": session_id,
            "task_type": task_type
        }
    )

    explainability_tool.invoke(
        {
            "session_id": session_id
        }
    )

    return report_tool.invoke(
        {
            "session_id": session_id
        }
    )
