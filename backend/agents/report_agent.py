import os
import json
import sys
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

class AIReport(BaseModel):
    executive_summary: str = Field(description="Business-friendly executive summary explaining the purpose and main outcomes of the AutoML process.")
    dataset_analysis: str = Field(description="Analysis of the dataset characteristics, shapes, and columns.")
    data_quality_analysis: str = Field(description="Analysis of data quality issues, duplicates, missing values, cardinality.")
    model_selection_reasoning: str = Field(description="Detailed explanation of why FLAML selected this winning estimator, tradeoffs between accuracy and complexity, and characteristics influencing the choice.")
    performance_analysis: str = Field(description="Interpretation of evaluation metrics, both in technical terms and non-technical language.")
    feature_analysis: str = Field(description="Detailed feature importance analysis, indicating which features drive the model's predictions.")
    deployment_recommendation: str = Field(description="Clear recommendation on whether the model is ready to deploy based on metrics.")
    next_steps: str = Field(description="Concrete action items for technical and non-technical stakeholders.")

def generate_fallback_report(
    dataset_summary: dict,
    model_summary: dict,
    evaluation_summary: dict,
    feature_importance: dict,
    task_type: str
) -> dict:
    """
    Constructs a deterministic, highly detailed fallback report when Groq is unavailable.
    Does not hallucinate metrics and uses actual dataset and model results.
    """
    rows = dataset_summary.get("shape", {}).get("rows", "N/A")
    cols = dataset_summary.get("shape", {}).get("columns", "N/A")
    
    missing_vals = 0
    missing_raw = dataset_summary.get("missing_values")
    if isinstance(missing_raw, dict):
        missing_vals = sum(missing_raw.values())
    elif isinstance(missing_raw, int):
        missing_vals = missing_raw
        
    dup_rows = dataset_summary.get("duplicate_rows", 0)
    best_est = model_summary.get("best_estimator", "N/A")
    best_loss = model_summary.get("best_loss", "N/A")
    
    # Format evaluation metrics
    metrics_str = ", ".join([f"{k.upper()}: {v:.4f}" if isinstance(v, float) else f"{k.upper()}: {v}" for k, v in evaluation_summary.items()])
    
    # Format top features
    top_feats = ", ".join([f"'{k}' ({v:.4f})" if isinstance(v, float) else f"'{k}' ({v})" for k, v in feature_importance.items()])
    
    # Model selection explanation logic
    # Estimate complexity vs accuracy tradeoff based on model type
    if "xgb" in str(best_est).lower() or "lgb" in str(best_est).lower():
        tradeoff_desc = (
            f"The chosen ensemble model ({best_est}) offers superior accuracy/loss optimization "
            f"by constructing boosting trees that iteratively correct errors. However, this comes with "
            f"higher computational complexity and slightly longer inference latency compared to simple models."
        )
    elif "rf" in str(best_est).lower() or "extra" in str(best_est).lower():
        tradeoff_desc = (
            f"The selected bagging ensemble model ({best_est}) achieves high stability and generalizes well "
            f"by averaging predictions across multiple decision trees, mitigating overfitting. It has moderate "
            f"inference complexity."
        )
    else:
        tradeoff_desc = (
            f"The selected model ({best_est}) balances performance and execution speed. It represents "
            f"the optimal choice found by FLAML's optimization search within the budget."
        )
        
    # Deployment readiness recommendation
    ready = "Ready for deployment"
    if task_type == "classification":
        acc = evaluation_summary.get("accuracy", 1.0)
        f1 = evaluation_summary.get("f1", 1.0)
        if acc < 0.6 or f1 < 0.6:
            ready = "Not recommended for direct production deployment without further improvements"
    else:
        r2 = evaluation_summary.get("r2", 1.0)
        if r2 < 0.5:
            ready = "Not recommended for direct production deployment without further tuning"

    return {
        "executive_summary": (
            f"An AutoML training pipeline was executed to solve a {task_type} task. "
            f"The search space exploration evaluated multiple candidate estimators, resulting in the selection "
            f"of '{best_est}' as the champion model. The champion model achieved a training search loss of {best_loss} "
            f"and demonstrates solid generalization capability on the evaluation subset."
        ),
        "dataset_analysis": (
            f"The dataset profiles {rows} rows across {cols} features. "
            f"It is composed of {len(dataset_summary.get('numerical_columns', []))} numerical fields and "
            f"{len(dataset_summary.get('categorical_columns', []))} categorical fields. "
            f"The target analysis was automatically aligned with the task type of '{task_type}'."
        ),
        "data_quality_analysis": (
            f"Data profiling detected a total of {missing_vals} missing entries across all features. "
            f"Additionally, {dup_rows} duplicate rows were identified in the source dataset. "
            f"Identifier and high-cardinality features were dynamically excluded/processed to prevent model memorization."
        ),
        "model_selection_reasoning": (
            f"FLAML successfully selected '{best_est}' as the champion estimator. "
            f"The decision was driven by the validation performance where it minimized overall error/loss. "
            f"{tradeoff_desc} The selection was tailored to the dataset scale ({rows} samples) and feature structure."
        ),
        "performance_analysis": (
            f"Evaluation on the holdout partition produced the following metrics: {metrics_str}. "
            f"These results indicate that the model exhibits stable performance, and the balanced scoring "
            f"suggests robust generalization without significant overfitting."
        ),
        "feature_analysis": (
            f"Feature importance analysis identifies the top drivers of predictions as: {top_feats}. "
            f"These features represent the primary variables influencing the model output, and should be the "
            f"focus of business interpretations and domain analysis."
        ),
        "deployment_recommendation": (
            f"{ready}. The model meets the basic validation criteria with a structured schema configured for inputs. "
            f"Inference latencies are expected to be within normal operational bounds."
        ),
        "next_steps": (
            "1. Deploy model to a staging environment and validate using live traffic shadow testing.\n"
            "2. Establish continuous monitoring on inputs and predictions to detect data drift.\n"
            "3. If performance needs improvement, consider collecting additional historical samples or engineering extra interaction features."
        )
    }

def generate_report_agent(
    dataset_summary: dict,
    model_summary: dict,
    evaluation_summary: dict,
    feature_importance: dict,
    task_type: str
) -> dict:
    """
    Builds prompts, calls Groq LLM (llama-3.3-70b-versatile), and parses the response.
    Falls back to a detailed deterministic report if Groq is unavailable.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return generate_fallback_report(
            dataset_summary, model_summary, evaluation_summary, feature_importance, task_type
        )

    try:
        # Instantiate LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.2
        )
        
        system_prompt = (
            "You are an expert AI report generator for an AutoML platform. "
            "You write highly professional, business-friendly, and technical reports summarizing dataset, model, and performance characteristics. "
            "Do not fabricate or hallucinate any metrics or feature names. Use only the provided real information. "
            "Explain model selection reasoning using actual FLAML concepts (validation loss, search trials, tradeoffs between model complexity and accuracy/loss). "
            "Ensure the output conforms exactly to the requested JSON schema."
        )
        
        user_prompt = f"""
Generate a structured dataset and model report based on the following pipeline outputs.

Task Type: {task_type}

Dataset Summary:
{json.dumps(dataset_summary, indent=2)}

Model Summary:
{json.dumps(model_summary, indent=2)}

Evaluation Summary:
{json.dumps(evaluation_summary, indent=2)}

Feature Importance:
{json.dumps(feature_importance, indent=2)}

Please write comprehensive, detailed descriptions for each of the fields in the requested schema.
"""
        
        # Invoke structured model
        structured_llm = llm.with_structured_output(AIReport)
        response = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        return response.model_dump()
        
    except Exception as e:
        print(f"Warning: Groq API call failed: {e}. Using deterministic fallback generator.", file=sys.stderr)
        return generate_fallback_report(
            dataset_summary, model_summary, evaluation_summary, feature_importance, task_type
        )
