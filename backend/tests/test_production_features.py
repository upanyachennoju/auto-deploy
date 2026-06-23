import os
import shutil
import pandas as pd
from pathlib import Path
from backend.services.analyser import analyse_dataset
from backend.services.preprocessor import preprocess_dataset
from backend.services.trainer import train_automl
from backend.services.model_registry import (
    save_model,
    load_model,
    save_preprocessor,
    load_preprocessor
)
from backend.services.predictor import predict
from backend.services.schema import (
    build_schema,
    validate_payload,
    save_schema,
    load_schema
)
from backend.services.explainability import (
    get_top_features,
    generate_shap_explanation
)

def run_production_flow():
    print("=== STARTING PRODUCTION AUTOML WORKFLOW TEST ===")
    
    file_path = "backend/tests/churn.csv"
    target_column = "Churn"
    temp_dir = "backend/tests/temp_models"
    
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 1. Dataset Profiling
        print("\n1. Profiling dataset...")
        profile = analyse_dataset(Path(file_path))
        
        print("   Identifier columns found:", profile.get("identifier_columns"))
        print("   High-cardinality columns found:", profile.get("high_cardinality_columns"))
        print("   Suggested targets count:", len(profile.get("suggested_targets", [])))
        print("   Recommended task:", profile.get("recommended_task"))
        
        assert "identifier_columns" in profile
        assert "high_cardinality_columns" in profile
        assert "suggested_targets" in profile
        assert "recommended_task" in profile
        
        # 2. Preprocessing (with ColumnTransformer Pipeline)
        print("\n2. Running Preprocessing Pipeline...")
        X_train, X_test, y_train, y_test, preprocessor = preprocess_dataset(
            Path(file_path),
            target_column
        )
        print("   X_train shape:", X_train.shape)
        print("   X_test shape:", X_test.shape)
        
        # 3. Train Model
        print("\n3. Training AutoML model...")
        train_results = train_automl(X_train, y_train, profile["recommended_task"], time_budget=15)
        model = train_results["model"]
        print("   Best estimator:", train_results["best_estimator"])
        print("   Best loss:", train_results["best_loss"])
        
        # 4. Save Model + Preprocessor + Schema
        print("\n4. Saving assets to model registry...")
        model_path = os.path.join(temp_dir, "model.pkl")
        prep_path = os.path.join(temp_dir, "preprocessor.pkl")
        schema_path = os.path.join(temp_dir, "schema.json")
        
        save_model(model, model_path)
        save_preprocessor(preprocessor, prep_path)
        
        # Read the raw inputs (excluding target) for building validation schema
        raw_df = pd.read_csv(file_path)
        # Drop rows where target is missing
        raw_df = raw_df.dropna(subset=[target_column])
        X_raw = raw_df.drop(columns=[target_column])
        
        schema = build_schema(X_raw)
        save_schema(schema, schema_path)
        print("   Assets saved successfully to", temp_dir)
        
        # 5. Load Model + Preprocessor + Schema
        print("\n5. Loading assets from model registry...")
        loaded_model = load_model(model_path)
        loaded_prep = load_preprocessor(prep_path)
        loaded_schema = load_schema(schema_path)
        print("   Assets loaded successfully.")
        
        # 6. Deployment Validation (Schemas)
        print("\n6. Running Schema Validation Tests...")
        # Get one raw sample for inference
        raw_sample = X_raw.head(2).copy()
        
        # Validate correct payload
        val_res = validate_payload(raw_sample, loaded_schema)
        print("   Correct payload validation status:", val_res["valid"])
        assert val_res["valid"] is True
        
        # Validate missing columns
        missing_payload = raw_sample.drop(columns=["Day Mins"])
        val_res_missing = validate_payload(missing_payload, loaded_schema)
        print("   Missing column payload validation status:", val_res_missing["valid"])
        print("   Errors caught:", val_res_missing["errors"])
        assert val_res_missing["valid"] is False
        assert any("Day Mins" in err for err in val_res_missing["errors"])
        
        # Validate extra columns
        extra_payload = raw_sample.copy()
        extra_payload["Extra_Column"] = [9.9, 8.8]
        val_res_extra = validate_payload(extra_payload, loaded_schema)
        print("   Extra column payload validation status:", val_res_extra["valid"])
        print("   Errors caught:", val_res_extra["errors"])
        assert val_res_extra["valid"] is False
        
        # Validate datatype mismatches
        mismatch_payload = raw_sample.copy()
        mismatch_payload["Day Mins"] = ["Not a float", "Another string"]
        val_res_mismatch = validate_payload(mismatch_payload, loaded_schema)
        print("   Datatype mismatch payload validation status:", val_res_mismatch["valid"])
        print("   Errors caught:", val_res_mismatch["errors"])
        assert val_res_mismatch["valid"] is False
        
        # 7. Predict on new data
        print("\n7. Running Predictor Service...")
        pred_res = predict(model_path, prep_path, raw_sample)
        print("   Predictions:", pred_res["predictions"])
        print("   Probabilities (sample 1):", pred_res["probabilities"][0] if pred_res["probabilities"] else None)
        assert len(pred_res["predictions"]) == 2
        
        # 8. Explain Predictions on Loaded Model
        print("\n8. Running Explainability on loaded model...")
        top_feats = get_top_features(loaded_model, X_train.columns.tolist(), 5)
        print("   Top features:", top_feats)
        
        shap_res = generate_shap_explanation(loaded_model, X_test.head(3))
        print("   SHAP values shape type:", type(shap_res["shap_values"]))
        print("   SHAP values list length:", len(shap_res["shap_values"]))
        assert len(shap_res["shap_values"]) == 3
        
        print("\n=== ALL PRODUCTION FLOW TESTS PASSED SUCCESSFULLY ===")
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    run_production_flow()
