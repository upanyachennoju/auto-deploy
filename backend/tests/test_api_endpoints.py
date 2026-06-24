import os
import shutil
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.model_registry import ARTIFACTS_DIR, get_model_dir

client = TestClient(app)

def test_full_automl_api_lifecycle():
    # 1. Create a mini dataset
    temp_dir = Path("backend/tests/temp_test")
    os.makedirs(temp_dir, exist_ok=True)
    csv_path = temp_dir / "mini_churn.csv"
    
    # 10 mock rows for classification
    data = {
        "Age": [25, 47, 31, 22, 54, 38, 41, 30, 48, 35],
        "CustServ_Calls": [1, 4, 1, 0, 5, 2, 1, 0, 3, 2],
        "Day_Mins": [120.5, 290.1, 150.2, 95.4, 310.8, 180.3, 210.5, 140.0, 260.4, 195.2],
        "User_ID": ["U101", "U102", "U103", "U104", "U105", "U106", "U107", "U108", "U109", "U110"], # High cardinality ID
        "Churn": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0] # Target
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    print("\n--- [STEP 1] Mini dataset created at:", csv_path)
    
    try:
        # 2. Upload file to retrieve session_id
        with open(csv_path, "rb") as f:
            response = client.post(
                "/upload-single/",
                files={"file": ("mini_churn.csv", f, "text/csv")}
            )
        
        assert response.status_code == 201
        res_json = response.json()
        session_id = res_json["session_id"]
        assert session_id is not None
        print(f"--- [STEP 2] File uploaded. session_id: {session_id}")
        
        # 3. Train model
        train_payload = {
            "session_id": session_id,
            "target_column": "Churn"
        }
        train_response = client.post("/train", json=train_payload)
        
        assert train_response.status_code == 200
        train_res = train_response.json()
        model_id = train_res["model_id"]
        assert model_id is not None
        assert train_res["session_id"] == session_id
        assert "dataset_summary" in train_res
        assert "model_summary" in train_res
        assert "evaluation_summary" in train_res
        print(f"--- [STEP 3] Model trained successfully. model_id: {model_id}")
        
        # 4. Verify artifact directory structure
        model_dir = get_model_dir(model_id)
        assert model_dir.exists()
        assert (model_dir / "model.pkl").exists()
        assert (model_dir / "preprocessor.pkl").exists()
        assert (model_dir / "schema.json").exists()
        assert (model_dir / "metadata.json").exists()
        print(f"--- [STEP 4] Artifact directory structure verified at {model_dir}")
        
        # 5. List models
        list_response = client.get("/models")
        assert list_response.status_code == 200
        list_res = list_response.json()
        model_ids = [m["model_id"] for m in list_res]
        assert model_id in model_ids
        print(f"--- [STEP 5] Model listing verified. Found {len(list_res)} models.")
        
        # 6. Retrieve model details
        detail_response = client.get(f"/models/{model_id}")
        assert detail_response.status_code == 200
        detail_res = detail_response.json()
        assert detail_res["metadata"]["model_id"] == model_id
        assert detail_res["schema"] is not None
        assert "model_info" in detail_res
        assert "evaluation_summary" in detail_res
        print("--- [STEP 6] Detailed model information retrieved successfully.")
        
        # 7. Predict single record
        single_payload = {
            "model_id": model_id,
            "data": {
                "Age": 28,
                "CustServ_Calls": 2,
                "Day_Mins": 175.5,
                "User_ID": "U123" # preprocessor will automatically ignore this
            }
        }
        pred_single_resp = client.post("/predict", json=single_payload)
        assert pred_single_resp.status_code == 200
        pred_single_res = pred_single_resp.json()
        assert "prediction" in pred_single_res
        assert "probability" in pred_single_res
        print(f"--- [STEP 7] Single record prediction verified. Prediction: {pred_single_res['prediction']}, Probability: {pred_single_res['probability']}")
        
        # 8. Predict batch of records
        batch_payload = {
            "model_id": model_id,
            "data": [
                {"Age": 28, "CustServ_Calls": 2, "Day_Mins": 175.5, "User_ID": "U123"},
                {"Age": 55, "CustServ_Calls": 5, "Day_Mins": 300.2, "User_ID": "U124"}
            ]
        }
        pred_batch_resp = client.post("/predict", json=batch_payload)
        assert pred_batch_resp.status_code == 200
        pred_batch_res = pred_batch_resp.json()
        assert "predictions" in pred_batch_res
        assert "probabilities" in pred_batch_res
        assert len(pred_batch_res["predictions"]) == 2
        print(f"--- [STEP 8] Batch prediction verified. Predictions: {pred_batch_res['predictions']}, Probabilities: {pred_batch_res['probabilities']}")
        
        # 9. Delete model
        del_response = client.delete(f"/models/{model_id}")
        assert del_response.status_code == 200
        assert not model_dir.exists()
        print("--- [STEP 9] Model directory deleted and verified.")
        
        # 10. Verify 404 after deletion
        get_after_del = client.get(f"/models/{model_id}")
        assert get_after_del.status_code == 404
        print("--- [STEP 10] Model 404 validation verified.")
        
    finally:
        # Clean up temp file
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_full_automl_api_lifecycle()
    print("\nALL API INTEGRATION TESTS PASSED SUCCESSFULLY!")
