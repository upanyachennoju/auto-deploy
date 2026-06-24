import os
import shutil
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.model_registry import (
    get_model_dir,
    get_metadata_path,
    get_dataset_summary_path,
    get_feature_importance_path,
    get_model_summary_path,
    get_ai_report_path
)
from backend.services.ai_report_generator import generate_ai_report
from backend.agents.report_agent import generate_fallback_report, generate_report_agent, AIReport

client = TestClient(app)

DUMMY_DATASET = {
    "shape": {"rows": 100, "columns": 5},
    "numerical_columns": ["Age", "Income"],
    "categorical_columns": ["Gender"],
    "missing_values": {"Age": 0, "Income": 2, "Gender": 0},
    "duplicate_rows": 0,
    "possible_target_columns": ["Churn"]
}

DUMMY_MODEL = {
    "best_estimator": "xgboost",
    "best_loss": 0.15
}

DUMMY_EVAL = {
    "accuracy": 0.85,
    "f1": 0.84,
    "precision": 0.86,
    "recall": 0.85
}

DUMMY_IMPORTANCE = {
    "Income": 0.6,
    "Age": 0.4
}

class TestAIReportGenerator(unittest.TestCase):
    def test_fallback_report_structure(self):
        """Verify fallback generator generates correct keys with correct actual values."""
        report = generate_fallback_report(
            dataset_summary=DUMMY_DATASET,
            model_summary=DUMMY_MODEL,
            evaluation_summary=DUMMY_EVAL,
            feature_importance=DUMMY_IMPORTANCE,
            task_type="classification"
        )
        
        required_keys = [
            "executive_summary",
            "dataset_analysis",
            "data_quality_analysis",
            "model_selection_reasoning",
            "performance_analysis",
            "feature_analysis",
            "deployment_recommendation",
            "next_steps"
        ]
        
        for key in required_keys:
            self.assertIn(key, report)
            self.assertIsInstance(report[key], str)
            self.assertTrue(len(report[key]) > 0)
            
        self.assertIn("100", report["dataset_analysis"])
        self.assertIn("5", report["dataset_analysis"])
        self.assertIn("2", report["data_quality_analysis"])
        self.assertIn("xgboost", report["executive_summary"])
        self.assertIn("0.15", report["executive_summary"])
        self.assertIn("ACCURACY: 0.85", report["performance_analysis"])
        self.assertIn("'Income' (0.60", report["feature_analysis"])

    @patch("backend.agents.report_agent.ChatGroq")
    def test_report_agent_groq_success(self, mock_chat_groq):
        """Verify report_agent returns correct format when Groq call succeeds."""
        mock_llm_instance = MagicMock()
        mock_structured_llm = MagicMock()
        
        mock_chat_groq.return_value = mock_llm_instance
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm
        
        mock_ai_report_instance = AIReport(
            executive_summary="Executive Summary Text",
            dataset_analysis="Dataset Analysis Text",
            data_quality_analysis="Data Quality Text",
            model_selection_reasoning="Model Selection Text",
            performance_analysis="Performance Text",
            feature_analysis="Feature Text",
            deployment_recommendation="Deployment Text",
            next_steps="Next Steps Text"
        )
        mock_structured_llm.invoke.return_value = mock_ai_report_instance
        
        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-api-key"}):
            report = generate_report_agent(
                dataset_summary=DUMMY_DATASET,
                model_summary=DUMMY_MODEL,
                evaluation_summary=DUMMY_EVAL,
                feature_importance=DUMMY_IMPORTANCE,
                task_type="classification"
            )
            
        self.assertEqual(report["executive_summary"], "Executive Summary Text")
        self.assertEqual(report["next_steps"], "Next Steps Text")
        mock_structured_llm.invoke.assert_called_once()

    @patch("backend.agents.report_agent.ChatGroq")
    def test_report_agent_groq_failure_fallback(self, mock_chat_groq):
        """Verify report_agent falls back gracefully if Groq throws an error."""
        mock_chat_groq.side_effect = Exception("API Error")
        
        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-api-key"}):
            report = generate_report_agent(
                dataset_summary=DUMMY_DATASET,
                model_summary=DUMMY_MODEL,
                evaluation_summary=DUMMY_EVAL,
                feature_importance=DUMMY_IMPORTANCE,
                task_type="classification"
            )
            
        self.assertIn("xgboost", report["executive_summary"])
        self.assertIn("0.15", report["executive_summary"])

    def test_ai_report_generator_caching(self):
        """Verify report caching. If model_id is given and report exists, it's loaded from disk without generating."""
        model_id = "test-caching-model-id"
        report_dir = get_model_dir(model_id)
        report_path = get_ai_report_path(model_id)
        
        os.makedirs(report_dir, exist_ok=True)
        
        cached_report = {
            "executive_summary": "Cached Summary",
            "dataset_analysis": "Cached Dataset",
            "data_quality_analysis": "Cached Quality",
            "model_selection_reasoning": "Cached Selection",
            "performance_analysis": "Cached Performance",
            "feature_analysis": "Cached Feature",
            "deployment_recommendation": "Cached Deployment",
            "next_steps": "Cached Next"
        }
        
        with open(report_path, "w") as f:
            json.dump(cached_report, f)
            
        try:
            with patch("backend.services.ai_report_generator.generate_report_agent") as mock_agent:
                report = generate_ai_report(
                    dataset_summary=DUMMY_DATASET,
                    model_summary=DUMMY_MODEL,
                    evaluation_summary=DUMMY_EVAL,
                    feature_importance=DUMMY_IMPORTANCE,
                    task_type="classification",
                    model_id=model_id
                )
                mock_agent.assert_not_called()
                self.assertEqual(report["executive_summary"], "Cached Summary")
        finally:
            shutil.rmtree(report_dir, ignore_errors=True)

    def test_report_endpoint_not_found(self):
        """Verify endpoint returns 404 for unknown model_id."""
        resp = client.get("/report/unknown-model-id-12345")
        self.assertEqual(resp.status_code, 404)

    def test_report_endpoint_success(self):
        """Verify endpoint loads dependencies correctly and returns a generated report."""
        model_id = "test-endpoint-model-id"
        model_dir = get_model_dir(model_id)
        
        os.makedirs(model_dir, exist_ok=True)
        
        metadata = {
            "model_id": model_id,
            "created_at": "2026-06-24 12:00:00",
            "target_column": "Churn",
            "task_type": "classification",
            "best_estimator": "lightgbm",
            "metrics": DUMMY_EVAL
        }
        
        with open(get_metadata_path(model_id), "w") as f:
            json.dump(metadata, f)
            
        with open(get_dataset_summary_path(model_id), "w") as f:
            json.dump(DUMMY_DATASET, f)
            
        with open(get_feature_importance_path(model_id), "w") as f:
            json.dump(DUMMY_IMPORTANCE, f)
            
        with open(get_model_summary_path(model_id), "w") as f:
            json.dump({"best_estimator": "lightgbm", "best_loss": 0.15}, f)
            
        try:
            resp = client.get(f"/report/{model_id}")
            self.assertEqual(resp.status_code, 200)
            
            report = resp.json()
            self.assertIn("executive_summary", report)
            self.assertIn("lightgbm", report["executive_summary"])
            self.assertTrue("accuracy" in report["performance_analysis"] or "ACCURACY" in report["performance_analysis"])
            
            self.assertTrue(get_ai_report_path(model_id).exists())
        finally:
            shutil.rmtree(model_dir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
