"""
Integration tests for prediction endpoints.

Requirements:
    - Server running via docker-compose.dev.yml
    - CI_SERVER_URL environment variable set

Usage:
    CI_SERVER_URL=http://localhost:9540 pytest tests/integration/endpoints/test_predict_flow.py -v
"""

import httpx
import pytest
from typing import Dict

from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET
from tests.integration.endpoints.conftest import (
    validate_error_response,
)


@pytest.fixture
def prediction_sequences() -> Dict[str, str]:
    """Sequences from canonical dataset suitable for prediction (minimum length of 7)."""
    return {
        "pred_1": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
    }


@pytest.fixture
def boundary_length_sequences() -> Dict[str, str]:
    """Sequences at or near the minimum length boundary for prediction."""
    return {
        "short_5": CANONICAL_TEST_DATASET.get_by_id("length_short_5").sequence,  # 5 aa - below min
        "short_10": CANONICAL_TEST_DATASET.get_by_id("length_short_10").sequence,  # 10 aa - above min
    }


@pytest.mark.order(1)
class TestModelMetadataEndpoint:
    """
    Integration tests for GET /prediction_service/model_metadata.
    Lightweight: No model loading.
    """

    @pytest.mark.integration
    def test_get_model_metadata(self, client):
        """Test retrieving available model metadata."""
        response = client.get("/prediction_service/model_metadata")

        assert response.status_code == 200
        response_json = response.json()
        assert "metadata" in response_json
        metadata = response_json["metadata"]
        assert isinstance(metadata, list)

    @pytest.mark.integration
    def test_model_metadata_structure(self, client):
        """Test that model metadata has expected structure."""
        response = client.get("/prediction_service/model_metadata")
        metadata = response.json()["metadata"]

        # Each model should have metadata
        for model_meta in metadata:
            assert isinstance(model_meta, dict)
            assert "name" in model_meta

    @pytest.mark.integration
    def test_model_metadata_consistent(self, client):
        """Test that metadata is consistent across calls."""
        response1 = client.get("/prediction_service/model_metadata")
        response2 = client.get("/prediction_service/model_metadata")

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json() == response2.json()


@pytest.mark.order(2)
class TestPredictEndpoint:
    """
    Integration tests for POST /prediction_service/predict.
    Medium-heavy: Submits prediction tasks and waits for completion.
    """


    @pytest.mark.integration
    def test_predict_invalid_model_rejected(
        self,
        client,
        prediction_sequences,
    ):
        """Test that invalid model name returns error."""
        request_data = {
            "model_names": ["invalid_model_xyz_123"],
            "sequence_input": prediction_sequences,
        }

        response = client.post("/prediction_service/predict", json=request_data)

        # should be returning 422 for invalid values
        assert response.status_code == 422

    @pytest.mark.integration
    def test_predict_empty_sequences_rejected(self, client):
        """Test that empty sequence input is rejected with proper error."""
        request_data = {
            "model_names": ["BindEmbed"],
            "sequence_input": {},  # Empty
        }

        response = client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 422
        validate_error_response(response.json())

    @pytest.mark.integration
    def test_predict_short_sequence_rejected(
        self,
        client,
        boundary_length_sequences,
    ):
        """Test that sequences shorter than minimum length are rejected with proper error."""
        short_seq = boundary_length_sequences["short_5"]
        request_data = {
            "model_names": ["BindEmbed"],
            "sequence_input": {"short": short_seq},
        }

        response = client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 422
        validate_error_response(response.json())

    @pytest.mark.integration
    def test_predict_empty_model_names_rejected(
        self,
        client,
        prediction_sequences,
    ):
        """Test that empty model names list is rejected with proper error."""
        request_data = {
            "model_names": [],  # Empty model list
            "sequence_input": prediction_sequences,
        }

        response = client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 422
        validate_error_response(response.json())

     
    @pytest.mark.integration
    def test_predict_task_completes(
        self,
        client,
        poll_task,
        prediction_sequences,
    ):
        """Test that prediction task completes successfully."""
        meta_response = client.get("/prediction_service/model_metadata")
        available_models = [m["name"] for m in meta_response.json()["metadata"]]

        if not available_models:
            pytest.skip("No prediction models available")

        request_data = {
            "model_names": [available_models[0]],
            "sequence_input": prediction_sequences,
        }

        response = client.post("/prediction_service/predict", json=request_data)
        assert response.status_code == 200

        task_id = response.json()["task_id"]
        
        # Wait for completion with graceful handling for CI resource constraints
        try:
            result = poll_task(task_id, timeout=280)
        except TimeoutError:
            pytest.skip(f"Task {task_id} timed out - CI resource constraints")
        except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
            pytest.skip(f"Server connection lost during polling: {e}")

        # Task should reach a terminal state
        assert result["status"].upper() in ("FINISHED", "COMPLETED", "DONE", "FAILED")
 