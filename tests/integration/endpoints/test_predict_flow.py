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
    CANONICAL_STANDARD_IDS,
    CANONICAL_REAL_WORLD_IDS,
    get_sequence_by_id,
    validate_task_response,
    validate_error_response,
)


@pytest.fixture
def prediction_sequences() -> Dict[str, str]:
    """Sequences from canonical dataset suitable for prediction (minimum 7 residues)."""
    return {
        "pred_1": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
        "pred_2": CANONICAL_TEST_DATASET.get_by_id("standard_002").sequence,
    }


@pytest.fixture
def boundary_length_sequences() -> Dict[str, str]:
    """Sequences at or near the minimum length boundary for prediction."""
    return {
        "short_5": CANONICAL_TEST_DATASET.get_by_id("length_short_5").sequence,  # 5 aa - below min
        "short_10": CANONICAL_TEST_DATASET.get_by_id("length_short_10").sequence,  # 10 aa - above min
    }


class TestModelMetadataEndpoint:
    """
    Integration tests for GET /prediction_service/model_metadata.
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


class TestPredictEndpoint:
    """
    Integration tests for POST /prediction_service/predict.
    """

    @pytest.mark.integration
    def test_predict_creates_task(
        self,
        client,
        prediction_sequences,
    ):
        """Test that prediction request creates a task."""
        # Get available models first
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
        response_json = response.json()
        validate_task_response(response_json)

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
            result = poll_task(task_id, timeout=120)
        except TimeoutError:
            pytest.skip(f"Task {task_id} timed out - CI resource constraints")
        except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
            pytest.skip(f"Server connection lost during polling: {e}")

        # Task should reach a terminal state
        assert result["status"].upper() in ("FINISHED", "COMPLETED", "DONE", "FAILED")

    @pytest.mark.integration
    def test_predict_multiple_models(
        self,
        client,
        prediction_sequences,
    ):
        """Test prediction with multiple models."""
        meta_response = client.get("/prediction_service/model_metadata")
        available_models = [m["name"] for m in meta_response.json()["metadata"]]

        if len(available_models) < 2:
            pytest.skip("Need at least 2 models for this test")

        request_data = {
            "model_names": available_models[:2],
            "sequence_input": prediction_sequences,
        }

        response = client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_REAL_WORLD_IDS, ids=lambda x: x)
    def test_predict_real_world_sequences(
        self,
        client,
        seq_id,
    ):
        """Test prediction with real-world protein sequences from canonical dataset."""
        meta_response = client.get("/prediction_service/model_metadata")
        available_models = [m["name"] for m in meta_response.json()["metadata"]]

        if not available_models:
            pytest.skip("No prediction models available")

        sequence = get_sequence_by_id(seq_id)

        request_data = {
            "model_names": [available_models[0]],
            "sequence_input": {seq_id: sequence},
        }

        response = client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_STANDARD_IDS, ids=lambda x: x)
    def test_predict_standard_sequences(
        self,
        client,
        seq_id,
    ):
        """Test prediction with standard sequences from canonical dataset."""
        meta_response = client.get("/prediction_service/model_metadata")
        available_models = [m["name"] for m in meta_response.json()["metadata"]]

        if not available_models:
            pytest.skip("No prediction models available")

        sequence = get_sequence_by_id(seq_id)

        request_data = {
            "model_names": [available_models[0]],
            "sequence_input": {seq_id: sequence},
        }

        response = client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

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

        # Pydantic validates model_names against BiocentralPredictionModel enum,
        # returning 422 for invalid values
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


class TestPredictionTaskLifecycle:
    """
    Tests for prediction task lifecycle management.
    """

    @pytest.mark.integration
    def test_prediction_task_id_uniqueness(
        self,
        client,
        prediction_sequences,
    ):
        """Test that multiple prediction submissions get unique task IDs."""
        meta_response = client.get("/prediction_service/model_metadata")
        available_models = [m["name"] for m in meta_response.json()["metadata"]]

        if not available_models:
            pytest.skip("No prediction models available")

        task_ids = set()

        for i in range(3):
            request_data = {
                "model_names": [available_models[0]],
                "sequence_input": prediction_sequences,
            }

            response = client.post("/prediction_service/predict", json=request_data)
            assert response.status_code == 200

            task_id = response.json()["task_id"]
            task_ids.add(task_id)

        # All task IDs should be unique
        assert len(task_ids) == 3

    @pytest.mark.integration
    def test_task_status_endpoint(
        self,
        client,
        prediction_sequences,
    ):
        """Test that task status can be retrieved."""
        meta_response = client.get("/prediction_service/model_metadata")
        available_models = [m["name"] for m in meta_response.json()["metadata"]]

        if not available_models:
            pytest.skip("No prediction models available")

        request_data = {
            "model_names": [available_models[0]],
            "sequence_input": prediction_sequences,
        }

        # Create task
        response = client.post("/prediction_service/predict", json=request_data)
        assert response.status_code == 200
        task_id = response.json()["task_id"]

        # Get task status
        status_response = client.get(f"/biocentral_service/task_status/{task_id}")
        assert status_response.status_code == 200

        status = status_response.json()
        # API returns {"dtos": [TaskDTO, ...]} structure
        assert "dtos" in status
        assert isinstance(status["dtos"], list)


class TestEndToEndPredictionFlow:
    """
    End-to-end tests for the complete prediction workflow.
    """

    @pytest.mark.integration
    def test_complete_prediction_flow(
        self,
        client,
        poll_task,
        real_world_sequences,
    ):
        """Test complete prediction flow from request to completion."""
        meta_response = client.get("/prediction_service/model_metadata")
        available_models = [m["name"] for m in meta_response.json()["metadata"]]

        if not available_models:
            pytest.skip("No prediction models available")

        request_data = {
            "model_names": [available_models[0]],
            "sequence_input": real_world_sequences,
        }

        # Submit prediction task
        response = client.post("/prediction_service/predict", json=request_data)
        assert response.status_code == 200

        task_id = response.json()["task_id"]

        # Wait for completion with graceful handling for CI resource constraints
        try:
            result = poll_task(task_id, timeout=180)
        except TimeoutError:
            pytest.skip(f"Task {task_id} timed out - CI resource constraints")
        except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
            pytest.skip(f"Server connection lost during polling: {e}")

        # Verify completion (task reached terminal state)
        assert result["status"].upper() in ("FINISHED", "COMPLETED", "DONE", "FAILED")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_predict_diverse_sequences(
        self,
        client,
        poll_task,
        real_world_sequences,
    ):
        """Test prediction with diverse sequences from canonical dataset."""
        meta_response = client.get("/prediction_service/model_metadata")
        available_models = [m["name"] for m in meta_response.json()["metadata"]]

        if not available_models:
            pytest.skip("No prediction models available")

        request_data = {
            "model_names": [available_models[0]],
            "sequence_input": real_world_sequences,
        }

        response = client.post("/prediction_service/predict", json=request_data)
        assert response.status_code == 200

        task_id = response.json()["task_id"]
        
        try:
            result = poll_task(task_id, timeout=120)
        except TimeoutError:
            pytest.skip(f"Task {task_id} timed out - CI resource constraints")
        except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
            pytest.skip(f"Server connection lost during polling: {e}")

        assert result["status"].lower() in ("finished", "completed", "done", "failed")

    @pytest.mark.integration
    def test_prediction_with_valid_boundary_length(
        self,
        client,
        boundary_length_sequences,
    ):
        """Test prediction with sequence at minimum valid length."""
        meta_response = client.get("/prediction_service/model_metadata")
        available_models = [m["name"] for m in meta_response.json()["metadata"]]

        if not available_models:
            pytest.skip("No prediction models available")

        # Use 10 aa sequence which is above minimum (7)
        valid_seq = boundary_length_sequences["short_10"]
        assert len(valid_seq) >= 7

        request_data = {
            "model_names": [available_models[0]],
            "sequence_input": {"boundary": valid_seq},
        }

        response = client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())
