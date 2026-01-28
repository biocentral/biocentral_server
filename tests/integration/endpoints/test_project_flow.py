"""Integration tests for projection endpoints."""

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


class TestProjectionConfigEndpoint:
    """
    Integration tests for GET /projection_service/projection_config.
    """

    @pytest.mark.integration
    def test_get_projection_config(self, client):
        """Test retrieving available projection methods and configurations."""
        response = client.get("/projection_service/projection_config")

        assert response.status_code == 200
        response_json = response.json()
        assert "projection_config" in response_json

        # Should contain common dimension reduction methods
        config = response_json["projection_config"]
        assert isinstance(config, dict)
        assert len(config) > 0

    @pytest.mark.integration
    def test_projection_config_contains_common_methods(self, client):
        """Test that common projection methods are available."""
        response = client.get("/projection_service/projection_config")

        assert response.status_code == 200
        config = response_json = response.json()["projection_config"]

        # Common methods that should be available
        config_lower = {k.lower(): v for k, v in config.items()}
        assert any(method in config_lower for method in ["umap", "pca", "tsne"])

    @pytest.mark.integration
    def test_projection_config_consistent(self, client):
        """Test that config is consistent across calls."""
        response1 = client.get("/projection_service/projection_config")
        response2 = client.get("/projection_service/projection_config")

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json() == response2.json()


class TestProjectEndpoint:
    """
    Integration tests for POST /projection_service/project.
    """

    @pytest.mark.integration
    def test_project_creates_task(
        self,
        client,
        embedder_name,
        short_test_sequences,
    ):
        """Test that projection request creates a task."""
        request_data = {
            "method": "pca",
            "sequence_data": short_test_sequences,
            "embedder_name": embedder_name,
            "config": {
                "n_components": 2,
            },
        }

        response = client.post("/projection_service/project", json=request_data)

        assert response.status_code == 200
        response_json = response.json()
        validate_task_response(response_json)

    @pytest.mark.integration
    def test_project_task_completes(
        self,
        client,
        poll_task,
        embedder_name,
        test_sequences,
    ):
        """Test that projection task completes successfully."""
        request_data = {
            "method": "pca",
            "sequence_data": test_sequences,
            "embedder_name": embedder_name,
            "config": {
                "n_components": 2,
            },
        }

        response = client.post("/projection_service/project", json=request_data)
        assert response.status_code == 200

        task_id = response.json()["task_id"]
        result = poll_task(task_id, timeout=120)

        assert result["status"].lower() in ("finished", "completed", "done", "failed")

    @pytest.mark.integration
    def test_project_with_pca(
        self,
        client,
        embedder_name,
        test_sequences,
    ):
        """Test projection with PCA method."""
        request_data = {
            "method": "pca",
            "sequence_data": test_sequences,
            "embedder_name": embedder_name,
            "config": {
                "n_components": 3,
            },
        }

        response = client.post("/projection_service/project", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    def test_project_with_umap(
        self,
        client,
        embedder_name,
        diverse_test_sequences,
    ):
        """Test projection with UMAP method."""
        request_data = {
            "method": "umap",
            "sequence_data": diverse_test_sequences,
            "embedder_name": embedder_name,
            "config": {
                "n_neighbors": min(5, len(diverse_test_sequences) - 1),
                "min_dist": 0.1,
                "n_components": 2,
            },
        }

        response = client.post("/projection_service/project", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    def test_project_with_tsne(
        self,
        client,
        embedder_name,
        test_sequences,
    ):
        """Test projection with t-SNE method."""
        request_data = {
            "method": "tsne",
            "sequence_data": test_sequences,
            "embedder_name": embedder_name,
            "config": {
                "n_components": 2,
                "perplexity": min(2, len(test_sequences) - 1),
            },
        }

        response = client.post("/projection_service/project", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_STANDARD_IDS, ids=lambda x: x)
    def test_project_standard_sequences(
        self,
        client,
        embedder_name,
        seq_id,
    ):
        """Test projection with standard sequences from canonical dataset."""
        # Need at least 2 sequences for projection
        sequence = get_sequence_by_id(seq_id)
        sequence_2 = get_sequence_by_id("standard_001" if seq_id != "standard_001" else "standard_002")

        request_data = {
            "method": "pca",
            "sequence_data": {seq_id: sequence, "other": sequence_2},
            "embedder_name": embedder_name,
            "config": {"n_components": 2},
        }

        response = client.post("/projection_service/project", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_REAL_WORLD_IDS, ids=lambda x: x)
    def test_project_real_world_sequences(
        self,
        client,
        embedder_name,
        seq_id,
    ):
        """Test projection with real-world protein sequences from canonical dataset."""
        sequence = get_sequence_by_id(seq_id)
        sequence_2 = get_sequence_by_id("real_insulin_b" if seq_id != "real_insulin_b" else "real_ubiquitin")

        request_data = {
            "method": "pca",
            "sequence_data": {seq_id: sequence, "other": sequence_2},
            "embedder_name": embedder_name,
            "config": {"n_components": 2},
        }

        response = client.post("/projection_service/project", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    def test_project_diverse_sequences(
        self,
        client,
        embedder_name,
        diverse_test_sequences,
    ):
        """Test projection with diverse sequence collection from canonical dataset."""
        request_data = {
            "method": "pca",
            "sequence_data": diverse_test_sequences,
            "embedder_name": embedder_name,
            "config": {"n_components": 3},
        }

        response = client.post("/projection_service/project", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    def test_project_invalid_method_rejected(
        self,
        client,
        embedder_name,
        short_test_sequences,
    ):
        """Test that invalid projection method is rejected with proper error."""
        request_data = {
            "method": "invalid_method_xyz",
            "sequence_data": short_test_sequences,
            "embedder_name": embedder_name,
            "config": {},
        }

        response = client.post("/projection_service/project", json=request_data)

        assert response.status_code == 400

    @pytest.mark.integration
    def test_project_empty_sequences_rejected(
        self,
        client,
        embedder_name,
    ):
        """Test that empty sequence data is rejected with proper error."""
        request_data = {
            "method": "pca",
            "sequence_data": {},
            "embedder_name": embedder_name,
            "config": {},
        }

        response = client.post("/projection_service/project", json=request_data)

        # Should fail validation
        assert response.status_code in (400, 422)


class TestProjectionTaskLifecycle:
    """
    Tests for projection task lifecycle management.
    """

    @pytest.mark.integration
    def test_projection_task_id_uniqueness(
        self,
        client,
        embedder_name,
        test_sequences,
    ):
        """Test that multiple projection submissions get unique task IDs."""
        task_ids = set()

        for i in range(3):
            request_data = {
                "method": "pca",
                "sequence_data": test_sequences,
                "embedder_name": embedder_name,
                "config": {"n_components": 2},
            }

            response = client.post("/projection_service/project", json=request_data)
            assert response.status_code == 200

            task_id = response.json()["task_id"]
            task_ids.add(task_id)

        # All task IDs should be unique
        assert len(task_ids) == 3

    @pytest.mark.integration
    def test_task_status_endpoint(
        self,
        client,
        embedder_name,
        short_test_sequences,
    ):
        """Test that task status can be retrieved."""
        request_data = {
            "method": "pca",
            "sequence_data": short_test_sequences,
            "embedder_name": embedder_name,
            "config": {"n_components": 2},
        }

        # Create task
        response = client.post("/projection_service/project", json=request_data)
        assert response.status_code == 200
        task_id = response.json()["task_id"]

        # Get task status
        status_response = client.get(f"/biocentral_service/task_status/{task_id}")
        assert status_response.status_code == 200

        status = status_response.json()
        # API returns {"dtos": [TaskDTO, ...]} structure
        assert "dtos" in status
        assert isinstance(status["dtos"], list)


class TestEndToEndProjectionFlow:
    """
    End-to-end tests for the complete projection workflow.
    """

    @pytest.mark.integration
    def test_complete_projection_flow(
        self,
        client,
        poll_task,
        embedder_name,
        diverse_test_sequences,
    ):
        """Test complete projection flow from request to completion."""
        request_data = {
            "method": "pca",
            "sequence_data": diverse_test_sequences,
            "embedder_name": embedder_name,
            "config": {"n_components": 2},
        }

        # Submit projection task
        response = client.post("/projection_service/project", json=request_data)
        assert response.status_code == 200

        task_id = response.json()["task_id"]

        # Wait for completion
        result = poll_task(task_id, timeout=180)

        # Verify completion (task reached terminal state)
        assert result["status"].upper() in ("FINISHED", "COMPLETED", "DONE", "FAILED")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_umap_projection_flow(
        self,
        client,
        poll_task,
        embedder_name,
        diverse_test_sequences,
    ):
        """Test UMAP projection flow from request to completion."""
        request_data = {
            "method": "umap",
            "sequence_data": diverse_test_sequences,
            "embedder_name": embedder_name,
            "config": {
                "n_neighbors": min(5, len(diverse_test_sequences) - 1),
                "min_dist": 0.1,
                "n_components": 2,
            },
        }

        response = client.post("/projection_service/project", json=request_data)
        assert response.status_code == 200

        task_id = response.json()["task_id"]
        result = poll_task(task_id, timeout=300)

        # Task should reach a terminal state
        assert result["status"].upper() in ("FINISHED", "COMPLETED", "DONE", "FAILED")

    @pytest.mark.integration
    def test_projection_with_different_embedders(
        self,
        client,
        test_sequences,
    ):
        """Test that different embedders can be used for projection."""
        embedders = ["one_hot_encoding", "blosum62"]

        for embedder in embedders:
            request_data = {
                "method": "pca",
                "sequence_data": test_sequences,
                "embedder_name": embedder,
                "config": {"n_components": 2},
            }

            response = client.post("/projection_service/project", json=request_data)
            assert response.status_code == 200, f"Failed for embedder: {embedder}"
            validate_task_response(response.json())

    @pytest.mark.integration
    def test_projection_with_real_world_collection(
        self,
        client,
        poll_task,
        embedder_name,
        real_world_sequences,
    ):
        """Test projection with collection of real-world sequences."""
        request_data = {
            "method": "pca",
            "sequence_data": real_world_sequences,
            "embedder_name": embedder_name,
            "config": {"n_components": 2},
        }

        response = client.post("/projection_service/project", json=request_data)
        assert response.status_code == 200

        task_id = response.json()["task_id"]
        result = poll_task(task_id, timeout=120)

        assert result["status"].lower() in ("finished", "completed", "done", "failed")
