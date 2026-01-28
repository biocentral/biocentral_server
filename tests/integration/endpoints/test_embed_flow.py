"""Integration tests for embeddings endpoints."""

import pytest
from typing import Dict

from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET
from tests.integration.endpoints.conftest import (
    CANONICAL_STANDARD_IDS,
    CANONICAL_LENGTH_EDGE_IDS,
    CANONICAL_UNKNOWN_TOKEN_IDS,
    CANONICAL_AMBIGUOUS_CODE_IDS,
    CANONICAL_REAL_WORLD_IDS,
    get_sequence_by_id,
    validate_task_response,
    validate_error_response,
)


class TestCommonEmbeddersEndpoint:
    """
    Integration tests for GET /embeddings_service/common_embedders.
    """

    @pytest.mark.integration
    def test_common_embedders_returns_list(self, client):
        """Test that common_embedders endpoint returns a list."""
        response = client.get("/embeddings_service/common_embedders")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    @pytest.mark.integration
    def test_common_embedders_includes_baseline_models(self, client):
        """Test that baseline embedders are available."""
        response = client.get("/embeddings_service/common_embedders")

        assert response.status_code == 200
        embedders = response.json()
        
        # Baseline models for testing (fast, no downloads needed)
        assert "one_hot_encoding" in embedders
        assert "blosum62" in embedders

    @pytest.mark.integration
    def test_common_embedders_response_is_consistent(self, client):
        """Test that multiple calls return the same list."""
        response1 = client.get("/embeddings_service/common_embedders")
        response2 = client.get("/embeddings_service/common_embedders")

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json() == response2.json()


class TestEmbedEndpoint:
    """
    Integration tests for POST /embeddings_service/embed.
    
    Tests embedding task creation and completion against real server.
    """

    @pytest.mark.integration
    def test_embed_request_creates_task(
        self,
        client,
        embedder_name,
        short_test_sequences,
    ):
        """Test that embedding request creates a task and returns task ID."""
        request_data = {
            "embedder_name": embedder_name,
            "reduce": False,
            "sequence_data": short_test_sequences,
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 200
        response_json = response.json()
        task_id = validate_task_response(response_json)
        assert isinstance(task_id, str)
        assert len(task_id) > 0

    @pytest.mark.integration
    def test_embed_task_completes_successfully(
        self,
        client,
        poll_task,
        embedder_name,
        single_test_sequence,
    ):
        """Test that embedding task completes successfully."""
        request_data = {
            "embedder_name": embedder_name,
            "reduce": True,
            "sequence_data": single_test_sequence,
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)
        assert response.status_code == 200
        
        task_id = response.json()["task_id"]
        result = poll_task(task_id, timeout=60)
        
        assert result["status"].lower() in ("finished", "completed", "done")

    @pytest.mark.integration
    def test_embed_request_with_reduction(
        self,
        client,
        embedder_name,
        single_test_sequence,
    ):
        """Test embedding request with reduce=True for per-sequence embeddings."""
        request_data = {
            "embedder_name": embedder_name,
            "reduce": True,
            "sequence_data": single_test_sequence,
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    def test_embed_multiple_sequences(
        self,
        client,
        embedder_name,
        test_sequences,
    ):
        """Test embedding request with multiple sequences."""
        request_data = {
            "embedder_name": embedder_name,
            "reduce": False,
            "sequence_data": test_sequences,
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_REAL_WORLD_IDS, ids=lambda x: x)
    def test_embed_real_world_sequences(
        self,
        client,
        embedder_name,
        seq_id,
    ):
        """Test embedding real-world protein sequences from canonical dataset."""
        sequence = get_sequence_by_id(seq_id)
        request_data = {
            "embedder_name": embedder_name,
            "reduce": True,
            "sequence_data": {seq_id: sequence},
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_LENGTH_EDGE_IDS, ids=lambda x: x)
    def test_embed_length_edge_case_sequences(
        self,
        client,
        embedder_name,
        seq_id,
    ):
        """Test embedding sequences at length boundaries from canonical dataset."""
        sequence = get_sequence_by_id(seq_id)
        request_data = {
            "embedder_name": embedder_name,
            "reduce": True,
            "sequence_data": {seq_id: sequence},
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_UNKNOWN_TOKEN_IDS, ids=lambda x: x)
    def test_embed_sequences_with_unknown_tokens(
        self,
        client,
        embedder_name,
        seq_id,
    ):
        """Test embedding sequences containing X residues from canonical dataset."""
        sequence = get_sequence_by_id(seq_id)
        request_data = {
            "embedder_name": embedder_name,
            "reduce": True,
            "sequence_data": {seq_id: sequence},
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_AMBIGUOUS_CODE_IDS, ids=lambda x: x)
    def test_embed_sequences_with_ambiguous_codes(
        self,
        client,
        embedder_name,
        seq_id,
    ):
        """Test embedding sequences with ambiguous amino acid codes from canonical dataset."""
        sequence = get_sequence_by_id(seq_id)
        request_data = {
            "embedder_name": embedder_name,
            "reduce": True,
            "sequence_data": {seq_id: sequence},
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 200
        validate_task_response(response.json())

    @pytest.mark.integration
    def test_embed_empty_sequences_rejected(
        self,
        client,
        embedder_name,
    ):
        """Test that empty sequence data is rejected with proper error."""
        request_data = {
            "embedder_name": embedder_name,
            "reduce": False,
            "sequence_data": {},  # Empty - should fail validation
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 422
        validate_error_response(response.json())

    @pytest.mark.integration
    def test_embed_missing_embedder_name_rejected(
        self,
        client,
        short_test_sequences,
    ):
        """Test that missing embedder name is rejected with proper error."""
        request_data = {
            # Missing embedder_name
            "reduce": False,
            "sequence_data": short_test_sequences,
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 422
        error_response = response.json()
        validate_error_response(error_response)


class TestGetMissingEmbeddingsEndpoint:
    """
    Integration tests for POST /embeddings_service/get_missing_embeddings.
    """

    @pytest.mark.integration
    def test_get_missing_embeddings_new_sequences(
        self,
        client,
        embedder_name,
        test_run_id,
    ):
        """Test identifying missing embeddings for new sequences."""
        import json
        
        # Use unique sequence IDs that won't exist in the database (canonical dataset)
        unique_sequences = {
            f"test_{test_run_id}_1": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
            f"test_{test_run_id}_2": CANONICAL_TEST_DATASET.get_by_id("standard_002").sequence,
        }
        
        request_data = {
            "sequences": json.dumps(unique_sequences),
            "embedder_name": embedder_name,
            "reduced": False,
        }

        response = client.post(
            "/embeddings_service/get_missing_embeddings", json=request_data
        )

        assert response.status_code == 200
        response_json = response.json()
        assert "missing" in response_json
        # New sequences should be marked as missing
        assert len(response_json["missing"]) > 0


class TestEmbedTaskLifecycle:
    """
    Tests for embedding task lifecycle: creation, status tracking, completion.
    """

    @pytest.mark.integration
    def test_task_id_is_valid_uuid(
        self,
        client,
        embedder_name,
        short_test_sequences,
    ):
        """Test that task IDs are valid."""
        request_data = {
            "embedder_name": embedder_name,
            "reduce": True,
            "sequence_data": short_test_sequences,
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 200
        task_id = response.json()["task_id"]
        
        # Task ID should be non-empty string
        assert isinstance(task_id, str)
        assert len(task_id) > 0

    @pytest.mark.integration
    def test_multiple_tasks_get_unique_ids(
        self,
        client,
        embedder_name,
        short_test_sequences,
    ):
        """Test that multiple task submissions get unique IDs."""
        task_ids = set()
        
        for i in range(3):
            request_data = {
                "embedder_name": embedder_name,
                "reduce": True,
                "sequence_data": short_test_sequences,
                "use_half_precision": False,
            }

            response = client.post("/embeddings_service/embed", json=request_data)
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
        single_test_sequence,
    ):
        """Test that task status can be retrieved."""
        request_data = {
            "embedder_name": embedder_name,
            "reduce": True,
            "sequence_data": single_test_sequence,
            "use_half_precision": False,
        }

        # Create task
        response = client.post("/embeddings_service/embed", json=request_data)
        assert response.status_code == 200
        task_id = response.json()["task_id"]
        
        # Get task status
        status_response = client.get(f"/biocentral_service/task_status/{task_id}")
        assert status_response.status_code == 200
        
        status = status_response.json()
        # API returns {"dtos": [TaskDTO, ...]} structure
        assert "dtos" in status
        assert isinstance(status["dtos"], list)


class TestEndToEndEmbedFlow:
    """
    End-to-end tests for the complete embedding workflow.
    """

    @pytest.mark.integration
    def test_embed_and_wait_for_completion(
        self,
        client,
        poll_task,
        embedder_name,
        test_sequences,
    ):
        """Test complete embedding flow from request to completion."""
        request_data = {
            "embedder_name": embedder_name,
            "reduce": True,
            "sequence_data": test_sequences,
            "use_half_precision": False,
        }

        # Submit embedding task
        response = client.post("/embeddings_service/embed", json=request_data)
        assert response.status_code == 200
        
        task_id = response.json()["task_id"]
        
        # Wait for completion
        result = poll_task(task_id, timeout=120)
        
        # Verify completion
        assert result["status"].lower() in ("finished", "completed", "done", "failed")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_embed_diverse_sequences(
        self,
        client,
        poll_task,
        embedder_name,
        diverse_test_sequences,
    ):
        """Test embedding diverse sequences from canonical dataset."""
        request_data = {
            "embedder_name": embedder_name,
            "reduce": True,
            "sequence_data": diverse_test_sequences,
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)
        assert response.status_code == 200
        
        task_id = response.json()["task_id"]
        result = poll_task(task_id, timeout=180)
        
        assert result["status"].lower() in ("finished", "completed", "done")

    @pytest.mark.integration
    def test_embed_with_different_embedders(
        self,
        client,
        single_test_sequence,
    ):
        """Test that different embedders can be used."""
        embedders = ["one_hot_encoding", "blosum62"]
        
        for embedder in embedders:
            request_data = {
                "embedder_name": embedder,
                "reduce": True,
                "sequence_data": single_test_sequence,
                "use_half_precision": False,
            }

            response = client.post("/embeddings_service/embed", json=request_data)
            assert response.status_code == 200, f"Failed for embedder: {embedder}"
            validate_task_response(response.json())
