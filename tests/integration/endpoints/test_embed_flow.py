"""Integration tests for embeddings endpoints."""

import pytest

from tests.integration.endpoints.conftest import (
    validate_error_response,
)


@pytest.mark.order(1)
class TestCommonEmbeddersEndpoint:
    """
    Integration tests for GET /embeddings_service/common_embedders.
    Lightweight: No embedding computation.
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


@pytest.mark.order(2)
class TestEmbedEndpoint:
    """
    Integration tests for POST /embeddings_service/embed.
    Validates request/response structure and error handling.
    """

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
            "use_half_precision": True,
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
            "use_half_precision": True,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 422
        error_response = response.json()
        validate_error_response(error_response)


@pytest.mark.order(2)
class TestEndToEndEmbedFlow:
    """
    End-to-end tests for the complete embedding workflow.
    Heavier: Waits for task completion.
    """

    @pytest.mark.integration
    def test_embed_and_wait_for_completion(
        self,
        client,
        poll_task,
        embedder_name,
        shared_embedding_sequences,
        verify_embedding_cache,
    ):
        """
        Test complete embedding flow from request to completion.
        
        IMPORTANT: This test pre-computes reduced embeddings for sequences
        that will be reused by projection tests.
        """
        # Check cache BEFORE embedding
        print("\n[BEFORE EMBEDDING] Checking cache status...")
        verify_embedding_cache(expect_cached=False)
       
        request_data = {
            "embedder_name": embedder_name,
            "reduce": True,
            "sequence_data": shared_embedding_sequences,
            "use_half_precision": False,
        }

        # Submit embedding task
        response = client.post("/embeddings_service/embed", json=request_data)
        assert response.status_code == 200, f"Failed to submit embedding task: {response.text}"
        
        task_id = response.json()["task_id"]
        print(f"[EMBEDDING] Submitted task {task_id}, waiting for completion...")
        
        # Wait for completion - poll_task handles retries internally
        result = poll_task(task_id, timeout=480, max_consecutive_errors=15)
        
        # Verify task succeeded (not just reached terminal state)
        task_status = result["status"].upper()
        assert task_status in ("FINISHED", "COMPLETED", "DONE"), \
            f"Embedding task failed with status '{task_status}': {result.get('error', 'unknown error')}"
        
        print("\n[AFTER EMBEDDING] Checking cache status...")
        after = verify_embedding_cache(expect_cached=True)
        print(f"[CACHE POPULATED] {after['cached']}/{after['total']} embeddings now cached")
 
    @pytest.mark.integration
    def test_embed_with_different_embedders(
        self,
        client,
        single_test_sequence,
        validate_task_response,
    ):
        """Test that different embedders can be used."""
        embedders = ["one_hot_encoding", "blosum62"]
        
        for embedder in embedders:
            request_data = {
                "embedder_name": embedder,
                "reduce": True,
                "sequence_data": single_test_sequence,
                "use_half_precision": True,
            }

            response = client.post("/embeddings_service/embed", json=request_data)
            assert response.status_code == 200, f"Failed for embedder: {embedder}"
            validate_task_response(response.json())
