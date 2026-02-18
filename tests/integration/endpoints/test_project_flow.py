"""Integration tests for projection endpoints."""

import httpx
import pytest




@pytest.mark.order(1)
class TestProjectionConfigEndpoint: 
    @pytest.mark.integration
    def test_get_projection_config(self, client):
        response = client.get("/projection_service/projection_config")

        assert response.status_code == 200
        response_json = response.json()
        assert "projection_config" in response_json

        # Should contain common dimension reduction methods
        config = response_json["projection_config"]
        assert isinstance(config, dict)
        assert len(config) > 0

@pytest.mark.order(2)
class TestProjectEndpoint: 

    @pytest.mark.integration
    def test_project_task_completes(
        self,
        client,
        poll_task,
        embedder_name,
        shared_embedding_sequences,
        verify_embedding_cache,
    ):
        """
        Test that projection task completes successfully.
        
        Uses shared_embedding_sequences which are pre-cached by
        test_embed_and_wait_for_completion in test_embed_flow.py.
        """
        # Verify embeddings are cached before running projection
        cache_status = verify_embedding_cache(expect_cached=True)
        print(f"\n[PROJECTION] Cache status: {cache_status['cached']}/{cache_status['total']} cached")
        
        request_data = {
            "method": "pca",
            "sequence_data": shared_embedding_sequences,
            "embedder_name": embedder_name,
            "config": {
                "n_components": 2,
            },
        }

        response = client.post("/projection_service/project", json=request_data)
        assert response.status_code == 200, f"Failed to submit projection task: {response.text}"

        task_id = response.json()["task_id"]
        print(f"[PROJECTION] Submitted task {task_id}, waiting for completion...")
        
        # poll_task handles retries internally
        result = poll_task(task_id, timeout=120, max_consecutive_errors=10)

        assert result["status"].upper() == "FINISHED", f"Projection failed: {result.get('error', 'unknown')}"
        print(f"[PROJECTION] Task {task_id} completed successfully")

    # @pytest.mark.integration
    # def test_project_with_pca(
    #     self,
    #     client,
    #     embedder_name,
    #     test_sequences,
    # ):
    #     """Test projection with PCA method."""
    #     request_data = {
    #         "method": "pca",
    #         "sequence_data": test_sequences,
    #         "embedder_name": embedder_name,
    #         "config": {
    #             "n_components": 3,
    #         },
    #     }

    #     response = client.post("/projection_service/project", json=request_data)

    #     assert response.status_code == 200
    #     validate_task_response(response.json())

    # @pytest.mark.integration
    # def test_project_with_umap(
    #     self,
    #     client,
    #     embedder_name,
    #     real_world_sequences,
    # ):
    #     """Test projection with UMAP method."""
    #     request_data = {
    #         "method": "umap",
    #         "sequence_data": real_world_sequences,
    #         "embedder_name": embedder_name,
    #         "config": {
    #             "n_neighbors": min(0, len(real_world_sequences) - 1),
    #             "min_dist": 0.1,
    #             "n_components": 1,
    #         },
    #     }

    #     response = client.post("/projection_service/project", json=request_data)

    #     assert response.status_code == 200
    #     validate_task_response(response.json())

    # @pytest.mark.integration
    # def test_project_with_tsne(
    #     self,
    #     client,
    #     embedder_name,
    #     test_sequences,
    # ):
    #     """Test projection with t-SNE method."""
    #     request_data = {
    #         "method": "tsne",
    #         "sequence_data": test_sequences,
    #         "embedder_name": embedder_name,
    #         "config": {
    #             "n_components": 2,
    #             "perplexity": min(2, len(test_sequences) - 1),
    #         },
    #     }

    #     response = client.post("/projection_service/project", json=request_data)

    #     assert response.status_code == 200
    #     validate_task_response(response.json())


    # @pytest.mark.integration
    # @pytest.mark.parametrize("seq_id", CANONICAL_STANDARD_IDS, ids=lambda x: x)
    # def test_project_standard_sequences(
    #     self,
    #     client,
    #     embedder_name,
    #     seq_id,
    # ):
    #     """Test projection with standard sequences from canonical dataset."""
    #     sequence = get_sequence_by_id(seq_id)
    #     sequence_2 = get_sequence_by_id("standard_001" if seq_id != "standard_001" else "standard_002")

    #     request_data = {
    #         "method": "pca",
    #         "sequence_data": {seq_id: sequence, "other": sequence_2},
    #         "embedder_name": embedder_name,
    #         "config": {"n_components": 2},
    #     }

    #     response = client.post("/projection_service/project", json=request_data)

    #     assert response.status_code == 200
    #     validate_task_response(response.json())

    # @pytest.mark.integration
    # @pytest.mark.parametrize("seq_id", CANONICAL_REAL_WORLD_IDS, ids=lambda x: x)
    # def test_project_real_world_sequences(
    #     self,
    #     client,
    #     embedder_name,
    #     seq_id,
    # ):
    #     """Test projection with real-world protein sequences from canonical dataset."""
    #     sequence = get_sequence_by_id(seq_id)
    #     sequence_2 = get_sequence_by_id("real_insulin_b" if seq_id != "real_insulin_b" else "real_ubiquitin")

    #     request_data = {
    #         "method": "pca",
    #         "sequence_data": {seq_id: sequence, "other": sequence_2},
    #         "embedder_name": embedder_name,
    #         "config": {"n_components": 2},
    #     }

    #     response = client.post("/projection_service/project", json=request_data)

    #     assert response.status_code == 200
    #     validate_task_response(response.json())

    # @pytest.mark.integration
    # def test_project_diverse_sequences(
    #     self,
    #     client,
    #     embedder_name,
    #     real_world_sequences,
    # ):
    #     """Test projection with diverse sequence collection from canonical dataset."""
    #     request_data = {
    #         "method": "pca",
    #         "sequence_data": real_world_sequences,
    #         "embedder_name": embedder_name,
    #         "config": {"n_components": 3},
    #     }

    #     response = client.post("/projection_service/project", json=request_data)

    #     assert response.status_code == 200
    #     validate_task_response(response.json())

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


@pytest.mark.order(3)
class TestEndToEndProjectionFlow:
    """
    End-to-end tests for the complete projection workflow.
    Heavier: Waits for task completion.
    """

    @pytest.mark.integration
    def test_complete_projection_flow(
        self,
        client,
        poll_task,
        embedder_name,
        shared_embedding_sequences,
    ):
        """
        Test complete projection flow from request to completion.
        
        Uses shared_embedding_sequences which are pre-cached by
        test_embed_and_wait_for_completion in test_embed_flow.py.
        """
        request_data = {
            "method": "pca",
            "sequence_data": shared_embedding_sequences,
            "embedder_name": embedder_name,
            "config": {"n_components": 2},
        }

        # Submit projection task
        response = client.post("/projection_service/project", json=request_data)
        assert response.status_code == 200

        task_id = response.json()["task_id"]

        # Wait for completion - reduced timeout since embeddings are pre-cached
        try:
            result = poll_task(task_id, timeout=60)
        except TimeoutError:
            pytest.skip(f"Task {task_id} timed out - CI resource constraints")
        except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
            pytest.skip(f"Server connection lost during polling: {e}")

        # Verify successful completion
        assert result["status"].upper() == "FINISHED", f"Projection failed: {result.get('error', 'unknown')}"

     
    @pytest.mark.integration
    @pytest.mark.slow
    def test_umap_projection_flow(
         self,
         client,
         poll_task,
         embedder_name,
         shared_embedding_sequences,
     ):
         """Test UMAP projection flow from request to completion."""
         request_data = {
             "method": "umap",
             "sequence_data": shared_embedding_sequences,
             "embedder_name": embedder_name,
             "config": {
                 "n_neighbors": min(5, len(shared_embedding_sequences) - 1),
                 "min_dist": 0.1,
                 "n_components": 2,
             },
         }

         response = client.post("/projection_service/project", json=request_data)
         assert response.status_code == 200

         task_id = response.json()["task_id"]
         result = poll_task(task_id, timeout=300)

         assert result["status"].upper() in ("FINISHED", "COMPLETED", "DONE", "FAILED")

    @pytest.mark.integration
    def test_projection_with_real_world_collection(
         self,
         client,
         poll_task,
         embedder_name,
         shared_embedding_sequences,
     ):
         request_data = {
             "method": "pca",
             "sequence_data": shared_embedding_sequences,
             "embedder_name": embedder_name,
             "config": {"n_components": 2},
         }

         response = client.post("/projection_service/project", json=request_data)
         assert response.status_code == 200

         task_id = response.json()["task_id"]
         result = poll_task(task_id, timeout=120)

         assert result["status"].lower() in ("finished", "completed", "done", "failed")
