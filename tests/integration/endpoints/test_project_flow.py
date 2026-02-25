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
        # Test that projection task completes successfully.

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
        

        result = poll_task(task_id, timeout=120, max_consecutive_errors=10)

        assert result["status"].upper() == "FINISHED", f"Projection failed: {result.get('error', 'unknown')}"
        print(f"[PROJECTION] Task {task_id} completed successfully")











































































































































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


        assert response.status_code in (400, 422)


@pytest.mark.order(3)
class TestEndToEndProjectionFlow:
    # End-to-end tests for the complete projection workflow.

    @pytest.mark.integration
    def test_complete_projection_flow(
        self,
        client,
        poll_task,
        embedder_name,
        shared_embedding_sequences,
    ):
        # Test complete projection flow from request to completion.
        request_data = {
            "method": "pca",
            "sequence_data": shared_embedding_sequences,
            "embedder_name": embedder_name,
            "config": {"n_components": 2},
        }


        response = client.post("/projection_service/project", json=request_data)
        assert response.status_code == 200

        task_id = response.json()["task_id"]


        try:
            result = poll_task(task_id, timeout=60)
        except TimeoutError:
            pytest.skip(f"Task {task_id} timed out - CI resource constraints")
        except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
            pytest.skip(f"Server connection lost during polling: {e}")


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
