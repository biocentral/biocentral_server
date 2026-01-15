"""
Integration tests for projection endpoints.

Tests the /projection_service/* endpoints:
- GET /projection_config - Get available projection methods and configs
- POST /project - Calculate dimensionality reduction projections

Uses configurable embedder backend (FixedEmbedder or ESM-2 8M).
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biocentral_server.embeddings import projection_router


@pytest.fixture(scope="module")
def projection_app():
    """Create a FastAPI app with projection router for testing."""
    app = FastAPI()
    app.include_router(projection_router)
    return app


@pytest.fixture(scope="module")
def client(projection_app):
    """Create test client."""
    return TestClient(projection_app)


class TestProjectionConfigEndpoint:
    """
    Integration tests for GET /projection_service/projection_config.
    """

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.projection_endpoint.RateLimiter")
    def test_get_projection_config(
        self,
        mock_rate_limiter,
        client,
    ):
        """Test retrieving available projection methods and configurations."""
        mock_rate_limiter.return_value = lambda: None

        response = client.get("/projection_service/projection_config")

        assert response.status_code == 200
        response_json = response.json()
        assert "projection_config" in response_json
        
        # Should contain common dimension reduction methods
        config = response_json["projection_config"]
        assert isinstance(config, dict)
        # ProtSpace typically supports umap, tsne, pca
        assert len(config) > 0

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.projection_endpoint.RateLimiter")
    def test_projection_config_contains_umap(
        self,
        mock_rate_limiter,
        client,
    ):
        """Test that UMAP configuration is available."""
        mock_rate_limiter.return_value = lambda: None

        response = client.get("/projection_service/projection_config")

        assert response.status_code == 200
        config = response.json()["projection_config"]
        
        # UMAP should be available as a common method
        assert "umap" in config or "UMAP" in config


class TestProjectEndpoint:
    """
    Integration tests for POST /projection_service/project.
    """

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.projection_endpoint.TaskManager")
    @patch("biocentral_server.embeddings.projection_endpoint.UserManager")
    @patch("biocentral_server.embeddings.projection_endpoint.RateLimiter")
    def test_project_creates_task(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_task_manager,
        client,
        embedder_name,
        short_test_sequences,
    ):
        """Test that projection request creates a task."""
        mock_rate_limiter.return_value = lambda: None
        mock_task_manager.return_value.add_task.return_value = "project-task-123"
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")

        request_data = {
            "method": "umap",
            "sequence_data": short_test_sequences,
            "embedder_name": embedder_name,
            "config": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "n_components": 2,
            },
        }

        response = client.post("/projection_service/project", json=request_data)

        assert response.status_code == 200
        response_json = response.json()
        assert "task_id" in response_json
        assert response_json["task_id"] == "project-task-123"

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.projection_endpoint.TaskManager")
    @patch("biocentral_server.embeddings.projection_endpoint.UserManager")
    @patch("biocentral_server.embeddings.projection_endpoint.RateLimiter")
    def test_project_with_pca(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_task_manager,
        client,
        embedder_name,
        test_sequences,
    ):
        """Test projection with PCA method."""
        mock_rate_limiter.return_value = lambda: None
        mock_task_manager.return_value.add_task.return_value = "pca-task-123"
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")

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
        assert "task_id" in response.json()

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.projection_endpoint.TaskManager")
    @patch("biocentral_server.embeddings.projection_endpoint.UserManager")
    @patch("biocentral_server.embeddings.projection_endpoint.RateLimiter")
    def test_project_with_tsne(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_task_manager,
        client,
        embedder_name,
        test_sequences,
    ):
        """Test projection with t-SNE method."""
        mock_rate_limiter.return_value = lambda: None
        mock_task_manager.return_value.add_task.return_value = "tsne-task-123"
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")

        request_data = {
            "method": "tsne",
            "sequence_data": test_sequences,
            "embedder_name": embedder_name,
            "config": {
                "n_components": 2,
                "perplexity": 30,
            },
        }

        response = client.post("/projection_service/project", json=request_data)

        assert response.status_code == 200
        assert "task_id" in response.json()

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.projection_endpoint.RateLimiter")
    def test_project_invalid_method_rejected(
        self,
        mock_rate_limiter,
        client,
        embedder_name,
        short_test_sequences,
    ):
        """Test that invalid projection method is rejected."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "method": "invalid_method_xyz",
            "sequence_data": short_test_sequences,
            "embedder_name": embedder_name,
            "config": {},
        }

        response = client.post("/projection_service/project", json=request_data)

        assert response.status_code == 400

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.projection_endpoint.RateLimiter")
    def test_project_empty_sequences_rejected(
        self,
        mock_rate_limiter,
        client,
        embedder_name,
    ):
        """Test that empty sequence data is rejected."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "method": "umap",
            "sequence_data": {},
            "embedder_name": embedder_name,
            "config": {},
        }

        response = client.post("/projection_service/project", json=request_data)

        # Should fail validation
        assert response.status_code in (400, 422)


class TestProjectionWithEmbeddings:
    """
    Tests that verify projection logic with actual embeddings.
    
    These tests use the embedder fixture to generate embeddings
    and verify projection properties.
    """

    @pytest.mark.integration
    def test_embeddings_suitable_for_projection(
        self,
        embedder,
        embedding_dim,
        test_sequences,
    ):
        """Test that generated embeddings are suitable for projection."""
        # Generate pooled embeddings (required for projection)
        embeddings = embedder.embed_dict(test_sequences, pooled=True)
        
        # Should have same number of embeddings as sequences
        assert len(embeddings) == len(test_sequences)
        
        # All embeddings should have same dimension
        dims = [emb.shape[0] for emb in embeddings.values()]
        assert all(d == embedding_dim for d in dims)
        
        # Embeddings should be finite (no NaN/Inf)
        for emb in embeddings.values():
            assert np.isfinite(emb).all()

    @pytest.mark.integration
    def test_embedding_matrix_construction(
        self,
        embedder,
        test_sequences,
    ):
        """Test constructing embedding matrix for projection algorithms."""
        embeddings = embedder.embed_dict(test_sequences, pooled=True)
        
        # Convert to matrix form (n_samples, n_features)
        seq_ids = list(test_sequences.keys())
        embedding_matrix = np.stack([embeddings[sid] for sid in seq_ids])
        
        assert embedding_matrix.shape[0] == len(test_sequences)
        assert embedding_matrix.ndim == 2
        
        # Matrix should have no NaN values
        assert not np.isnan(embedding_matrix).any()

    @pytest.mark.integration
    def test_embedding_variance_for_projection(
        self,
        embedder,
        test_sequences,
    ):
        """Test that embeddings have sufficient variance for projection."""
        embeddings = embedder.embed_dict(test_sequences, pooled=True)
        
        # Convert to matrix
        embedding_matrix = np.stack(list(embeddings.values()))
        
        # Check variance across samples (should be non-zero for projection)
        sample_variance = np.var(embedding_matrix, axis=0)
        assert np.mean(sample_variance) > 0
        
        # At least some features should have significant variance
        assert np.sum(sample_variance > 1e-6) > 0

    @pytest.mark.integration  
    @pytest.mark.slow
    def test_actual_umap_projection(
        self,
        embedder,
        test_sequences,
    ):
        """
        Test actual UMAP projection on generated embeddings.
        
        Marked as slow because UMAP can take time.
        """
        try:
            import umap
        except ImportError:
            pytest.skip("UMAP not installed")
        
        # Generate embeddings
        embeddings = embedder.embed_dict(test_sequences, pooled=True)
        embedding_matrix = np.stack(list(embeddings.values()))
        
        # Run UMAP
        reducer = umap.UMAP(
            n_neighbors=min(3, len(test_sequences) - 1),
            n_components=2,
            min_dist=0.1,
            random_state=42,
        )
        projected = reducer.fit_transform(embedding_matrix)
        
        # Verify output shape
        assert projected.shape == (len(test_sequences), 2)
        
        # Projected points should be finite
        assert np.isfinite(projected).all()

    @pytest.mark.integration
    def test_pca_projection(
        self,
        embedder,
        test_sequences,
    ):
        """Test PCA projection on generated embeddings."""
        from sklearn.decomposition import PCA
        
        # Generate embeddings
        embeddings = embedder.embed_dict(test_sequences, pooled=True)
        embedding_matrix = np.stack(list(embeddings.values()))
        
        # Run PCA
        n_components = min(2, len(test_sequences) - 1, embedding_matrix.shape[1])
        pca = PCA(n_components=n_components)
        projected = pca.fit_transform(embedding_matrix)
        
        # Verify output shape
        assert projected.shape == (len(test_sequences), n_components)
        
        # Check explained variance
        assert pca.explained_variance_ratio_.sum() > 0
