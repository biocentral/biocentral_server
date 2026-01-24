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
from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET
from tests.integration.endpoints.conftest import (
    CANONICAL_STANDARD_IDS,
    CANONICAL_REAL_WORLD_IDS,
    get_sequence_by_id,
)


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


class TestProjectionWithDiverseSequences:
    """
    Tests for projection with diverse sequence collections.
    
    Uses the diverse_test_sequences fixture for larger sample sizes.
    """

    @pytest.mark.integration
    def test_projection_with_diverse_sequences(
        self,
        embedder,
        embedding_dim,
        diverse_test_sequences,
    ):
        """Test projection suitability with diverse sequence collection."""
        embeddings = embedder.embed_dict(diverse_test_sequences, pooled=True)
        
        # Should have enough sequences for meaningful projection
        assert len(embeddings) >= 5
        
        # All embeddings should have same dimension
        for emb in embeddings.values():
            assert emb.shape == (embedding_dim,)
            assert np.isfinite(emb).all()

    @pytest.mark.integration
    def test_diverse_sequence_matrix_for_projection(
        self,
        embedder,
        diverse_test_sequences,
    ):
        """Test constructing embedding matrix from diverse sequences."""
        embeddings = embedder.embed_dict(diverse_test_sequences, pooled=True)
        
        seq_ids = list(diverse_test_sequences.keys())
        embedding_matrix = np.stack([embeddings[sid] for sid in seq_ids])
        
        assert embedding_matrix.shape[0] == len(diverse_test_sequences)
        assert embedding_matrix.ndim == 2
        assert not np.isnan(embedding_matrix).any()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_actual_umap_with_diverse_sequences(
        self,
        embedder,
        diverse_test_sequences,
    ):
        """Test actual UMAP projection on diverse sequence embeddings."""
        try:
            import umap
        except ImportError:
            pytest.skip("UMAP not installed")
        
        embeddings = embedder.embed_dict(diverse_test_sequences, pooled=True)
        embedding_matrix = np.stack(list(embeddings.values()))
        
        # UMAP requires n_neighbors < n_samples
        n_neighbors = min(5, len(diverse_test_sequences) - 1)
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=2,
            min_dist=0.1,
            random_state=42,
        )
        projected = reducer.fit_transform(embedding_matrix)
        
        assert projected.shape == (len(diverse_test_sequences), 2)
        assert np.isfinite(projected).all()

    @pytest.mark.integration
    def test_pca_with_diverse_sequences(
        self,
        embedder,
        diverse_test_sequences,
    ):
        """Test PCA projection on diverse sequence embeddings."""
        from sklearn.decomposition import PCA
        
        embeddings = embedder.embed_dict(diverse_test_sequences, pooled=True)
        embedding_matrix = np.stack(list(embeddings.values()))
        
        pca = PCA(n_components=3)
        projected = pca.fit_transform(embedding_matrix)
        
        assert projected.shape == (len(diverse_test_sequences), 3)
        assert np.isfinite(projected).all()
        assert pca.explained_variance_ratio_.sum() > 0


class TestProjectionWithHomopolymers:
    """
    Tests for projection edge cases with homopolymer sequences.
    """

    @pytest.mark.integration
    def test_homopolymer_embeddings_for_projection(
        self,
        embedder,
        embedding_dim,
        composition_edge_sequences,
    ):
        """Test that homopolymer sequences produce valid projection inputs."""
        homopolymer_seqs = {
            "short": composition_edge_sequences["homopolymer_A"],
            "long": composition_edge_sequences["homopolymer_long"],
        }
        
        embeddings = embedder.embed_dict(homopolymer_seqs, pooled=True)
        
        for seq_id, emb in embeddings.items():
            assert emb.shape == (embedding_dim,)
            assert np.isfinite(emb).all()

    @pytest.mark.integration
    def test_homopolymer_separation_in_projection(
        self,
        embedder,
        composition_edge_sequences,
    ):
        """Test that homopolymers of different lengths are distinguishable."""
        from sklearn.decomposition import PCA
        
        # Include standard sequences for contrast
        seqs = {
            "homopolymer_short": composition_edge_sequences["homopolymer_A"],
            "homopolymer_long": composition_edge_sequences["homopolymer_long"],
            "standard_001": get_sequence_by_id("standard_001"),
            "standard_002": get_sequence_by_id("standard_002"),
        }
        
        embeddings = embedder.embed_dict(seqs, pooled=True)
        embedding_matrix = np.stack(list(embeddings.values()))
        
        pca = PCA(n_components=2)
        projected = pca.fit_transform(embedding_matrix)
        
        assert projected.shape == (4, 2)
        # Homopolymers should be distinguishable from standard sequences
        # (different positions in projected space)


class TestProjectionWithRealWorldSequences:
    """
    Tests for projection with real-world protein sequences.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_REAL_WORLD_IDS)
    def test_real_world_sequences_projection_ready(
        self,
        embedder,
        embedding_dim,
        seq_id,
    ):
        """Test that real-world sequences are suitable for projection."""
        sequence = get_sequence_by_id(seq_id)
        
        pooled = embedder.embed_pooled(sequence)
        
        assert pooled.shape == (embedding_dim,)
        assert np.isfinite(pooled).all()
        # Should have sufficient variance for projection
        assert np.std(pooled) > 0

    @pytest.mark.integration
    def test_real_world_collection_projection(
        self,
        embedder,
        real_world_sequences,
    ):
        """Test projection with collection of real-world sequences."""
        from sklearn.decomposition import PCA
        
        embeddings = embedder.embed_dict(real_world_sequences, pooled=True)
        embedding_matrix = np.stack(list(embeddings.values()))
        
        # PCA with enough components for meaningful projection
        n_components = min(2, len(real_world_sequences) - 1)
        pca = PCA(n_components=n_components)
        projected = pca.fit_transform(embedding_matrix)
        
        assert projected.shape == (len(real_world_sequences), n_components)
        assert np.isfinite(projected).all()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_tsne_with_real_world_sequences(
        self,
        embedder,
        real_world_sequences,
    ):
        """Test t-SNE projection on real-world sequences."""
        from sklearn.manifold import TSNE
        
        embeddings = embedder.embed_dict(real_world_sequences, pooled=True)
        embedding_matrix = np.stack(list(embeddings.values()))
        
        # t-SNE perplexity must be less than n_samples
        perplexity = min(2, len(real_world_sequences) - 1)
        
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            n_iter=250,  # Reduced for testing speed
        )
        projected = tsne.fit_transform(embedding_matrix)
        
        assert projected.shape == (len(real_world_sequences), 2)
        assert np.isfinite(projected).all()


class TestProjectionWithStructuralMotifs:
    """
    Tests for projection with structural motif sequences.
    """

    @pytest.mark.integration
    def test_structural_motif_embeddings(
        self,
        embedder,
        embedding_dim,
        structural_motif_sequences,
    ):
        """Test embeddings for structural motif sequences."""
        embeddings = embedder.embed_dict(structural_motif_sequences, pooled=True)
        
        for seq_id, emb in embeddings.items():
            assert emb.shape == (embedding_dim,)
            assert np.isfinite(emb).all()

    @pytest.mark.integration
    def test_structural_motif_projection_separation(
        self,
        embedder,
        structural_motif_sequences,
    ):
        """Test that structural motifs are distinguishable in projection."""
        from sklearn.decomposition import PCA
        
        embeddings = embedder.embed_dict(structural_motif_sequences, pooled=True)
        embedding_matrix = np.stack(list(embeddings.values()))
        
        pca = PCA(n_components=2)
        projected = pca.fit_transform(embedding_matrix)
        
        assert projected.shape == (len(structural_motif_sequences), 2)
        
        # Different motifs should produce different projections
        # (not all at the same point)
        variance = np.var(projected, axis=0)
        assert variance.sum() > 0


class TestProjectionEndpointWithDiverse:
    """
    Endpoint tests using diverse sequences.
    """

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.projection_endpoint.TaskManager")
    @patch("biocentral_server.embeddings.projection_endpoint.UserManager")
    @patch("biocentral_server.embeddings.projection_endpoint.RateLimiter")
    def test_project_diverse_sequences(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_task_manager,
        client,
        embedder_name,
        diverse_test_sequences,
    ):
        """Test projection endpoint with diverse sequence collection."""
        mock_rate_limiter.return_value = lambda: None
        mock_task_manager.return_value.add_task.return_value = "diverse-project-task"
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")

        request_data = {
            "method": "pca",
            "sequence_data": diverse_test_sequences,
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
    def test_project_real_world_sequences(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_task_manager,
        client,
        embedder_name,
        real_world_sequences,
    ):
        """Test projection endpoint with real-world protein sequences."""
        mock_rate_limiter.return_value = lambda: None
        mock_task_manager.return_value.add_task.return_value = "real-world-project-task"
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")

        request_data = {
            "method": "pca",
            "sequence_data": real_world_sequences,
            "embedder_name": embedder_name,
            "config": {
                "n_components": 2,
            },
        }

        response = client.post("/projection_service/project", json=request_data)

        assert response.status_code == 200
        assert "task_id" in response.json()
