"""
Unit tests for embeddings endpoint.

Tests endpoints:
- POST /embeddings_service/embed
- POST /embeddings_service/get_missing_embeddings
- POST /embeddings_service/add_embeddings
"""

import json
import base64
import io
import h5py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biocentral_server.embeddings import embeddings_router


@pytest.fixture
def embeddings_app():
    """Create a FastAPI app with embeddings router for testing."""
    app = FastAPI()
    app.include_router(embeddings_router)
    return app


@pytest.fixture
def embeddings_client(embeddings_app):
    """Create test client with mocked dependencies."""
    return TestClient(embeddings_app)


class TestEmbedEndpoint:
    """Tests for POST /embeddings_service/embed"""

    @patch("biocentral_server.embeddings.embeddings_endpoint.TaskManager")
    @patch("biocentral_server.embeddings.embeddings_endpoint.UserManager")
    @patch("biocentral_server.embeddings.embeddings_endpoint.MetricsCollector")
    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_embed_valid_request(
        self,
        mock_rate_limiter,
        mock_metrics,
        mock_user_manager,
        mock_task_manager,
        embeddings_client,
    ):
        """Test embedding request with valid sequence data."""
        # Setup mocks
        mock_rate_limiter.return_value = lambda: None
        mock_task_manager.return_value.add_task.return_value = "task-123"
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="user-1")

        request_data = {
            "embedder_name": "prot_t5_xl_uniref50",
            "reduce": False,
            "sequence_data": {
                "seq1": "MVLSPADKTNVKAAWGKVGAHAGE",
                "seq2": "MGHFTEEDKATITSLWGKVNVE",
            },
            "use_half_precision": False,
        }

        response = embeddings_client.post(
            "/embeddings_service/embed", json=request_data
        )

        # Verify task was submitted
        assert response.status_code == 200
        assert "task_id" in response.json()

    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_embed_empty_sequences(self, mock_rate_limiter, embeddings_client):
        """Test embedding request with empty sequence data fails validation."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "embedder_name": "prot_t5_xl_uniref50",
            "reduce": False,
            "sequence_data": {},  # Empty - should fail
            "use_half_precision": False,
        }

        response = embeddings_client.post(
            "/embeddings_service/embed", json=request_data
        )

        assert response.status_code == 422  # Validation error

    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_embed_missing_embedder_name(self, mock_rate_limiter, embeddings_client):
        """Test embedding request without embedder name fails."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "reduce": False,
            "sequence_data": {"seq1": "MVLSPADKTNVKAAWGKVGAHAGE"},
            "use_half_precision": False,
        }

        response = embeddings_client.post(
            "/embeddings_service/embed", json=request_data
        )

        assert response.status_code == 422  # Validation error


class TestGetMissingEmbeddingsEndpoint:
    """Tests for POST /embeddings_service/get_missing_embeddings"""

    @patch("biocentral_server.embeddings.embeddings_endpoint.EmbeddingDatabaseFactory")
    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_get_missing_embeddings_valid(
        self, mock_rate_limiter, mock_db_factory, embeddings_client
    ):
        """Test checking missing embeddings with valid request."""
        mock_rate_limiter.return_value = lambda: None

        mock_db = MagicMock()
        mock_db.get_missing_embeddings.return_value = ["seq2"]
        mock_db_factory.get_database.return_value = mock_db

        request_data = {
            "sequences": json.dumps({"seq1": "MVLSPAD", "seq2": "MGHFTEE"}),
            "embedder_name": "prot_t5_xl_uniref50",
            "reduced": True,
        }

        response = embeddings_client.post(
            "/embeddings_service/get_missing_embeddings", json=request_data
        )

        assert response.status_code == 200
        assert "missing" in response.json()

    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_get_missing_embeddings_invalid_json(
        self, mock_rate_limiter, embeddings_client
    ):
        """Test with invalid JSON in sequences field."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "sequences": "not valid json {",
            "embedder_name": "prot_t5_xl_uniref50",
            "reduced": True,
        }

        response = embeddings_client.post(
            "/embeddings_service/get_missing_embeddings", json=request_data
        )

        assert response.status_code == 422  # Pydantic validation error

    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_get_missing_embeddings_sequences_not_dict(
        self, mock_rate_limiter, embeddings_client
    ):
        """Test with sequences that parse to non-dict JSON."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "sequences": json.dumps(["seq1", "seq2"]),  # List instead of dict
            "embedder_name": "prot_t5_xl_uniref50",
            "reduced": True,
        }

        response = embeddings_client.post(
            "/embeddings_service/get_missing_embeddings", json=request_data
        )

        assert response.status_code == 422  # Validation error


class TestAddEmbeddingsEndpoint:
    """Tests for POST /embeddings_service/add_embeddings"""

    @staticmethod
    def _create_h5_bytes(embeddings_dict: dict) -> str:
        """Helper to create base64-encoded HDF5 file."""
        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as f:
            for key, value in embeddings_dict.items():
                f.create_dataset(key, data=value)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    @patch("biocentral_server.embeddings.embeddings_endpoint.EmbeddingDatabaseFactory")
    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_add_embeddings_valid(
        self, mock_rate_limiter, mock_db_factory, embeddings_client
    ):
        """Test adding embeddings with valid HDF5 data."""
        mock_rate_limiter.return_value = lambda: None
        mock_db = MagicMock()
        mock_db_factory.get_database.return_value = mock_db

        import numpy as np

        embeddings_data = {
            "seq1": np.random.rand(1024).astype(np.float32),
            "seq2": np.random.rand(1024).astype(np.float32),
        }
        h5_bytes = self._create_h5_bytes(embeddings_data)

        request_data = {
            "h5_bytes": h5_bytes,
            "embedder_name": "prot_t5_xl_uniref50",
            "reduced": True,
        }

        response = embeddings_client.post(
            "/embeddings_service/add_embeddings", json=request_data
        )

        # Should succeed or return appropriate status
        assert response.status_code in [200, 201]

    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_add_embeddings_invalid_base64(
        self, mock_rate_limiter, embeddings_client
    ):
        """Test adding embeddings with invalid base64 data."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "h5_bytes": "not-valid-base64!!!",
            "embedder_name": "prot_t5_xl_uniref50",
            "reduced": True,
        }

        response = embeddings_client.post(
            "/embeddings_service/add_embeddings", json=request_data
        )

        # Should fail with validation or processing error
        assert response.status_code in [400, 422, 500]
