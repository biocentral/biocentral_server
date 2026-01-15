"""
Integration tests for prediction endpoints.

Tests the /prediction_service/* endpoints:
- GET /model_metadata - Get available prediction models
- POST /predict - Submit prediction task

Uses configurable embedder backend (FixedEmbedder or ESM-2 8M).
"""

import pytest
import numpy as np
from typing import Dict, List
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biocentral_server.predict import predict_router


@pytest.fixture(scope="module")
def predict_app():
    """Create a FastAPI app with predict router for testing."""
    app = FastAPI()
    app.include_router(predict_router)
    return app


@pytest.fixture(scope="module")
def client(predict_app):
    """Create test client."""
    return TestClient(predict_app)


@pytest.fixture
def prediction_sequences() -> Dict[str, str]:
    """Sequences suitable for prediction (minimum 7 residues)."""
    return {
        "pred_1": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGN",
        "pred_2": "MKKLVLSLSLVLAFSSATAAFAAIPQNIRAQYPAVVKEQRQVVRSQNGDLADNIKKISDNLKAKIYAMHYVDVFYNKSLEKIMKDIQVTNATKTVYISINDLKRRMGGWKYPNMQVLLGRKGKKGKKAKRQ",
    }


class TestModelMetadataEndpoint:
    """
    Integration tests for GET /prediction_service/model_metadata.
    """

    @pytest.mark.integration
    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_get_model_metadata(
        self,
        mock_rate_limiter,
        client,
    ):
        """Test retrieving available model metadata."""
        mock_rate_limiter.return_value = lambda: None

        response = client.get("/prediction_service/model_metadata")

        assert response.status_code == 200
        response_json = response.json()
        assert "metadata" in response_json
        metadata = response_json["metadata"]
        assert isinstance(metadata, dict)
        assert len(metadata) > 0

    @pytest.mark.integration
    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_model_metadata_structure(
        self,
        mock_rate_limiter,
        client,
    ):
        """Test that model metadata has expected structure."""
        mock_rate_limiter.return_value = lambda: None

        response = client.get("/prediction_service/model_metadata")
        metadata = response.json()["metadata"]

        # Each model should have metadata
        for model_name, model_meta in metadata.items():
            assert isinstance(model_name, str)
            assert isinstance(model_meta, dict)
            # Should have basic metadata fields
            assert len(model_meta) > 0

    @pytest.mark.integration
    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_expected_models_available(
        self,
        mock_rate_limiter,
        client,
    ):
        """Test that expected prediction models are available."""
        mock_rate_limiter.return_value = lambda: None

        response = client.get("/prediction_service/model_metadata")
        metadata = response.json()["metadata"]
        model_names_lower = {name.lower() for name in metadata.keys()}

        # At least some of these common models should be available
        expected_models = ["bindembed", "tmbed", "seth"]
        available = [m for m in expected_models if m in model_names_lower]
        
        # At least one model should be available
        assert len(available) > 0


class TestPredictEndpoint:
    """
    Integration tests for POST /prediction_service/predict.
    """

    @pytest.mark.integration
    @patch("biocentral_server.predict.predict_endpoint.TaskManager")
    @patch("biocentral_server.predict.predict_endpoint.UserManager")
    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_predict_creates_task(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_task_manager,
        client,
        prediction_sequences,
    ):
        """Test that prediction request creates a task."""
        mock_rate_limiter.return_value = lambda: None
        mock_task_manager.return_value.add_task.return_value = "predict-task-123"
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")

        # Get available models first
        with patch("biocentral_server.predict.predict_endpoint.RateLimiter") as rl:
            rl.return_value = lambda: None
            meta_response = client.get("/prediction_service/model_metadata")
            available_models = list(meta_response.json()["metadata"].keys())

        if not available_models:
            pytest.skip("No prediction models available")

        request_data = {
            "model_names": [available_models[0]],
            "sequence_input": prediction_sequences,
        }

        response = client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 200
        response_json = response.json()
        assert "task_id" in response_json
        assert response_json["task_id"] == "predict-task-123"

    @pytest.mark.integration
    @patch("biocentral_server.predict.predict_endpoint.TaskManager")
    @patch("biocentral_server.predict.predict_endpoint.UserManager")
    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_predict_multiple_models(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_task_manager,
        client,
        prediction_sequences,
    ):
        """Test prediction with multiple models."""
        mock_rate_limiter.return_value = lambda: None
        mock_task_manager.return_value.add_task.return_value = "multi-predict-task"
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")

        # Get available models
        with patch("biocentral_server.predict.predict_endpoint.RateLimiter") as rl:
            rl.return_value = lambda: None
            meta_response = client.get("/prediction_service/model_metadata")
            available_models = list(meta_response.json()["metadata"].keys())

        if len(available_models) < 2:
            pytest.skip("Need at least 2 models for this test")

        request_data = {
            "model_names": available_models[:2],
            "sequence_input": prediction_sequences,
        }

        response = client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 200
        assert "task_id" in response.json()

    @pytest.mark.integration
    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_predict_invalid_model_rejected(
        self,
        mock_rate_limiter,
        client,
        prediction_sequences,
    ):
        """Test that invalid model name returns error."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "model_names": ["invalid_model_xyz_123"],
            "sequence_input": prediction_sequences,
        }

        response = client.post("/prediction_service/predict", json=request_data)

        # Should return 404 (not found) for unknown model
        assert response.status_code == 200  # Returns NotFoundErrorResponse with 200
        response_json = response.json()
        assert "error" in response_json

    @pytest.mark.integration
    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_predict_empty_sequences_rejected(
        self,
        mock_rate_limiter,
        client,
    ):
        """Test that empty sequence input is rejected."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "model_names": ["BindEmbed"],
            "sequence_input": {},  # Empty
        }

        response = client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 422

    @pytest.mark.integration
    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_predict_short_sequence_rejected(
        self,
        mock_rate_limiter,
        client,
    ):
        """Test that sequences shorter than minimum length are rejected."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "model_names": ["BindEmbed"],
            "sequence_input": {"short": "MKT"},  # Too short (< 7)
        }

        response = client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 422

    @pytest.mark.integration
    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_predict_empty_model_names_rejected(
        self,
        mock_rate_limiter,
        client,
        prediction_sequences,
    ):
        """Test that empty model names list is rejected."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "model_names": [],  # Empty model list
            "sequence_input": prediction_sequences,
        }

        response = client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 422


class TestPredictionWithEmbeddings:
    """
    Tests that verify prediction prerequisites with embeddings.
    
    These tests use the embedder fixture to verify that embeddings
    generated are suitable for prediction models.
    """

    @pytest.mark.integration
    def test_embeddings_for_prediction_sequences(
        self,
        embedder,
        embedding_dim,
        prediction_sequences,
    ):
        """Test generating embeddings for prediction sequences."""
        embeddings = embedder.embed_dict(prediction_sequences, pooled=True)
        
        assert len(embeddings) == len(prediction_sequences)
        for seq_id in prediction_sequences.keys():
            assert seq_id in embeddings
            assert embeddings[seq_id].shape == (embedding_dim,)
            assert np.isfinite(embeddings[seq_id]).all()

    @pytest.mark.integration
    def test_per_residue_embeddings_for_prediction(
        self,
        embedder,
        embedding_dim,
        prediction_sequences,
    ):
        """Test per-residue embeddings for residue-level predictions."""
        embeddings = embedder.embed_dict(prediction_sequences, pooled=False)
        
        for seq_id, seq in prediction_sequences.items():
            assert seq_id in embeddings
            emb = embeddings[seq_id]
            assert emb.shape == (len(seq), embedding_dim)
            assert np.isfinite(emb).all()

    @pytest.mark.integration
    def test_embedding_batch_consistency(
        self,
        embedder,
        prediction_sequences,
    ):
        """Test that batch and individual embeddings are consistent."""
        # Generate individually
        individual = {}
        for seq_id, seq in prediction_sequences.items():
            individual[seq_id] = embedder.embed_pooled(seq)
        
        # Generate as batch
        batch = embedder.embed_dict(prediction_sequences, pooled=True)
        
        # Should produce same results
        for seq_id in prediction_sequences.keys():
            np.testing.assert_array_almost_equal(
                individual[seq_id],
                batch[seq_id],
                decimal=5,
            )


class TestPredictionModelRegistry:
    """
    Tests for prediction model registry and metadata.
    """

    @pytest.mark.integration
    def test_model_registry_exists(self):
        """Test that model registry is populated."""
        from biocentral_server.predict.models import MODEL_REGISTRY
        
        assert isinstance(MODEL_REGISTRY, dict)
        assert len(MODEL_REGISTRY) > 0

    @pytest.mark.integration
    def test_model_metadata_accessible(self):
        """Test that model metadata can be retrieved."""
        from biocentral_server.predict.models import get_metadata_for_all_models
        
        metadata = get_metadata_for_all_models()
        
        assert isinstance(metadata, dict)
        assert len(metadata) > 0

    @pytest.mark.integration
    def test_model_filtering(self):
        """Test filtering models by name."""
        from biocentral_server.predict.models import filter_models, MODEL_REGISTRY
        
        # Get first available model
        model_names = list(MODEL_REGISTRY.keys())
        if not model_names:
            pytest.skip("No models in registry")
        
        filtered = filter_models([model_names[0]])
        
        assert len(filtered) == 1
        assert model_names[0] in filtered


class TestEndToEndPredictionFlow:
    """
    End-to-end tests for the complete prediction flow.
    """

    @pytest.mark.integration
    def test_sequence_validation_for_prediction(
        self,
        prediction_sequences,
    ):
        """Test that sequences meet prediction requirements."""
        min_length = 7
        max_length = 5000
        
        for seq_id, seq in prediction_sequences.items():
            assert len(seq) >= min_length, f"{seq_id} too short"
            assert len(seq) <= max_length, f"{seq_id} too long"
            # Should only contain valid amino acids
            valid_aas = set("ACDEFGHIKLMNPQRSTVWYX")
            assert all(aa in valid_aas for aa in seq.upper())

    @pytest.mark.integration
    def test_prediction_input_format(
        self,
        prediction_sequences,
    ):
        """Test prediction input format matches API requirements."""
        # Validate structure
        assert isinstance(prediction_sequences, dict)
        assert all(isinstance(k, str) for k in prediction_sequences.keys())
        assert all(isinstance(v, str) for v in prediction_sequences.values())

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_prediction_flow_mocked(
        self,
        embedder,
        prediction_sequences,
    ):
        """
        Test complete prediction flow with mocked components.
        
        This tests the flow: sequences -> embeddings -> prediction format
        """
        # Step 1: Generate embeddings
        pooled_embeddings = embedder.embed_dict(prediction_sequences, pooled=True)
        per_residue_embeddings = embedder.embed_dict(prediction_sequences, pooled=False)
        
        # Step 2: Verify embeddings are suitable
        for seq_id, seq in prediction_sequences.items():
            pooled = pooled_embeddings[seq_id]
            per_residue = per_residue_embeddings[seq_id]
            
            # Pooled should be 1D
            assert pooled.ndim == 1
            
            # Per-residue should match sequence length
            assert per_residue.shape[0] == len(seq)
            
            # All values should be finite
            assert np.isfinite(pooled).all()
            assert np.isfinite(per_residue).all()
