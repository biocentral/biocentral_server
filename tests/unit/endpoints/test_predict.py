"""
Unit tests for predict endpoint.

Tests endpoints:
- GET /prediction_service/model_metadata
- POST /prediction_service/predict
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biocentral_server.predict import router as predict_router


@pytest.fixture
def predict_app():
    """Create a FastAPI app with predict router for testing."""
    app = FastAPI()
    app.include_router(predict_router)
    return app


@pytest.fixture
def predict_client(predict_app):
    """Create test client for predict endpoints."""
    return TestClient(predict_app)


class TestModelMetadataEndpoint:
    """Tests for GET /prediction_service/model_metadata"""

    @patch("biocentral_server.predict.predict_endpoint.get_metadata_for_all_models")
    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_model_metadata_returns_available_models(
        self, mock_rate_limiter, mock_get_metadata, predict_client
    ):
        """Test that model metadata endpoint returns available models."""
        mock_rate_limiter.return_value = lambda: None

        # Create mock metadata objects
        mock_metadata = MagicMock()
        mock_metadata.to_dict.return_value = {
            "name": "BindEmbed",
            "description": "Binding site prediction",
            "input_type": "sequence",
            "output_type": "per_residue",
        }
        mock_get_metadata.return_value = {"BindEmbed": mock_metadata}

        response = predict_client.get("/prediction_service/model_metadata")

        assert response.status_code == 200
        data = response.json()
        assert "metadata" in data
        assert "BindEmbed" in data["metadata"]

    @patch("biocentral_server.predict.predict_endpoint.get_metadata_for_all_models")
    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_model_metadata_empty_when_no_models(
        self, mock_rate_limiter, mock_get_metadata, predict_client
    ):
        """Test metadata endpoint returns empty dict when no models available."""
        mock_rate_limiter.return_value = lambda: None
        mock_get_metadata.return_value = {}

        response = predict_client.get("/prediction_service/model_metadata")

        assert response.status_code == 200
        assert response.json()["metadata"] == {}


class TestPredictEndpoint:
    """Tests for POST /prediction_service/predict"""

    @patch("biocentral_server.predict.predict_endpoint.TaskManager")
    @patch("biocentral_server.predict.predict_endpoint.UserManager")
    @patch("biocentral_server.predict.predict_endpoint.filter_models")
    @patch("biocentral_server.predict.predict_endpoint.get_metadata_for_all_models")
    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_predict_valid_request(
        self,
        mock_rate_limiter,
        mock_get_metadata,
        mock_filter_models,
        mock_user_manager,
        mock_task_manager,
        predict_client,
    ):
        """Test prediction with valid model and sequence data."""
        mock_rate_limiter.return_value = lambda: None
        mock_get_metadata.return_value = {"bindembed": MagicMock()}
        mock_filter_models.return_value = [MagicMock()]
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="user-1")
        mock_task_manager.return_value.add_task.return_value = "predict-task-123"

        request_data = {
            "model_names": ["BindEmbed"],
            "sequence_input": {
                "protein1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            },
        }

        response = predict_client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 200
        assert "task_id" in response.json()

    @patch("biocentral_server.predict.predict_endpoint.get_metadata_for_all_models")
    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_predict_unknown_model(
        self, mock_rate_limiter, mock_get_metadata, predict_client
    ):
        """Test prediction with non-existent model returns 404."""
        mock_rate_limiter.return_value = lambda: None
        mock_get_metadata.return_value = {"bindembed": MagicMock()}

        request_data = {
            "model_names": ["NonExistentModel"],
            "sequence_input": {
                "protein1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            },
        }

        response = predict_client.post("/prediction_service/predict", json=request_data)

        # Should return NotFoundErrorResponse
        assert response.status_code == 200  # FastAPI returns model, not HTTP error
        data = response.json()
        assert "error" in data or "error_code" in data

    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_predict_empty_model_names(self, mock_rate_limiter, predict_client):
        """Test prediction with empty model names list fails validation."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "model_names": [],
            "sequence_input": {"protein1": "MVLSPADKTNVKAAWGKVGAHAGE"},
        }

        response = predict_client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 422  # Validation error

    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_predict_empty_sequences(self, mock_rate_limiter, predict_client):
        """Test prediction with empty sequence data fails validation."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "model_names": ["BindEmbed"],
            "sequence_input": {},
        }

        response = predict_client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 422  # Validation error

    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_predict_sequence_too_short(self, mock_rate_limiter, predict_client):
        """Test prediction with sequence shorter than minimum fails."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "model_names": ["BindEmbed"],
            "sequence_input": {"protein1": "MVL"},  # Too short (min is 7)
        }

        response = predict_client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 422  # Validation error

    @patch("biocentral_server.predict.predict_endpoint.RateLimiter")
    def test_predict_sequence_too_long(self, mock_rate_limiter, predict_client):
        """Test prediction with sequence longer than maximum fails."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "model_names": ["BindEmbed"],
            "sequence_input": {"protein1": "M" * 5001},  # Too long (max is 5000)
        }

        response = predict_client.post("/prediction_service/predict", json=request_data)

        assert response.status_code == 422  # Validation error
