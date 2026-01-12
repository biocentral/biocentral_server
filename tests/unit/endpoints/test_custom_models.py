"""
Unit tests for custom models endpoint.

Tests endpoints:
- GET /custom_models_service/protocols
- GET /custom_models_service/config_options/{protocol}
- POST /custom_models_service/verify_config
- POST /custom_models_service/start_training
- POST /custom_models_service/model_files
- POST /custom_models_service/start_inference
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biocentral_server.custom_models import router as custom_models_router


@pytest.fixture
def custom_models_app():
    """Create a FastAPI app with custom models router for testing."""
    app = FastAPI()
    app.include_router(custom_models_router)
    return app


@pytest.fixture
def custom_models_client(custom_models_app):
    """Create test client for custom models endpoints."""
    return TestClient(custom_models_app)


class TestProtocolsEndpoint:
    """Tests for GET /custom_models_service/protocols"""

    @patch("biocentral_server.custom_models.custom_models_endpoint.Protocol")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_protocols_returns_list(
        self, mock_rate_limiter, mock_protocol, custom_models_client
    ):
        """Test protocols endpoint returns list of available protocols."""
        mock_rate_limiter.return_value = lambda: None
        mock_protocol.all.return_value = [
            MagicMock(__str__=lambda self: "residue_to_class"),
            MagicMock(__str__=lambda self: "sequence_to_class"),
            MagicMock(__str__=lambda self: "sequence_to_value"),
        ]

        response = custom_models_client.get("/custom_models_service/protocols")

        assert response.status_code == 200
        data = response.json()
        assert "protocols" in data
        assert isinstance(data["protocols"], list)


class TestConfigOptionsEndpoint:
    """Tests for GET /custom_models_service/config_options/{protocol}"""

    @patch("biocentral_server.custom_models.custom_models_endpoint.Configurator")
    @patch("biocentral_server.custom_models.custom_models_endpoint.BiotrainerTask")
    @patch("biocentral_server.custom_models.custom_models_endpoint.Protocol")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_config_options_valid_protocol(
        self,
        mock_rate_limiter,
        mock_protocol,
        mock_biotrainer_task,
        mock_configurator,
        custom_models_client,
    ):
        """Test config options for valid protocol returns options."""
        mock_rate_limiter.return_value = lambda: None
        mock_protocol.from_string.return_value = MagicMock()
        mock_configurator.get_option_dicts_by_protocol.return_value = [
            {"name": "learning_rate", "type": "float", "default": 0.001},
            {"name": "batch_size", "type": "int", "default": 32},
        ]
        mock_biotrainer_task.get_config_presets.return_value = {}

        response = custom_models_client.get(
            "/custom_models_service/config_options/residue_to_class"
        )

        assert response.status_code == 200
        data = response.json()
        assert "options" in data
        assert len(data["options"]) == 2

    @patch("biocentral_server.custom_models.custom_models_endpoint.Protocol")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_config_options_invalid_protocol(
        self, mock_rate_limiter, mock_protocol, custom_models_client
    ):
        """Test config options for invalid protocol returns 400."""
        mock_rate_limiter.return_value = lambda: None
        mock_protocol.from_string.side_effect = ValueError("Invalid protocol")

        response = custom_models_client.get(
            "/custom_models_service/config_options/invalid_protocol"
        )

        assert response.status_code == 400
        assert "Invalid protocol" in response.json()["detail"]


class TestVerifyConfigEndpoint:
    """Tests for POST /custom_models_service/verify_config"""

    @patch("biocentral_server.custom_models.custom_models_endpoint.verify_biotrainer_config")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_verify_config_valid(
        self, mock_rate_limiter, mock_verify, custom_models_client
    ):
        """Test config verification with valid config returns no error."""
        mock_rate_limiter.return_value = lambda: None
        mock_verify.return_value = ({}, "")  # Empty error means valid

        request_data = {
            "config_dict": {
                "protocol": "residue_to_class",
                "learning_rate": 0.001,
            }
        }

        response = custom_models_client.post(
            "/custom_models_service/verify_config/", json=request_data
        )

        assert response.status_code == 200
        assert response.json()["error"] == ""

    @patch("biocentral_server.custom_models.custom_models_endpoint.verify_biotrainer_config")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_verify_config_invalid(
        self, mock_rate_limiter, mock_verify, custom_models_client
    ):
        """Test config verification with invalid config returns error message."""
        mock_rate_limiter.return_value = lambda: None
        mock_verify.return_value = ({}, "Missing required field: protocol")

        request_data = {"config_dict": {"learning_rate": 0.001}}

        response = custom_models_client.post(
            "/custom_models_service/verify_config/", json=request_data
        )

        assert response.status_code == 200
        assert "Missing required field" in response.json()["error"]

    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_verify_config_empty_dict(self, mock_rate_limiter, custom_models_client):
        """Test config verification with empty dict fails validation."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {"config_dict": {}}

        response = custom_models_client.post(
            "/custom_models_service/verify_config/", json=request_data
        )

        assert response.status_code == 422  # Validation error (min_length=1)


class TestStartTrainingEndpoint:
    """Tests for POST /custom_models_service/start_training"""

    @patch("biocentral_server.custom_models.custom_models_endpoint.TaskManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.verify_biotrainer_config")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_training_valid(
        self,
        mock_rate_limiter,
        mock_verify,
        mock_user_manager,
        mock_file_manager,
        mock_task_manager,
        custom_models_client,
    ):
        """Test starting training with valid config and data."""
        mock_rate_limiter.return_value = lambda: None
        mock_verify.return_value = ({"protocol": "residue_to_class"}, "")
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="user-1")
        mock_file_manager.return_value.get_biotrainer_model_path.return_value = "/path"
        mock_task_manager.return_value.get_unique_task_id.return_value = "train-123"
        mock_task_manager.return_value.add_task.return_value = "train-123"

        request_data = {
            "config_dict": {"protocol": "residue_to_class"},
            "training_data": [
                {
                    "seq_id": "seq1",
                    "sequence": "MVLSPADKTNVKAAWGKVGAHAGE",
                    "label": "BBBBBBBBBBBBBBBBBBBBBBBB",
                    "set": "train",
                }
            ],
        }

        response = custom_models_client.post(
            "/custom_models_service/start_training", json=request_data
        )

        assert response.status_code == 200
        assert "task_id" in response.json()

    @patch("biocentral_server.custom_models.custom_models_endpoint.verify_biotrainer_config")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_training_invalid_config(
        self, mock_rate_limiter, mock_verify, custom_models_client
    ):
        """Test starting training with invalid config returns 400."""
        mock_rate_limiter.return_value = lambda: None
        mock_verify.return_value = ({}, "Invalid configuration")

        request_data = {
            "config_dict": {"invalid": "config"},
            "training_data": [
                {
                    "seq_id": "seq1",
                    "sequence": "MVLSPADKTNVKAAWGKVGAHAGE",
                    "label": "A",
                    "set": "train",
                }
            ],
        }

        response = custom_models_client.post(
            "/custom_models_service/start_training", json=request_data
        )

        assert response.status_code == 400

    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_training_empty_training_data(
        self, mock_rate_limiter, custom_models_client
    ):
        """Test starting training with empty training data fails."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "config_dict": {"protocol": "residue_to_class"},
            "training_data": [],
        }

        response = custom_models_client.post(
            "/custom_models_service/start_training", json=request_data
        )

        assert response.status_code == 422  # Validation error


class TestModelFilesEndpoint:
    """Tests for POST /custom_models_service/model_files"""

    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_model_files_found(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_file_manager,
        custom_models_client,
    ):
        """Test retrieving model files for valid model hash."""
        mock_rate_limiter.return_value = lambda: None
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="user-1")
        mock_file_manager.return_value.get_biotrainer_result_files.return_value = {
            "out_config": "config content",
            "logging_out": "log content",
            "out_file": "output content",
        }

        request_data = {"model_hash": "valid-model-hash-123"}

        response = custom_models_client.post(
            "/custom_models_service/model_files", json=request_data
        )

        assert response.status_code == 200
        data = response.json()
        assert "out_config" in data or "logging_out" in data

    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_model_files_not_found(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_file_manager,
        custom_models_client,
    ):
        """Test retrieving model files for invalid hash returns 404."""
        mock_rate_limiter.return_value = lambda: None
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="user-1")
        mock_file_manager.return_value.get_biotrainer_result_files.return_value = {}

        request_data = {"model_hash": "invalid-hash"}

        response = custom_models_client.post(
            "/custom_models_service/model_files", json=request_data
        )

        assert response.status_code == 404


class TestStartInferenceEndpoint:
    """Tests for POST /custom_models_service/start_inference"""

    @patch("biocentral_server.custom_models.custom_models_endpoint.TaskManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_inference_valid(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_file_manager,
        mock_task_manager,
        custom_models_client,
    ):
        """Test starting inference with valid model and sequences."""
        mock_rate_limiter.return_value = lambda: None
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="user-1")
        mock_file_manager.return_value.get_biotrainer_model_path.return_value = "/path/to/model"
        mock_task_manager.return_value.add_task.return_value = "inference-task-123"

        request_data = {
            "model_hash": "valid-model-hash",
            "sequence_data": {
                "seq1": "MVLSPADKTNVKAAWGKVGAHAGE",
                "seq2": "MGHFTEEDKATITSLWGKVNVE",
            },
        }

        response = custom_models_client.post(
            "/custom_models_service/start_inference", json=request_data
        )

        assert response.status_code == 200
        assert "task_id" in response.json()

    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_inference_model_not_found(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_file_manager,
        custom_models_client,
    ):
        """Test starting inference with invalid model hash returns 404."""
        mock_rate_limiter.return_value = lambda: None
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="user-1")
        mock_file_manager.return_value.get_biotrainer_model_path.return_value = None

        request_data = {
            "model_hash": "invalid-model-hash",
            "sequence_data": {"seq1": "MVLSPADKTNVKAAWGKVGAHAGE"},
        }

        response = custom_models_client.post(
            "/custom_models_service/start_inference", json=request_data
        )

        assert response.status_code == 404

    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_inference_empty_sequences(
        self, mock_rate_limiter, custom_models_client
    ):
        """Test starting inference with empty sequence data fails."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "model_hash": "valid-model-hash",
            "sequence_data": {},
        }

        response = custom_models_client.post(
            "/custom_models_service/start_inference", json=request_data
        )

        assert response.status_code == 422  # Validation error
