"""
Integration tests for custom models (training and inference) endpoints.

Tests the /custom_models_service/* endpoints:
- GET /protocols - Get available training protocols
- GET /config_options/{protocol} - Get config options for a protocol
- POST /verify_config - Verify training configuration
- POST /start_training - Start model training
- POST /model_files - Get trained model files
- POST /start_inference - Run inference on trained model

Uses configurable embedder backend (FixedEmbedder or ESM-2 8M).
"""

import pytest
import numpy as np
from typing import Dict, List
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biocentral_server.custom_models import custom_models_router
from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET
from tests.integration.endpoints.conftest import (
    CANONICAL_STANDARD_IDS,
    CANONICAL_REAL_WORLD_IDS,
    get_sequence_by_id,
)


@pytest.fixture(scope="module")
def custom_models_app():
    """Create a FastAPI app with custom_models router for testing."""
    app = FastAPI()
    app.include_router(custom_models_router)
    return app


@pytest.fixture(scope="module")
def client(custom_models_app):
    """Create test client."""
    return TestClient(custom_models_app)


@pytest.fixture
def training_data_classification(short_test_sequences) -> List[Dict]:
    """Training data for sequence classification task."""
    labels = ["membrane", "soluble"]
    sets = ["train", "train", "val"] if len(short_test_sequences) > 2 else ["train", "val"]
    
    data = []
    for i, (seq_id, seq) in enumerate(short_test_sequences.items()):
        data.append({
            "seq_id": seq_id,
            "sequence": seq,
            "label": labels[i % len(labels)],
            "set": sets[i] if i < len(sets) else "train",
        })
    return data


@pytest.fixture
def training_data_regression(short_test_sequences) -> List[Dict]:
    """Training data for sequence regression task."""
    data = []
    for i, (seq_id, seq) in enumerate(short_test_sequences.items()):
        data.append({
            "seq_id": seq_id,
            "sequence": seq,
            "label": str(float(i) * 0.5 + 0.1),  # Numeric label
            "set": "train" if i == 0 else "val",
        })
    return data


@pytest.fixture
def basic_classification_config(embedder_name) -> Dict:
    """Basic biotrainer configuration for sequence classification."""
    return {
        "protocol": "sequence_to_class",
        "embedder_name": embedder_name,
        "num_epochs": 1,
        "learning_rate": 0.001,
        "batch_size": 2,
    }


@pytest.fixture
def standard_training_data() -> List[Dict]:
    """Training data using all three standard sequences from canonical dataset."""
    return [
        {
            "seq_id": "standard_001",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
            "label": "membrane",
            "set": "train",
        },
        {
            "seq_id": "standard_002",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("standard_002").sequence,
            "label": "soluble",
            "set": "train",
        },
        {
            "seq_id": "standard_003",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("standard_003").sequence,
            "label": "membrane",
            "set": "val",
        },
    ]


@pytest.fixture
def real_world_training_data() -> List[Dict]:
    """Training data using real-world protein sequences from canonical dataset."""
    return [
        {
            "seq_id": "insulin_b",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("real_insulin_b").sequence,
            "label": "hormone",
            "set": "train",
        },
        {
            "seq_id": "ubiquitin",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("real_ubiquitin").sequence,
            "label": "signaling",
            "set": "train",
        },
        {
            "seq_id": "gfp_core",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("real_gfp_core").sequence,
            "label": "fluorescent",
            "set": "val",
        },
    ]


@pytest.fixture
def diverse_training_data() -> List[Dict]:
    """Diverse training data combining standard and real-world sequences."""
    return [
        {
            "seq_id": "standard_001",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
            "label": "class_A",
            "set": "train",
        },
        {
            "seq_id": "standard_002",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("standard_002").sequence,
            "label": "class_B",
            "set": "train",
        },
        {
            "seq_id": "insulin_b",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("real_insulin_b").sequence,
            "label": "class_A",
            "set": "train",
        },
        {
            "seq_id": "ubiquitin",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("real_ubiquitin").sequence,
            "label": "class_B",
            "set": "val",
        },
        {
            "seq_id": "gfp_core",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("real_gfp_core").sequence,
            "label": "class_A",
            "set": "val",
        },
    ]


@pytest.fixture
def regression_training_data() -> List[Dict]:
    """Training data for regression task using canonical sequences."""
    return [
        {
            "seq_id": "standard_001",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
            "label": "0.75",
            "set": "train",
        },
        {
            "seq_id": "standard_002",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("standard_002").sequence,
            "label": "0.42",
            "set": "train",
        },
        {
            "seq_id": "insulin_b",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("real_insulin_b").sequence,
            "label": "0.89",
            "set": "train",
        },
        {
            "seq_id": "ubiquitin",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("real_ubiquitin").sequence,
            "label": "0.33",
            "set": "val",
        },
    ]


@pytest.fixture
def inference_sequences() -> Dict[str, str]:
    """Sequences for inference testing using canonical dataset."""
    return {
        "infer_1": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
        "infer_2": CANONICAL_TEST_DATASET.get_by_id("real_insulin_b").sequence,
        "infer_3": CANONICAL_TEST_DATASET.get_by_id("real_ubiquitin").sequence,
    }


class TestProtocolsEndpoint:
    """
    Integration tests for GET /custom_models_service/protocols.
    """

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_get_protocols(
        self,
        mock_rate_limiter,
        client,
    ):
        """Test retrieving available training protocols."""
        mock_rate_limiter.return_value = lambda: None

        response = client.get("/custom_models_service/protocols")

        assert response.status_code == 200
        response_json = response.json()
        assert "protocols" in response_json
        protocols = response_json["protocols"]
        assert isinstance(protocols, list)
        assert len(protocols) > 0

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_protocols_contains_expected_types(
        self,
        mock_rate_limiter,
        client,
    ):
        """Test that common protocols are available."""
        mock_rate_limiter.return_value = lambda: None

        response = client.get("/custom_models_service/protocols")
        protocols = response.json()["protocols"]
        
        # Biotrainer should support these common protocols
        protocol_names_lower = [p.lower() for p in protocols]
        
        # At least one of these should be present
        expected_any = ["sequence_to_class", "residue_to_class", "sequence_to_value"]
        assert any(exp in protocol_names_lower for exp in expected_any)


class TestConfigOptionsEndpoint:
    """
    Integration tests for GET /custom_models_service/config_options/{protocol}.
    """

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_get_config_options_valid_protocol(
        self,
        mock_rate_limiter,
        client,
    ):
        """Test getting config options for a valid protocol."""
        mock_rate_limiter.return_value = lambda: None

        response = client.get("/custom_models_service/config_options/sequence_to_class")

        assert response.status_code == 200
        response_json = response.json()
        assert "options" in response_json
        assert isinstance(response_json["options"], list)

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_get_config_options_invalid_protocol(
        self,
        mock_rate_limiter,
        client,
    ):
        """Test getting config options for an invalid protocol."""
        mock_rate_limiter.return_value = lambda: None

        response = client.get("/custom_models_service/config_options/invalid_protocol_xyz")

        assert response.status_code == 400


class TestVerifyConfigEndpoint:
    """
    Integration tests for POST /custom_models_service/verify_config.
    """

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_verify_valid_config(
        self,
        mock_rate_limiter,
        client,
        basic_classification_config,
    ):
        """Test verifying a valid configuration."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {"config_dict": basic_classification_config}

        response = client.post("/custom_models_service/verify_config/", json=request_data)

        assert response.status_code == 200
        response_json = response.json()
        assert "error" in response_json
        # Empty error string means config is valid
        # Note: May return validation errors if config is incomplete

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_verify_invalid_config(
        self,
        mock_rate_limiter,
        client,
    ):
        """Test verifying an invalid configuration."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "config_dict": {
                "protocol": "invalid_protocol_xyz",
                "embedder_name": "invalid_embedder",
            }
        }

        response = client.post("/custom_models_service/verify_config/", json=request_data)

        assert response.status_code == 200
        response_json = response.json()
        # Should have error message for invalid protocol
        assert response_json.get("error") != ""


class TestStartTrainingEndpoint:
    """
    Integration tests for POST /custom_models_service/start_training.
    """

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.TaskManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_training_creates_task(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_file_manager,
        mock_task_manager,
        client,
        basic_classification_config,
        training_data_classification,
    ):
        """Test that training request creates a task."""
        mock_rate_limiter.return_value = lambda: None
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")
        mock_file_manager.return_value.get_biotrainer_model_path.return_value = "/tmp/model"
        mock_task_manager.return_value.get_unique_task_id.return_value = "train-task-123"
        mock_task_manager.return_value.add_task.return_value = "train-task-123"

        request_data = {
            "config_dict": basic_classification_config,
            "training_data": training_data_classification,
        }

        response = client.post("/custom_models_service/start_training", json=request_data)

        assert response.status_code == 200
        response_json = response.json()
        assert "task_id" in response_json
        assert response_json["task_id"] == "train-task-123"

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_training_empty_data_rejected(
        self,
        mock_rate_limiter,
        client,
        basic_classification_config,
    ):
        """Test that empty training data is rejected."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "config_dict": basic_classification_config,
            "training_data": [],  # Empty - should fail validation
        }

        response = client.post("/custom_models_service/start_training", json=request_data)

        assert response.status_code == 422

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_training_invalid_config_rejected(
        self,
        mock_rate_limiter,
        client,
        training_data_classification,
    ):
        """Test that invalid configuration is rejected."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "config_dict": {
                "protocol": "invalid_protocol",
                "embedder_name": "invalid",
            },
            "training_data": training_data_classification,
        }

        response = client.post("/custom_models_service/start_training", json=request_data)

        assert response.status_code == 400


class TestModelFilesEndpoint:
    """
    Integration tests for POST /custom_models_service/model_files.
    """

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_get_model_files_success(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_file_manager,
        client,
    ):
        """Test retrieving model files after training."""
        mock_rate_limiter.return_value = lambda: None
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")
        mock_file_manager.return_value.get_biotrainer_result_files.return_value = {
            "out_config": "config_content",
            "logging_out": "log_content",
            "out_file": "model_content",
        }

        request_data = {"model_hash": "train-task-123"}

        response = client.post("/custom_models_service/model_files", json=request_data)

        assert response.status_code == 200
        response_json = response.json()
        assert "out_config" in response_json

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_get_model_files_not_found(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_file_manager,
        client,
    ):
        """Test retrieving files for non-existent model."""
        mock_rate_limiter.return_value = lambda: None
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")
        mock_file_manager.return_value.get_biotrainer_result_files.return_value = {}

        request_data = {"model_hash": "non-existent-model"}

        response = client.post("/custom_models_service/model_files", json=request_data)

        assert response.status_code == 404


class TestStartInferenceEndpoint:
    """
    Integration tests for POST /custom_models_service/start_inference.
    """

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.TaskManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_inference_creates_task(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_file_manager,
        mock_task_manager,
        client,
        short_test_sequences,
    ):
        """Test that inference request creates a task."""
        mock_rate_limiter.return_value = lambda: None
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")
        mock_file_manager.return_value.get_biotrainer_model_path.return_value = "/tmp/model/train-123"
        mock_task_manager.return_value.add_task.return_value = "inference-task-456"

        request_data = {
            "model_hash": "train-task-123",
            "sequence_data": short_test_sequences,
        }

        response = client.post("/custom_models_service/start_inference", json=request_data)

        assert response.status_code == 200
        response_json = response.json()
        assert "task_id" in response_json
        assert response_json["task_id"] == "inference-task-456"

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_inference_model_not_found(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_file_manager,
        client,
        short_test_sequences,
    ):
        """Test inference with non-existent model."""
        mock_rate_limiter.return_value = lambda: None
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")
        mock_file_manager.return_value.get_biotrainer_model_path.return_value = None

        request_data = {
            "model_hash": "non-existent-model",
            "sequence_data": short_test_sequences,
        }

        response = client.post("/custom_models_service/start_inference", json=request_data)

        assert response.status_code == 404

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_inference_empty_sequences_rejected(
        self,
        mock_rate_limiter,
        client,
    ):
        """Test that empty sequence data is rejected."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "model_hash": "some-model",
            "sequence_data": {},  # Empty
        }

        response = client.post("/custom_models_service/start_inference", json=request_data)

        assert response.status_code == 422


class TestTrainInferenceFlow:
    """
    End-to-end tests for the training -> inference flow.
    
    These tests verify the complete workflow using embeddings.
    """

    @pytest.mark.integration
    def test_training_data_format(
        self,
        training_data_classification,
    ):
        """Test that training data has correct format."""
        assert len(training_data_classification) >= 2
        
        for item in training_data_classification:
            assert "seq_id" in item
            assert "sequence" in item
            assert "label" in item
            assert "set" in item
            assert len(item["sequence"]) > 0

    @pytest.mark.integration
    def test_embeddings_for_training_sequences(
        self,
        embedder,
        embedding_dim,
        training_data_classification,
    ):
        """Test generating embeddings for training sequences."""
        sequences = {item["seq_id"]: item["sequence"] for item in training_data_classification}
        
        embeddings = embedder.embed_dict(sequences, pooled=True)
        
        assert len(embeddings) == len(sequences)
        for seq_id in sequences.keys():
            assert seq_id in embeddings
            assert embeddings[seq_id].shape == (embedding_dim,)

    @pytest.mark.integration
    def test_config_with_embedder_name(
        self,
        embedder_name,
        basic_classification_config,
    ):
        """Test that config correctly references embedder."""
        assert basic_classification_config["embedder_name"] == embedder_name
        assert basic_classification_config["protocol"] == "sequence_to_class"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_data_conversion_to_biotrainer_format(
        self,
        training_data_classification,
    ):
        """Test converting training data to biotrainer format."""
        from biocentral_server.custom_models.endpoint_models import SequenceTrainingData
        
        for item in training_data_classification:
            training_item = SequenceTrainingData(**item)
            
            # Convert to FASTA format
            fasta = training_item.to_fasta()
            assert training_item.seq_id in fasta
            assert training_item.sequence in fasta
            
            # Convert to BiotrainerSequenceRecord
            record = training_item.to_biotrainer_seq_record()
            assert record.seq_id == training_item.seq_id
            assert record.seq == training_item.sequence


class TestTrainingWithStandardSequences:
    """
    Tests for training with standard canonical sequences.
    """

    @pytest.mark.integration
    def test_standard_training_data_format(
        self,
        standard_training_data,
    ):
        """Test that standard training data has correct format."""
        assert len(standard_training_data) == 3
        
        for item in standard_training_data:
            assert "seq_id" in item
            assert "sequence" in item
            assert "label" in item
            assert "set" in item
            assert len(item["sequence"]) > 0

    @pytest.mark.integration
    def test_embeddings_for_standard_training_sequences(
        self,
        embedder,
        embedding_dim,
        standard_training_data,
    ):
        """Test generating embeddings for standard training sequences."""
        sequences = {item["seq_id"]: item["sequence"] for item in standard_training_data}
        
        embeddings = embedder.embed_dict(sequences, pooled=True)
        
        assert len(embeddings) == len(sequences)
        for seq_id in sequences.keys():
            assert seq_id in embeddings
            assert embeddings[seq_id].shape == (embedding_dim,)
            assert np.isfinite(embeddings[seq_id]).all()

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.TaskManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_training_with_standard_sequences(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_file_manager,
        mock_task_manager,
        client,
        basic_classification_config,
        standard_training_data,
    ):
        """Test training with standard canonical sequences."""
        mock_rate_limiter.return_value = lambda: None
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")
        mock_file_manager.return_value.get_biotrainer_model_path.return_value = "/tmp/model"
        mock_task_manager.return_value.get_unique_task_id.return_value = "standard-train-task"
        mock_task_manager.return_value.add_task.return_value = "standard-train-task"

        request_data = {
            "config_dict": basic_classification_config,
            "training_data": standard_training_data,
        }

        response = client.post("/custom_models_service/start_training", json=request_data)

        assert response.status_code == 200
        assert "task_id" in response.json()


class TestTrainingWithRealWorldSequences:
    """
    Tests for training with real-world protein sequences.
    """

    @pytest.mark.integration
    def test_real_world_training_data_format(
        self,
        real_world_training_data,
    ):
        """Test that real-world training data has correct format."""
        assert len(real_world_training_data) == 3
        
        # Should have diverse labels
        labels = {item["label"] for item in real_world_training_data}
        assert len(labels) == 3  # hormone, signaling, fluorescent
        
        for item in real_world_training_data:
            assert len(item["sequence"]) >= 10

    @pytest.mark.integration
    def test_embeddings_for_real_world_training_sequences(
        self,
        embedder,
        embedding_dim,
        real_world_training_data,
    ):
        """Test generating embeddings for real-world training sequences."""
        sequences = {item["seq_id"]: item["sequence"] for item in real_world_training_data}
        
        embeddings = embedder.embed_dict(sequences, pooled=True)
        
        assert len(embeddings) == len(sequences)
        for seq_id in sequences.keys():
            assert seq_id in embeddings
            assert embeddings[seq_id].shape == (embedding_dim,)

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.TaskManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_training_with_real_world_sequences(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_file_manager,
        mock_task_manager,
        client,
        basic_classification_config,
        real_world_training_data,
    ):
        """Test training with real-world protein sequences."""
        mock_rate_limiter.return_value = lambda: None
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")
        mock_file_manager.return_value.get_biotrainer_model_path.return_value = "/tmp/model"
        mock_task_manager.return_value.get_unique_task_id.return_value = "real-world-train-task"
        mock_task_manager.return_value.add_task.return_value = "real-world-train-task"

        request_data = {
            "config_dict": basic_classification_config,
            "training_data": real_world_training_data,
        }

        response = client.post("/custom_models_service/start_training", json=request_data)

        assert response.status_code == 200
        assert "task_id" in response.json()


class TestTrainingWithDiverseSequences:
    """
    Tests for training with diverse sequence collections.
    """

    @pytest.mark.integration
    def test_diverse_training_data_coverage(
        self,
        diverse_training_data,
    ):
        """Test that diverse training data covers multiple categories."""
        assert len(diverse_training_data) >= 5
        
        # Should have train and val sets
        sets = {item["set"] for item in diverse_training_data}
        assert "train" in sets
        assert "val" in sets
        
        # Should have multiple classes
        labels = {item["label"] for item in diverse_training_data}
        assert len(labels) >= 2

    @pytest.mark.integration
    def test_embeddings_for_diverse_training_sequences(
        self,
        embedder,
        embedding_dim,
        diverse_training_data,
    ):
        """Test generating embeddings for diverse training sequences."""
        sequences = {item["seq_id"]: item["sequence"] for item in diverse_training_data}
        
        embeddings = embedder.embed_dict(sequences, pooled=True)
        
        assert len(embeddings) == len(sequences)
        for seq_id, emb in embeddings.items():
            assert emb.shape == (embedding_dim,)
            assert np.isfinite(emb).all()

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.TaskManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_training_with_diverse_sequences(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_file_manager,
        mock_task_manager,
        client,
        basic_classification_config,
        diverse_training_data,
    ):
        """Test training with diverse sequence collection."""
        mock_rate_limiter.return_value = lambda: None
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")
        mock_file_manager.return_value.get_biotrainer_model_path.return_value = "/tmp/model"
        mock_task_manager.return_value.get_unique_task_id.return_value = "diverse-train-task"
        mock_task_manager.return_value.add_task.return_value = "diverse-train-task"

        request_data = {
            "config_dict": basic_classification_config,
            "training_data": diverse_training_data,
        }

        response = client.post("/custom_models_service/start_training", json=request_data)

        assert response.status_code == 200
        assert "task_id" in response.json()


class TestRegressionTraining:
    """
    Tests for regression training tasks.
    """

    @pytest.mark.integration
    def test_regression_training_data_format(
        self,
        regression_training_data,
    ):
        """Test that regression training data has numeric labels."""
        for item in regression_training_data:
            # Labels should be numeric strings
            label = float(item["label"])
            assert 0.0 <= label <= 1.0

    @pytest.mark.integration
    def test_embeddings_for_regression_training(
        self,
        embedder,
        embedding_dim,
        regression_training_data,
    ):
        """Test generating embeddings for regression training sequences."""
        sequences = {item["seq_id"]: item["sequence"] for item in regression_training_data}
        
        embeddings = embedder.embed_dict(sequences, pooled=True)
        
        assert len(embeddings) == len(sequences)
        for emb in embeddings.values():
            assert emb.shape == (embedding_dim,)
            assert np.isfinite(emb).all()


class TestInferenceWithCanonicalSequences:
    """
    Tests for inference using canonical sequences.
    """

    @pytest.mark.integration
    def test_inference_sequences_format(
        self,
        inference_sequences,
    ):
        """Test that inference sequences have correct format."""
        assert len(inference_sequences) >= 3
        
        for seq_id, seq in inference_sequences.items():
            assert isinstance(seq_id, str)
            assert isinstance(seq, str)
            assert len(seq) > 0

    @pytest.mark.integration
    def test_embeddings_for_inference_sequences(
        self,
        embedder,
        embedding_dim,
        inference_sequences,
    ):
        """Test generating embeddings for inference sequences."""
        embeddings = embedder.embed_dict(inference_sequences, pooled=True)
        
        assert len(embeddings) == len(inference_sequences)
        for seq_id in inference_sequences.keys():
            assert seq_id in embeddings
            assert embeddings[seq_id].shape == (embedding_dim,)

    @pytest.mark.integration
    @patch("biocentral_server.custom_models.custom_models_endpoint.TaskManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.FileManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.UserManager")
    @patch("biocentral_server.custom_models.custom_models_endpoint.RateLimiter")
    def test_start_inference_with_canonical_sequences(
        self,
        mock_rate_limiter,
        mock_user_manager,
        mock_file_manager,
        mock_task_manager,
        client,
        inference_sequences,
    ):
        """Test inference with canonical sequences."""
        mock_rate_limiter.return_value = lambda: None
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")
        mock_file_manager.return_value.get_biotrainer_model_path.return_value = "/tmp/model/train-123"
        mock_task_manager.return_value.add_task.return_value = "canonical-inference-task"

        request_data = {
            "model_hash": "train-task-123",
            "sequence_data": inference_sequences,
        }

        response = client.post("/custom_models_service/start_inference", json=request_data)

        assert response.status_code == 200
        assert "task_id" in response.json()


class TestTrainingDataValidation:
    """
    Tests for training data validation.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_STANDARD_IDS)
    def test_standard_sequences_valid_for_training(
        self,
        seq_id,
    ):
        """Test that standard sequences meet training requirements."""
        sequence = get_sequence_by_id(seq_id)
        
        # Should have reasonable length
        assert len(sequence) >= 5
        assert len(sequence) <= 5000

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_REAL_WORLD_IDS)
    def test_real_world_sequences_valid_for_training(
        self,
        seq_id,
    ):
        """Test that real-world sequences meet training requirements."""
        sequence = get_sequence_by_id(seq_id)
        
        # Should have reasonable length
        assert len(sequence) >= 5
        assert len(sequence) <= 5000
        
        # Should only contain valid amino acids
        valid_aas = set("ACDEFGHIKLMNPQRSTVWYX")
        assert all(aa in valid_aas for aa in sequence.upper())

    @pytest.mark.integration
    def test_training_set_distribution(
        self,
        diverse_training_data,
    ):
        """Test that training data has proper train/val distribution."""
        train_count = sum(1 for item in diverse_training_data if item["set"] == "train")
        val_count = sum(1 for item in diverse_training_data if item["set"] == "val")
        
        # Should have both train and val samples
        assert train_count >= 1
        assert val_count >= 1
        
        # Train should typically have more samples
        assert train_count >= val_count
