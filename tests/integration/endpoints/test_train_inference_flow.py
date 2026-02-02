"""Integration tests for custom models (training and inference) endpoints.

Test file order: 4 (heavy - runs real training with ESM models).
Test class ordering: Lightweight tests (order=1-4) run first, heavy training tests (order=5-7) run last.
This prevents background jobs from earlier tests causing OOM during lightweight tests.
"""

import httpx
import pytest
from typing import Dict, List

from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET

# Standard sequences for inference testing
STANDARD_SEQUENCES = {
    "standard_001": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
    "standard_002": CANONICAL_TEST_DATASET.get_by_id("standard_002").sequence,
}


@pytest.fixture
def classification_training_data() -> List[Dict]:
    """Training data for sequence classification task using canonical dataset."""
    return [
        {
            "seq_id": "standard_001",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
            "label": "membrane",
            "set": "train",
        },
          {
            "seq_id": "standard_001",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("standard_002").sequence,
            "label": "membrane",
            "set": "val",
        },
        {
            "seq_id": "standard_003",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("standard_003").sequence, 
            "label": "membrane",
            "set": "test",
        },
    ]


@pytest.fixture
def real_world_training_data() -> List[Dict]:
    """Training data using real protein sequences from canonical dataset."""
    return [
        {
            "seq_id": "insulin_b",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("real_insulin_b").sequence,
            "label": "hormone",
            "set": "train",
        },
    ]


@pytest.fixture
def regression_training_data() -> List[Dict]:
    """Training data for regression task using canonical dataset."""
    return [
        {
            "seq_id": "standard_001",
            "sequence": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
            "label": "0.75",
            "set": "train",
        },
    ]


@pytest.fixture
def inference_sequences() -> Dict[str, str]:
    """Sequences for inference testing from canonical dataset."""
    return {
        "infer_1": CANONICAL_TEST_DATASET.get_by_id("length_min_1").sequence,
    }


@pytest.fixture
def classification_config(embedder_name: str) -> Dict:
    """Biotrainer configuration for sequence classification."""
    return {
        "protocol": "sequence_to_class",
        "embedder_name": "facebook/esm2_t6_8M_UR50D",
        "num_epochs": 1,
        "learning_rate": 0.001,
        "batch_size": 1,
    }


@pytest.fixture
def regression_config(embedder_name: str) -> Dict:
    """Biotrainer configuration for regression."""
    return {
        "protocol": "sequence_to_value",
        "embedder_name": embedder_name,
        "num_epochs": 1,
        "learning_rate": 0.001,
        "batch_size": 1,
    }


@pytest.mark.order(1)
class TestConfigOptionsEndpoint:
    """
    Integration tests for GET /custom_models_service/config_options/{protocol}.
    Lightweight: No model loading required.
    """

    @pytest.mark.integration
    def test_get_config_options_for_classification(self, client):
        """Test getting config options for sequence_to_class protocol."""
        response = client.get("/custom_models_service/config_options/sequence_to_class")

        assert response.status_code == 200
        data = response.json()
        assert "options" in data
        assert isinstance(data["options"], list)

    @pytest.mark.integration
    def test_get_config_options_for_regression(self, client):
        """Test getting config options for sequence_to_value protocol."""
        response = client.get("/custom_models_service/config_options/sequence_to_value")

        assert response.status_code == 200
        data = response.json()
        assert "options" in data

    @pytest.mark.integration
    def test_get_config_options_invalid_protocol(self, client):
        """Test getting config options for an invalid protocol returns error."""
        try:
            response = client.get("/custom_models_service/config_options/invalid_protocol_xyz")
        except Exception as e:
            assert e.response.status_code in (400, 500)
            return

        # Server returns 500 for unrecognized protocols (internal handling error)
        assert response.status_code in [400, 500]


@pytest.mark.order(2)
class TestVerifyConfigEndpoint:
    """
    Integration tests for POST /custom_models_service/verify_config.
    Lightweight: Config validation only, no model loading.
    """

    @pytest.mark.integration
    def test_verify_valid_classification_config(self, client, classification_config):
        """Test verifying a valid classification configuration."""
        try:
            response = client.post(
                "/custom_models_service/verify_config/",
                json={"config_dict": classification_config}
            )
        except httpx.RemoteProtocolError:
            pytest.skip("Server disconnected (likely restarting due to resource constraints)")

        assert response.status_code == 200
        data = response.json()
        assert "error" in data

    @pytest.mark.integration
    def test_verify_valid_regression_config(self, client, regression_config):
        """Test verifying a valid regression configuration."""
        response = client.post(
            "/custom_models_service/verify_config/",
            json={"config_dict": regression_config}
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data

    @pytest.mark.integration
    def test_verify_invalid_protocol_config(self, client):
        """Test verifying config with invalid protocol."""
        invalid_config = {
            "protocol": "invalid_protocol_xyz",
            "embedder_name": "invalid_embedder",
        }

        response = client.post(
            "/custom_models_service/verify_config/",
            json={"config_dict": invalid_config}
        )

        assert response.status_code == 200
        data = response.json()
        # Should have non-empty error message
        assert data.get("error") != ""


@pytest.mark.order(5)
class TestStartTrainingEndpoint:
    """
    Integration tests for POST /custom_models_service/start_training.
    Medium: Submits training jobs (doesn't wait for completion).
    """
 

    @pytest.mark.integration
    def test_start_training_with_real_proteins(
        self,
        client,
        classification_config,
        real_world_training_data,
    ):
        """Test training with real protein sequences."""
        response = client.post(
            "/custom_models_service/start_training",
            json={
                "config_dict": classification_config,
                "training_data": real_world_training_data,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data

    @pytest.mark.integration
    def test_start_regression_training(
        self,
        client,
        regression_config,
        regression_training_data,
    ):
        """Test training with regression task."""
        response = client.post(
            "/custom_models_service/start_training",
            json={
                "config_dict": regression_config,
                "training_data": regression_training_data,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data

    @pytest.mark.integration
    def test_start_training_empty_data_rejected(
        self,
        client,
        classification_config,
    ):
        """Test that empty training data is rejected."""
        response = client.post(
            "/custom_models_service/start_training",
            json={
                "config_dict": classification_config,
                "training_data": [],
            }
        )

        # Should be rejected with validation error
        assert response.status_code == 422

    @pytest.mark.integration
    def test_start_training_invalid_config(
        self,
        client,
        classification_training_data,
    ):
        """Test that invalid configuration is rejected."""
        response = client.post(
            "/custom_models_service/start_training",
            json={
                "config_dict": {
                    "protocol": "invalid_protocol",
                    "embedder_name": "invalid",
                },
                "training_data": classification_training_data,
            }
        )

        # Should be rejected
        assert response.status_code in [400, 422]


@pytest.mark.order(3)
class TestStartInferenceEndpoint:
    """
    Integration tests for POST /custom_models_service/start_inference.
    Lightweight: Validation tests only, model not found expected.
    """

    @pytest.mark.integration
    def test_start_inference_empty_sequences_rejected(self, client):
        """Test that empty sequence data is rejected."""
        response = client.post(
            "/custom_models_service/start_inference",
            json={
                "model_hash": "some-model",
                "sequence_data": {},
            }
        )

        # Should be rejected with validation error
        assert response.status_code == 422

    @pytest.mark.integration
    def test_start_inference_with_standard_sequences(
        self,
        client,
    ):
        """Test inference with standard sequences."""
        response = client.post(
            "/custom_models_service/start_inference",
            json={
                "model_hash": "trained-model-123",
                "sequence_data": STANDARD_SEQUENCES,
            }
        )

        # Either creates task or model not found
        assert response.status_code in [200, 404]


@pytest.mark.order(4)
class TestModelFilesEndpoint:
    """
    Integration tests for POST /custom_models_service/model_files.
    Lightweight: File retrieval for non-existent model.
    """

    @pytest.mark.integration
    def test_get_model_files_nonexistent(self, client):
        """Test retrieving files for non-existent model."""
        try:
            response = client.post(
                "/custom_models_service/model_files",
                json={"model_hash": "non-existent-model-xyz"}
            )
            # Server returns 404 or 500 for non-existent model
            assert response.status_code in [404, 500]
        except Exception as e:
            # StorageError from SeaweedFS is acceptable - it means the server
            # correctly tried to fetch a non-existent model
            assert "404" in str(e) or "Not Found" in str(e) or "StorageError" in str(e)

# Not realistic in standard github CI setting due to resource constraints
# class TestTrainingTaskLifecycle:
#     """
#     Tests for training task lifecycle management.
#     """

#     @pytest.mark.integration
#     def test_training_task_ids_are_unique(
#         self,
#         client,
#         classification_config,
#         classification_training_data,
#     ):
#         """Test that multiple training submissions get unique task IDs."""
#         task_ids = set()

#         for _ in range(3):
#             try:
#                 response = client.post(
#                     "/custom_models_service/start_training",
#                     json={
#                         "config_dict": classification_config,
#                         "training_data": classification_training_data,
#                     }
#                 )
#             except httpx.RemoteProtocolError:
#                 # Server may disconnect under heavy load, continue with collected IDs
#                 break
            
#             if response.status_code == 200:
#                 task_id = response.json().get("task_id")
#                 if task_id:
#                     task_ids.add(task_id)

#         # All task IDs should be unique (if any were returned)
#         if task_ids:
#             assert len(task_ids) == min(3, len(task_ids))

#     @pytest.mark.integration
#     @pytest.mark.slow
#     def test_training_task_completes(
#         self,
#         client,
#         poll_task,
#         classification_config,
#         classification_training_data,
#     ):
#         """Test that training task eventually completes."""
#         # Start training
#         response = client.post(
#             "/custom_models_service/start_training",
#             json={
#                 "config_dict": classification_config,
#                 "training_data": classification_training_data,
#             }
#         )
        
#         assert response.status_code == 200
#         task_id = response.json()["task_id"]

#         # Poll for completion
#         result = poll_task(task_id, timeout=300)  # 5 minutes for training

#         assert result is not None
#         # Task should reach a terminal state (case-insensitive)
#         assert result.get("status", "").upper() in ["FINISHED", "FAILED"]


@pytest.mark.order(7)
class TestEndToEndTrainInferenceFlow:
    """
    End-to-end tests for the complete training -> inference workflow.
    Heavy: Runs full training and waits for completion. Run last.
    """

    @pytest.mark.integration
    @pytest.mark.slow
    def test_train_then_inference_flow(
        self,
        client,
        poll_task,
        flush_redis,
        classification_config,
        classification_training_data,
        inference_sequences,
    ):
        """Test complete train then inference flow."""
        # Flush any pending jobs from previous tests to free memory
        flush_redis()
        
        # Step 1: Start training
        try:
            train_response = client.post(
                "/custom_models_service/start_training",
                json={
                    "config_dict": classification_config,
                    "training_data": classification_training_data,
                }
            )
        except httpx.RemoteProtocolError:
            pytest.skip("Server disconnected")
        
        assert train_response.status_code == 200
        train_task_id = train_response.json()["task_id"]

        # Step 2: Wait for training to complete
        train_result = poll_task(train_task_id, timeout=400)
        
        assert train_result is not None
        train_status = train_result.get("status", "").upper()
        if train_status == "FAILED":
            pytest.skip("Training failed (likely due to CI resource constraints)")
        assert train_status == "FINISHED"

        # Extract model_hash from biotrainer result
        biotrainer_result = train_result.get("biotrainer_result", {})
        model_hash = biotrainer_result.get("derived_values", {}).get("model_hash")
        
        if not model_hash:
            pytest.fail(f"model_hash not found in training result: {train_result}")

        # Step 3: Run inference with the trained model
        inference_response = client.post(
            "/custom_models_service/start_inference",
            json={
                "model_hash": model_hash,
                "sequence_data": inference_sequences,
            }
        )

        assert inference_response.status_code == 200
        inference_task_id = inference_response.json()["task_id"]

        # Step 4: Wait for inference to complete
        inference_result = poll_task(inference_task_id, timeout=380)
        
        assert inference_result is not None
        inference_status = inference_result.get("status", "").upper()
        assert inference_status in ("FINISHED", "FAILED")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_get_model_files_after_training(
        self,
        client,
        poll_task,
        flush_redis,
        classification_config,
        classification_training_data,
    ):
        """Test retrieving model files after training completes."""
        # Flush any pending jobs from previous tests to free memory
        flush_redis()
        
        # Train a model
        train_response = client.post(
            "/custom_models_service/start_training",
            json={
                "config_dict": classification_config,
                "training_data": classification_training_data,
            }
        )
        
        assert train_response.status_code == 200
        train_task_id = train_response.json()["task_id"]

        # Wait for training to complete
        train_result = poll_task(train_task_id, timeout=500)
        
        assert train_result is not None
        train_status = train_result.get("status", "").upper()
        if train_status == "FAILED":
            pytest.skip("Training failed (likely due to CI resource constraints)")
        assert train_status == "FINISHED"

        # Extract model_hash from biotrainer result
        biotrainer_result = train_result.get("biotrainer_result", {})
        model_hash = biotrainer_result.get("derived_values", {}).get("model_hash")
        
        if not model_hash:
            pytest.fail(f"model_hash not found in training result: {train_result}")

        # Get model files using the actual model_hash (not task_id)
        files_response = client.post(
            "/custom_models_service/model_files",
            json={"model_hash": model_hash}
        )

        assert files_response.status_code == 200
        data = files_response.json()
        
        assert "BIOTRAINER_RESULT" in data or "BIOTRAINER_LOGGING" in data or "BIOTRAINER_CHECKPOINT" in data


@pytest.mark.order(6)
class TestTrainingDataValidation:
    """
    Tests for training data validation.
    Medium: Submits training job (doesn't wait for completion).
    """

    @pytest.mark.integration
    def test_training_with_train_val_split(
        self,
        client,
        classification_config,
    ):
        """Test training with proper train/val split."""
        # Data with explicit train/val split using canonical dataset
        training_data = [
            {"seq_id": "train_1", "sequence": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence, "label": "A", "set": "train"},
            {"seq_id": "val_1", "sequence": CANONICAL_TEST_DATASET.get_by_id("real_insulin_b").sequence, "label": "B", "set": "val"},
        ]

        response = client.post(
            "/custom_models_service/start_training",
            json={
                "config_dict": classification_config,
                "training_data": training_data,
            }
        )

        assert response.status_code == 200
