"""Prediction oracle tests for output consistency and validity."""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pytest

from tests.fixtures.test_dataset import (
    CANONICAL_TEST_DATASET,
)

_prediction_oracle_results: List[Dict[str, Any]] = []
pytestmark = pytest.mark.property

# Skip real model tests when CI uses FixedEmbedder
_using_fixed_embedder_ci = os.environ.get("CI_EMBEDDER") == "FixedEmbedder"
skip_in_fixed_embedder_ci = pytest.mark.skipif(
    _using_fixed_embedder_ci,
    reason="Testing with CI_EMBEDDER=FixedEmbedder",
)

def get_prediction_oracle_results() -> List[Dict[str, Any]]:
    """Get accumulated prediction oracle results."""
    return _prediction_oracle_results

def add_prediction_oracle_result(result: Dict[str, Any]) -> None:
    """Add a result to the prediction oracle results collection."""
    if "timestamp" not in result:
        result["timestamp"] = datetime.now().isoformat()
    _prediction_oracle_results.append(result)

def clear_prediction_oracle_results() -> None:
    """Clear all accumulated prediction oracle results."""
    _prediction_oracle_results.clear()

@dataclass
class PredictionOracleConfig:
    """Configuration for prediction oracles."""

    model_name: str
    output_type: str  # "per_residue" or "per_sequence"
    is_classification: bool = True
    num_classes: Optional[int] = None
    value_range: Optional[tuple] = None

PREDICTION_ORACLE_CONFIGS = {
    "ProtT5SecondaryStructure": PredictionOracleConfig(
        model_name="ProtT5SecondaryStructure",
        output_type="per_residue",
        is_classification=True,
        num_classes=3,
    ),
    "BindEmbed": PredictionOracleConfig(
        model_name="BindEmbed",
        output_type="per_residue",
        is_classification=True,
        num_classes=2,
    ),
    "TMbed": PredictionOracleConfig(
        model_name="TMbed",
        output_type="per_residue",
        is_classification=True,
        num_classes=4,
    ),
    "Seth": PredictionOracleConfig(
        model_name="Seth",
        output_type="per_residue",
        is_classification=False,
        value_range=(0.0, 1.0),
    ),
}

class PredictorProtocol(Protocol):
    """Protocol defining the predictor interface for oracle tests."""

    def predict(self, sequence: str) -> Dict[str, Any]:
        """Predict for a single sequence, returning prediction dict."""
        ...

    def predict_batch(self, sequences: List[str]) -> List[Dict[str, Any]]:
        """Predict for multiple sequences."""
        ...

class MockPredictor:
    """
    Mock predictor that generates deterministic predictions for testing.
    
    Uses sequence hash to generate reproducible outputs, similar to FixedEmbedder.
    """

    def __init__(self, config: PredictionOracleConfig):
        self.config = config

    def _get_seed(self, sequence: str) -> int:
        """Get deterministic seed from sequence."""
        import hashlib
        hash_bytes = hashlib.sha256(sequence.encode()).digest()
        return int.from_bytes(hash_bytes[:4], "big")

    def predict(self, sequence: str) -> Dict[str, Any]:
        """Generate deterministic mock prediction."""
        seed = self._get_seed(sequence)
        rng = np.random.default_rng(seed)

        if self.config.output_type == "per_residue":
            if self.config.is_classification:

                raw = rng.random((len(sequence), self.config.num_classes))
                probs = raw / raw.sum(axis=1, keepdims=True)
                predictions = probs.argmax(axis=1)
                return {
                    "predictions": predictions.tolist(),
                    "probabilities": probs.tolist(),
                    "sequence_length": len(sequence),
                }
            else:

                low, high = self.config.value_range or (0.0, 1.0)
                values = rng.uniform(low, high, len(sequence))
                return {
                    "predictions": values.tolist(),
                    "sequence_length": len(sequence),
                }
        else:
            if self.config.is_classification:
                raw = rng.random(self.config.num_classes)
                probs = raw / raw.sum()
                prediction = int(probs.argmax())
                return {
                    "prediction": prediction,
                    "probabilities": probs.tolist(),
                }
            else:
                low, high = self.config.value_range or (0.0, 1.0)
                value = rng.uniform(low, high)
                return {"prediction": float(value)}

    def predict_batch(self, sequences: List[str]) -> List[Dict[str, Any]]:
        """Predict for batch of sequences."""
        return [self.predict(seq) for seq in sequences]


class PredictionDeterminismOracle:
    """
    Oracle verifying that predictions are deterministic.

    Tests that predicting the same sequence multiple times yields
    identical results.
    """

    def __init__(
        self,
        predictor: PredictorProtocol,
        config: PredictionOracleConfig,
        num_runs: int = 3,
    ):
        self.predictor = predictor
        self.config = config
        self.num_runs = num_runs

    def verify(self, sequence: str) -> Dict[str, Any]:
        """
        Verify prediction determinism for a sequence.

        Args:
            sequence: The sequence to test

        Returns:
            Result dictionary with pass/fail and details
        """
        predictions = []
        for _ in range(self.num_runs):
            pred = self.predictor.predict(sequence)
            predictions.append(pred)


        first = predictions[0]
        all_identical = all(
            self._predictions_equal(first, p) for p in predictions[1:]
        )

        result = {
            "model": self.config.model_name,
            "test_type": "determinism",
            "sequence_length": len(sequence),
            "num_runs": self.num_runs,
            "passed": all_identical,
        }
        add_prediction_oracle_result(result)
        return result

    def _predictions_equal(self, p1: Dict, p2: Dict) -> bool:
        """Check if two prediction dicts are equal."""
        if set(p1.keys()) != set(p2.keys()):
            return False
        for key in p1:
            v1, v2 = p1[key], p2[key]
            if isinstance(v1, (list, np.ndarray)):
                if not np.allclose(v1, v2, rtol=1e-5):
                    return False
            elif isinstance(v1, float):
                if not np.isclose(v1, v2, rtol=1e-5):
                    return False
            elif v1 != v2:
                return False
        return True


class OutputValidityOracle:
    """
    Oracle verifying that prediction outputs are valid.

    Tests:
    - Classification probabilities sum to 1 and are in [0, 1]
    - Per-residue predictions have correct length
    - Continuous outputs are within expected range
    """

    def __init__(
        self,
        predictor: PredictorProtocol,
        config: PredictionOracleConfig,
    ):
        self.predictor = predictor
        self.config = config

    def verify(self, sequence: str) -> Dict[str, Any]:
        """
        Verify output validity for a sequence.

        Args:
            sequence: The sequence to test

        Returns:
            Result dictionary with pass/fail and details
        """
        pred = self.predictor.predict(sequence)
        issues = []


        if self.config.output_type == "per_residue":
            if "predictions" in pred:
                if len(pred["predictions"]) != len(sequence):
                    issues.append(
                        f"predictions length {len(pred['predictions'])} != "
                        f"sequence length {len(sequence)}"
                    )


        if self.config.is_classification and "probabilities" in pred:
            probs = np.array(pred["probabilities"])
            

            if np.any(probs < 0) or np.any(probs > 1):
                issues.append("probabilities outside [0, 1] range")
            

            if self.config.output_type == "per_residue":
                sums = probs.sum(axis=1)
                if not np.allclose(sums, 1.0, atol=1e-5):
                    issues.append("per-residue probabilities don't sum to 1")
            else:
                if not np.isclose(probs.sum(), 1.0, atol=1e-5):
                    issues.append("sequence probabilities don't sum to 1")


        if not self.config.is_classification and self.config.value_range:
            low, high = self.config.value_range
            values = np.array(pred.get("predictions", [pred.get("prediction", 0)]))
            if np.any(values < low) or np.any(values > high):
                issues.append(f"values outside expected range [{low}, {high}]")

        passed = len(issues) == 0

        result = {
            "model": self.config.model_name,
            "test_type": "output_validity",
            "sequence_length": len(sequence),
            "issues": issues,
            "passed": passed,
        }
        add_prediction_oracle_result(result)
        return result


class ShapeInvarianceOracle:
    """
    Oracle verifying that output shapes are consistent with model specification.

    Tests that:
    - Per-residue outputs match sequence length
    - Number of classes matches model specification
    - Batch outputs maintain individual shapes
    """

    def __init__(
        self,
        predictor: PredictorProtocol,
        config: PredictionOracleConfig,
    ):
        self.predictor = predictor
        self.config = config

    def verify(self, sequences: List[str]) -> Dict[str, Any]:
        """
        Verify shape invariance across sequences of different lengths.

        Args:
            sequences: List of sequences with varying lengths

        Returns:
            Result dictionary with pass/fail and details
        """
        issues = []

        for seq in sequences:
            pred = self.predictor.predict(seq)

            if self.config.output_type == "per_residue":

                if "predictions" in pred:
                    pred_len = len(pred["predictions"])
                    if pred_len != len(seq):
                        issues.append(
                            f"seq len {len(seq)}: pred len {pred_len} mismatch"
                        )


                if "probabilities" in pred and self.config.num_classes:
                    probs = np.array(pred["probabilities"])
                    expected_shape = (len(seq), self.config.num_classes)
                    if probs.shape != expected_shape:
                        issues.append(
                            f"seq len {len(seq)}: probs shape {probs.shape} "
                            f"!= expected {expected_shape}"
                        )

        passed = len(issues) == 0

        result = {
            "model": self.config.model_name,
            "test_type": "shape_invariance",
            "num_sequences": len(sequences),
            "issues": issues,
            "passed": passed,
        }
        add_prediction_oracle_result(result)
        return result


@pytest.fixture(scope="module")
def ss_oracle_config() -> PredictionOracleConfig:
    """Oracle configuration for secondary structure prediction."""
    return PREDICTION_ORACLE_CONFIGS["ProtT5SecondaryStructure"]


@pytest.fixture(scope="module")
def binding_oracle_config() -> PredictionOracleConfig:
    """Oracle configuration for binding site prediction."""
    return PREDICTION_ORACLE_CONFIGS["BindEmbed"]


@pytest.fixture(scope="module")
def disorder_oracle_config() -> PredictionOracleConfig:
    """Oracle configuration for disorder prediction."""
    return PREDICTION_ORACLE_CONFIGS["Seth"]


@pytest.fixture(scope="module")
def mock_ss_predictor(ss_oracle_config) -> MockPredictor:
    """Mock predictor for secondary structure."""
    return MockPredictor(ss_oracle_config)


@pytest.fixture(scope="module")
def mock_binding_predictor(binding_oracle_config) -> MockPredictor:
    """Mock predictor for binding sites."""
    return MockPredictor(binding_oracle_config)


@pytest.fixture(scope="module")
def mock_disorder_predictor(disorder_oracle_config) -> MockPredictor:
    """Mock predictor for disorder."""
    return MockPredictor(disorder_oracle_config)


@pytest.fixture(scope="module")
def standard_test_sequences() -> List[str]:
    """Standard test sequences from canonical dataset."""
    return [
        CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
        CANONICAL_TEST_DATASET.get_by_id("standard_002").sequence,
        CANONICAL_TEST_DATASET.get_by_id("standard_003").sequence,
    ]


@pytest.fixture(scope="module")
def varied_length_sequences() -> List[str]:
    """Sequences with varied lengths for shape testing."""
    return [
        CANONICAL_TEST_DATASET.get_by_id("length_short_10").sequence,
        CANONICAL_TEST_DATASET.get_by_id("length_medium_50").sequence,
        CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
    ]


class TestPredictionDeterminism:
    """Tests for prediction determinism oracle."""

    def test_secondary_structure_determinism(
        self,
        mock_ss_predictor: MockPredictor,
        ss_oracle_config: PredictionOracleConfig,
        standard_test_sequences: List[str],
    ):
        """Verify secondary structure predictions are deterministic."""
        oracle = PredictionDeterminismOracle(
            predictor=mock_ss_predictor,
            config=ss_oracle_config,
        )

        for seq in standard_test_sequences:
            result = oracle.verify(seq)
            assert result["passed"], (
                f"Determinism failed: model={result['model']}, "
                f"sequence_length={len(seq)}, runs={result['num_runs']}"
            )

    def test_binding_site_determinism(
        self,
        mock_binding_predictor: MockPredictor,
        binding_oracle_config: PredictionOracleConfig,
        standard_test_sequences: List[str],
    ):
        """Verify binding site predictions are deterministic."""
        oracle = PredictionDeterminismOracle(
            predictor=mock_binding_predictor,
            config=binding_oracle_config,
        )

        for seq in standard_test_sequences:
            result = oracle.verify(seq)
            assert result["passed"], (
                f"Determinism failed: model={result['model']}, "
                f"sequence_length={len(seq)}, runs={result['num_runs']}"
            )

    def test_disorder_determinism(
        self,
        mock_disorder_predictor: MockPredictor,
        disorder_oracle_config: PredictionOracleConfig,
        standard_test_sequences: List[str],
    ):
        """Verify disorder predictions are deterministic."""
        oracle = PredictionDeterminismOracle(
            predictor=mock_disorder_predictor,
            config=disorder_oracle_config,
        )

        for seq in standard_test_sequences:
            result = oracle.verify(seq)
            assert result["passed"], (
                f"Determinism failed: model={result['model']}, "
                f"sequence_length={len(seq)}, runs={result['num_runs']}"
            )


class TestOutputValidity:
    """Tests for output validity oracle."""

    def test_secondary_structure_output_validity(
        self,
        mock_ss_predictor: MockPredictor,
        ss_oracle_config: PredictionOracleConfig,
        standard_test_sequences: List[str],
    ):
        """Verify secondary structure outputs are valid."""
        oracle = OutputValidityOracle(
            predictor=mock_ss_predictor,
            config=ss_oracle_config,
        )

        for seq in standard_test_sequences:
            result = oracle.verify(seq)
            assert result["passed"], (
                f"Output validity failed: model={result['model']}, "
                f"sequence_length={len(seq)}, issues={result['issues']}"
            )

    def test_binding_site_output_validity(
        self,
        mock_binding_predictor: MockPredictor,
        binding_oracle_config: PredictionOracleConfig,
        standard_test_sequences: List[str],
    ):
        """Verify binding site outputs are valid."""
        oracle = OutputValidityOracle(
            predictor=mock_binding_predictor,
            config=binding_oracle_config,
        )

        for seq in standard_test_sequences:
            result = oracle.verify(seq)
            assert result["passed"], (
                f"Output validity failed: model={result['model']}, "
                f"sequence_length={len(seq)}, issues={result['issues']}"
            )

    def test_disorder_output_validity(
        self,
        mock_disorder_predictor: MockPredictor,
        disorder_oracle_config: PredictionOracleConfig,
        standard_test_sequences: List[str],
    ):
        """Verify disorder outputs are valid (values in [0,1])."""
        oracle = OutputValidityOracle(
            predictor=mock_disorder_predictor,
            config=disorder_oracle_config,
        )

        for seq in standard_test_sequences:
            result = oracle.verify(seq)
            assert result["passed"], (
                f"Output validity failed: model={result['model']}, "
                f"sequence_length={len(seq)}, issues={result['issues']}"
            )


class TestShapeInvariance:
    """Tests for shape invariance oracle."""

    def test_secondary_structure_shape_invariance(
        self,
        mock_ss_predictor: MockPredictor,
        ss_oracle_config: PredictionOracleConfig,
        varied_length_sequences: List[str],
    ):
        """Verify secondary structure output shapes match sequence lengths."""
        oracle = ShapeInvarianceOracle(
            predictor=mock_ss_predictor,
            config=ss_oracle_config,
        )

        result = oracle.verify(varied_length_sequences)
        assert result["passed"], (
            f"Shape invariance failed: model={result['model']}, "
            f"num_sequences={result['num_sequences']}, issues={result['issues']}"
        )

    def test_binding_site_shape_invariance(
        self,
        mock_binding_predictor: MockPredictor,
        binding_oracle_config: PredictionOracleConfig,
        varied_length_sequences: List[str],
    ):
        """Verify binding site output shapes match sequence lengths."""
        oracle = ShapeInvarianceOracle(
            predictor=mock_binding_predictor,
            config=binding_oracle_config,
        )

        result = oracle.verify(varied_length_sequences)
        assert result["passed"], (
            f"Shape invariance failed: model={result['model']}, "
            f"num_sequences={result['num_sequences']}, issues={result['issues']}"
        )


# =============================================================================
# Real Model Predictor Wrapper
# =============================================================================


class RealPredictor:
    """
    Wrapper that makes HTTP requests to the real running server.
    
    Submits prediction requests to /prediction_service/predict and polls
    for completion, just like a real client would. This tests the full
    end-to-end prediction pipeline including embeddings and ONNX inference.
    
    Requires:
        - Server running (docker-compose.dev.yml or CI_SERVER_URL)
        - Pre-cached ProtT5 embeddings in the database
    """

    def __init__(
        self,
        model_name: str,
        config: PredictionOracleConfig,
        server_url: str = None,
        timeout: int = 120,
        poll_interval: float = 2.0,
    ):
        self.model_name = model_name
        self.config = config
        self.timeout = timeout
        self.poll_interval = poll_interval
        
        # Get server URL from env or parameter
        self.server_url = server_url or os.environ.get(
            "CI_SERVER_URL", "http://localhost:9540"
        )
        self._client = None

    def _ensure_initialized(self):
        """Lazy initialization of HTTP client."""
        if self._client is not None:
            return
        
        import httpx
        self._client = httpx.Client(
            base_url=self.server_url,
            timeout=30.0,
        )
        
        # Verify server is reachable
        try:
            response = self._client.get("/health")
            if response.status_code != 200:
                raise RuntimeError(f"Server health check failed: {response.status_code}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to server at {self.server_url}. "
                f"Ensure server is running: {e}"
            )

    def _poll_task(self, task_id: str) -> Dict[str, Any]:
        """Poll task until completion."""
        import time
        
        start = time.time()
        while time.time() - start < self.timeout:
            response = self._client.get(f"/biocentral_service/task_status/{task_id}")
            
            if response.status_code != 200:
                time.sleep(self.poll_interval)
                continue
            
            dtos = response.json().get("dtos", [])
            if not dtos:
                time.sleep(self.poll_interval)
                continue
            
            latest = dtos[-1]
            status = latest.get("status", "").upper()
            
            if status in ("FINISHED", "COMPLETED", "DONE"):
                return latest
            elif status in ("FAILED", "ERROR", "CANCELLED"):
                raise RuntimeError(
                    f"Prediction task failed: {latest.get('error', 'unknown')}"
                )
            
            time.sleep(self.poll_interval)
        
        raise TimeoutError(f"Task {task_id} did not complete within {self.timeout}s")

    def predict(self, sequence: str) -> Dict[str, Any]:
        """Predict for a single sequence via server HTTP API."""
        self._ensure_initialized()
        
        seq_id = "oracle_test_seq"
        
        # Submit prediction request
        request_data = {
            "model_names": [self.model_name],
            "sequence_input": {seq_id: sequence},
        }
        
        response = self._client.post("/prediction_service/predict", json=request_data)
        
        if response.status_code != 200:
            raise RuntimeError(
                f"Prediction request failed: {response.status_code} - {response.text}"
            )
        
        task_id = response.json().get("task_id")
        if not task_id:
            raise RuntimeError("No task_id in prediction response")
        
        # Poll for completion
        result = self._poll_task(task_id)
        
        # Extract predictions from result
        return self._format_server_response(result, seq_id, sequence)

    def _format_server_response(
        self,
        result: Dict[str, Any],
        seq_id: str,
        sequence: str,
    ) -> Dict[str, Any]:
        """Format server response to match PredictorProtocol interface."""
        # Server returns predictions nested under model name
        predictions_by_model = result.get("predictions", {})
        model_predictions = predictions_by_model.get(self.model_name, {})
        seq_predictions = model_predictions.get(seq_id, [])
        
        if self.config.is_classification:
            # Per-residue classification
            predictions = []
            for residue_pred in seq_predictions:
                if isinstance(residue_pred, dict):
                    predictions.append(residue_pred.get("class", residue_pred.get("prediction", 0)))
                else:
                    predictions.append(int(residue_pred))
            
            return {
                "predictions": predictions,
                "sequence_length": len(sequence),
            }
        else:
            # Regression output
            values = [
                float(p.get("value", p) if isinstance(p, dict) else p)
                for p in seq_predictions
            ]
            return {
                "predictions": values,
                "sequence_length": len(sequence),
            }

    def predict_batch(self, sequences: List[str]) -> List[Dict[str, Any]]:
        """Predict for multiple sequences."""
        return [self.predict(seq) for seq in sequences]


# =============================================================================
# Real Model Fixtures (Server Integration)
# =============================================================================


def _get_available_models() -> List[str]:
    """Get list of models available for testing."""
    return ["BindEmbed", "TMbed", "Seth"]


@pytest.fixture(scope="module")
def real_bindembed_predictor():
    """
    Create predictor that calls the real server's BindEmbed endpoint.
    
    Requires server running with pre-cached ProtT5 embeddings.
    Set CI_SERVER_URL environment variable or uses localhost:9540.
    """
    try:
        config = PREDICTION_ORACLE_CONFIGS["BindEmbed"]
        predictor = RealPredictor(
            model_name="BindEmbed",
            config=config,
        )
        predictor._ensure_initialized()
        return predictor
    except Exception as e:
        pytest.skip(f"Server not available for BindEmbed: {e}")


@pytest.fixture(scope="module")
def real_tmbed_predictor():
    """Create predictor that calls the real server's TMbed endpoint."""
    try:
        config = PREDICTION_ORACLE_CONFIGS["TMbed"]
        predictor = RealPredictor(
            model_name="TMbed",
            config=config,
        )
        predictor._ensure_initialized()
        return predictor
    except Exception as e:
        pytest.skip(f"Server not available for TMbed: {e}")


@pytest.fixture(scope="module")
def real_seth_predictor():
    """Create predictor that calls the real server's Seth endpoint."""
    try:
        config = PREDICTION_ORACLE_CONFIGS["Seth"]
        predictor = RealPredictor(
            model_name="Seth",
            config=config,
        )
        predictor._ensure_initialized()
        return predictor
    except Exception as e:
        pytest.skip(f"Server not available for Seth: {e}")


@pytest.fixture(scope="module", params=["BindEmbed"])
def real_predictor(request):
    """
    Parametrized fixture to test multiple models via server.
    
    Add model names to params list to test more models:
    params=["BindEmbed", "TMbed", "Seth"]
    """
    model_name = request.param
    config = PREDICTION_ORACLE_CONFIGS[model_name]
    try:
        predictor = RealPredictor(
            model_name=model_name,
            config=config,
        )
        predictor._ensure_initialized()
        return predictor, config
    except Exception as e:
        pytest.skip(f"Server not available for {model_name}: {e}")


# =============================================================================
# Server Integration Oracle Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@skip_in_fixed_embedder_ci
class TestPredictionDeterminismRealModels:
    """
    Prediction determinism tests via the real running server.
    
    Tests end-to-end prediction flow: HTTP request -> embeddings -> ONNX -> response.
    Requires server running with pre-cached ProtT5 embeddings.
    """

    def test_bindembed_determinism(
        self,
        real_bindembed_predictor: RealPredictor,
        binding_oracle_config: PredictionOracleConfig,
    ):
        """Verify BindEmbed predictions are deterministic via server."""
        oracle = PredictionDeterminismOracle(
            predictor=real_bindembed_predictor,
            config=binding_oracle_config,
        )
        
        # Use shorter sequence for speed
        sequence = CANONICAL_TEST_DATASET.get_by_id("length_medium_50").sequence
        result = oracle.verify(sequence)
        
        assert result["passed"], (
            f"Determinism failed: model={result['model']}, "
            f"sequence_length={result['sequence_length']}, runs={result['num_runs']}"
        )


@pytest.mark.slow
@pytest.mark.integration
@skip_in_fixed_embedder_ci
class TestOutputValidityRealModels:
    """
    Output validity tests via the real running server.
    
    Validates that server returns properly formatted prediction outputs.
    """

    def test_bindembed_output_validity(
        self,
        real_bindembed_predictor: RealPredictor,
        binding_oracle_config: PredictionOracleConfig,
    ):
        """Verify BindEmbed outputs are valid via server."""
        oracle = OutputValidityOracle(
            predictor=real_bindembed_predictor,
            config=binding_oracle_config,
        )
        
        sequence = CANONICAL_TEST_DATASET.get_by_id("length_medium_50").sequence
        result = oracle.verify(sequence)
        
        assert result["passed"], (
            f"Output validity failed: model={result['model']}, "
            f"sequence_length={len(sequence)}, issues={result['issues']}"
        )


@pytest.mark.slow
@pytest.mark.integration
@skip_in_fixed_embedder_ci
class TestShapeInvarianceRealModels:
    """
    Shape invariance tests via the real running server.
    
    Validates that server returns predictions with correct shapes.
    """

    def test_bindembed_shape_invariance(
        self,
        real_bindembed_predictor: RealPredictor,
        binding_oracle_config: PredictionOracleConfig,
    ):
        """Verify BindEmbed output shapes match sequence lengths via server."""
        oracle = ShapeInvarianceOracle(
            predictor=real_bindembed_predictor,
            config=binding_oracle_config,
        )
        
        # Test with varied lengths
        sequences = [
            CANONICAL_TEST_DATASET.get_by_id("length_short_10").sequence,
            CANONICAL_TEST_DATASET.get_by_id("length_medium_50").sequence,
        ]
        
        result = oracle.verify(sequences)
        assert result["passed"], (
            f"Shape invariance failed: model={result['model']}, "
            f"num_sequences={result['num_sequences']}, issues={result['issues']}"
        )


@pytest.mark.slow
@pytest.mark.integration
@skip_in_fixed_embedder_ci
class TestParametrizedRealModelOracles:
    """
    Parametrized oracle tests via the real running server.
    
    Tests multiple prediction models through the server's HTTP API.
    Skipped when CI_EMBEDDER=FixedEmbedder (no server available).
    To add more models, modify the `real_predictor` fixture params.
    """

    def test_real_model_determinism(
        self,
        real_predictor,
    ):
        """Verify predictions are deterministic for any model via server."""
        predictor, config = real_predictor
        oracle = PredictionDeterminismOracle(
            predictor=predictor,
            config=config,
        )
        
        sequence = CANONICAL_TEST_DATASET.get_by_id("length_medium_50").sequence
        result = oracle.verify(sequence)
        
        assert result["passed"], (
            f"Determinism failed: model={result['model']}, "
            f"sequence_length={result['sequence_length']}"
        )

    def test_real_model_output_validity(
        self,
        real_predictor,
    ):
        """Verify outputs are valid for any model via server."""
        predictor, config = real_predictor
        oracle = OutputValidityOracle(
            predictor=predictor,
            config=config,
        )
        
        sequence = CANONICAL_TEST_DATASET.get_by_id("length_medium_50").sequence
        result = oracle.verify(sequence)
        
        assert result["passed"], (
            f"Output validity failed: model={result['model']}, "
            f"issues={result['issues']}"
        )


@pytest.fixture(scope="module", autouse=True)
def cleanup_prediction_results():
    """Clean up prediction oracle results after module completes."""
    yield
    results = get_prediction_oracle_results()
    if results:
        print(f"\n📊 Prediction oracle tests completed: {len(results)} results")
    clear_prediction_oracle_results()
