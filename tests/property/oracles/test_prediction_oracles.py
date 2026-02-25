"""Prediction oracle tests for output consistency and validity."""

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
    Wrapper to run real prediction models (BindEmbed, TMbed, etc.).
    
    Handles embedding generation and model inference for oracle tests.
    Requires ProtT5 embeddings for most models.
    """

    def __init__(
        self,
        model_name: str,
        config: PredictionOracleConfig,
        device: str = "cpu",
        batch_size: int = 1,
    ):
        self.model_name = model_name
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._embedding_service = None

    def _ensure_initialized(self):
        """Lazy initialization of model and embedding service."""
        if self._model is not None:
            return

        from biocentral_server.predict.models import MODEL_MAP
        from biotrainer.embedders import get_embedding_service

        # Get model class from MODEL_MAP
        model_class = MODEL_MAP.get(self.model_name)
        if model_class is None:
            raise ValueError(f"Unknown model: {self.model_name}")

        # Initialize model with ONNX backend
        self._model = model_class(batch_size=self.batch_size, backend="onnx")
        
        # Get embedder name from model metadata
        embedder_name = self._model.get_metadata().embedder
        
        # Initialize embedding service
        self._embedding_service = get_embedding_service(
            embedder_name=embedder_name,
            use_half_precision=False,
            custom_tokenizer_config=None,
            device=self.device,
        )

    def predict(self, sequence: str) -> Dict[str, Any]:
        """Predict for a single sequence."""
        self._ensure_initialized()
        
        # Generate embedding
        results = list(
            self._embedding_service.generate_embeddings(
                input_data=[sequence],
                reduce=False,  # Per-residue embeddings for BindEmbed
            )
        )
        
        if not results:
            raise RuntimeError("Embedding generation failed")
        
        seq_id, embedding = results[0]
        embeddings = {seq_id: np.array(embedding)}
        sequences = {seq_id: sequence}
        
        # Run prediction
        predictions = self._model.predict(sequences=sequences, embeddings=embeddings)
        
        # Convert to oracle-expected format
        pred_result = predictions.get(seq_id, [])
        return self._format_prediction(pred_result, sequence)

    def _format_prediction(
        self,
        model_output: List,
        sequence: str,
    ) -> Dict[str, Any]:
        """Format model output to match PredictorProtocol interface."""
        if self.config.is_classification:
            # Per-residue classification (e.g., BindEmbed returns list of Prediction objects)
            predictions = []
            probabilities = []
            
            for residue_pred in model_output:
                # Each residue_pred should have class index and probabilities
                if hasattr(residue_pred, 'output_value'):
                    # Prediction object format
                    predictions.append(residue_pred.output_value)
                elif isinstance(residue_pred, dict):
                    predictions.append(residue_pred.get('class', 0))
                    if 'probabilities' in residue_pred:
                        probabilities.append(residue_pred['probabilities'])
                else:
                    predictions.append(int(residue_pred))
            
            result = {
                "predictions": predictions,
                "sequence_length": len(sequence),
            }
            if probabilities:
                result["probabilities"] = probabilities
            return result
        else:
            # Regression output
            values = [
                float(p.output_value) if hasattr(p, 'output_value') else float(p)
                for p in model_output
            ]
            return {
                "predictions": values,
                "sequence_length": len(sequence),
            }

    def predict_batch(self, sequences: List[str]) -> List[Dict[str, Any]]:
        """Predict for multiple sequences."""
        return [self.predict(seq) for seq in sequences]


# =============================================================================
# Real Model Fixtures
# =============================================================================


def _get_available_models() -> List[str]:
    """Get list of models available for testing."""
    return ["BindEmbed", "TMbed", "Seth"]


@pytest.fixture(scope="module")
def real_bindembed_predictor():
    """
    Load real BindEmbed model with ProtT5 embedder.
    
    Fails explicitly if dependencies cannot be loaded.
    """
    try:
        config = PREDICTION_ORACLE_CONFIGS["BindEmbed"]
        predictor = RealPredictor(
            model_name="BindEmbed",
            config=config,
            device="cpu",
        )
        # Force initialization to fail fast if model unavailable
        predictor._ensure_initialized()
        return predictor
    except ImportError as e:
        pytest.fail(
            f"Failed to import prediction model dependencies. "
            f"Ensure biotrainer and model files are available: {e}"
        )
    except Exception as e:
        pytest.fail(
            f"Failed to load BindEmbed model. "
            f"Model must be available for oracle tests: {e}"
        )


@pytest.fixture(scope="module")
def real_tmbed_predictor():
    """Load real TMbed model with ProtT5 embedder."""
    try:
        config = PREDICTION_ORACLE_CONFIGS["TMbed"]
        predictor = RealPredictor(
            model_name="TMbed",
            config=config,
            device="cpu",
        )
        predictor._ensure_initialized()
        return predictor
    except Exception as e:
        pytest.fail(f"Failed to load TMbed model: {e}")


@pytest.fixture(scope="module")
def real_seth_predictor():
    """Load real Seth (disorder) model with ProtT5 embedder."""
    try:
        config = PREDICTION_ORACLE_CONFIGS["Seth"]
        predictor = RealPredictor(
            model_name="Seth",
            config=config,
            device="cpu",
        )
        predictor._ensure_initialized()
        return predictor
    except Exception as e:
        pytest.fail(f"Failed to load Seth model: {e}")


@pytest.fixture(scope="module", params=["BindEmbed"])
def real_predictor(request):
    """
    Parametrized fixture to test multiple real models.
    
    Add model names to params list to test more models:
    params=["BindEmbed", "TMbed", "Seth"]
    """
    model_name = request.param
    config = PREDICTION_ORACLE_CONFIGS[model_name]
    try:
        predictor = RealPredictor(
            model_name=model_name,
            config=config,
            device="cpu",
        )
        predictor._ensure_initialized()
        return predictor, config
    except Exception as e:
        pytest.skip(f"Model {model_name} not available: {e}")


# =============================================================================
# Real Model Oracle Tests
# =============================================================================


@pytest.mark.slow
class TestPredictionDeterminismRealModels:
    """Prediction determinism tests using real models."""

    def test_bindembed_determinism(
        self,
        real_bindembed_predictor: RealPredictor,
        binding_oracle_config: PredictionOracleConfig,
    ):
        """Verify BindEmbed predictions are deterministic."""
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
class TestOutputValidityRealModels:
    """Output validity tests using real models."""

    def test_bindembed_output_validity(
        self,
        real_bindembed_predictor: RealPredictor,
        binding_oracle_config: PredictionOracleConfig,
    ):
        """Verify BindEmbed outputs are valid."""
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
class TestShapeInvarianceRealModels:
    """Shape invariance tests using real models."""

    def test_bindembed_shape_invariance(
        self,
        real_bindembed_predictor: RealPredictor,
        binding_oracle_config: PredictionOracleConfig,
    ):
        """Verify BindEmbed output shapes match sequence lengths."""
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
class TestParametrizedRealModelOracles:
    """
    Parametrized oracle tests that run against multiple real models.
    
    To add more models, modify the `real_predictor` fixture params.
    """

    def test_real_model_determinism(
        self,
        real_predictor,
    ):
        """Verify predictions are deterministic for any real model."""
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
        """Verify outputs are valid for any real model."""
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
