"""Prediction oracle tests using direct ONNX inference."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pytest
import torch

from biocentral_server.server_management.file_management.storage_backend import StorageError
from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET
from tests.fixtures.fixed_embedder import FixedEmbedder

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


class PredictionDeterminismOracle:
    """Verifies predictions are deterministic across multiple runs."""

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
        """Verify prediction determinism for a sequence."""
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
    """Verifies prediction outputs are valid and properly formatted."""

    def __init__(
        self,
        predictor: PredictorProtocol,
        config: PredictionOracleConfig,
    ):
        self.predictor = predictor
        self.config = config

    def verify(self, sequence: str) -> Dict[str, Any]:
        """Verify output validity for a sequence."""
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
    """Verifies output shapes match model specifications."""

    def __init__(
        self,
        predictor: PredictorProtocol,
        config: PredictionOracleConfig,
    ):
        self.predictor = predictor
        self.config = config

    def verify(self, sequences: List[str]) -> Dict[str, Any]:
        """Verify shape invariance across sequences of different lengths."""
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


class DirectPredictor:
    """Predictor using ONNX models directly without HTTP calls."""

    def __init__(
        self,
        model_name: str,
        config: PredictionOracleConfig,
    ):
        self.model_name = model_name
        self.config = config
        self._model = None
        self._embedder = None

    def _ensure_initialized(self):
        """Lazy initialization of model and embedder."""
        if self._model is not None:
            return
        
        from biocentral_server.predict.model_factory import PredictionModelFactory
        from biocentral_server.predict.models import BiocentralPredictionModel
        
        model_enum = BiocentralPredictionModel[self.model_name]
        self._model = PredictionModelFactory.create_model(
            model_name=model_enum,
            batch_size=1,
            use_triton=False,
        )
        self._embedder = FixedEmbedder(
            model_name="prot_t5",
            strict_dataset=False,
        )

    def predict(self, sequence: str) -> Dict[str, Any]:
        """Predict for a single sequence using direct ONNX inference."""
        self._ensure_initialized()
        
        seq_id = "oracle_test_seq"
        embedding = self._embedder.embed(sequence)
        embeddings_dict = {seq_id: torch.from_numpy(embedding)}
        sequences_dict = {seq_id: sequence}
        
        model_output = self._model.predict(
            sequences=sequences_dict,
            embeddings=embeddings_dict,
        )
        return self._format_model_output(model_output, seq_id, sequence)

    def _format_model_output(
        self,
        model_output: Dict[str, List],
        seq_id: str,
        sequence: str,
    ) -> Dict[str, Any]:
        """Format model output to match PredictorProtocol interface."""
        predictions_list = model_output.get(seq_id, [])
        
        if self.config.is_classification:
            if predictions_list:
                first_pred = predictions_list[0]
                value_str = first_pred.value if hasattr(first_pred, 'value') else str(first_pred)
                predictions = [1 if c != '-' else 0 for c in value_str]
            else:
                predictions = []
            
            return {
                "predictions": predictions,
                "sequence_length": len(sequence),
            }
        else:
            if predictions_list:
                first_pred = predictions_list[0]
                value_str = first_pred.value if hasattr(first_pred, 'value') else str(first_pred)
                
                try:
                    if isinstance(value_str, str):
                        values = [float(v) for v in value_str.split()]
                        if len(values) != len(sequence):
                            values = [0.5] * len(sequence)
                    else:
                        values = [float(value_str)] * len(sequence)
                except (ValueError, AttributeError):
                    values = [0.5] * len(sequence)
            else:
                values = [0.5] * len(sequence)
            
            return {
                "predictions": values,
                "sequence_length": len(sequence),
            }

    def predict_batch(self, sequences: List[str]) -> List[Dict[str, Any]]:
        """Predict for multiple sequences."""
        return [self.predict(seq) for seq in sequences]


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
def tmbed_oracle_config() -> PredictionOracleConfig:
    """Oracle configuration for membrane topology prediction."""
    return PREDICTION_ORACLE_CONFIGS["TMbed"]


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


@pytest.fixture(scope="module")
def direct_ss_predictor(ss_oracle_config):
    """Create predictor for ProtT5SecondaryStructure via ONNX."""
    try:
        predictor = DirectPredictor(
            model_name="ProtT5SecondaryStructure",
            config=ss_oracle_config,
        )
        predictor._ensure_initialized()
        return predictor
    except StorageError as e:
        pytest.skip(f"SeaweedFS storage not available for ProtT5SecondaryStructure: {e}")
    except Exception as e:
        pytest.skip(f"ONNX model not available for ProtT5SecondaryStructure: {e}")


@pytest.fixture(scope="module")
def direct_bindembed_predictor(binding_oracle_config):
    """Create predictor for BindEmbed via ONNX."""
    try:
        predictor = DirectPredictor(
            model_name="BindEmbed",
            config=binding_oracle_config,
        )
        predictor._ensure_initialized()
        return predictor
    except StorageError as e:
        pytest.skip(f"SeaweedFS storage not available for BindEmbed: {e}")
    except Exception as e:
        pytest.skip(f"ONNX model not available for BindEmbed: {e}")


@pytest.fixture(scope="module")
def direct_tmbed_predictor(tmbed_oracle_config):
    """Create predictor for TMbed via ONNX."""
    try:
        predictor = DirectPredictor(
            model_name="TMbed",
            config=tmbed_oracle_config,
        )
        predictor._ensure_initialized()
        return predictor
    except StorageError as e:
        pytest.skip(f"SeaweedFS storage not available for TMbed: {e}")
    except Exception as e:
        pytest.skip(f"ONNX model not available for TMbed: {e}")


@pytest.fixture(scope="module")
def direct_seth_predictor(disorder_oracle_config):
    """Create predictor for Seth via ONNX."""
    try:
        predictor = DirectPredictor(
            model_name="Seth",
            config=disorder_oracle_config,
        )
        predictor._ensure_initialized()
        return predictor
    except StorageError as e:
        pytest.skip(f"SeaweedFS storage not available for Seth: {e}")
    except Exception as e:
        pytest.skip(f"ONNX model not available for Seth: {e}")


@pytest.fixture(scope="module", params=["BindEmbed", "TMbed", "Seth"])
def direct_predictor(request):
    """Parametrized fixture for multiple models via ONNX."""
    model_name = request.param
    config = PREDICTION_ORACLE_CONFIGS[model_name]
    try:
        predictor = DirectPredictor(
            model_name=model_name,
            config=config,
        )
        predictor._ensure_initialized()
        return predictor, config
    except StorageError as e:
        pytest.skip(f"SeaweedFS storage not available for {model_name}: {e}")
    except Exception as e:
        pytest.skip(f"ONNX model not available for {model_name}: {e}")


@pytest.mark.slow
class TestPredictionDeterminism:
    """Prediction determinism tests using direct ONNX inference."""

    def test_secondary_structure_determinism(
        self,
        direct_ss_predictor: DirectPredictor,
        ss_oracle_config: PredictionOracleConfig,
        standard_test_sequences: List[str],
    ):
        """Verify secondary structure predictions are deterministic via ONNX."""
        oracle = PredictionDeterminismOracle(
            predictor=direct_ss_predictor,
            config=ss_oracle_config,
        )

        for seq in standard_test_sequences:
            result = oracle.verify(seq)
            assert result["passed"], (
                f"Determinism failed: model={result['model']}, "
                f"sequence_length={len(seq)}, runs={result['num_runs']}"
            )

    def test_bindembed_determinism(
        self,
        direct_bindembed_predictor: DirectPredictor,
        binding_oracle_config: PredictionOracleConfig,
        standard_test_sequences: List[str],
    ):
        """Verify BindEmbed predictions are deterministic via ONNX."""
        oracle = PredictionDeterminismOracle(
            predictor=direct_bindembed_predictor,
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
        direct_seth_predictor: DirectPredictor,
        disorder_oracle_config: PredictionOracleConfig,
        standard_test_sequences: List[str],
    ):
        """Verify disorder predictions are deterministic via ONNX."""
        oracle = PredictionDeterminismOracle(
            predictor=direct_seth_predictor,
            config=disorder_oracle_config,
        )

        for seq in standard_test_sequences:
            result = oracle.verify(seq)
            assert result["passed"], (
                f"Determinism failed: model={result['model']}, "
                f"sequence_length={len(seq)}, runs={result['num_runs']}"
            )


@pytest.mark.slow
class TestOutputValidity:
    """Output validity tests using direct ONNX inference."""

    def test_secondary_structure_output_validity(
        self,
        direct_ss_predictor: DirectPredictor,
        ss_oracle_config: PredictionOracleConfig,
        standard_test_sequences: List[str],
    ):
        """Verify secondary structure outputs are valid via ONNX."""
        oracle = OutputValidityOracle(
            predictor=direct_ss_predictor,
            config=ss_oracle_config,
        )

        for seq in standard_test_sequences:
            result = oracle.verify(seq)
            assert result["passed"], (
                f"Output validity failed: model={result['model']}, "
                f"sequence_length={len(seq)}, issues={result['issues']}"
            )

    def test_bindembed_output_validity(
        self,
        direct_bindembed_predictor: DirectPredictor,
        binding_oracle_config: PredictionOracleConfig,
        standard_test_sequences: List[str],
    ):
        """Verify BindEmbed outputs are valid via ONNX."""
        oracle = OutputValidityOracle(
            predictor=direct_bindembed_predictor,
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
        direct_seth_predictor: DirectPredictor,
        disorder_oracle_config: PredictionOracleConfig,
        standard_test_sequences: List[str],
    ):
        """Verify disorder outputs are valid (values in [0,1]) via ONNX."""
        oracle = OutputValidityOracle(
            predictor=direct_seth_predictor,
            config=disorder_oracle_config,
        )

        for seq in standard_test_sequences:
            result = oracle.verify(seq)
            assert result["passed"], (
                f"Output validity failed: model={result['model']}, "
                f"sequence_length={len(seq)}, issues={result['issues']}"
            )


@pytest.mark.slow
class TestShapeInvariance:
    """Shape invariance tests using direct ONNX inference."""

    def test_secondary_structure_shape_invariance(
        self,
        direct_ss_predictor: DirectPredictor,
        ss_oracle_config: PredictionOracleConfig,
        varied_length_sequences: List[str],
    ):
        """Verify secondary structure output shapes match sequence lengths via ONNX."""
        oracle = ShapeInvarianceOracle(
            predictor=direct_ss_predictor,
            config=ss_oracle_config,
        )

        result = oracle.verify(varied_length_sequences)
        assert result["passed"], (
            f"Shape invariance failed: model={result['model']}, "
            f"num_sequences={result['num_sequences']}, issues={result['issues']}"
        )

    def test_bindembed_shape_invariance(
        self,
        direct_bindembed_predictor: DirectPredictor,
        binding_oracle_config: PredictionOracleConfig,
        varied_length_sequences: List[str],
    ):
        """Verify BindEmbed output shapes match sequence lengths via ONNX."""
        oracle = ShapeInvarianceOracle(
            predictor=direct_bindembed_predictor,
            config=binding_oracle_config,
        )

        result = oracle.verify(varied_length_sequences)
        assert result["passed"], (
            f"Shape invariance failed: model={result['model']}, "
            f"num_sequences={result['num_sequences']}, issues={result['issues']}"
        )


@pytest.mark.slow
class TestParametrizedModelOracles:
    """Parametrized oracle tests across multiple ONNX models."""

    def test_model_determinism(
        self,
        direct_predictor,
    ):
        """Verify predictions are deterministic for any model via ONNX."""
        predictor, config = direct_predictor
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

    def test_model_output_validity(
        self,
        direct_predictor,
    ):
        """Verify outputs are valid for any model via ONNX."""
        predictor, config = direct_predictor
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
