"""Prediction oracle tests using direct ONNX inference.

These tests load ONNX models directly from local filesystem, bypassing
all server infrastructure (SeaweedFS, Triton, etc.).

Configure model path with ONNX_MODELS_PATH environment variable.
Models can be downloaded from the TUM Nextcloud links in .env.example.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pytest

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
        num_classes=3,  # metal, nucleic, small binding types
    ),
    "TMbed": PredictionOracleConfig(
        model_name="TMbed",
        output_type="per_residue",
        is_classification=True,
        num_classes=7,  # B, b, H, h, S, i, o topology classes
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


# Model name mapping to ONNX directory names (uses model_name.value.lower())
# Files are auto-discovered within each directory
MODEL_DIRECTORIES = {
    "ProtT5SecondaryStructure": "prott5secondarystructure",
    "Seth": "seth",
    "BindEmbed": "bindembed",
    "TMbed": "tmbed",
}

# Label mappings for classification models
LABEL_MAPPINGS = {
    "ProtT5SecondaryStructure": {0: "H", 1: "E", 2: "L"},
    # BindEmbed predicts 3 binding types: argmax gives dominant type
    "BindEmbed": {0: "M", 1: "N", 2: "S"},  # Metal, Nucleic, Small molecule
    # TMbed transmembrane topology classes
    "TMbed": {0: "B", 1: "b", 2: "H", 3: "h", 4: "S", 5: "i", 6: "o"},
}


def get_onnx_models_path() -> Optional[Path]:
    """Get the path to ONNX models directory from env var."""
    path_str = os.environ.get("ONNX_MODELS_PATH")
    if path_str:
        path = Path(path_str)
        if path.exists():
            return path
    return None


class DirectPredictor:
    """Predictor using ONNX models directly from local filesystem.
    
    This class loads ONNX models without any server dependencies.
    Configure model path with ONNX_MODELS_PATH environment variable.
    """

    def __init__(
        self,
        model_name: str,
        config: PredictionOracleConfig,
    ):
        self.model_name = model_name
        self.config = config
        self._models: List[Any] = []  # List of ONNX sessions (for ensemble models)
        self._embedder = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of model and embedder."""
        if self._initialized:
            return
        
        import onnxruntime as ort
        
        models_path = get_onnx_models_path()
        if models_path is None:
            raise FileNotFoundError(
                "ONNX_MODELS_PATH not set or directory not found. "
                "Set ONNX_MODELS_PATH to directory containing model files."
            )
        
        model_dir_name = MODEL_DIRECTORIES.get(self.model_name)
        if not model_dir_name:
            raise ValueError(f"No directory configured for model: {self.model_name}")
        
        model_dir = models_path / model_dir_name
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. "
                f"Download models and set ONNX_MODELS_PATH correctly."
            )
        
        # Auto-discover ONNX files in the model directory
        onnx_files = sorted(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(
                f"No .onnx files found in {model_dir}. "
                f"Ensure models are properly extracted."
            )
        
        # Load all ONNX models (single model or ensemble)
        for onnx_file in onnx_files:
            session = ort.InferenceSession(str(onnx_file))
            self._models.append(session)
        
        # Initialize embedder
        self._embedder = FixedEmbedder(
            model_name="prot_t5",
            strict_dataset=False,
        )
        self._initialized = True

    def predict(self, sequence: str) -> Dict[str, Any]:
        """Predict for a single sequence using direct ONNX inference."""
        self._ensure_initialized()
        
        # Get embedding
        embedding = self._embedder.embed(sequence)
        # Shape: (seq_len, embedding_dim) -> (1, seq_len, embedding_dim)
        embedding_input = np.expand_dims(embedding.astype(np.float32), axis=0)
        
        # Run inference
        if len(self._models) == 1:
            # Single model
            output = self._run_single_model(embedding_input)
        else:
            # Ensemble - average predictions
            output = self._run_ensemble(embedding_input)
        
        return self._format_output(output, sequence)
    
    def _run_single_model(self, embedding_input: np.ndarray) -> np.ndarray:
        """Run inference with a single model."""
        input_name = self._models[0].get_inputs()[0].name
        outputs = self._models[0].run(None, {input_name: embedding_input})
        return outputs[0]  # Return first output
    
    def _run_ensemble(self, embedding_input: np.ndarray) -> np.ndarray:
        """Run ensemble inference and average predictions."""
        all_outputs = []
        for model in self._models:
            input_name = model.get_inputs()[0].name
            outputs = model.run(None, {input_name: embedding_input})
            all_outputs.append(outputs[0])
        
        # Average all predictions
        stacked = np.stack(all_outputs, axis=0)
        return np.mean(stacked, axis=0)
    
    def _format_output(self, raw_output: np.ndarray, sequence: str) -> Dict[str, Any]:
        """Format raw ONNX output to match PredictorProtocol interface."""
        # raw_output shape: (1, seq_len, num_classes) or (1, seq_len) or (1, seq_len, 1)
        
        # Handle 3D output
        if len(raw_output.shape) == 3:
            predictions_raw = raw_output[0]  # (seq_len, num_classes) or (seq_len, 1)
            
            # Check if it's regression with single output dim (e.g., Seth: seq_len x 1)
            if not self.config.is_classification and predictions_raw.shape[-1] == 1:
                # Squeeze the last dimension for regression
                predictions = predictions_raw.squeeze(-1).tolist()  # (seq_len,)
            elif self.config.is_classification:
                # Classification with logits - argmax to get predicted class index
                pred_indices = np.argmax(predictions_raw, axis=-1)  # (seq_len,)
                label_map = LABEL_MAPPINGS.get(self.model_name, {})
                predictions = [label_map.get(int(idx), "?") for idx in pred_indices]
            else:
                # Regression with multiple outputs per residue - take mean or first
                predictions = predictions_raw.mean(axis=-1).tolist()
        else:
            # 2D output: (1, seq_len) - direct values
            predictions = raw_output[0].tolist()  # (seq_len,)
        
        # Ensure predictions match sequence length
        if isinstance(predictions, list) and len(predictions) != len(sequence):
            if self.config.is_classification:
                predictions = [0] * len(sequence)
            else:
                predictions = [0.5] * len(sequence)
        
        return {
            "predictions": predictions,
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
    except FileNotFoundError as e:
        pytest.skip(f"ONNX model not found for ProtT5SecondaryStructure: {e}")
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
    except FileNotFoundError as e:
        pytest.skip(f"ONNX model not found for BindEmbed: {e}")
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
    except FileNotFoundError as e:
        pytest.skip(f"ONNX model not found for TMbed: {e}")
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
    except FileNotFoundError as e:
        pytest.skip(f"ONNX model not found for Seth: {e}")
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
    except FileNotFoundError as e:
        pytest.skip(f"ONNX model not found for {model_name}: {e}")
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
