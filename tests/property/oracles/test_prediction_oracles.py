"""
Prediction Oracle Tests for Output Consistency and Validity.

This module implements test oracles that verify critical prediction properties:
1. Determinism: Same input always produces the same prediction output.
2. Output Range Validity: Classification probabilities in [0,1], per-residue 
   predictions have correct length, etc.
3. Shape Invariance: Output shapes match expected dimensions for the model.

Uses the canonical test dataset for reproducible testing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pytest

from tests.fixtures.test_dataset import (
    CANONICAL_TEST_DATASET,
    get_test_sequences,
)


# ============================================================================
# ORACLE RESULT STORAGE
# ============================================================================

_prediction_oracle_results: List[Dict[str, Any]] = []


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


# ============================================================================
# ORACLE CONFIGURATION
# ============================================================================


@dataclass
class PredictionOracleConfig:
    """Configuration for prediction oracles."""

    model_name: str
    output_type: str  # "per_residue" or "per_sequence"
    is_classification: bool = True
    num_classes: Optional[int] = None
    value_range: Optional[tuple] = None  # For regression outputs


# Pre-defined configurations for common prediction models
PREDICTION_ORACLE_CONFIGS = {
    "ProtT5SecondaryStructure": PredictionOracleConfig(
        model_name="ProtT5SecondaryStructure",
        output_type="per_residue",
        is_classification=True,
        num_classes=3,  # Helix, Sheet, Coil
    ),
    "BindEmbed": PredictionOracleConfig(
        model_name="BindEmbed",
        output_type="per_residue",
        is_classification=True,
        num_classes=2,  # Binding, Non-binding
    ),
    "TMbed": PredictionOracleConfig(
        model_name="TMbed",
        output_type="per_residue",
        is_classification=True,
        num_classes=4,  # Membrane topology classes
    ),
    "Seth": PredictionOracleConfig(
        model_name="Seth",
        output_type="per_residue",
        is_classification=False,
        value_range=(0.0, 1.0),  # Disorder probability
    ),
}


# ============================================================================
# PREDICTOR PROTOCOL
# ============================================================================


class PredictorProtocol(Protocol):
    """Protocol defining the predictor interface for oracle tests."""

    def predict(self, sequence: str) -> Dict[str, Any]:
        """Predict for a single sequence, returning prediction dict."""
        ...

    def predict_batch(self, sequences: List[str]) -> List[Dict[str, Any]]:
        """Predict for multiple sequences."""
        ...


# ============================================================================
# MOCK PREDICTOR FOR TESTING
# ============================================================================


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
                # Generate class probabilities for each residue
                raw = rng.random((len(sequence), self.config.num_classes))
                probs = raw / raw.sum(axis=1, keepdims=True)  # Normalize to sum to 1
                predictions = probs.argmax(axis=1)
                return {
                    "predictions": predictions.tolist(),
                    "probabilities": probs.tolist(),
                    "sequence_length": len(sequence),
                }
            else:
                # Generate continuous values in range
                low, high = self.config.value_range or (0.0, 1.0)
                values = rng.uniform(low, high, len(sequence))
                return {
                    "predictions": values.tolist(),
                    "sequence_length": len(sequence),
                }
        else:  # per_sequence
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


# ============================================================================
# DETERMINISM ORACLE
# ============================================================================


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

        # Check all predictions are identical
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


# ============================================================================
# OUTPUT VALIDITY ORACLE
# ============================================================================


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

        # Check per-residue length
        if self.config.output_type == "per_residue":
            if "predictions" in pred:
                if len(pred["predictions"]) != len(sequence):
                    issues.append(
                        f"predictions length {len(pred['predictions'])} != "
                        f"sequence length {len(sequence)}"
                    )

        # Check classification probabilities
        if self.config.is_classification and "probabilities" in pred:
            probs = np.array(pred["probabilities"])
            
            # Check range [0, 1]
            if np.any(probs < 0) or np.any(probs > 1):
                issues.append("probabilities outside [0, 1] range")
            
            # Check sum to 1 (per position for per-residue)
            if self.config.output_type == "per_residue":
                sums = probs.sum(axis=1)
                if not np.allclose(sums, 1.0, atol=1e-5):
                    issues.append("per-residue probabilities don't sum to 1")
            else:
                if not np.isclose(probs.sum(), 1.0, atol=1e-5):
                    issues.append("sequence probabilities don't sum to 1")

        # Check continuous value range
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


# ============================================================================
# SHAPE INVARIANCE ORACLE
# ============================================================================


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
                # Check predictions length
                if "predictions" in pred:
                    pred_len = len(pred["predictions"])
                    if pred_len != len(seq):
                        issues.append(
                            f"seq len {len(seq)}: pred len {pred_len} mismatch"
                        )

                # Check probability shape
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


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


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


# ============================================================================
# TEST CLASSES
# ============================================================================


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
                f"Determinism failed for sequence of length {len(seq)}"
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
                f"Determinism failed for sequence of length {len(seq)}"
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
                f"Determinism failed for sequence of length {len(seq)}"
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
                f"Output validity failed: {result['issues']}"
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
                f"Output validity failed: {result['issues']}"
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
                f"Output validity failed: {result['issues']}"
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
            f"Shape invariance failed: {result['issues']}"
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
            f"Shape invariance failed: {result['issues']}"
        )


# ============================================================================
# SESSION CLEANUP
# ============================================================================


@pytest.fixture(scope="module", autouse=True)
def cleanup_prediction_results():
    """Clean up prediction oracle results after module completes."""
    yield
    results = get_prediction_oracle_results()
    if results:
        print(f"\n📊 Prediction oracle tests completed: {len(results)} results")
    clear_prediction_oracle_results()
