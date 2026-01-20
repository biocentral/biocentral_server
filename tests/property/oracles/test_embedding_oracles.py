"""
Embedding Oracle Tests for Batch Invariance and Masking Robustness.

This module implements test oracles that verify critical embedding properties:
1. Batch Invariance: Embedding a sequence alone yields the same vector as
   embedding it within a larger batch.
2. Masking Robustness: Embeddings remain stable when input tokens are
   progressively replaced with the unknown token 'X'.

Supports both FixedEmbedder (deterministic mock) and real ESM2-T6-8M model.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
import pytest

from tests.fixtures.fixed_embedder import FixedEmbedder
from tests.fixtures.test_dataset import (
    CANONICAL_TEST_DATASET,
    get_test_sequences,
)
from tests.property.oracles.embedding_metrics import (
    compute_all_metrics,
    format_metrics_table,
    get_default_report_path,
    write_metrics_csv,
)


# ============================================================================
# ORACLE RESULT STORAGE
# ============================================================================

# Global storage for collecting results across tests
_oracle_results: List[Dict[str, Any]] = []


def get_oracle_results() -> List[Dict[str, Any]]:
    """Get accumulated oracle results."""
    return _oracle_results


def add_oracle_result(result: Dict[str, Any]) -> None:
    """Add a result to the oracle results collection."""
    if "timestamp" not in result:
        result["timestamp"] = datetime.now().isoformat()
    _oracle_results.append(result)


def clear_oracle_results() -> None:
    """Clear all accumulated oracle results."""
    _oracle_results.clear()


# ============================================================================
# EMBEDDER PROTOCOL
# ============================================================================


class EmbedderProtocol(Protocol):
    """Protocol defining the embedder interface for oracle tests."""

    def embed(self, sequence: str) -> np.ndarray:
        """Embed a single sequence, returning per-residue embeddings."""
        ...

    def embed_pooled(self, sequence: str) -> np.ndarray:
        """Embed a single sequence, returning pooled embedding."""
        ...

    def embed_batch(
        self, sequences: List[str], pooled: bool = False
    ) -> List[np.ndarray]:
        """Embed multiple sequences."""
        ...


# ============================================================================
# ORACLE CONFIGURATION
# ============================================================================


@dataclass
class OracleConfig:
    """Configuration for embedding oracles."""

    embedder_name: str
    cosine_threshold: float
    batch_sizes: List[int] = field(default_factory=lambda: [1, 5, 10])
    masking_ratios: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])


# Pre-defined configurations for supported embedders
ORACLE_CONFIGS = {
    "fixed_embedder": OracleConfig(
        embedder_name="fixed_embedder",
        cosine_threshold=0.1,
    ),
    "esm2_t6_8m": OracleConfig(
        embedder_name="esm2_t6_8m",
        cosine_threshold=0.2,
    ),
}


# ============================================================================
# BATCH INVARIANCE ORACLE
# ============================================================================


class BatchInvarianceOracle:
    """
    Oracle verifying that embeddings are invariant to batch composition.

    Tests that embedding sequence A alone yields the same vector as
    embedding A within a larger batch of sequences.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        config: OracleConfig,
    ):
        """
        Initialize BatchInvarianceOracle.

        Args:
            embedder: Embedder instance implementing EmbedderProtocol
            config: Oracle configuration with thresholds and batch sizes
        """
        self.embedder = embedder
        self.config = config

    def verify(
        self,
        target_sequence: str,
        filler_sequences: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Verify batch invariance for a target sequence.

        Args:
            target_sequence: The sequence to test for batch invariance
            filler_sequences: Additional sequences to form batches

        Returns:
            List of result dictionaries with metrics for each batch size
        """
        results = []

        # Get the "ground truth" embedding: sequence embedded alone
        single_embedding = self.embedder.embed_pooled(target_sequence)

        for batch_size in self.config.batch_sizes:
            # Create batch with target at a random position
            batch = self._create_batch(target_sequence, filler_sequences, batch_size)
            target_idx = batch.index(target_sequence)

            # Embed the batch
            batch_embeddings = self.embedder.embed_batch(batch, pooled=True)
            batched_embedding = batch_embeddings[target_idx]

            # Compute all metrics
            metrics = compute_all_metrics(single_embedding, batched_embedding)

            # Check if passed (using cosine distance)
            passed = metrics["cosine_distance"] <= self.config.cosine_threshold

            result = {
                "embedder": self.config.embedder_name,
                "test_type": "batch_invariance",
                "parameter": f"batch_{batch_size}",
                "cosine_distance": metrics["cosine_distance"],
                "l2_distance": metrics["l2_distance"],
                "kl_divergence": metrics["kl_divergence"],
                "threshold": self.config.cosine_threshold,
                "passed": passed,
            }
            results.append(result)
            add_oracle_result(result)

        return results

    def _create_batch(
        self,
        target: str,
        fillers: List[str],
        batch_size: int,
    ) -> List[str]:
        """Create a batch with target sequence at a random position."""
        if batch_size == 1:
            return [target]

        # Select filler sequences (with repetition if needed)
        n_fillers = batch_size - 1
        selected_fillers = []
        for i in range(n_fillers):
            selected_fillers.append(fillers[i % len(fillers)])

        # Insert target at random position
        batch = selected_fillers.copy()
        insert_pos = random.randint(0, len(batch))
        batch.insert(insert_pos, target)

        return batch


# ============================================================================
# MASKING ROBUSTNESS ORACLE
# ============================================================================


class MaskingRobustnessOracle:
    """
    Oracle verifying embedding stability under progressive token masking.

    Tests that embeddings remain stable (below divergence threshold) when
    input tokens are progressively replaced with the unknown token 'X'.
    """

    MASK_TOKEN = "X"

    def __init__(
        self,
        embedder: EmbedderProtocol,
        config: OracleConfig,
    ):
        """
        Initialize MaskingRobustnessOracle.

        Args:
            embedder: Embedder instance implementing EmbedderProtocol
            config: Oracle configuration with thresholds and masking ratios
        """
        self.embedder = embedder
        self.config = config

    def verify(
        self,
        sequence: str,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Verify masking robustness for a sequence.

        Args:
            sequence: The sequence to test for masking robustness
            seed: Random seed for reproducible masking

        Returns:
            List of result dictionaries with metrics for each masking ratio
        """
        results = []
        random.seed(seed)

        # Get the "ground truth" embedding: original unmasked sequence
        original_embedding = self.embedder.embed_pooled(sequence)

        for ratio in self.config.masking_ratios:
            # Create masked sequence
            masked_sequence = self._mask_sequence(sequence, ratio, seed)

            # Embed the masked sequence
            masked_embedding = self.embedder.embed_pooled(masked_sequence)

            # Compute all metrics
            metrics = compute_all_metrics(original_embedding, masked_embedding)

            # Check if passed (using cosine distance)
            passed = metrics["cosine_distance"] <= self.config.cosine_threshold

            result = {
                "embedder": self.config.embedder_name,
                "test_type": "masking_robustness",
                "parameter": f"mask_{int(ratio * 100)}%",
                "cosine_distance": metrics["cosine_distance"],
                "l2_distance": metrics["l2_distance"],
                "kl_divergence": metrics["kl_divergence"],
                "threshold": self.config.cosine_threshold,
                "passed": passed,
            }
            results.append(result)
            add_oracle_result(result)

        return results

    def _mask_sequence(
        self,
        sequence: str,
        ratio: float,
        seed: int,
    ) -> str:
        """
        Mask a portion of the sequence with unknown token 'X'.

        Args:
            sequence: Original sequence
            ratio: Fraction of positions to mask (0.0 to 1.0)
            seed: Random seed for position selection

        Returns:
            Masked sequence string
        """
        if ratio <= 0:
            return sequence
        if ratio >= 1:
            return self.MASK_TOKEN * len(sequence)

        random.seed(seed + int(ratio * 1000))  # Different seed per ratio
        seq_list = list(sequence)
        n_mask = int(len(sequence) * ratio)

        # Select positions to mask
        positions = random.sample(range(len(sequence)), n_mask)

        for pos in positions:
            seq_list[pos] = self.MASK_TOKEN

        return "".join(seq_list)


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def fixed_embedder_oracle_config() -> OracleConfig:
    """Oracle configuration for FixedEmbedder."""
    return ORACLE_CONFIGS["fixed_embedder"]


@pytest.fixture(scope="module")
def esm2_t6_8m_oracle_config() -> OracleConfig:
    """Oracle configuration for ESM2-T6-8M."""
    return ORACLE_CONFIGS["esm2_t6_8m"]


@pytest.fixture(scope="module")
def esm2_t6_8m_embedder():
    """
    Load real ESM2-T6-8M embedder.

    Fails explicitly if the model cannot be loaded.
    """
    try:
        from biotrainer.embedders import get_embedding_service

        embedding_service = get_embedding_service(
            embedder_name="facebook/esm2_t6_8M_UR50D",
            use_half_precision=False,
            custom_tokenizer_config=None,
            device="cpu",
        )

        # Wrap in a compatible interface
        return ESM2EmbedderWrapper(embedding_service)

    except ImportError as e:
        pytest.fail(
            f"Failed to import biotrainer embedders. "
            f"Ensure biotrainer is installed: {e}"
        )
    except Exception as e:
        pytest.fail(
            f"Failed to load ESM2-T6-8M model. "
            f"Model must be available for oracle tests: {e}"
        )


class ESM2EmbedderWrapper:
    """Wrapper to adapt biotrainer EmbeddingService to EmbedderProtocol."""

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    def embed(self, sequence: str) -> np.ndarray:
        """Embed single sequence, returning per-residue embeddings."""
        # generate_embeddings yields tuples
        results = list(
            self.embedding_service.generate_embeddings(
                input_data=[sequence], reduce=False
            )
        )
        if results:
            _, embedding = results[0]
            return np.array(embedding)
        return np.array([])

    def embed_pooled(self, sequence: str) -> np.ndarray:
        """Embed single sequence, returning pooled embedding."""
        results = list(
            self.embedding_service.generate_embeddings(
                input_data=[sequence], reduce=True
            )
        )
        if results:
            _, embedding = results[0]
            return np.array(embedding)
        return np.array([])

    def embed_batch(
        self, sequences: List[str], pooled: bool = False
    ) -> List[np.ndarray]:
        """Embed multiple sequences."""
        results = list(
            self.embedding_service.generate_embeddings(
                input_data=sequences, reduce=pooled
            )
        )
        return [np.array(embedding) for _, embedding in results]


@pytest.fixture(scope="module")
def test_sequences() -> List[str]:
    """Test sequences for oracle verification from canonical dataset."""
    return get_test_sequences(categories=["standard"])


@pytest.fixture(scope="module")
def filler_sequences() -> List[str]:
    """Filler sequences for batch creation from canonical dataset."""
    # Use edge case sequences as fillers for diversity
    return get_test_sequences(categories=["edge_case"])


# ============================================================================
# TEST CLASSES
# ============================================================================


class TestBatchInvarianceFixedEmbedder:
    """Batch invariance tests using FixedEmbedder."""

    def test_embedding_matches_across_batch_sizes(
        self,
        fixed_embedder_esm2_t6: FixedEmbedder,
        fixed_embedder_oracle_config: OracleConfig,
        test_sequences: List[str],
        filler_sequences: List[str],
    ):
        """Verify embeddings are identical regardless of batch composition."""
        oracle = BatchInvarianceOracle(
            embedder=fixed_embedder_esm2_t6,
            config=fixed_embedder_oracle_config,
        )

        all_results = []
        for seq in test_sequences[:3]:  # Test first 3 sequences
            results = oracle.verify(seq, filler_sequences)
            all_results.extend(results)

        # Print results table
        table = format_metrics_table(
            all_results,
            title="Batch Invariance Oracle - FixedEmbedder",
        )
        print(table)

        # Assert all tests passed
        for result in all_results:
            assert result["passed"], (
                f"Batch invariance failed for {result['parameter']}: "
                f"cosine_distance={result['cosine_distance']:.6f} > "
                f"threshold={result['threshold']:.4f}"
            )


class TestMaskingRobustnessFixedEmbedder:
    """Masking robustness tests using FixedEmbedder."""

    def test_embedding_stable_under_progressive_masking(
        self,
        fixed_embedder_esm2_t6: FixedEmbedder,
        fixed_embedder_oracle_config: OracleConfig,
        test_sequences: List[str],
    ):
        """Verify embeddings remain stable under progressive masking."""
        oracle = MaskingRobustnessOracle(
            embedder=fixed_embedder_esm2_t6,
            config=fixed_embedder_oracle_config,
        )

        all_results = []
        for seq in test_sequences[:3]:  # Test first 3 sequences
            results = oracle.verify(seq)
            all_results.extend(results)

        # Print results table
        table = format_metrics_table(
            all_results,
            title="Masking Robustness Oracle - FixedEmbedder",
        )
        print(table)

        # Assert all tests passed
        for result in all_results:
            assert result["passed"], (
                f"Masking robustness failed for {result['parameter']}: "
                f"cosine_distance={result['cosine_distance']:.6f} > "
                f"threshold={result['threshold']:.4f}"
            )


@pytest.mark.slow
class TestBatchInvarianceESM2:
    """Batch invariance tests using real ESM2-T6-8M model."""

    def test_embedding_matches_across_batch_sizes(
        self,
        esm2_t6_8m_embedder,
        esm2_t6_8m_oracle_config: OracleConfig,
        test_sequences: List[str],
        filler_sequences: List[str],
    ):
        """Verify embeddings are identical regardless of batch composition."""
        oracle = BatchInvarianceOracle(
            embedder=esm2_t6_8m_embedder,
            config=esm2_t6_8m_oracle_config,
        )

        all_results = []
        for seq in test_sequences[:2]:  # Test first 2 sequences (real model is the slower one)
            results = oracle.verify(seq, filler_sequences)
            all_results.extend(results)

        # Print results table
        table = format_metrics_table(
            all_results,
            title="Batch Invariance Oracle - ESM2-T6-8M",
        )
        print(table)

        # Assert all tests passed
        for result in all_results:
            assert result["passed"], (
                f"Batch invariance failed for {result['parameter']}: "
                f"cosine_distance={result['cosine_distance']:.6f} > "
                f"threshold={result['threshold']:.4f}"
            )


@pytest.mark.slow
class TestMaskingRobustnessESM2:
    """Masking robustness tests using real ESM2-T6-8M model."""

    def test_embedding_stable_under_progressive_masking(
        self,
        esm2_t6_8m_embedder,
        esm2_t6_8m_oracle_config: OracleConfig,
        test_sequences: List[str],
    ):
        """Verify embeddings remain stable under progressive masking."""
        oracle = MaskingRobustnessOracle(
            embedder=esm2_t6_8m_embedder,
            config=esm2_t6_8m_oracle_config,
        )

        all_results = []
        for seq in test_sequences[:2]:  # Test first 2 sequences (again real model is slower)
            results = oracle.verify(seq)
            all_results.extend(results)

        # Print results table
        table = format_metrics_table(
            all_results,
            title="Masking Robustness Oracle - ESM2-T6-8M",
        )
        print(table)

        # Assert all tests passed
        for result in all_results:
            assert result["passed"], (
                f"Masking robustness failed for {result['parameter']}: "
                f"cosine_distance={result['cosine_distance']:.6f} > "
                f"threshold={result['threshold']:.4f}"
            )


# ============================================================================
# SESSION CLEANUP - WRITE CSV REPORT
# ============================================================================


@pytest.fixture(scope="module", autouse=True)
def write_oracle_report(request):
    """Write accumulated oracle results to CSV after module completes."""
    yield

    # Write results to CSV
    results = get_oracle_results()
    if results:
        report_path = get_default_report_path()
        write_metrics_csv(results, report_path)
        print(f"\n📊 Oracle metrics report written to: {report_path}")

    # Clear results for next module
    clear_oracle_results()
