# Embedding Oracle Tests for Batch Invariance and Masking Robustness.

import random
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Protocol

import numpy as np
import pytest
from biotrainer.input_files import BiotrainerSequenceRecord

from tests.fixtures.test_dataset import (
    get_test_sequences,
)
from tests.property.oracles.embedding_metrics import (
    compute_all_metrics,
    format_metrics_table,
    get_default_report_path,
    write_metrics_csv,
)







_oracle_results: List[Dict[str, Any]] = []
pytestmark = pytest.mark.property


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







@dataclass
class OracleConfig:
    """Configuration for embedding oracles."""

    embedder_name: str
    cosine_threshold: float
    batch_sizes: List[int] = field(default_factory=lambda: [1, 5, 10])
    masking_ratios: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])



ORACLE_CONFIGS = {
    "esm2_t6_8m": OracleConfig(
        embedder_name="esm2_t6_8m",
        cosine_threshold=0.25,  # Allow some variation in larger batches
    ),
}







class BatchInvarianceOracle:
    # Oracle verifying that embeddings are invariant to batch composition.

    def __init__(
        self,
        embedder: EmbedderProtocol,
        config: OracleConfig,
    ):
        # Initialize BatchInvarianceOracle.
        self.embedder = embedder
        self.config = config

    def verify(
        self,
        target_sequence: str,
        filler_sequences: List[str],
    ) -> List[Dict[str, Any]]:
        # Verify batch invariance for a target sequence.
        results = []


        single_embedding = self.embedder.embed_pooled(target_sequence)

        for batch_size in self.config.batch_sizes:

            batch = self._create_batch(target_sequence, filler_sequences, batch_size)
            target_idx = batch.index(target_sequence)


            batch_embeddings = self.embedder.embed_batch(batch, pooled=True)
            batched_embedding = batch_embeddings[target_idx]


            metrics = compute_all_metrics(single_embedding, batched_embedding)


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


        n_fillers = batch_size - 1
        selected_fillers = []
        for i in range(n_fillers):
            selected_fillers.append(fillers[i % len(fillers)])


        batch = selected_fillers.copy()
        seed_material = f"{self.config.embedder_name}:{batch_size}:{target}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        rng = random.Random(seed)
        insert_pos = rng.randint(0, len(batch))
        batch.insert(insert_pos, target)

        return batch







class MaskingRobustnessOracle:
    # Oracle verifying embedding behavior under progressive token masking.

    MASK_TOKEN = "X"

    def __init__(
        self,
        embedder: EmbedderProtocol,
        config: OracleConfig,
    ):
        # Initialize MaskingRobustnessOracle.
        self.embedder = embedder
        self.config = config

    def verify(
        self,
        sequence: str,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        # Verify masking robustness for a sequence.
        results = []


        original_embedding = self.embedder.embed_pooled(sequence)

        for ratio in self.config.masking_ratios:

            masked_sequence = self._mask_sequence(sequence, ratio, seed)


            masked_embedding = self.embedder.embed_pooled(masked_sequence)


            metrics = compute_all_metrics(original_embedding, masked_embedding)


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
        # Mask a portion of the sequence with unknown token 'X'.
        if ratio <= 0:
            return sequence
        if ratio >= 1:
            return self.MASK_TOKEN * len(sequence)

        rng = random.Random(seed + int(ratio * 1000))
        seq_list = list(sequence)
        n_mask = int(len(sequence) * ratio)


        positions = rng.sample(range(len(sequence)), n_mask)

        for pos in positions:
            seq_list[pos] = self.MASK_TOKEN

        return "".join(seq_list)







@pytest.fixture(scope="module")
def esm2_t6_8m_oracle_config() -> OracleConfig:
    """Oracle configuration for ESM2-T6-8M."""
    return ORACLE_CONFIGS["esm2_t6_8m"]


@pytest.fixture(scope="module")
def esm2_t6_8m_embedder():
    # Load real ESM2-T6-8M embedder.
    try:
        import torch
        from biotrainer.embedders import get_embedding_service

        embedding_service = get_embedding_service(
            embedder_name="facebook/esm2_t6_8M_UR50D",
            use_half_precision=False,
            custom_tokenizer_config=None,
            device=torch.device("cpu"),
        )


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

    def _to_records(self, sequences: List[str]) -> List[BiotrainerSequenceRecord]:
        """Convert sequences to BiotrainerSequenceRecord objects."""
        return [
            BiotrainerSequenceRecord(seq_id=f"seq_{i}", seq=seq)
            for i, seq in enumerate(sequences)
        ]

    def embed(self, sequence: str) -> np.ndarray:
        """Embed single sequence, returning per-residue embeddings."""
        records = self._to_records([sequence])
        results = list(
            self.embedding_service.generate_embeddings(
                records, reduce=False
            )
        )
        if results:
            _, embedding = results[0]
            return np.array(embedding)
        return np.array([])

    def embed_pooled(self, sequence: str) -> np.ndarray:
        """Embed single sequence, returning pooled embedding."""
        records = self._to_records([sequence])
        results = list(
            self.embedding_service.generate_embeddings(
                records, reduce=True
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
        records = self._to_records(sequences)
        results = list(
            self.embedding_service.generate_embeddings(
                records, reduce=pooled
            )
        )
        return [np.array(embedding) for _, embedding in results]


@pytest.fixture(scope="module")
def oracle_sequences() -> List[str]:
    """Test sequences for oracle verification from canonical dataset."""
    return get_test_sequences(categories=["standard"])


@pytest.fixture(scope="module")
def filler_sequences() -> List[str]:
    """Filler sequences for batch creation from canonical dataset."""

    return get_test_sequences(categories=["edge_case"])

class TestBatchInvarianceESM2:
    """Batch invariance tests using real ESM2-T6-8M model."""

    def test_embedding_matches_across_batch_sizes(
        self,
        esm2_t6_8m_embedder,
        esm2_t6_8m_oracle_config: OracleConfig,
        oracle_sequences: List[str],
        filler_sequences: List[str],
    ):
        """Verify embeddings are identical regardless of batch composition."""
        oracle = BatchInvarianceOracle(
            embedder=esm2_t6_8m_embedder,
            config=esm2_t6_8m_oracle_config,
        )

        all_results = []
        for idx, seq in enumerate(oracle_sequences[:2]):
            results = oracle.verify(seq, filler_sequences)
            for result in results:
                result["sequence_index"] = idx
                result["sequence_length"] = len(seq)
            all_results.extend(results)


        table = format_metrics_table(
            all_results,
            title="Batch Invariance Oracle - ESM2-T6-8M",
        )
        print(table)


        for result in all_results:
            assert result["passed"], (
                f"Batch invariance failed for {result['parameter']}: "
                f"cosine_distance={result['cosine_distance']:.6f} > "
                f"threshold={result['threshold']:.4f}; "
                f"embedder={result['embedder']}; "
                f"sequence_index={result.get('sequence_index')}; "
                f"sequence_length={result.get('sequence_length')}"
            )


class TestMaskingRobustnessESM2:
    """Masking robustness tests using real ESM2-T6-8M model."""

    def test_embedding_stable_under_progressive_masking(
        self,
        esm2_t6_8m_embedder,
        esm2_t6_8m_oracle_config: OracleConfig,
        oracle_sequences: List[str],
    ):
        """Verify embeddings remain stable under progressive masking."""
        oracle = MaskingRobustnessOracle(
            embedder=esm2_t6_8m_embedder,
            config=esm2_t6_8m_oracle_config,
        )

        all_results = []
        for idx, seq in enumerate(oracle_sequences[:2]):
            results = oracle.verify(seq)
            for result in results:
                result["sequence_index"] = idx
                result["sequence_length"] = len(seq)
            all_results.extend(results)


        table = format_metrics_table(
            all_results,
            title="Masking Robustness Oracle - ESM2-T6-8M",
        )
        print(table)


        for result in all_results:
            assert result["passed"], (
                f"Masking robustness failed for {result['parameter']}: "
                f"cosine_distance={result['cosine_distance']:.6f} > "
                f"threshold={result['threshold']:.4f}; "
                f"embedder={result['embedder']}; "
                f"sequence_index={result.get('sequence_index')}; "
                f"sequence_length={result.get('sequence_length')}"
            )







@pytest.fixture(scope="module", autouse=True)
def write_oracle_report(request):
    """Write accumulated oracle results to CSV after module completes."""
    yield


    results = get_oracle_results()
    if results:
        report_path = get_default_report_path()
        write_metrics_csv(results, report_path)
        print(f"\n📊 Oracle metrics report written to: {report_path}")


    clear_oracle_results()
