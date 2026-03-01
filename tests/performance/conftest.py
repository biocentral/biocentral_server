import pytest
import numpy as np
from typing import List

from biotrainer.input_files import BiotrainerSequenceRecord
from tests.fixtures.fixed_embedder import FixedEmbedder
from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET, get_test_sequences


class ESM2EmbedderAdapter:
    # Adapter to provide a consistent embed() API for biotrainer's EmbeddingService.

    def __init__(self, embedding_service):
        self._service = embedding_service

    def _to_records(self, sequences: List[str]) -> List[BiotrainerSequenceRecord]:
        """Convert sequences to BiotrainerSequenceRecord objects."""
        return [
            BiotrainerSequenceRecord(seq_id=f"seq_{i}", seq=seq)
            for i, seq in enumerate(sequences)
        ]

    def embed(self, sequence: str) -> np.ndarray:
        # Embed a single sequence, returning per-residue embeddings.
        records = self._to_records([sequence])
        results = list(
            self._service.generate_embeddings(
                records,
                reduce=False,
            )
        )
        if results:
            _, embedding = results[0]
            return np.array(embedding)
        raise ValueError(f"Failed to embed sequence: {sequence[:50]}...")

    def embed_pooled(self, sequence: str) -> np.ndarray:
        # Embed a single sequence, returning pooled (reduced) embedding.
        records = self._to_records([sequence])
        results = list(
            self._service.generate_embeddings(
                records,
                reduce=True,
            )
        )
        if results:
            _, embedding = results[0]
            return np.array(embedding)
        raise ValueError(f"Failed to embed sequence: {sequence[:50]}...")

    def embed_batch(
        self, sequences: List[str], pooled: bool = False
    ) -> List[np.ndarray]:
        # Embed multiple sequences.
        records = self._to_records(sequences)
        results = list(
            self._service.generate_embeddings(
                records,
                reduce=pooled,
            )
        )
        return [np.array(emb) for _, emb in results]


@pytest.fixture(scope="module")
def perf_embedder() -> FixedEmbedder:
    """Module-scoped embedder for performance tests."""
    return FixedEmbedder(model_name="prot_t5", seed_base=42, strict_dataset=False)


@pytest.fixture(scope="module")
def esm2_embedder():
    # Real ESM2-t6-8M embedder for performance benchmarking.
    import torch
    from biotrainer.embedders import get_embedding_service

    service = get_embedding_service(
        embedder_name="facebook/esm2_t6_8M_UR50D",
        use_half_precision=False,
        custom_tokenizer_config=None,
        device=torch.device("cpu"),
    )
    return ESM2EmbedderAdapter(service)


@pytest.fixture
def canonical_sequences() -> List[str]:
    return get_test_sequences()


@pytest.fixture
def short_sequence() -> str:
    return CANONICAL_TEST_DATASET.get_by_id("length_short_10").sequence


@pytest.fixture
def medium_sequence() -> str:
    # 79 aa
    return CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence


@pytest.fixture
def long_sequence() -> str:
    # 211 aa
    return CANONICAL_TEST_DATASET.get_by_id("length_long_200").sequence


@pytest.fixture
def very_long_sequence() -> str:
    # 400 aa
    return CANONICAL_TEST_DATASET.get_by_id("length_very_long_400").sequence


@pytest.fixture
def small_batch() -> List[str]:
    # 5 sequences
    return get_test_sequences()[:5]


@pytest.fixture
def medium_batch() -> List[str]:
    # 15 sequences
    return get_test_sequences()[:15]


@pytest.fixture
def large_batch() -> List[str]:
    # All sequences from the dataset
    return get_test_sequences()


@pytest.fixture
def variable_length_sequences() -> List[str]:
    # Sequences with varying lengths from the dataset
    ids = [
        "length_min_1",
        "length_short_5",
        "length_short_10",
        "length_medium_50",
        "standard_001",
        "length_long_200",
        "length_very_long_400",
    ]
    return [CANONICAL_TEST_DATASET.get_by_id(id).sequence for id in ids]
