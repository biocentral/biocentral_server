"""Performance test fixtures."""

import pytest
from typing import List

from tests.fixtures.fixed_embedder import FixedEmbedder
from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET, get_test_sequences


@pytest.fixture(scope="module")
def perf_embedder() -> FixedEmbedder:
    """Module-scoped embedder for performance tests."""
    return FixedEmbedder(model_name="prot_t5", seed_base=42)


@pytest.fixture(scope="module")
def esm2_embedder():
    """
    Real ESM2-t6-8M embedder for performance benchmarking.
    
    Uses biotrainer's embedding service to load the actual model.
    Module-scoped to avoid reloading the model for each test.
    """
    from biotrainer.embedders import get_embedding_service
    
    embedder = get_embedding_service(
        embedder_name="facebook/esm2_t6_8M_UR50D",
        use_half_precision=False,
        custom_tokenizer_config=None,
        device="cpu",
    )
    return embedder


@pytest.fixture
def canonical_sequences() -> List[str]:
    """All sequences from canonical test dataset."""
    return get_test_sequences()


@pytest.fixture
def short_sequence() -> str:
    """Short sequence (10 aa) from canonical dataset."""
    return CANONICAL_TEST_DATASET.get_by_id("length_short_10").sequence


@pytest.fixture
def medium_sequence() -> str:
    """Medium sequence (~79 aa) from canonical dataset."""
    return CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence


@pytest.fixture
def long_sequence() -> str:
    """Long sequence (~211 aa) from canonical dataset."""
    return CANONICAL_TEST_DATASET.get_by_id("length_long_200").sequence


@pytest.fixture
def very_long_sequence() -> str:
    """Very long sequence (400 aa) from canonical dataset."""
    return CANONICAL_TEST_DATASET.get_by_id("length_very_long_400").sequence


@pytest.fixture
def small_batch() -> List[str]:
    """Small batch: first 5 sequences from canonical dataset."""
    return get_test_sequences()[:5]


@pytest.fixture
def medium_batch() -> List[str]:
    """Medium batch: first 15 sequences from canonical dataset."""
    return get_test_sequences()[:15]


@pytest.fixture
def large_batch() -> List[str]:
    """Large batch: all sequences from canonical dataset."""
    return get_test_sequences()


@pytest.fixture
def variable_length_sequences() -> List[str]:
    """Sequences with varying lengths from canonical dataset."""
    ids = [
        "length_min_1",      # 1 aa
        "length_short_5",    # 5 aa
        "length_short_10",   # 10 aa
        "length_medium_50",  # 49 aa
        "standard_001",      # 79 aa
        "length_long_200",   # 211 aa
        "length_very_long_400",  # 400 aa
    ]
    return [CANONICAL_TEST_DATASET.get_by_id(id).sequence for id in ids]
