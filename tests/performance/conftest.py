"""Performance test fixtures."""

import pytest
from typing import List

from tests.fixtures.fixed_embedder import FixedEmbedder


@pytest.fixture(scope="module")
def perf_embedder() -> FixedEmbedder:
    """Module-scoped embedder for performance tests."""
    return FixedEmbedder(model_name="prot_t5", seed_base=42)


@pytest.fixture
def small_batch() -> List[str]:
    """10 sequences, ~100 residues each."""
    return ["MKTAYIAK" * 12 + "M" * i for i in range(10)]


@pytest.fixture
def medium_batch() -> List[str]:
    """100 sequences, ~100 residues each."""
    return ["MKTAYIAK" * 12 + "M" * (i % 20) for i in range(100)]


@pytest.fixture
def large_batch() -> List[str]:
    """1000 sequences, ~100 residues each."""
    return ["MKTAYIAK" * 12 + "M" * (i % 20) for i in range(1000)]


@pytest.fixture
def variable_length_sequences() -> List[str]:
    """Sequences with varying lengths for scaling tests."""
    return [
        "M" * 10,
        "M" * 50,
        "M" * 100,
        "M" * 200,
        "M" * 500,
    ]
