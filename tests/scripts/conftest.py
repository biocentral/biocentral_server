# Shared fixtures for invariant / metamorphic-relation experiment scripts.

import pytest
import numpy as np
from pathlib import Path
from typing import List

from tests.fixtures.fixed_embedder import FixedEmbedder
from tests.fixtures.test_dataset import get_test_sequences


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test to only run with --run-slow")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow", default=False):
        skip_slow = pytest.mark.skip(reason="needs --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Report output directory
# ---------------------------------------------------------------------------

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


@pytest.fixture(scope="session")
def reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixed_embedder() -> FixedEmbedder:
    # Lightweight deterministic embedder (no GPU).
    return FixedEmbedder(model_name="esm2_t6", seed_base=42, strict_dataset=False)


@pytest.fixture(scope="session")
def esm2_embedder():
    # Real ESM2-T6-8M embedder. Skipped when model unavailable.
    try:
        from biotrainer.embedders import get_embedding_service

        svc = get_embedding_service(
            embedder_name="facebook/esm2_t6_8M_UR50D",
            use_half_precision=False,
            custom_tokenizer_config=None,
            device="cpu",
        )
        return _ESM2Wrapper(svc)
    except Exception as exc:
        pytest.skip(f"ESM2-T6-8M unavailable: {exc}")


class _ESM2Wrapper:
    # Adapt biotrainer EmbeddingService to the same interface as FixedEmbedder.

    def __init__(self, service):
        self._svc = service

    def embed(self, sequence: str) -> np.ndarray:
        results = list(self._svc.generate_embeddings(input_data=[sequence], reduce=False))
        return np.array(results[0][1]) if results else np.array([])

    def embed_pooled(self, sequence: str) -> np.ndarray:
        results = list(self._svc.generate_embeddings(input_data=[sequence], reduce=True))
        return np.array(results[0][1]) if results else np.array([])

    def embed_batch(self, sequences: List[str], pooled: bool = False) -> List[np.ndarray]:
        results = list(self._svc.generate_embeddings(input_data=sequences, reduce=pooled))
        return [np.array(emb) for _, emb in results]


# ---------------------------------------------------------------------------
# Canonical sequences
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def standard_sequences() -> List[str]:
    # Three 'standard' protein sequences from the canonical test dataset.
    return get_test_sequences(categories=["standard"])


@pytest.fixture(scope="session")
def diverse_sequences() -> List[str]:
    # A broader selection covering standard + real-world sequences.
    seqs = get_test_sequences(categories=["standard", "real"])
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in seqs:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


@pytest.fixture(scope="session")
def filler_sequences() -> List[str]:
    # Edge-case sequences used as batch fillers.
    return get_test_sequences(categories=["edge_case"])
