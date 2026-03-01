import pytest
import numpy as np
from pathlib import Path
from typing import List

from biotrainer.input_files import BiotrainerSequenceRecord
from tests.fixtures.test_dataset import get_test_sequences



REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


@pytest.fixture(scope="session")
def reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


@pytest.fixture(scope="session")
def esm2_embedder():
    try:
        import torch
        from biotrainer.embedders import get_embedding_service

        svc = get_embedding_service(
            embedder_name="facebook/esm2_t6_8M_UR50D",
            use_half_precision=False,
            custom_tokenizer_config=None,
            device=torch.device("cpu"),
        )
        return _ESM2Wrapper(svc)
    except Exception as exc:
        pytest.skip(f"ESM2-T6-8M unavailable: {exc}")


class _ESM2Wrapper:

    def __init__(self, service):
        self._svc = service

    def _to_records(self, sequences: List[str]) -> List[BiotrainerSequenceRecord]:
        return [
            BiotrainerSequenceRecord(seq_id=f"seq_{i}", seq=seq)
            for i, seq in enumerate(sequences)
        ]

    def embed(self, sequence: str) -> np.ndarray:
        records = self._to_records([sequence])
        results = list(self._svc.generate_embeddings(records, reduce=False))
        return np.array(results[0][1]) if results else np.array([])

    def embed_pooled(self, sequence: str) -> np.ndarray:
        records = self._to_records([sequence])
        results = list(self._svc.generate_embeddings(records, reduce=True))
        return np.array(results[0][1]) if results else np.array([])

    def embed_batch(
        self, sequences: List[str], pooled: bool = False
    ) -> List[np.ndarray]:
        records = self._to_records(sequences)
        results = list(self._svc.generate_embeddings(records, reduce=pooled))
        return [np.array(emb) for _, emb in results]


@pytest.fixture(scope="session")
def standard_sequences() -> List[str]:
    return get_test_sequences(categories=["standard"])


@pytest.fixture(scope="session")
def diverse_sequences() -> List[str]:
    seqs = get_test_sequences(categories=["standard", "real"])
    seen = set()
    unique = []
    for s in seqs:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


@pytest.fixture(scope="session")
def filler_sequences() -> List[str]:
    return get_test_sequences(categories=["edge_case"])
