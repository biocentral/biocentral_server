"""Performance test fixtures."""

import pytest
import numpy as np
from typing import List


from tests.fixtures.fixed_embedder import FixedEmbedder
from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET, get_test_sequences


class ESM2EmbedderAdapter:
    """
    Adapter to provide a consistent embed() API for biotrainer's EmbeddingService.
    
    biotrainer's EmbeddingService uses generate_embeddings() which yields tuples,
    but our tests expect a simple embed(sequence) -> np.ndarray interface.
    """
    
    def __init__(self, embedding_service):
        self._service = embedding_service
    
    def embed(self, sequence: str) -> np.ndarray:
        """
        Embed a single sequence, returning per-residue embeddings.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            np.ndarray of shape (sequence_length, embedding_dim)
        """
        # generate_embeddings yields (id, embedding) tuples
        # For a single sequence, we pass it as a dict and get back one result
        results = list(self._service.generate_embeddings(
            input_data={sequence: sequence},
            reduce=False,
        ))
        if results:
            _, embedding = results[0]
            return np.array(embedding)
        raise ValueError(f"Failed to embed sequence: {sequence[:50]}...")
    
    def embed_pooled(self, sequence: str) -> np.ndarray:
        """
        Embed a single sequence, returning pooled (reduced) embedding.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            np.ndarray of shape (embedding_dim,)
        """
        results = list(self._service.generate_embeddings(
            input_data={sequence: sequence},
            reduce=True,
        ))
        if results:
            _, embedding = results[0]
            return np.array(embedding)
        raise ValueError(f"Failed to embed sequence: {sequence[:50]}...")
    
    def embed_batch(self, sequences: List[str], pooled: bool = False) -> List[np.ndarray]:
        """
        Embed multiple sequences.
        
        Args:
            sequences: List of protein sequence strings
            pooled: If True, return pooled embeddings
            
        Returns:
            List of np.ndarray embeddings
        """
        input_data = {seq: seq for seq in sequences}
        results = list(self._service.generate_embeddings(
            input_data=input_data,
            reduce=pooled,
        ))
        return [np.array(emb) for _, emb in results]


@pytest.fixture(scope="module")
def perf_embedder() -> FixedEmbedder:
    """Module-scoped embedder for performance tests."""
    return FixedEmbedder(model_name="prot_t5", seed_base=42, strict_dataset=False)


@pytest.fixture(scope="module")
def esm2_embedder():
    """
    Real ESM2-t6-8M embedder for performance benchmarking.
    
    Uses biotrainer's embedding service to load the actual model.
    Module-scoped to avoid reloading the model for each test.
    
    Returns an ESM2EmbedderAdapter that provides a consistent embed() API.
    """
    from biotrainer.embedders import get_embedding_service
    
    service = get_embedding_service(
        embedder_name="facebook/esm2_t6_8M_UR50D",
        use_half_precision=False,
        custom_tokenizer_config=None,
        device="cpu",
    )
    return ESM2EmbedderAdapter(service)


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
