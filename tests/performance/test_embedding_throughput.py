"""
Benchmark embedding generation throughput.

Tests measure:
- Single sequence embedding latency at various lengths
- Batch embedding throughput at various batch sizes
- Pooled vs per-residue embedding performance
"""

import pytest
import numpy as np

from tests.fixtures.fixed_embedder import FixedEmbedder


@pytest.mark.performance
class TestSingleSequenceLatency:
    """Measure single sequence embedding latency."""

    def test_short_sequence_latency(self, perf_embedder, benchmark):
        """Benchmark: 10 residue sequence."""
        sequence = "MKTAYIAKQR"
        result = benchmark(perf_embedder.embed, sequence)
        assert result.shape == (10, 1024)

    def test_medium_sequence_latency(self, perf_embedder, benchmark):
        """Benchmark: 100 residue sequence."""
        sequence = "MKTAYIAK" * 12 + "MKTM"
        result = benchmark(perf_embedder.embed, sequence)
        assert result.shape == (100, 1024)

    def test_long_sequence_latency(self, perf_embedder, benchmark):
        """Benchmark: 500 residue sequence."""
        sequence = "MKTAYIAK" * 62 + "MKTM"
        result = benchmark(perf_embedder.embed, sequence)
        assert result.shape == (500, 1024)

    def test_very_long_sequence_latency(self, perf_embedder, benchmark):
        """Benchmark: 1000 residue sequence."""
        sequence = "MKTAYIAK" * 125
        result = benchmark(perf_embedder.embed, sequence)
        assert result.shape == (1000, 1024)


@pytest.mark.performance
class TestBatchThroughput:
    """Measure batch embedding throughput."""

    def test_small_batch_throughput(self, perf_embedder, small_batch, benchmark):
        """Benchmark: 10 sequences."""
        result = benchmark(perf_embedder.embed_batch, small_batch)
        assert len(result) == 10

    def test_medium_batch_throughput(self, perf_embedder, medium_batch, benchmark):
        """Benchmark: 100 sequences."""
        result = benchmark(perf_embedder.embed_batch, medium_batch)
        assert len(result) == 100

    def test_large_batch_throughput(self, perf_embedder, large_batch, benchmark):
        """Benchmark: 1000 sequences."""
        result = benchmark(perf_embedder.embed_batch, large_batch)
        assert len(result) == 1000


@pytest.mark.performance
class TestPooledEmbeddingThroughput:
    """Measure pooled embedding throughput."""

    def test_pooled_single_sequence(self, perf_embedder, benchmark):
        """Benchmark: single pooled embedding."""
        sequence = "MKTAYIAK" * 12
        result = benchmark(perf_embedder.embed_pooled, sequence)
        assert result.shape == (1024,)

    def test_pooled_batch(self, perf_embedder, medium_batch, benchmark):
        """Benchmark: batch pooled embeddings."""
        result = benchmark(perf_embedder.embed_batch, medium_batch, True)
        assert len(result) == 100
        assert all(emb.shape == (1024,) for emb in result)


@pytest.mark.performance
class TestDictEmbeddingThroughput:
    """Measure dictionary-format embedding throughput."""

    def test_dict_embedding_throughput(self, perf_embedder, medium_batch, benchmark):
        """Benchmark: dict format embeddings."""
        sequences_dict = {f"seq{i}": seq for i, seq in enumerate(medium_batch)}

        result = benchmark(perf_embedder.embed_dict, sequences_dict)

        assert len(result) == 100
        assert all(isinstance(k, str) for k in result.keys())
