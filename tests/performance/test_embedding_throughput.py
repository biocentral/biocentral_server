"""
Benchmark embedding generation throughput.

Tests measure:
- Single sequence embedding latency at various lengths
- Batch embedding throughput at various batch sizes
- Pooled vs per-residue embedding performance

All tests use sequences from the canonical test dataset.
"""

import pytest
import numpy as np

from tests.fixtures.fixed_embedder import FixedEmbedder


@pytest.mark.performance
class TestSingleSequenceLatency:
    """Measure single sequence embedding latency."""

    def test_short_sequence_latency(self, perf_embedder, short_sequence, benchmark):
        """Benchmark: short sequence (10 aa) from canonical dataset."""
        result = benchmark(perf_embedder.embed, short_sequence)
        assert result.shape == (len(short_sequence), 1024)

    def test_medium_sequence_latency(self, perf_embedder, medium_sequence, benchmark):
        """Benchmark: medium sequence (~79 aa) from canonical dataset."""
        result = benchmark(perf_embedder.embed, medium_sequence)
        assert result.shape == (len(medium_sequence), 1024)

    def test_long_sequence_latency(self, perf_embedder, long_sequence, benchmark):
        """Benchmark: long sequence (~211 aa) from canonical dataset."""
        result = benchmark(perf_embedder.embed, long_sequence)
        assert result.shape == (len(long_sequence), 1024)

    def test_very_long_sequence_latency(self, perf_embedder, very_long_sequence, benchmark):
        """Benchmark: very long sequence (400 aa) from canonical dataset."""
        result = benchmark(perf_embedder.embed, very_long_sequence)
        assert result.shape == (len(very_long_sequence), 1024)


@pytest.mark.performance
class TestBatchThroughput:
    """Measure batch embedding throughput."""

    def test_small_batch_throughput(self, perf_embedder, small_batch, benchmark):
        """Benchmark: small batch (5 sequences) from canonical dataset."""
        result = benchmark(perf_embedder.embed_batch, small_batch)
        assert len(result) == len(small_batch)

    def test_medium_batch_throughput(self, perf_embedder, medium_batch, benchmark):
        """Benchmark: medium batch (15 sequences) from canonical dataset."""
        result = benchmark(perf_embedder.embed_batch, medium_batch)
        assert len(result) == len(medium_batch)

    def test_large_batch_throughput(self, perf_embedder, large_batch, benchmark):
        """Benchmark: large batch (all sequences) from canonical dataset."""
        result = benchmark(perf_embedder.embed_batch, large_batch)
        assert len(result) == len(large_batch)


@pytest.mark.performance
class TestPooledEmbeddingThroughput:
    """Measure pooled embedding throughput."""

    def test_pooled_single_sequence(self, perf_embedder, medium_sequence, benchmark):
        """Benchmark: single pooled embedding from canonical dataset."""
        result = benchmark(perf_embedder.embed_pooled, medium_sequence)
        assert result.shape == (1024,)

    def test_pooled_batch(self, perf_embedder, medium_batch, benchmark):
        """Benchmark: batch pooled embeddings from canonical dataset."""
        result = benchmark(perf_embedder.embed_batch, medium_batch, True)
        assert len(result) == len(medium_batch)
        assert all(emb.shape == (1024,) for emb in result)


@pytest.mark.performance
class TestDictEmbeddingThroughput:
    """Measure dictionary-format embedding throughput."""

    def test_dict_embedding_throughput(self, perf_embedder, medium_batch, benchmark):
        """Benchmark: dict format embeddings from canonical dataset."""
        sequences_dict = {f"seq{i}": seq for i, seq in enumerate(medium_batch)}

        result = benchmark(perf_embedder.embed_dict, sequences_dict)

        assert len(result) == len(medium_batch)
        assert all(isinstance(k, str) for k in result.keys())
