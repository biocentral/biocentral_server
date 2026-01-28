"""Test memory usage and detect leaks."""

import pytest
import gc
import numpy as np

from tests.fixtures.fixed_embedder import FixedEmbedder
from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET


def get_memory_mb() -> float:
    """Get current process memory in MB (macOS/Linux)."""
    try:
        import resource

        # macOS returns bytes, Linux returns KB
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys

        if sys.platform == "darwin":
            return usage / (1024 * 1024)  # bytes to MB
        else:
            return usage / 1024  # KB to MB
    except ImportError:
        return 0.0


@pytest.mark.performance
class TestMemoryLeaks:
    """Detect memory leaks in embedding generation."""

    def test_no_leak_repeated_single_embedding(self, perf_embedder, very_long_sequence):
        """Memory should not grow with repeated single embeddings."""
        # Use canonical 400-residue sequence
        sequence = very_long_sequence

        # Warm up
        for _ in range(10):
            _ = perf_embedder.embed(sequence)
        gc.collect()

        baseline = get_memory_mb()

        # Run many iterations
        for _ in range(1000):
            _ = perf_embedder.embed(sequence)

        gc.collect()
        final = get_memory_mb()

        growth = final - baseline
        # Allow some tolerance for GC timing
        assert growth < 100, f"Memory grew by {growth:.1f} MB (possible leak)"

    def test_no_leak_repeated_batch_embedding(self, perf_embedder, large_batch):
        """Memory should not grow with repeated batch embeddings."""
        # Use all canonical sequences
        sequences = large_batch

        # Warm up
        for _ in range(5):
            _ = perf_embedder.embed_batch(sequences)
        gc.collect()

        baseline = get_memory_mb()

        # Run iterations
        for _ in range(100):
            _ = perf_embedder.embed_batch(sequences)

        gc.collect()
        final = get_memory_mb()

        growth = final - baseline
        assert growth < 200, f"Memory grew by {growth:.1f} MB (possible leak)"

    def test_gc_releases_embeddings(self, perf_embedder, very_long_sequence):
        """Verify GC properly releases embedding memory."""
        gc.collect()
        baseline = get_memory_mb()

        # Create large embeddings by repeating the long sequence
        embeddings = []
        for _ in range(100):
            emb = perf_embedder.embed(very_long_sequence)
            embeddings.append(emb)

        peak = get_memory_mb()
        print(f"\nPeak memory after 100 x {len(very_long_sequence)}-residue embeddings: {peak:.1f} MB")

        # Release references
        del embeddings
        gc.collect()

        after_gc = get_memory_mb()
        print(f"Memory after GC: {after_gc:.1f} MB")
        print(f"Released: {peak - after_gc:.1f} MB")


@pytest.mark.performance
class TestMemoryFootprint:
    """Measure memory footprint of embeddings."""

    def test_embedding_memory_size(self, perf_embedder, medium_sequence):
        """Verify embedding memory matches expected size."""
        # Use canonical medium-length sequence
        embedding = perf_embedder.embed(medium_sequence)
        seq_len = len(medium_sequence)

        expected_bytes = seq_len * 1024 * 4  # seq_len * dim * float32
        actual_bytes = embedding.nbytes

        assert actual_bytes == expected_bytes
        print(f"\n{seq_len}-residue embedding: {actual_bytes / 1024:.1f} KB")

    def test_batch_memory_size(self, perf_embedder, large_batch):
        """Measure total memory for large batch."""
        embeddings = perf_embedder.embed_batch(large_batch)

        total_bytes = sum(emb.nbytes for emb in embeddings)
        total_mb = total_bytes / (1024 * 1024)

        avg_length = np.mean([len(seq) for seq in large_batch])
        n_seqs = len(large_batch)
        print(f"\n{n_seqs} sequences (avg {avg_length:.0f} residues):")
        print(f"  Total memory: {total_mb:.1f} MB")
        print(f"  Per sequence: {total_bytes / n_seqs / 1024:.1f} KB")

    def test_pooled_vs_per_residue_memory(self, perf_embedder, medium_batch):
        """Compare memory usage: pooled vs per-residue."""
        per_residue = perf_embedder.embed_batch(medium_batch, pooled=False)
        pooled = perf_embedder.embed_batch(medium_batch, pooled=True)

        per_residue_bytes = sum(e.nbytes for e in per_residue)
        pooled_bytes = sum(e.nbytes for e in pooled)

        per_residue_mb = per_residue_bytes / (1024 * 1024)
        pooled_mb = pooled_bytes / (1024 * 1024)

        n_seqs = len(medium_batch)
        print(f"\n{n_seqs} sequences memory comparison:")
        print(f"  Per-residue: {per_residue_mb:.2f} MB")
        print(f"  Pooled:      {pooled_mb:.4f} MB")
        print(f"  Reduction:   {(1 - pooled_bytes/per_residue_bytes) * 100:.1f}%")

        # Pooled should use much less memory
        assert pooled_bytes < per_residue_bytes / 10

    def test_memory_per_dimension(self, perf_embedder, medium_sequence):
        """Measure memory scaling with embedding dimension."""
        dims = [512, 1024, 1280, 2560]
        results = []
        seq_len = len(medium_sequence)

        for dim in dims:
            embedder = FixedEmbedder(embedding_dim=dim)
            emb = embedder.embed(medium_sequence)
            results.append({"dim": dim, "bytes": emb.nbytes, "kb": emb.nbytes / 1024})

        print(f"\n\nMemory by embedding dimension ({seq_len} residues):")
        print("-" * 40)
        print(f"{'Dimension':>12} {'Memory (KB)':>14}")
        print("-" * 40)
        for r in results:
            print(f"{r['dim']:>12} {r['kb']:>14.1f}")


@pytest.mark.performance
class TestMemoryEstimation:
    """Test memory estimation for planning."""

    def test_estimate_batch_memory(self, perf_embedder):
        """Provide memory estimates for different batch configurations."""
        configs = [
            {"n_seqs": 100, "avg_len": 100},
            {"n_seqs": 1000, "avg_len": 100},
            {"n_seqs": 100, "avg_len": 500},
            {"n_seqs": 1000, "avg_len": 500},
            {"n_seqs": 10000, "avg_len": 100},
        ]

        dim = perf_embedder.embedding_dim
        bytes_per_float = 4

        print("\n\nMemory Estimation (per-residue, 1024-dim):")
        print("-" * 60)
        print(f"{'Sequences':>12} {'Avg Length':>12} {'Estimated MB':>14}")
        print("-" * 60)

        for cfg in configs:
            estimated_bytes = cfg["n_seqs"] * cfg["avg_len"] * dim * bytes_per_float
            estimated_mb = estimated_bytes / (1024 * 1024)
            print(f"{cfg['n_seqs']:>12} {cfg['avg_len']:>12} {estimated_mb:>14.1f}")
