import pytest
import gc
import numpy as np

from tests.fixtures.fixed_embedder import FixedEmbedder


def get_memory_mb() -> float:
    try:
        import resource


        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys

        if sys.platform == "darwin":
            return usage / (1024 * 1024)
        else:
            return usage / 1024
    except ImportError:
        return 0.0

@pytest.mark.slow
@pytest.mark.performance
class TestMemoryLeaks:
    def test_no_leak_repeated_single_embedding(self, esm2_embedder, very_long_sequence):
        # Memory should not grow with repeated single embeddings 

        sequence = very_long_sequence

        for _ in range(3):
            _ = esm2_embedder.embed(sequence)
        gc.collect()

        baseline = get_memory_mb()


        for _ in range(20):
            _ = esm2_embedder.embed(sequence)

        gc.collect()
        final = get_memory_mb()

        growth = final - baseline

        assert growth < 7, f"Memory grew by {growth:.1f} MB (possible leak)"

    def test_no_leak_repeated_batch_embedding(self, esm2_embedder, large_batch):
        # Memory should not grow with repeated batch embeddings.

        sequences = large_batch


        for _ in range(3):
            _ = esm2_embedder.embed_batch(sequences)
        gc.collect()

        baseline = get_memory_mb()


        for _ in range(10):
            _ = esm2_embedder.embed_batch(sequences)

        gc.collect()
        final = get_memory_mb()

        growth = final - baseline
        assert growth < 4, f"Memory grew by {growth:.1f} MB (possible leak)"

    def test_gc_releases_embeddings(self, esm2_embedder, very_long_sequence):
        # Verify GC properly releases embedding memory.
        gc.collect()
        baseline = get_memory_mb()


        embeddings = []
        for _ in range(10):
            emb = esm2_embedder.embed(very_long_sequence)
            embeddings.append(emb)

        peak = get_memory_mb()
        print(f"\nPeak memory after 10 x {len(very_long_sequence)}-residue embeddings: {peak:.1f} MB")


        del embeddings
        gc.collect()

        after_gc = get_memory_mb()
        print(f"Memory baseline: {baseline:.1f} MB")
        print(f"Memory after GC: {after_gc:.1f} MB")
        print(f"Released: {peak - after_gc:.1f} MB")

@pytest.mark.slow
@pytest.mark.performance
class TestMemoryFootprint:
    # Measure memory footprint of embeddings.

    def test_embedding_memory_size(self, esm2_embedder, medium_sequence):
        # Verify embedding memory matches expected size.

        embedding = esm2_embedder.embed(medium_sequence)
        seq_len = len(medium_sequence)
        embedding_dim = embedding.shape[-1]  # Get actual dimension from embedding

        expected_bytes = seq_len * embedding_dim * 4
        actual_bytes = embedding.nbytes

        assert actual_bytes == expected_bytes
        print(f"\n{seq_len}-residue embedding ({embedding_dim}-dim): {actual_bytes / 1024:.1f} KB")

    def test_batch_memory_size(self, esm2_embedder, large_batch):
        # Measure total memory for large batch.
        embeddings = esm2_embedder.embed_batch(large_batch)

        total_bytes = sum(emb.nbytes for emb in embeddings)
        total_mb = total_bytes / (1024 * 1024)

        avg_length = np.mean([len(seq) for seq in large_batch])
        n_seqs = len(large_batch)
        print(f"\n{n_seqs} sequences (avg {avg_length:.0f} residues):")
        print(f"  Total memory: {total_mb:.1f} MB")
        print(f"  Per sequence: {total_bytes / n_seqs / 1024:.1f} KB")

    def test_pooled_vs_per_residue_memory(self, esm2_embedder, medium_batch):
        # Compare memory usage: pooled vs per-residue
        per_residue = esm2_embedder.embed_batch(medium_batch, pooled=False)
        pooled = esm2_embedder.embed_batch(medium_batch, pooled=True)

        per_residue_bytes = sum(e.nbytes for e in per_residue)
        pooled_bytes = sum(e.nbytes for e in pooled)

        per_residue_mb = per_residue_bytes / (1024 * 1024)
        pooled_mb = pooled_bytes / (1024 * 1024)

        n_seqs = len(medium_batch)
        print(f"\n{n_seqs} sequences memory comparison:")
        print(f"  Per-residue: {per_residue_mb:.2f} MB")
        print(f"  Pooled:      {pooled_mb:.4f} MB")
        print(f"  Reduction:   {(1 - pooled_bytes/per_residue_bytes) * 100:.1f}%")


        assert pooled_bytes < per_residue_bytes / 10

    def test_memory_per_dimension(self, esm2_embedder, medium_sequence):
        #Measure memory scaling with embedding dimension.
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

@pytest.mark.slow
@pytest.mark.performance
class TestMemoryEstimation:
    # Test memory estimation for planning.

    def test_estimate_batch_memory(self, esm2_embedder):
        # Provide memory estimates for different batch configurations.
        configs = [
            {"n_seqs": 100, "avg_len": 100},
            {"n_seqs": 1000, "avg_len": 100},
            {"n_seqs": 100, "avg_len": 500},
            {"n_seqs": 1000, "avg_len": 500},
            {"n_seqs": 10000, "avg_len": 100},
        ]

        # Get dimension from a sample embedding
        sample_emb = esm2_embedder.embed("M")
        dim = sample_emb.shape[-1]
        bytes_per_float = 4

        print(f"\n\nMemory Estimation (per-residue, {dim}-dim):")
        print("-" * 60)
        print(f"{'Sequences':>12} {'Avg Length':>12} {'Estimated MB':>14}")
        print("-" * 60)

        for cfg in configs:
            estimated_bytes = cfg["n_seqs"] * cfg["avg_len"] * dim * bytes_per_float
            estimated_mb = estimated_bytes / (1024 * 1024)
            print(f"{cfg['n_seqs']:>12} {cfg['avg_len']:>12} {estimated_mb:>14.1f}")
