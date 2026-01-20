"""
Test scaling behavior with sequence length and batch size.

Verifies that embedding generation scales linearly O(n) with:
- Sequence length
- Batch size

All tests use sequences from the canonical test dataset.
"""

import pytest
import time
import numpy as np

from tests.fixtures.fixed_embedder import FixedEmbedder
from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET


@pytest.mark.performance
class TestSequenceLengthScaling:
    """Verify embedding time scales linearly with sequence length."""

    def test_linear_scaling_with_length(self, perf_embedder, variable_length_sequences):
        """Time should scale O(n) with sequence length."""
        times = []
        lengths = []
        iterations = 5

        for sequence in variable_length_sequences:
            lengths.append(len(sequence))
            start = time.perf_counter()
            for _ in range(iterations):
                perf_embedder.embed(sequence)
            elapsed = time.perf_counter() - start
            times.append(elapsed / iterations)

        # Check scaling factor between consecutive lengths
        for i in range(1, len(lengths)):
            length_ratio = lengths[i] / lengths[i - 1]
            time_ratio = times[i] / times[i - 1]

            # Allow up to 3x deviation from linear
            assert time_ratio < length_ratio * 3, (
                f"Non-linear scaling: length {lengths[i-1]}->{lengths[i]}, "
                f"expected ~{length_ratio:.1f}x, got {time_ratio:.1f}x"
            )

    def test_collect_scaling_data(self, perf_embedder, variable_length_sequences):
        """Collect scaling data for analysis."""
        results = []

        for sequence in variable_length_sequences:
            times = []
            for _ in range(10):
                start = time.perf_counter()
                perf_embedder.embed(sequence)
                times.append(time.perf_counter() - start)

            results.append(
                {
                    "length": len(sequence),
                    "mean_ms": np.mean(times) * 1000,
                    "std_ms": np.std(times) * 1000,
                    "min_ms": np.min(times) * 1000,
                    "max_ms": np.max(times) * 1000,
                }
            )

        # Print results table
        print("\n\nSequence Length Scaling:")
        print("-" * 60)
        print(f"{'Length':>8} {'Mean (ms)':>12} {'Std (ms)':>12} {'Min-Max (ms)':>16}")
        print("-" * 60)
        for r in results:
            print(
                f"{r['length']:>8} {r['mean_ms']:>12.3f} {r['std_ms']:>12.3f} "
                f"{r['min_ms']:>7.3f}-{r['max_ms']:.3f}"
            )


@pytest.mark.performance
class TestBatchSizeScaling:
    """Verify embedding time scales linearly with batch size."""

    def test_linear_scaling_with_batch_size(self, perf_embedder, canonical_sequences):
        """Time should scale O(n) with batch size."""
        # Use subsets of canonical sequences
        batch_sizes = [2, 5, 10, 15, len(canonical_sequences)]
        times = []

        for size in batch_sizes:
            sequences = canonical_sequences[:size]

            start = time.perf_counter()
            perf_embedder.embed_batch(sequences)
            elapsed = time.perf_counter() - start

            times.append(elapsed)

        # Check scaling
        for i in range(1, len(batch_sizes)):
            size_ratio = batch_sizes[i] / batch_sizes[i - 1]
            time_ratio = times[i] / times[i - 1]

            assert time_ratio < size_ratio * 3, (
                f"Non-linear scaling: batch {batch_sizes[i-1]}->{batch_sizes[i]}, "
                f"expected ~{size_ratio:.1f}x, got {time_ratio:.1f}x"
            )

    def test_collect_batch_scaling_data(self, perf_embedder, canonical_sequences):
        """Collect batch scaling data for analysis."""
        # Use increasing subsets of canonical sequences
        total = len(canonical_sequences)
        batch_sizes = [1, 5, 10, 15, total]
        results = []

        for size in batch_sizes:
            sequences = canonical_sequences[:size]

            times = []
            for _ in range(5):
                start = time.perf_counter()
                perf_embedder.embed_batch(sequences)
                times.append(time.perf_counter() - start)

            results.append(
                {
                    "batch_size": size,
                    "mean_ms": np.mean(times) * 1000,
                    "per_seq_ms": np.mean(times) * 1000 / size,
                }
            )

        # Print results
        print("\n\nBatch Size Scaling:")
        print("-" * 50)
        print(f"{'Batch Size':>12} {'Total (ms)':>12} {'Per Seq (ms)':>14}")
        print("-" * 50)
        for r in results:
            print(
                f"{r['batch_size']:>12} {r['mean_ms']:>12.3f} {r['per_seq_ms']:>14.4f}"
            )


@pytest.mark.performance
class TestSequentialVsBatch:
    """Compare sequential vs batch embedding performance."""

    def test_batch_vs_sequential(self, perf_embedder, medium_batch):
        """Compare batch vs sequential performance."""
        # Sequential
        start = time.perf_counter()
        sequential_results = [perf_embedder.embed(seq) for seq in medium_batch]
        sequential_time = time.perf_counter() - start

        # Batch
        start = time.perf_counter()
        batch_results = perf_embedder.embed_batch(medium_batch)
        batch_time = time.perf_counter() - start

        # Results should be identical
        for seq_result, batch_result in zip(sequential_results, batch_results):
            np.testing.assert_array_equal(seq_result, batch_result)

        print(f"\n\nSequential vs Batch ({len(medium_batch)} sequences):")
        print(f"  Sequential: {sequential_time*1000:.2f} ms")
        print(f"  Batch:      {batch_time*1000:.2f} ms")
        print(f"  Ratio:      {sequential_time/batch_time:.2f}x")
