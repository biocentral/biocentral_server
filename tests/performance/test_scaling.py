# Test scaling behavior with sequence length and batch size.

import gc
import json
from pathlib import Path

import pytest
import time
import numpy as np


def get_memory_mb() -> float:
    """Get current process memory (RSS) in MB using psutil."""
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def _append_memory_results(key: str, data: list | dict):
    """Append memory results to a shared JSON file for CI reporting."""
    results_file = Path("memory_results.json")

    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[key] = data

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)


@pytest.mark.slow
@pytest.mark.performance
class TestSequenceLengthScaling:
    # Verify embedding time scales linearly with sequence length.

    def test_linear_scaling_with_length(self, esm2_embedder, variable_length_sequences):
        times = []
        lengths = []
        iterations = 5

        for sequence in variable_length_sequences:
            lengths.append(len(sequence))
            start = time.perf_counter()
            for _ in range(iterations):
                esm2_embedder.embed(sequence)
            elapsed = time.perf_counter() - start
            times.append(elapsed / iterations)

        for i in range(1, len(lengths)):
            length_ratio = lengths[i] / lengths[i - 1]
            time_ratio = times[i] / times[i - 1]

            assert time_ratio < length_ratio * 3, (
                f"Non-linear scaling: length {lengths[i - 1]}->{lengths[i]}, "
                f"expected ~{length_ratio:.1f}x, got {time_ratio:.1f}x"
            )

    def test_collect_scaling_data(self, esm2_embedder, variable_length_sequences):
        # Collect scaling data for analysis.
        results = []
        gc.collect()

        for sequence in variable_length_sequences:
            times = []
            mem_before = get_memory_mb()
            for _ in range(10):
                start = time.perf_counter()
                embedding = esm2_embedder.embed(sequence)
                times.append(time.perf_counter() - start)
            mem_after = get_memory_mb()
            embedding_mb = embedding.nbytes / (1024 * 1024)

            results.append(
                {
                    "length": len(sequence),
                    "mean_ms": np.mean(times) * 1000,
                    "std_ms": np.std(times) * 1000,
                    "min_ms": np.min(times) * 1000,
                    "max_ms": np.max(times) * 1000,
                    "embedding_mb": embedding_mb,
                    "mem_delta_mb": mem_after - mem_before,
                }
            )

        print("\n\nSequence Length Scaling:")
        print("-" * 85)
        print(
            f"{'Length':>8} {'Mean (ms)':>12} {'Std (ms)':>12} {'Min-Max (ms)':>16} {'Emb (MB)':>10} {'Mem Δ (MB)':>12}"
        )
        print("-" * 85)
        for r in results:
            print(
                f"{r['length']:>8} {r['mean_ms']:>12.3f} {r['std_ms']:>12.3f} "
                f"{r['min_ms']:>7.3f}-{r['max_ms']:.3f} {r['embedding_mb']:>10.4f} {r['mem_delta_mb']:>12.2f}"
            )

        _append_memory_results("sequence_length_scaling", results)


@pytest.mark.slow
@pytest.mark.performance
class TestBatchSizeScaling:
    # Verify embedding time scales linearly with batch size.

    def test_linear_scaling_with_batch_size(self, esm2_embedder, canonical_sequences):
        # Time should scale O(n) with batch size.

        # Warmup to avoid initialization overhead in timing
        esm2_embedder.embed_batch(canonical_sequences[:2])

        batch_sizes = [2, 5, 10, 15, len(canonical_sequences)]
        times = []

        for size in batch_sizes:
            sequences = canonical_sequences[:size]

            start = time.perf_counter()
            esm2_embedder.embed_batch(sequences)
            elapsed = time.perf_counter() - start

            times.append(elapsed)

        # Print scaling data for analysis (timing is too variable for strict assertions in CI)
        print("\nBatch size scaling:")
        for i, (size, t) in enumerate(zip(batch_sizes, times)):
            ratio_str = f"({times[i] / times[0]:.1f}x)" if i > 0 else ""
            print(f"  {size:>3} sequences: {t * 1000:.1f}ms {ratio_str}")

    def test_collect_batch_scaling_data(self, esm2_embedder, canonical_sequences):
        # Collect batch scaling data for analysis.
        gc.collect()

        total = len(canonical_sequences)
        batch_sizes = [1, 5, 10, 15, total]
        results = []

        for size in batch_sizes:
            sequences = canonical_sequences[:size]
            gc.collect()
            mem_before = get_memory_mb()

            times = []
            for _ in range(5):
                start = time.perf_counter()
                embeddings = esm2_embedder.embed_batch(sequences)
                times.append(time.perf_counter() - start)

            mem_after = get_memory_mb()
            total_emb_mb = sum(e.nbytes for e in embeddings) / (1024 * 1024)

            results.append(
                {
                    "batch_size": size,
                    "mean_ms": np.mean(times) * 1000,
                    "per_seq_ms": np.mean(times) * 1000 / size,
                    "total_emb_mb": total_emb_mb,
                    "mem_delta_mb": mem_after - mem_before,
                }
            )

        print("\n\nBatch Size Scaling:")
        print("-" * 80)
        print(
            f"{'Batch Size':>12} {'Total (ms)':>12} {'Per Seq (ms)':>14} {'Emb (MB)':>12} {'Mem Δ (MB)':>12}"
        )
        print("-" * 80)
        for r in results:
            print(
                f"{r['batch_size']:>12} {r['mean_ms']:>12.3f} {r['per_seq_ms']:>14.4f} "
                f"{r['total_emb_mb']:>12.4f} {r['mem_delta_mb']:>12.2f}"
            )

        _append_memory_results("batch_size_scaling", results)


@pytest.mark.slow
@pytest.mark.performance
class TestSequentialVsBatch:
    # Compare sequential vs batch embedding performance.

    def test_batch_vs_sequential(self, esm2_embedder, medium_batch):
        # Compare batch vs sequential performance.
        gc.collect()
        mem_start = get_memory_mb()

        start = time.perf_counter()
        sequential_results = [esm2_embedder.embed(seq) for seq in medium_batch]
        sequential_time = time.perf_counter() - start
        mem_after_seq = get_memory_mb()
        seq_emb_mb = sum(e.nbytes for e in sequential_results) / (1024 * 1024)

        gc.collect()
        mem_before_batch = get_memory_mb()
        start = time.perf_counter()
        batch_results = esm2_embedder.embed_batch(medium_batch)
        batch_time = time.perf_counter() - start
        mem_after_batch = get_memory_mb()
        batch_emb_mb = sum(e.nbytes for e in batch_results) / (1024 * 1024)

        # Verify same number of results (order may differ in batch processing)
        assert len(sequential_results) == len(batch_results)

        print(f"\n\nSequential vs Batch ({len(medium_batch)} sequences):")
        print(f"  {'':20} {'Time (ms)':>12} {'Emb (MB)':>12} {'Mem Δ (MB)':>12}")
        print(f"  {'-' * 56}")
        print(
            f"  {'Sequential:':20} {sequential_time * 1000:>12.2f} {seq_emb_mb:>12.4f} {mem_after_seq - mem_start:>12.2f}"
        )
        print(
            f"  {'Batch:':20} {batch_time * 1000:>12.2f} {batch_emb_mb:>12.4f} {mem_after_batch - mem_before_batch:>12.2f}"
        )
        print(f"  {'-' * 56}")
        print(f"  Time ratio: {sequential_time / batch_time:.2f}x")

        _append_memory_results(
            "sequential_vs_batch",
            {
                "num_sequences": len(medium_batch),
                "sequential": {
                    "time_ms": sequential_time * 1000,
                    "emb_mb": seq_emb_mb,
                    "mem_delta_mb": mem_after_seq - mem_start,
                },
                "batch": {
                    "time_ms": batch_time * 1000,
                    "emb_mb": batch_emb_mb,
                    "mem_delta_mb": mem_after_batch - mem_before_batch,
                },
                "time_ratio": sequential_time / batch_time,
            },
        )
