"""
Performance benchmarks for real ESM2-t6-8M model.

These tests measure actual model performance, not mock embedder performance.
All tests are marked as 'slow' since they require loading and running a real model.
All tests use sequences from the canonical test dataset.

Run with:
    uv run pytest tests/performance/test_esm2_throughput.py -v -s
    
Skip slow tests:
    uv run pytest tests/performance/ -m "not slow"
"""

from os import times
import pytest
import time
from typing import List, Dict
import numpy as np


@pytest.mark.slow
@pytest.mark.performance
class TestESM2SingleSequenceLatency:
    """Measure real ESM2 single sequence embedding latency."""

    def test_short_sequence_latency(self, esm2_embedder, short_sequence, benchmark):
        """Benchmark: short sequence (10 aa) with real ESM2."""
        def embed():
            return esm2_embedder.embed(short_sequence)
        
        result = benchmark(embed)
        assert result is not None
        # ESM2-t6-8M has embedding dimension 320
        assert result.shape[0] == len(short_sequence)
        assert result.shape[1] == 320

    def test_medium_sequence_latency(self, esm2_embedder, medium_sequence, benchmark):
        """Benchmark: medium sequence (~79 aa) with real ESM2."""
        def embed():
            return esm2_embedder.embed(medium_sequence)
        
        result = benchmark(embed)
        assert result is not None
        assert result.shape[0] == len(medium_sequence)
        assert result.shape[1] == 320

    def test_long_sequence_latency(self, esm2_embedder, long_sequence, benchmark):
        """Benchmark: long sequence (~211 aa) with real ESM2."""
        def embed():
            return esm2_embedder.embed(long_sequence)
        
        result = benchmark(embed)
        assert result is not None
        assert result.shape[0] == len(long_sequence)
        assert result.shape[1] == 320

    def test_very_long_sequence_latency(self, esm2_embedder, very_long_sequence, benchmark):
        """Benchmark: very long sequence (400 aa) with real ESM2."""
        def embed():
            return esm2_embedder.embed(very_long_sequence)
        
        result = benchmark(embed)
        assert result is not None
        assert result.shape[0] == len(very_long_sequence)
        assert result.shape[1] == 320


@pytest.mark.slow
@pytest.mark.performance
class TestESM2BatchThroughput:
    """Measure real ESM2 batch embedding throughput."""

    def test_canonical_dataset_throughput(self, esm2_embedder, canonical_sequences, benchmark):
        """Benchmark: embed all canonical test sequences."""
        
        def embed_all():
            results = []
            for seq in canonical_sequences:
                results.append(esm2_embedder.embed(seq))
            return results
        
        results = benchmark(embed_all)
        assert len(results) == len(canonical_sequences)

    def test_small_batch_throughput(self, esm2_embedder, small_batch, benchmark):
        """Benchmark: small batch from canonical dataset."""
        def embed_batch():
            return [esm2_embedder.embed(seq) for seq in small_batch]
        
        results = benchmark(embed_batch)
        assert len(results) == len(small_batch)


@pytest.mark.slow
@pytest.mark.performance
class TestESM2PooledEmbeddings:
    """Measure real ESM2 pooled embedding performance."""

    def test_pooled_embedding_latency(self, esm2_embedder, medium_sequence, benchmark):
        """Benchmark: pooled embedding for single sequence from canonical dataset."""
        def embed_pooled():
            emb = esm2_embedder.embed(medium_sequence)
            # Mean pooling
            return np.mean(emb, axis=0)
        
        result = benchmark(embed_pooled)
        assert result.shape == (320,)


@pytest.mark.slow
@pytest.mark.performance  
class TestESM2Comparison:
    """Compare ESM2 with FixedEmbedder performance."""

    def test_esm2_vs_fixed_embedder(self, esm2_embedder, perf_embedder, medium_sequence):
        """Compare real ESM2 latency against mock FixedEmbedder."""
        # Time FixedEmbedder
        fixed_times = []
        for _ in range(10):
            start = time.perf_counter()
            perf_embedder.embed(medium_sequence)
            fixed_times.append(time.perf_counter() - start)
        
        # Time ESM2
        esm2_times = []
        for _ in range(10):
            start = time.perf_counter()
            esm2_embedder.embed(medium_sequence)
            esm2_times.append(time.perf_counter() - start)
        
        fixed_mean = np.mean(fixed_times) * 1000  # ms
        esm2_mean = np.mean(esm2_times) * 1000  # ms
        
        print(f"\n{'='*50}")
        print(f"Performance Comparison (sequence length: {len(medium_sequence)})")
        print(f"{'='*50}")
        print(f"FixedEmbedder: {fixed_mean:.3f} ms (±{np.std(fixed_times)*1000:.3f} ms)")
        print(f"ESM2-t6-8M:    {esm2_mean:.3f} ms (±{np.std(esm2_times)*1000:.3f} ms)")
        print(f"Ratio:         {esm2_mean/fixed_mean:.1f}x slower")
        print(f"{'='*50}")
        
        # ESM2 should be slower but still complete
        assert esm2_mean > 0
        assert fixed_mean > 0


@pytest.mark.slow
@pytest.mark.performance
class TestESM2ScalingWithLength:
    """Measure how ESM2 performance scales with sequence length."""

    def test_scaling_report(self, esm2_embedder, variable_length_sequences):
        """Generate a scaling report for different sequence lengths from canonical dataset."""
        print(f"\n{'='*60}")
        print("ESM2-t6-8M Scaling with Sequence Length")
        print(f"{'='*60}")
        print(f"{'Length':<10} {'Mean (ms)':<15} {'Std (ms)':<15} {'Throughput (seq/s)':<20}")
        print(f"{'-'*60}")
        
        results = []
        for sequence in variable_length_sequences:
            times = []
            for _ in range(5):
                start = time.perf_counter()
                esm2_embedder.embed(sequence)
                times.append(time.perf_counter() - start)
            
            mean_ms = np.mean(times) * 1000
            std_ms = np.std(times) * 1000
            throughput = 1000 / mean_ms if mean_ms > 0 else 0
            
            results.append({
                "length": len(sequence),
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "throughput": throughput,
            })
            
            print(f"{len(sequence):<10} {mean_ms:<15.3f} {std_ms:<15.3f} {throughput:<20.1f}")
        
        print(f"{'='*60}")
        
        # Basic sanity check - all embeddings should complete
        assert all(r["mean_ms"] > 0 for r in results)
        std_ms = np.std(times) * 1000
        throughput = 1000 / mean_ms if mean_ms > 0 else 0
            
        results.append({
            "length": length,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "throughput": throughput,
        })
        
        print(f"{length:<10} {mean_ms:<15.3f} {std_ms:<15.3f} {throughput:<20.1f}")
        
        print(f"{'='*60}")
        
        # Basic sanity check - longer sequences should take more time (generally)
        assert all(r["mean_ms"] > 0 for r in results)
