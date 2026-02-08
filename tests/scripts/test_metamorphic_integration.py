#!/usr/bin/env python3
"""Pytest integration tests for metamorphic relations."""

import os

import numpy as np
import pytest

from tests.scripts.metamorphic_relations import (
    BatchVarianceRelation,
    IdempotencyRelation,
    ProgressiveMaskingRelation,
    ProjectionDeterminismRelation,
    RelationVerdict,
    ReversalRelation,
    compute_all_metrics,
    embeddings_are_identical,
    run_all_relations,
    summarize_results,
)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--use-real-embedder",
        action="store_true",
        default=False,
        help="Use real ESM2-T6-8M embedder instead of mock",
    )
    parser.addoption(
        "--use-server",
        action="store_true",
        default=True,
        help="Test against running server (requires CI_SERVER_URL)",
    )


@pytest.fixture(scope="module")
def use_real_embedder(request) -> bool:
    """Check if real embedder should be used."""
    cli_flag = request.config.getoption("--use-real-embedder", default=False)
    env_flag = os.environ.get("USE_REAL_EMBEDDER", "0") == "1"
    return cli_flag or env_flag


@pytest.fixture(scope="module")
def use_server(request) -> bool:
    """Check if server should be tested."""
    cli_flag = request.config.getoption("--use-server", default=True)
    env_flag = os.environ.get("CI_SERVER_URL") is not None
    return cli_flag or env_flag


@pytest.fixture(scope="module")
def embedder(use_server, use_real_embedder):
    """
    Get appropriate embedder based on configuration.
    
    Priority:
    1. FixedEmbedder if CI_EMBEDDER=fixed (fast mock, no server needed)
    2. ServerEmbedder if CI_SERVER_URL is set (integration test mode with esm2_t6_8M)
    3. Real ESM2 embedder if --use-real-embedder flag
    4. FixedEmbedder (default fallback)
    """
    from tests.fixtures.fixed_embedder import FixedEmbedder
    
    ci_embedder = os.environ.get("CI_EMBEDDER", "").lower()
    
    # CI fixed embedder mode: use FixedEmbedder directly without server
    if ci_embedder == "fixed":
        return FixedEmbedder(model_name="esm2_t6", strict_dataset=False)
    
    # Default: integration test mode - test the actual server with esm2_t6_8M
    if use_server:
        server_url = os.environ.get("CI_SERVER_URL")
        if not server_url:
            pytest.skip("CI_SERVER_URL not set, cannot test server")
        
        from tests.fixtures.server_embedder import ServerEmbedder
        embedder_name = os.environ.get("CI_EMBEDDER_NAME", "facebook/esm2_t6_8M_UR50D")
        return ServerEmbedder(base_url=server_url, embedder_name=embedder_name)
    
    # Real embedder mode: test actual model locally
    if use_real_embedder:
        try:
            from tests.scripts.run_metamorphic_experiments import get_real_embedder
            return get_real_embedder()
        except Exception as e:
            pytest.skip(f"Real embedder not available: {e}")
    
    # Fallback: use fast mock embedder
    return FixedEmbedder(model_name="esm2_t6", strict_dataset=False)


@pytest.fixture(scope="module")
def test_sequences():
    """Get test sequences from canonical dataset."""
    try:
        from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET
        return CANONICAL_TEST_DATASET.get_all_sequences()
    except ImportError:
        # Fallback sequences
        return [
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQ",
            "MKKLVLSLSLVLAFSSATAAFAAIPQNIRAQYPAVVKEQRQVVRSQNGDLADNIKKISDNLKAKIYAMHYVDVFYNKS",
            "M",
            "MK",
            "MKTAY",
            "XXXXX",
            "AAAAAAAAAA",
        ]


@pytest.fixture(scope="module")
def standard_sequences(test_sequences):
    """Get standard-length sequences (>10 aa) for batch tests."""
    return [s for s in test_sequences if len(s) >= 10]


@pytest.fixture(scope="module")
def short_sequences(test_sequences):
    """Get short sequences (<10 aa) for edge case tests."""
    return [s for s in test_sequences if len(s) < 10]


@pytest.fixture(scope="module")
def ci_scale():
    """
    Get CI scale setting from environment.
    
    Returns 'half' when CI_METAMORPHIC_SCALE=half, 'full' otherwise.
    """
    return os.environ.get("CI_METAMORPHIC_SCALE", "full")


@pytest.fixture(scope="module")
def ci_config(ci_scale):
    """
    Get scaled test configuration based on CI environment.
    
    When CI_METAMORPHIC_SCALE=half, reduces all test parameters by ~50%
    for faster CI execution without GPU.
    """
    if ci_scale == "half":
        return {
            # Idempotency
            "num_repetitions": 2,
            "idempotency_seq_count": 3,
            "per_residue_seq_count": 2,
            # Batch variance
            "batch_sizes": [2, 5],
            "num_permutations": 2,
            "batch_seq_count": 5,
            # Projection
            "projection_seeds": [42, 123],
            "projection_seq_count": 5,
            # Masking
            "masking_ratios": [0.0, 0.3, 0.5],
            "masking_ratios_full": [0.0, 0.2, 0.5, 0.8],
            # Integration
            "integration_seq_count": 5,
            # Reversal
            "reversal_seq_count": 3,
        }
    else:
        return {
            # Idempotency
            "num_repetitions": 3,
            "idempotency_seq_count": 5,
            "per_residue_seq_count": 3,
            # Batch variance
            "batch_sizes": [2, 5, 10],
            "num_permutations": 3,
            "batch_seq_count": 10,
            # Projection
            "projection_seeds": [42, 123, 999],
            "projection_seq_count": 10,
            # Masking
            "masking_ratios": [0.0, 0.3, 0.5],
            "masking_ratios_full": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            # Integration
            "integration_seq_count": 10,
            # Reversal
            "reversal_seq_count": 5,
        }


class TestIdempotencyRelation:
    """Tests for idempotency: embed(seq) == embed(seq)."""
    
    def test_single_sequence_idempotency(self, embedder, test_sequences, ci_config):
        """Verify that embedding the same sequence twice yields identical results."""
        relation = IdempotencyRelation(
            embedder, threshold=1e-6, num_repetitions=ci_config["num_repetitions"]
        )
        
        seq_count = ci_config["idempotency_seq_count"]
        for seq in test_sequences[:seq_count]:
            results = relation.verify(seq, use_pooled=True)
            
            for result in results:
                assert result.verdict == RelationVerdict.PASSED, (
                    f"Idempotency failed for sequence of length {len(seq)}: "
                    f"cosine_distance={result.metrics.cosine_distance}"
                )
    
    def test_per_residue_idempotency(self, embedder, test_sequences, ci_config):
        """Verify idempotency for per-residue (non-pooled) embeddings."""
        relation = IdempotencyRelation(embedder, threshold=1e-6, num_repetitions=2)
        
        seq_count = ci_config["per_residue_seq_count"]
        for seq in test_sequences[:seq_count]:
            results = relation.verify(seq, use_pooled=False)
            
            for result in results:
                assert result.verdict == RelationVerdict.PASSED, (
                    f"Per-residue idempotency failed for sequence of length {len(seq)}"
                )
    
    def test_idempotency_with_special_tokens(self, embedder):
        """Verify idempotency with sequences containing X tokens."""
        relation = IdempotencyRelation(embedder, threshold=1e-6)
        
        special_sequences = ["XXXXX", "MXKXAX", "XXXXXXXXXXX"]
        
        for seq in special_sequences:
            results = relation.verify(seq)
            
            for result in results:
                assert result.verdict == RelationVerdict.PASSED, (
                    f"Idempotency failed for special sequence '{seq}'"
                )
    
    def test_idempotency_batch(self, embedder, test_sequences):
        """Verify idempotency across all test sequences."""
        relation = IdempotencyRelation(embedder, threshold=1e-6)
        
        results = relation.verify_batch(test_sequences)
        
        failed = [r for r in results if r.verdict == RelationVerdict.FAILED]
        assert len(failed) == 0, (
            f"{len(failed)} idempotency tests failed out of {len(results)}"
        )


class TestBatchVarianceRelation:
    """Tests for batch invariance: embed([A,B])[i] == embed([A])[0]."""
    
    def test_single_vs_batch_embedding(self, embedder, standard_sequences, ci_config):
        """Verify embedding alone equals embedding in a batch."""
        relation = BatchVarianceRelation(
            embedder, threshold=0.1, batch_sizes=ci_config["batch_sizes"]
        )
        
        target = standard_sequences[0]
        fillers = standard_sequences[1:6]
        
        results = relation.verify(target, fillers)
        
        for result in results:
            assert result.verdict == RelationVerdict.PASSED, (
                f"Batch variance failed: {result.parameter}, "
                f"cosine_distance={result.metrics.cosine_distance}"
            )
    
    def test_batch_order_independence(self, embedder, standard_sequences, ci_config):
        """Verify batch order doesn't affect individual embeddings."""
        relation = BatchVarianceRelation(embedder, threshold=0.1)
        
        if len(standard_sequences) < 5:
            pytest.skip("Need at least 5 sequences for order test")
        
        results = relation.verify_order_independence(
            standard_sequences[:5], num_permutations=ci_config["num_permutations"]
        )
        
        for result in results:
            assert result.verdict == RelationVerdict.PASSED, (
                f"Order independence failed: {result.parameter}, "
                f"cosine_distance={result.metrics.cosine_distance}"
            )
    
    def test_batch_sizes(self, embedder, standard_sequences, ci_config):
        """Test with various batch sizes."""
        if len(standard_sequences) < 10:
            pytest.skip("Need at least 10 sequences")
        
        relation = BatchVarianceRelation(
            embedder, 
            threshold=0.1, 
            batch_sizes=ci_config["batch_sizes"]
        )
        
        target = standard_sequences[0]
        fillers = standard_sequences[1:15]
        
        results = relation.verify(target, fillers)
        
        # Group by batch size
        by_size = {}
        for r in results:
            size = r.details.get("batch_size", 0)
            if size not in by_size:
                by_size[size] = []
            by_size[size].append(r)
        
        for size, size_results in by_size.items():
            failed = [r for r in size_results if r.verdict == RelationVerdict.FAILED]
            assert len(failed) == 0, (
                f"Batch size {size} failed {len(failed)} tests"
            )
    
    def test_batch_variance_with_short_sequences(self, embedder, short_sequences):
        """Test batch variance with very short sequences."""
        if len(short_sequences) < 3:
            pytest.skip("Not enough short sequences")
        
        relation = BatchVarianceRelation(embedder, threshold=0.15, batch_sizes=[2])
        
        for target in short_sequences[:2]:
            fillers = [s for s in short_sequences if s != target]
            results = relation.verify(target, fillers)
            
            for result in results:
                assert result.verdict == RelationVerdict.PASSED, (
                    f"Batch variance failed for short sequence: {result.parameter}"
                )


class TestProjectionDeterminismRelation:
    """Tests for projection determinism with fixed seeds."""
    
    def test_pca_determinism(self, embedder, standard_sequences):
        """Verify PCA projection is deterministic."""
        if len(standard_sequences) < 5:
            pytest.skip("Need at least 5 sequences")
        
        relation = ProjectionDeterminismRelation(
            embedder, threshold=1e-6, methods=["pca"]
        )
        
        results = relation.verify(standard_sequences[:10], seed=42)
        
        pca_results = [r for r in results if "pca" in r.parameter]
        for result in pca_results:
            assert result.verdict == RelationVerdict.PASSED, (
                f"PCA determinism failed: frobenius_diff={result.details.get('frobenius_diff')}"
            )
    
    def test_umap_determinism(self, embedder, standard_sequences):
        """Verify UMAP projection is deterministic with fixed seed."""
        if len(standard_sequences) < 5:
            pytest.skip("Need at least 5 sequences")
        
        try:
            import umap
        except ImportError:
            pytest.skip("UMAP not installed")
        
        relation = ProjectionDeterminismRelation(
            embedder, threshold=1e-5, methods=["umap"]
        )
        
        results = relation.verify(standard_sequences[:10], seed=42)
        
        umap_results = [r for r in results if "umap" in r.parameter]
        for result in umap_results:
            assert result.verdict == RelationVerdict.PASSED, (
                f"UMAP determinism failed: frobenius_diff={result.details.get('frobenius_diff')}"
            )
    
    def test_multiple_seeds(self, embedder, standard_sequences, ci_config):
        """Verify different seeds produce same-seed determinism."""
        if len(standard_sequences) < 5:
            pytest.skip("Need at least 5 sequences")
        
        relation = ProjectionDeterminismRelation(
            embedder, threshold=1e-6, methods=["pca"]
        )
        
        for seed in ci_config["projection_seeds"]:
            results = relation.verify(standard_sequences[:5], seed=seed)
            
            for result in results:
                assert result.verdict == RelationVerdict.PASSED, (
                    f"Determinism failed for seed {seed}"
                )


@pytest.mark.slow
class TestReversalRelation:
    """Exploratory tests for sequence reversal effects."""
    
    def test_reversal_relationship(self, embedder, standard_sequences, ci_config):
        """Explore relationship between sequence and its reverse."""
        relation = ReversalRelation(embedder, threshold=0.5)
        
        seq_count = ci_config["reversal_seq_count"]
        results = relation.verify_batch(standard_sequences[:seq_count])
        
        # Collect statistics
        distances = [r.metrics.cosine_distance for r in results if r.metrics]
        
        assert len(distances) > 0, "No reversal results generated"
        
        avg_distance = np.mean(distances)
        print(f"\nReversal analysis:")
        print(f"  Average cosine distance: {avg_distance:.4f}")
        print(f"  Min distance: {np.min(distances):.4f}")
        print(f"  Max distance: {np.max(distances):.4f}")
    
    def test_palindrome_reversal(self, embedder):
        """Test reversal with palindromic sequences."""
        palindromes = ["MAAM", "AKKA", "GLLLG"]
        
        relation = ReversalRelation(embedder, threshold=0.5)
        
        for seq in palindromes:
            results = relation.verify(seq)
            
            # Palindromes should be more similar to their reverse
            for result in results:
                if result.metrics:
                    print(f"  Palindrome '{seq}': distance={result.metrics.cosine_distance:.4f}")


@pytest.mark.slow
class TestProgressiveMaskingRelation:
    """Exploratory tests for progressive X-masking effects."""
    
    def test_masking_degradation_curve(self, embedder, standard_sequences, ci_config):
        """Analyze embedding degradation as masking increases."""
        relation = ProgressiveMaskingRelation(
            embedder, 
            threshold=0.3,
            masking_ratios=ci_config["masking_ratios_full"]
        )
        
        seq = standard_sequences[0]
        results = relation.verify(seq)
        
        print(f"\nMasking degradation for {len(seq)}aa sequence:")
        
        prev_distance = 0.0
        for result in results:
            ratio = result.details.get("masking_ratio", 0)
            distance = result.metrics.cosine_distance if result.metrics else 0
            delta = distance - prev_distance
            
            print(f"  {int(ratio*100):3d}% masked: cosine={distance:.4f} (Δ={delta:+.4f})")
            prev_distance = distance
        
        # Verify monotonic increase (generally)
        distances = [r.metrics.cosine_distance for r in results if r.metrics]
        # Allow some tolerance for non-monotonicity
        mostly_increasing = sum(
            1 for i in range(len(distances)-1) 
            if distances[i+1] >= distances[i] - 0.01
        )
        assert mostly_increasing >= len(distances) - 2, (
            "Degradation curve should be mostly increasing"
        )
    
    def test_find_divergence_threshold(self, embedder, standard_sequences):
        """Find the masking ratio at which embeddings diverge significantly."""
        relation = ProgressiveMaskingRelation(embedder, threshold=0.3)
        
        seq = standard_sequences[0]
        threshold_info = relation.find_divergence_threshold(seq, granularity=0.1)
        
        print(f"\nDivergence threshold analysis:")
        print(f"  Sequence length: {threshold_info['sequence_length']}")
        print(f"  Threshold ratio: {threshold_info['threshold_ratio']}")
        
        if threshold_info['threshold_ratio'] is not None:
            assert 0 < threshold_info['threshold_ratio'] <= 1.0
    
    def test_masking_different_lengths(self, embedder, test_sequences, ci_config):
        """Compare masking effects across different sequence lengths."""
        relation = ProgressiveMaskingRelation(
            embedder, threshold=0.3, masking_ratios=ci_config["masking_ratios"]
        )
        
        # Group by length
        by_length = {"short": [], "medium": [], "long": []}
        
        for seq in test_sequences:
            if len(seq) <= 5:
                by_length["short"].append(seq)
            elif len(seq) <= 50:
                by_length["medium"].append(seq)
            else:
                by_length["long"].append(seq)
        
        print("\nMasking effects by sequence length:")
        
        for length_cat, seqs in by_length.items():
            if not seqs:
                continue
            
            all_distances = []
            for seq in seqs[:3]:  # Test up to 3 per category
                results = relation.verify(seq)
                # Get 50% masking distance
                for r in results:
                    if r.details.get("masking_ratio") == 0.5 and r.metrics:
                        all_distances.append(r.metrics.cosine_distance)
            
            if all_distances:
                avg = np.mean(all_distances)
                print(f"  {length_cat}: avg 50% mask distance = {avg:.4f}")


class TestMetamorphicIntegration:
    """Integration tests running multiple relations together."""
    
    def test_run_all_fundamental_relations(self, embedder, standard_sequences, ci_config):
        """Run all fundamental invariant relations."""
        if len(standard_sequences) < 5:
            pytest.skip("Need at least 5 sequences")
        
        seq_count = ci_config["integration_seq_count"]
        results = run_all_relations(
            embedder,
            standard_sequences[:seq_count],
            relations=["idempotency", "batch_variance"],
            config={
                "idempotency": {"threshold": 1e-6},
                "batch_variance": {"threshold": 0.1},
            }
        )
        
        summary = summarize_results(results)
        
        print(f"\nFundamental invariants summary:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")
        
        # All fundamental invariants should pass
        assert summary["failed"] == 0, (
            f"{summary['failed']} fundamental invariant tests failed"
        )
    
    def test_run_all_relations(self, embedder, standard_sequences, ci_config):
        """Run all relations and collect results."""
        seq_count = ci_config["integration_seq_count"]
        if len(standard_sequences) < seq_count:
            pytest.skip(f"Need at least {seq_count} sequences")
        
        results = run_all_relations(
            embedder,
            standard_sequences[:seq_count],
        )
        
        summary = summarize_results(results)
        
        print(f"\nAll relations summary:")
        for rel_name, rel_summary in summary["by_relation"].items():
            status = "✓" if rel_summary["failed"] == 0 else "✗"
            print(f"  {status} {rel_name}: {rel_summary['passed']}/{rel_summary['total']}")
        
        # Fundamental invariants should pass; exploratory can be inconclusive
        fundamental_failed = (
            summary["by_relation"].get("idempotency", {}).get("failed", 0) +
            summary["by_relation"].get("batch_variance", {}).get("failed", 0)
        )
        
        assert fundamental_failed == 0, (
            f"{fundamental_failed} fundamental tests failed"
        )


class TestEdgeCases:
    """Edge case tests for metamorphic relations."""
    
    def test_single_residue_sequence(self, embedder):
        """Test with minimum-length sequence."""
        relation = IdempotencyRelation(embedder, threshold=1e-6)
        
        results = relation.verify("M")
        
        for result in results:
            assert result.verdict == RelationVerdict.PASSED
    
    def test_all_x_sequence(self, embedder):
        """Test with sequence of all unknown tokens."""
        relation = IdempotencyRelation(embedder, threshold=1e-6)
        
        for seq in ["X", "XX", "XXXXX", "XXXXXXXXXX"]:
            results = relation.verify(seq)
            
            for result in results:
                assert result.verdict == RelationVerdict.PASSED
    
    def test_homopolymer(self, embedder):
        """Test with homopolymer sequences."""
        relation = BatchVarianceRelation(embedder, threshold=0.1, batch_sizes=[2])
        
        homopolymers = ["AAAAAAAAAA", "KKKKKKKKKK", "LLLLLLLLLL"]
        
        for target in homopolymers:
            fillers = [s for s in homopolymers if s != target]
            results = relation.verify(target, fillers)
            
            for result in results:
                assert result.verdict == RelationVerdict.PASSED
    
    def test_empty_batch_handling(self, embedder):
        """Test handling of edge cases in batch operations."""
        relation = BatchVarianceRelation(embedder, threshold=0.1, batch_sizes=[1])
        
        target = "MKTAY"
        fillers = []  # Empty fillers
        
        # Should handle gracefully (batch size 1 = solo embedding)
        results = relation.verify(target, fillers)
        
        # With empty fillers, only batch_size=1 should work
        assert len(results) > 0


class TestMetrics:
    """Tests for metric computation functions."""
    
    def test_identical_embeddings(self):
        """Identical embeddings should have zero distance."""
        emb = np.random.randn(100).astype(np.float32)
        
        metrics = compute_all_metrics(emb, emb.copy())
        
        assert metrics.cosine_distance < 1e-10
        assert metrics.l2_distance < 1e-10
    
    def test_opposite_embeddings(self):
        """Opposite embeddings should have maximum cosine distance."""
        emb = np.random.randn(100).astype(np.float32)
        
        metrics = compute_all_metrics(emb, -emb)
        
        assert abs(metrics.cosine_distance - 2.0) < 1e-6
    
    def test_embeddings_are_identical(self):
        """Test the numerical identity check."""
        emb = np.array([1.0, 2.0, 3.0])
        
        assert embeddings_are_identical(emb, emb.copy())
        assert not embeddings_are_identical(emb, emb + 0.1)
        assert embeddings_are_identical(emb, emb + 1e-9)  # Within tolerance
