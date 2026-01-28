#!/usr/bin/env python3
"""Metamorphic testing experiment runner for embedding services."""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.scripts.metamorphic_relations import (
    BatchVarianceRelation,
    IdempotencyRelation,
    ProgressiveMaskingRelation,
    ProjectionDeterminismRelation,
    RelationRegistry,
    RelationResult,
    RelationVerdict,
    ReversalRelation,
    run_all_relations,
    summarize_results,
)


def get_fixed_embedder(model_name: str = "esm2_t6", strict: bool = False):
    """
    Get a FixedEmbedder instance for testing.
    
    Args:
        model_name: Name of the model to emulate
        strict: Whether to restrict to canonical test dataset
    
    Returns:
        FixedEmbedder instance
    """
    from tests.fixtures.fixed_embedder import FixedEmbedder
    return FixedEmbedder(model_name=model_name, strict_dataset=strict)


def get_real_embedder(model_name: str = "facebook/esm2_t6_8M_UR50D"):
    """
    Get a real embedder using biotrainer.
    
    Args:
        model_name: HuggingFace model name
    
    Returns:
        Wrapped embedder instance
    """
    try:
        from biotrainer.embedders import get_embedding_service
    except ImportError:
        raise RuntimeError(
            "biotrainer not installed. Install with: pip install biotrainer"
        )
    
    embedding_service = get_embedding_service(
        embedder_name=model_name,
        use_half_precision=False,
        custom_tokenizer_config=None,
        device="cpu",
    )
    
    return RealEmbedderWrapper(embedding_service, model_name)


class RealEmbedderWrapper:
    """Wrapper to adapt biotrainer EmbeddingService to our protocol."""
    
    def __init__(self, embedding_service, model_name: str):
        self.embedding_service = embedding_service
        self.model_name = model_name
    
    def embed(self, sequence: str) -> np.ndarray:
        """Embed single sequence, returning per-residue embeddings."""
        results = list(
            self.embedding_service.generate_embeddings(
                input_data=[sequence], reduce=False
            )
        )
        if results:
            _, embedding = results[0]
            return np.array(embedding)
        return np.array([])
    
    def embed_pooled(self, sequence: str) -> np.ndarray:
        """Embed single sequence, returning pooled embedding."""
        results = list(
            self.embedding_service.generate_embeddings(
                input_data=[sequence], reduce=True
            )
        )
        if results:
            _, embedding = results[0]
            return np.array(embedding)
        return np.array([])
    
    def embed_batch(self, sequences: List[str], pooled: bool = False) -> List[np.ndarray]:
        """Embed multiple sequences."""
        results = list(
            self.embedding_service.generate_embeddings(
                input_data=sequences, reduce=pooled
            )
        )
        return [np.array(emb) for _, emb in results]


def get_test_sequences(use_canonical: bool = True) -> List[str]:
    """
    Get test sequences for experiments.
    
    Args:
        use_canonical: Whether to use canonical test dataset
    
    Returns:
        List of test sequences
    """
    if use_canonical:
        from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET
        return CANONICAL_TEST_DATASET.get_all_sequences()
    
    # Fallback sequences if canonical dataset not available
    return [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQ",
        "MKKLVLSLSLVLAFSSATAAFAAIPQNIRAQYPAVVKEQRQVVRSQNGDLADNIKKISDNLKAKIYAMHYVDVFYNKS",
        "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLA",
        "M",
        "MK",
        "MKTAY",
        "XXXXX",
        "MXKXAX",
        "AAAAAAAAAA",
        "KKKKKKKKKK",
    ]


def run_experiments(
    embedder,
    sequences: List[str],
    relations: List[str],
    config: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, List[RelationResult]]:
    """
    Run metamorphic experiments.
    
    Args:
        embedder: Embedder instance
        sequences: Test sequences
        relations: List of relation names to run
        config: Configuration for relations
        verbose: Whether to print progress
    
    Returns:
        Dictionary mapping relation names to results
    """
    all_results = {}
    
    for rel_name in relations:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running: {rel_name}")
            print(f"{'='*60}")
        
        rel_class = RelationRegistry.get_relation(rel_name)
        rel_config = config.get(rel_name, {})
        
        relation = rel_class(embedder, **rel_config)
        
        if rel_name == "idempotency":
            results = run_idempotency_experiments(relation, sequences, verbose)
        elif rel_name == "batch_variance":
            results = run_batch_variance_experiments(relation, sequences, verbose)
        elif rel_name == "projection_determinism":
            results = run_projection_experiments(relation, sequences, verbose)
        elif rel_name == "reversal":
            results = run_reversal_experiments(relation, sequences, verbose)
        elif rel_name == "progressive_masking":
            results = run_masking_experiments(relation, sequences, verbose)
        else:
            results = []
        
        all_results[rel_name] = results
        
        if verbose:
            print_relation_summary(rel_name, results)
    
    return all_results


def run_idempotency_experiments(
    relation: IdempotencyRelation,
    sequences: List[str],
    verbose: bool,
) -> List[RelationResult]:
    """Run idempotency experiments."""
    results = []
    
    for i, seq in enumerate(sequences):
        if verbose and i % 10 == 0:
            print(f"  Testing sequence {i+1}/{len(sequences)}...")
        
        try:
            seq_results = relation.verify(seq)
            results.extend(seq_results)
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to test sequence {i}: {e}")
    
    return results


def run_batch_variance_experiments(
    relation: BatchVarianceRelation,
    sequences: List[str],
    verbose: bool,
) -> List[RelationResult]:
    """Run batch variance experiments."""
    results = []
    
    # Test each sequence against others as fillers
    for i, target_seq in enumerate(sequences):
        if verbose and i % 5 == 0:
            print(f"  Testing sequence {i+1}/{len(sequences)}...")
        
        fillers = [s for j, s in enumerate(sequences) if j != i]
        
        try:
            seq_results = relation.verify(target_seq, fillers)
            results.extend(seq_results)
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed batch test for sequence {i}: {e}")
    
    # Also test order independence
    if len(sequences) >= 5:
        if verbose:
            print("  Testing batch order independence...")
        
        try:
            order_results = relation.verify_order_independence(sequences[:5])
            results.extend(order_results)
        except Exception as e:
            if verbose:
                print(f"  Warning: Order independence test failed: {e}")
    
    return results


def run_projection_experiments(
    relation: ProjectionDeterminismRelation,
    sequences: List[str],
    verbose: bool,
) -> List[RelationResult]:
    """Run projection determinism experiments."""
    results = []
    
    if len(sequences) < 5:
        if verbose:
            print("  Skipping: Need at least 5 sequences for projection")
        return results
    
    # Test with different numbers of sequences
    for n_seqs in [5, 10, min(20, len(sequences))]:
        if n_seqs > len(sequences):
            continue
        
        if verbose:
            print(f"  Testing with {n_seqs} sequences...")
        
        try:
            proj_results = relation.verify(sequences[:n_seqs])
            results.extend(proj_results)
        except Exception as e:
            if verbose:
                print(f"  Warning: Projection test with {n_seqs} sequences failed: {e}")
    
    return results


def run_reversal_experiments(
    relation: ReversalRelation,
    sequences: List[str],
    verbose: bool,
) -> List[RelationResult]:
    """Run sequence reversal experiments."""
    results = []
    
    for i, seq in enumerate(sequences):
        if verbose and i % 10 == 0:
            print(f"  Testing sequence {i+1}/{len(sequences)}...")
        
        try:
            seq_results = relation.verify(seq)
            results.extend(seq_results)
        except Exception as e:
            if verbose:
                print(f"  Warning: Reversal test for sequence {i} failed: {e}")
    
    return results


def run_masking_experiments(
    relation: ProgressiveMaskingRelation,
    sequences: List[str],
    verbose: bool,
) -> List[RelationResult]:
    """Run progressive masking experiments."""
    results = []
    
    for i, seq in enumerate(sequences):
        if verbose and i % 10 == 0:
            print(f"  Testing sequence {i+1}/{len(sequences)}...")
        
        try:
            seq_results = relation.verify(seq)
            results.extend(seq_results)
        except Exception as e:
            if verbose:
                print(f"  Warning: Masking test for sequence {i} failed: {e}")
    
    return results


def print_relation_summary(rel_name: str, results: List[RelationResult]) -> None:
    """Print summary for a single relation."""
    if not results:
        print(f"  No results for {rel_name}")
        return
    
    passed = sum(1 for r in results if r.verdict == RelationVerdict.PASSED)
    failed = sum(1 for r in results if r.verdict == RelationVerdict.FAILED)
    inconclusive = sum(1 for r in results if r.verdict == RelationVerdict.INCONCLUSIVE)
    total = len(results)
    
    print(f"\n  Summary for {rel_name}:")
    print(f"    Total tests: {total}")
    print(f"    Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"    Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"    Inconclusive: {inconclusive} ({inconclusive/total*100:.1f}%)")
    
    # Show metrics for failed tests
    failed_results = [r for r in results if r.verdict == RelationVerdict.FAILED]
    if failed_results:
        print(f"\n  Failed test details:")
        for r in failed_results[:5]:  # Show first 5
            if r.metrics:
                print(f"    - {r.test_case}/{r.parameter}: cosine={r.metrics.cosine_distance:.6f}")


def write_csv_report(
    results: Dict[str, List[RelationResult]],
    output_path: Path,
) -> None:
    """Write detailed CSV report."""
    all_rows = []
    
    for rel_name, rel_results in results.items():
        for result in rel_results:
            row = result.to_dict()
            all_rows.append(row)
    
    if not all_rows:
        print("No results to write")
        return
    
    # Get all unique columns
    columns = set()
    for row in all_rows:
        columns.update(row.keys())
    columns = sorted(columns)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"CSV report written to: {output_path}")


def write_json_summary(
    summary: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write JSON summary report."""
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"JSON summary written to: {output_path}")


def write_masking_analysis(
    results: List[RelationResult],
    output_path: Path,
) -> None:
    """Write special analysis for progressive masking experiments."""
    # Group by sequence
    by_sequence = {}
    for r in results:
        seq_key = r.test_case
        if seq_key not in by_sequence:
            by_sequence[seq_key] = []
        by_sequence[seq_key].append(r)
    
    analysis = {
        "sequences_analyzed": len(by_sequence),
        "degradation_curves": {},
        "divergence_thresholds": {},
    }
    
    for seq_key, seq_results in by_sequence.items():
        # Sort by masking ratio
        sorted_results = sorted(
            seq_results,
            key=lambda r: r.details.get("masking_ratio", 0)
        )
        
        ratios = []
        distances = []
        
        for r in sorted_results:
            ratio = r.details.get("masking_ratio", 0)
            distance = r.metrics.cosine_distance if r.metrics else 0
            ratios.append(ratio)
            distances.append(distance)
        
        analysis["degradation_curves"][seq_key] = {
            "ratios": ratios,
            "cosine_distances": distances,
        }
        
        # Find divergence threshold
        for i, (ratio, dist) in enumerate(zip(ratios, distances)):
            if dist > 0.3:  # Significant divergence threshold
                analysis["divergence_thresholds"][seq_key] = ratio
                break
    
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Masking analysis written to: {output_path}")


def print_final_summary(summary: Dict[str, Any]) -> None:
    """Print final summary to console."""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Inconclusive: {summary['inconclusive']}")
    print(f"Pass rate (excluding inconclusive): {summary['pass_rate']:.1f}%")
    
    print("\nBy relation:")
    for rel_name, rel_summary in summary["by_relation"].items():
        status = "✓" if rel_summary["failed"] == 0 else "✗"
        print(f"  {status} {rel_name}: {rel_summary['passed']}/{rel_summary['total']} passed")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run metamorphic testing experiments for embedding services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--use-real-embedder",
        action="store_true",
        help="Use real ESM2-T6-8M instead of mock embedder",
    )
    
    parser.add_argument(
        "--relations",
        nargs="+",
        choices=RelationRegistry.list_relations(),
        default=None,
        help="Specific relations to test (default: all)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/reports/metamorphic"),
        help="Output directory for reports",
    )
    
    parser.add_argument(
        "--idempotency-threshold",
        type=float,
        default=1e-6,
        help="Threshold for idempotency tests (default: 1e-6)",
    )
    
    parser.add_argument(
        "--batch-threshold",
        type=float,
        default=0.1,
        help="Threshold for batch variance tests (default: 0.1)",
    )
    
    parser.add_argument(
        "--masking-threshold",
        type=float,
        default=0.3,
        help="Threshold for masking significance (default: 0.3)",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("METAMORPHIC TESTING EXPERIMENTS")
    print("=" * 60)
    
    # Setup embedder
    if args.use_real_embedder:
        print("\nUsing: Real ESM2-T6-8M embedder")
        try:
            embedder = get_real_embedder()
        except Exception as e:
            print(f"Error loading real embedder: {e}")
            print("Falling back to mock embedder")
            embedder = get_fixed_embedder(strict=False)
    else:
        print("\nUsing: Mock FixedEmbedder")
        embedder = get_fixed_embedder(strict=False)
    
    # Get test sequences
    try:
        sequences = get_test_sequences(use_canonical=True)
        print(f"Loaded {len(sequences)} canonical test sequences")
    except Exception as e:
        print(f"Warning: Could not load canonical dataset: {e}")
        sequences = get_test_sequences(use_canonical=False)
        print(f"Using {len(sequences)} fallback sequences")
    
    # Configure relations
    relations = args.relations or RelationRegistry.list_relations()
    print(f"Relations to test: {', '.join(relations)}")
    
    config = {
        "idempotency": {"threshold": args.idempotency_threshold},
        "batch_variance": {"threshold": args.batch_threshold},
        "progressive_masking": {"threshold": args.masking_threshold},
        "reversal": {"threshold": 0.5},
        "projection_determinism": {"threshold": 1e-6},
    }
    
    # Run experiments
    verbose = not args.quiet
    results = run_experiments(embedder, sequences, relations, config, verbose)
    
    # Generate summary
    summary = summarize_results(results)
    summary["embedder"] = embedder.model_name
    summary["timestamp"] = timestamp
    summary["num_sequences"] = len(sequences)
    
    # Write reports
    csv_path = args.output_dir / f"metamorphic_results_{timestamp}.csv"
    json_path = args.output_dir / f"metamorphic_summary_{timestamp}.json"
    
    write_csv_report(results, csv_path)
    write_json_summary(summary, json_path)
    
    # Special analysis for masking experiments
    if "progressive_masking" in results and results["progressive_masking"]:
        masking_path = args.output_dir / f"masking_analysis_{timestamp}.json"
        write_masking_analysis(results["progressive_masking"], masking_path)
    
    # Print final summary
    print_final_summary(summary)
    
    # Return non-zero if any tests failed
    if summary["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
