import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tests.scripts.metamorphic_relations import (
    BatchVarianceRelation,
    IdempotencyRelation,
    ProgressiveMaskingRelation,
    ProjectionDeterminismRelation,
    RelationVerdict,
    ReversalRelation,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_embedder(use_real: bool = False, model_name: str = "esm2_t6"):
    """Get an embedder for experiments."""
    if use_real:
        try:
            from biotrainer.embedders import get_embedding_service
            from tests.scripts.run_metamorphic_experiments import RealEmbedderWrapper
            
            embedding_service = get_embedding_service(
                embedder_name="facebook/esm2_t6_8M_UR50D",
                use_half_precision=False,
                custom_tokenizer_config=None,
                device="cpu",
            )
            return RealEmbedderWrapper(embedding_service, "esm2_t6_8M")
        except ImportError as e:
            print(f"Warning: Real embedder not available ({e}), using mock")
    
    from tests.fixtures.fixed_embedder import FixedEmbedder
    return FixedEmbedder(model_name=model_name, strict_dataset=False)


def get_test_sequences() -> List[str]:
    """Get test sequences for exploration."""
    try:
        from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET
        return CANONICAL_TEST_DATASET.get_all_sequences()
    except ImportError:
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


def explore_idempotency(embedder, sequences: List[str], verbose: bool = True) -> Dict[str, Any]:
    """
    Explore idempotency: Does same sequence -> same embedding?
    
    This is a fundamental invariant that SHOULD always hold.
    """
    print("\n" + "=" * 70)
    print("EXPLORING IDEMPOTENCY")
    print("Question: Does embedding the same sequence always produce identical results?")
    print("=" * 70)
    
    relation = IdempotencyRelation(embedder, threshold=1e-6, num_repetitions=5)
    
    findings = {
        "invariant_holds": True,
        "max_distance_seen": 0.0,
        "violations": [],
        "sequence_analyses": [],
    }
    
    for seq in sequences[:10]:
        results = relation.verify(seq, use_pooled=True)
        
        distances = [r.metrics.cosine_distance for r in results if r.metrics]
        max_dist = max(distances) if distances else 0.0
        
        analysis = {
            "sequence_length": len(seq),
            "repetitions_tested": len(results) + 1,
            "max_cosine_distance": max_dist,
            "all_identical": all(r.verdict == RelationVerdict.PASSED for r in results),
        }
        findings["sequence_analyses"].append(analysis)
        
        if max_dist > findings["max_distance_seen"]:
            findings["max_distance_seen"] = max_dist
        
        for r in results:
            if r.verdict == RelationVerdict.FAILED:
                findings["invariant_holds"] = False
                findings["violations"].append({
                    "sequence_length": len(seq),
                    "cosine_distance": r.metrics.cosine_distance,
                })
        
        if verbose:
            status = "✓" if analysis["all_identical"] else "✗"
            print(f"  {status} {len(seq):3d}aa: max_distance={max_dist:.2e}")
    
    print(f"\n  CONCLUSION: {'Idempotency HOLDS' if findings['invariant_holds'] else 'Idempotency VIOLATED!'}")
    print(f"  Maximum distance across all tests: {findings['max_distance_seen']:.2e}")
    
    return findings


def explore_batch_variance(embedder, sequences: List[str], verbose: bool = True) -> Dict[str, Any]:
    """
    Explore batch variance: Does batching affect individual embeddings?
    
    This is a fundamental invariant that SHOULD always hold.
    """
    print("\n" + "=" * 70)
    print("EXPLORING BATCH INVARIANCE")
    print("Question: Does embedding A alone produce the same result as A in a batch?")
    print("=" * 70)
    
    relation = BatchVarianceRelation(embedder, threshold=0.1, batch_sizes=[2, 5, 10])
    
    findings = {
        "invariant_holds": True,
        "max_distance_seen": 0.0,
        "by_batch_size": {},
        "by_position": {},
        "violations": [],
    }
    
    standard_seqs = [s for s in sequences if len(s) >= 10][:10]
    
    for target in standard_seqs[:5]:
        fillers = [s for s in standard_seqs if s != target][:9]
        results = relation.verify(target, fillers)
        
        for r in results:
            batch_size = r.details.get("batch_size", 0)
            position = r.details.get("target_position", "unknown")
            distance = r.metrics.cosine_distance if r.metrics else 0
            
            if batch_size not in findings["by_batch_size"]:
                findings["by_batch_size"][batch_size] = []
            findings["by_batch_size"][batch_size].append(distance)
            
            if position not in findings["by_position"]:
                findings["by_position"][position] = []
            findings["by_position"][position].append(distance)
            
            if distance > findings["max_distance_seen"]:
                findings["max_distance_seen"] = distance
            
            if r.verdict == RelationVerdict.FAILED:
                findings["invariant_holds"] = False
                findings["violations"].append({
                    "batch_size": batch_size,
                    "position": position,
                    "cosine_distance": distance,
                })
    
    if verbose:
        print("\n  Distance by batch size:")
        for size, distances in sorted(findings["by_batch_size"].items()):
            avg = np.mean(distances)
            mx = np.max(distances)
            print(f"    Batch size {size:2d}: avg={avg:.2e}, max={mx:.2e}")
        
        print("\n  Distance by position:")
        for pos, distances in findings["by_position"].items():
            avg = np.mean(distances)
            mx = np.max(distances)
            print(f"    Position={pos:6s}: avg={avg:.2e}, max={mx:.2e}")
    
    print(f"\n  CONCLUSION: {'Batch invariance HOLDS' if findings['invariant_holds'] else 'Batch invariance VIOLATED!'}")
    
    return findings


def explore_reversal(embedder, sequences: List[str], verbose: bool = True) -> Dict[str, Any]:
    """
    Explore sequence reversal effects.
    
    This is EXPLORATORY - we're discovering what happens, not testing an invariant.
    """
    print("\n" + "=" * 70)
    print("EXPLORING SEQUENCE REVERSAL")
    print("Question: How similar is embed(seq) to embed(reverse(seq))?")
    print("=" * 70)
    
    relation = ReversalRelation(embedder, threshold=0.5)
    
    findings = {
        "by_length": {},
        "all_distances": [],
        "average_distance": 0.0,
        "palindrome_analysis": [],
    }
    
    for seq in sequences:
        results = relation.verify(seq)
        
        for r in results:
            distance = r.metrics.cosine_distance if r.metrics else 0
            length = len(seq)
            
            findings["all_distances"].append(distance)
            
            length_bucket = f"{(length // 10) * 10}-{(length // 10 + 1) * 10}"
            if length_bucket not in findings["by_length"]:
                findings["by_length"][length_bucket] = []
            findings["by_length"][length_bucket].append(distance)
            
            if verbose:
                relationship = r.details.get("relationship", "unknown")
                print(f"  {len(seq):3d}aa: distance={distance:.4f} ({relationship})")
    
    # Test palindromes
    palindromes = ["MAAM", "AKKA", "GLLLG", "MALMAM"]
    print("\n  Palindrome analysis (should be more similar to reverse):")
    for seq in palindromes:
        results = relation.verify(seq)
        for r in results:
            distance = r.metrics.cosine_distance if r.metrics else 0
            findings["palindrome_analysis"].append({
                "sequence": seq,
                "distance": distance,
            })
            if verbose:
                print(f"    '{seq}': distance={distance:.4f}")
    
    findings["average_distance"] = np.mean(findings["all_distances"]) if findings["all_distances"] else 0
    
    print(f"\n  FINDINGS:")
    print(f"    Average reversal distance: {findings['average_distance']:.4f}")
    print(f"    This suggests {'high' if findings['average_distance'] > 0.3 else 'moderate' if findings['average_distance'] > 0.1 else 'low'} sensitivity to sequence direction")
    
    return findings


def explore_progressive_masking(embedder, sequences: List[str], verbose: bool = True) -> Dict[str, Any]:
    """
    Explore progressive masking effects.
    
    This is EXPLORATORY - we're discovering the degradation curve as sequences
    are progressively masked with 'X' (unknown amino acid placeholder).
    """
    print("\n" + "=" * 70)
    print("EXPLORING PROGRESSIVE MASKING")
    print("Question: At what point does replacing amino acids with 'X' significantly")
    print("          change the embedding? How does the degradation curve look?")
    print("=" * 70)
    
    relation = ProgressiveMaskingRelation(
        embedder, 
        threshold=0.3,
        masking_ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    
    findings = {
        "divergence_thresholds": [],
        "degradation_curves": {},
        "by_length": {},
    }
    
    standard_seqs = [s for s in sequences if len(s) >= 10][:5]
    
    for seq in standard_seqs:
        results = relation.verify(seq)
        
        ratios = []
        distances = []
        
        if verbose:
            print(f"\n  Sequence ({len(seq)}aa):")
        
        for r in results:
            ratio = r.details.get("masking_ratio", 0)
            distance = r.metrics.cosine_distance if r.metrics else 0
            ratios.append(ratio)
            distances.append(distance)
            
            if verbose:
                bar = "█" * int(distance * 20)
                print(f"    {int(ratio*100):3d}% masked: {bar:20s} {distance:.4f}")
        
        findings["degradation_curves"][len(seq)] = {
            "ratios": ratios,
            "distances": distances,
        }
        
        # Find divergence threshold (where distance > 0.3)
        threshold_info = relation.find_divergence_threshold(seq, granularity=0.05)
        if threshold_info["threshold_ratio"] is not None:
            findings["divergence_thresholds"].append(threshold_info["threshold_ratio"])
    
    avg_threshold = np.mean(findings["divergence_thresholds"]) if findings["divergence_thresholds"] else None
    
    print(f"\n  FINDINGS:")
    if avg_threshold:
        print(f"    Average divergence threshold: {avg_threshold*100:.1f}% masking")
        print(f"    Interpretation: Embeddings remain meaningful up to ~{int(avg_threshold*100)}% unknown residues")
    else:
        print(f"    No significant divergence detected - embeddings are robust to masking")
    
    return findings


def explore_projection_determinism(embedder, sequences: List[str], verbose: bool = True) -> Dict[str, Any]:
    """
    Explore projection determinism with UMAP, PCA, t-SNE.
    
    This SHOULD hold with fixed seeds.
    """
    print("\n" + "=" * 70)
    print("EXPLORING PROJECTION DETERMINISM")
    print("Question: Do PCA/UMAP/t-SNE projections produce identical results")
    print("          when using the same random seed?")
    print("=" * 70)
    
    relation = ProjectionDeterminismRelation(
        embedder, 
        threshold=1e-6, 
        methods=["pca", "umap"]
    )
    
    findings = {
        "by_method": {},
        "invariant_holds": True,
    }
    
    standard_seqs = [s for s in sequences if len(s) >= 10][:16]  # Need 16 for UMAP
    
    if len(standard_seqs) < 5:
        print("  Not enough sequences for projection experiments")
        return findings
    
    results = relation.verify(standard_seqs, seed=42, num_repetitions=3)
    
    for r in results:
        method = r.details.get("method", "unknown")
        diff = r.details.get("frobenius_diff", 0)
        
        if method not in findings["by_method"]:
            findings["by_method"][method] = []
        findings["by_method"][method].append(diff)
        
        if r.verdict == RelationVerdict.FAILED:
            findings["invariant_holds"] = False
        
        if verbose:
            status = "✓" if r.verdict == RelationVerdict.PASSED else "✗"
            print(f"  {status} {method}: frobenius_diff={diff:.2e}")
    
    print(f"\n  CONCLUSION: {'Projection determinism HOLDS' if findings['invariant_holds'] else 'Projection determinism VIOLATED!'}")
    
    return findings


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive exploration of metamorphic relations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--use-real-embedder",
        action="store_true",
        help="Use real ESM2-T6-8M instead of mock embedder",
    )
    
    parser.add_argument(
        "--relation",
        choices=["idempotency", "batch_variance", "reversal", "progressive_masking", "projection_determinism", "all"],
        default="all",
        help="Specific relation to explore (default: all)",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for JSON findings",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("METAMORPHIC RELATIONS EXPLORER")
    print("Discovering properties of embedding services")
    print("=" * 70)
    
    embedder = get_embedder(use_real=args.use_real_embedder)
    print(f"\nUsing embedder: {embedder.model_name}")
    
    sequences = get_test_sequences()
    print(f"Loaded {len(sequences)} test sequences")
    
    verbose = not args.quiet
    all_findings = {"embedder": embedder.model_name}
    
    relations_to_explore = (
        ["idempotency", "batch_variance", "reversal", "progressive_masking", "projection_determinism"]
        if args.relation == "all"
        else [args.relation]
    )
    
    for rel in relations_to_explore:
        if rel == "idempotency":
            all_findings["idempotency"] = explore_idempotency(embedder, sequences, verbose)
        elif rel == "batch_variance":
            all_findings["batch_variance"] = explore_batch_variance(embedder, sequences, verbose)
        elif rel == "reversal":
            all_findings["reversal"] = explore_reversal(embedder, sequences, verbose)
        elif rel == "progressive_masking":
            all_findings["progressive_masking"] = explore_progressive_masking(embedder, sequences, verbose)
        elif rel == "projection_determinism":
            all_findings["projection_determinism"] = explore_projection_determinism(embedder, sequences, verbose)
    
    if args.output:
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        with open(args.output, "w") as f:
            json.dump(convert_types(all_findings), f, indent=2)
        print(f"\nFindings saved to: {args.output}")
    
    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)
    print("\nThese findings can inform testing strategies for embedding services.")
    print("Strict invariants (idempotency, batch invariance) should be tested in CI.")
    print("Exploratory relations (reversal, masking) help understand model behavior.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
