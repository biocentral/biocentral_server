# MR: Reversed sequence -- reverse amino-acid order and compare embeddings to the original.

import numpy as np
from typing import Any, Dict, List

from tests.property.oracles.embedding_metrics import (
    compute_all_metrics,
    format_metrics_table,
    write_metrics_csv,
)


def _reverse_sequence(seq: str) -> str:
    return seq[::-1]


def _is_repetitive(seq: str, max_motif_len: int = 5) -> bool:
    """Check if sequence is highly repetitive (e.g., 'KKRRKKRR' or 'AAAA')."""
    if len(seq) < 4:
        return False
    for motif_len in range(1, min(max_motif_len + 1, len(seq) // 2 + 1)):
        motif = seq[:motif_len]
        if motif * (len(seq) // motif_len) == seq[: motif_len * (len(seq) // motif_len)]:
            # At least 80% of sequence is repetitive
            if motif_len * (len(seq) // motif_len) >= len(seq) * 0.8:
                return True
    return False


def _run_reversal_experiment(
    embedder,
    embedder_label: str,
    sequences: List[str],
) -> List[Dict[str, Any]]:
    results = []

    for seq_idx, seq in enumerate(sequences):
        rev_seq = _reverse_sequence(seq)
        is_palindrome = seq == rev_seq
        is_repetitive = _is_repetitive(seq)

        orig_emb = embedder.embed_pooled(seq)
        rev_emb = embedder.embed_pooled(rev_seq)

        metrics = compute_all_metrics(orig_emb, rev_emb)

        results.append(
            {
                "embedder": embedder_label,
                "test_type": "reversed_sequence",
                "parameter": f"seq{seq_idx}_len{len(seq)}",
                "cosine_distance": metrics["cosine_distance"],
                "l2_distance": metrics["l2_distance"],
                "threshold": 0.0,
                "passed": True,
                "sequence_length": len(seq),
                "original_prefix": seq[:20],
                "reversed_prefix": rev_seq[:20],
                "is_palindrome": is_palindrome,
                "is_repetitive": is_repetitive,
            }
        )

    return results


def _run_double_reversal_experiment(
    embedder,
    embedder_label: str,
    sequences: List[str],
    tolerance: float = 1e-6,
) -> List[Dict[str, Any]]:
    results = []

    for seq_idx, seq in enumerate(sequences):
        rev_rev_seq = _reverse_sequence(_reverse_sequence(seq))
        assert rev_rev_seq == seq, "Bug: double-reversal changed the sequence"

        orig_emb = embedder.embed_pooled(seq)
        double_rev_emb = embedder.embed_pooled(rev_rev_seq)

        metrics = compute_all_metrics(orig_emb, double_rev_emb)

        results.append(
            {
                "embedder": embedder_label,
                "test_type": "double_reversal",
                "parameter": f"seq{seq_idx}",
                "cosine_distance": metrics["cosine_distance"],
                "l2_distance": metrics["l2_distance"],
                "threshold": tolerance,
                "passed": metrics["cosine_distance"] <= tolerance,
                "sequence_length": len(seq),
            }
        )

    return results


def _summarise_reversal(results: List[Dict[str, Any]], label: str) -> str:
    lines = [
        f"\n{'=' * 80}",
        f"  Reversed Sequence Summary — {label}",
        f"{'=' * 80}",
    ]

    rev_results = [r for r in results if r["test_type"] == "reversed_sequence"]
    if rev_results:
        cos_dists = [r["cosine_distance"] for r in rev_results]
        lines.append(
            f"  Reversal cosine distances: "
            f"min={min(cos_dists):.6f}, max={max(cos_dists):.6f}, "
            f"mean={np.mean(cos_dists):.6f}, std={np.std(cos_dists):.6f}"
        )
        if max(cos_dists) < 1e-6:
            lines.append("  → Model is ORDER-INSENSITIVE (bag-of-residues baseline?)")
        elif min(cos_dists) > 0.01:
            lines.append("  → Model is ORDER-SENSITIVE (positional encoding matters)")
        else:
            lines.append("  → Mixed sensitivity — depends on sequence")

    dbl_results = [r for r in results if r["test_type"] == "double_reversal"]
    if dbl_results:
        all_pass = all(r["passed"] for r in dbl_results)
        lines.append(
            f"  Double-reversal identity: {'ALL PASS' if all_pass else 'SOME FAILED'}"
        )

    lines.append("")
    return "\n".join(lines)


class TestReversedSequenceESM2:
    def test_reversal_significantly_different(
        self,
        esm2_embedder,
        extended_sequences: List[str],
        reports_dir,
    ):
        results = _run_reversal_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=extended_sequences,  # Use all extended sequences for substantial output
        )

        table = format_metrics_table(results, title="Reversed Sequence — ESM2-T6-8M")
        print(table)
        write_metrics_csv(results, reports_dir / "reversed_sequence_esm2.csv")

        for r in results:
            print(
                f"  {r['parameter']}: cosine_dist={r['cosine_distance']:.6f}, "
                f"l2={r['l2_distance']:.4f}"
            )
            # Skip assertion for palindromic or highly repetitive sequences
            # These have symmetric/near-symmetric patterns when reversed
            skip = r.get("is_palindrome", False) or r.get("is_repetitive", False)
            if not skip:
                assert r["cosine_distance"] > 0.001, (
                    f"ESM2 should be order-sensitive but cosine_dist "
                    f"is very small ({r['cosine_distance']:.8f})"
                )

    def test_double_reversal_is_identity(
        self,
        esm2_embedder,
        extended_sequences: List[str],
    ):
        results = _run_double_reversal_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=extended_sequences,
            tolerance=1e-5,
        )

        for r in results:
            assert r["passed"], (
                f"Double-reversal identity FAILED for {r['parameter']}: "
                f"cosine_dist={r['cosine_distance']:.10f}"
            )

    def test_reversal_summary_and_report(
        self,
        esm2_embedder,
        extended_sequences: List[str],
        reports_dir,
    ):
        rev_results = _run_reversal_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=extended_sequences,
        )
        dbl_results = _run_double_reversal_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=extended_sequences,
            tolerance=1e-5,
        )

        all_results = rev_results + dbl_results
        print(_summarise_reversal(all_results, "esm2_t6_8m (diverse)"))
        write_metrics_csv(
            all_results, reports_dir / "reversed_sequence_esm2_diverse.csv"
        )
