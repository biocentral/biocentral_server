# MR: Progressive X-masking — replace residues with 'X' and measure embedding divergence.

import random
from typing import Any, Dict, List, Optional

from tests.property.oracles.embedding_metrics import (
    compute_all_metrics,
    format_metrics_table,
    write_metrics_csv,
)


# Masking ratios to explore (0 % → 100 %)
MASKING_RATIOS = [
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.0,
]

# Cosine-distance threshold that we consider "significant divergence"
SIGNIFICANCE_THRESHOLD = 0.1

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_sequence(sequence: str, ratio: float, seed: int = SEED) -> str:
    # Replace ratio fraction of residues with 'X'.
    if ratio <= 0.0:
        return sequence
    if ratio >= 1.0:
        return "X" * len(sequence)

    rng = random.Random(seed)
    n_mask = max(1, int(len(sequence) * ratio))
    positions = rng.sample(range(len(sequence)), min(n_mask, len(sequence)))
    seq_list = list(sequence)
    for pos in positions:
        seq_list[pos] = "X"
    return "".join(seq_list)


def _run_masking_experiment(
    embedder,
    embedder_label: str,
    sequences: List[str],
    masking_ratios: List[float] = MASKING_RATIOS,
    significance_threshold: float = SIGNIFICANCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    # Run the progressive X-masking experiment and return per-step metrics.
    results: List[Dict[str, Any]] = []

    for seq_idx, seq in enumerate(sequences):
        original_emb = embedder.embed_pooled(seq)
        prev_cosine = 0.0

        for ratio in masking_ratios:
            masked_seq = _mask_sequence(seq, ratio, seed=SEED + seq_idx)
            masked_emb = embedder.embed_pooled(masked_seq)

            metrics = compute_all_metrics(original_emb, masked_emb)

            # Check monotonicity w.r.t. previous ratio
            monotonic = metrics["cosine_distance"] >= prev_cosine - 1e-9

            results.append(
                {
                    "embedder": embedder_label,
                    "test_type": "x_masking",
                    "parameter": f"seq{seq_idx}_mask{int(ratio * 100)}%",
                    "masking_ratio": ratio,
                    "cosine_distance": metrics["cosine_distance"],
                    "l2_distance": metrics["l2_distance"],
                    "kl_divergence": metrics["kl_divergence"],
                    "threshold": significance_threshold,
                    "significant": metrics["cosine_distance"] > significance_threshold,
                    "monotonic": monotonic,
                    "passed": True,  # exploratory — always pass
                    "sequence_length": len(seq),
                    "masked_sequence": masked_seq[:40]
                    + ("..." if len(masked_seq) > 40 else ""),
                }
            )

            prev_cosine = metrics["cosine_distance"]

    return results


def _find_critical_ratio(
    results: List[Dict[str, Any]], seq_idx: int
) -> Optional[float]:
    # Find the smallest masking ratio where cosine distance crosses the significance threshold.
    seq_results = [r for r in results if r["parameter"].startswith(f"seq{seq_idx}_")]
    for r in sorted(seq_results, key=lambda x: x["masking_ratio"]):
        if r["significant"]:
            return r["masking_ratio"]
    return None


def _summarise_experiment(results: List[Dict[str, Any]], embedder_label: str) -> str:
    # Produce a human-readable summary of the masking experiment.
    lines = [
        f"\n{'=' * 80}",
        f"  Progressive X-Masking Summary — {embedder_label}",
        f"{'=' * 80}",
    ]

    # Identify unique sequences
    seq_indices = sorted(
        {int(r["parameter"].split("_")[0].replace("seq", "")) for r in results}
    )

    for seq_idx in seq_indices:
        critical = _find_critical_ratio(results, seq_idx)
        seq_len = next(
            r["sequence_length"]
            for r in results
            if r["parameter"].startswith(f"seq{seq_idx}_")
        )
        # Check monotonicity violations
        seq_results = [
            r for r in results if r["parameter"].startswith(f"seq{seq_idx}_")
        ]
        violations = sum(1 for r in seq_results if not r["monotonic"])

        lines.append(
            f"  seq{seq_idx} (len={seq_len}): critical_ratio={critical or 'N/A'}, "
            f"monotonicity_violations={violations}/{len(seq_results)}"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ESM2 experiments
# ---------------------------------------------------------------------------


class TestProgressiveXMaskingESM2:
    # Characterise real ESM2-T6-8M's sensitivity to progressive X-masking.

    def test_masking_divergence_profile(
        self,
        esm2_embedder,
        standard_sequences: List[str],
        reports_dir,
    ):
        results = _run_masking_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=standard_sequences[:2],  # keep runtime manageable
            significance_threshold=SIGNIFICANCE_THRESHOLD,
        )

        table = format_metrics_table(results, title="X-Masking — ESM2-T6-8M")
        print(table)
        print(_summarise_experiment(results, "esm2_t6_8m"))
        write_metrics_csv(results, reports_dir / "x_masking_esm2.csv")

    def test_monotonicity_mostly_holds(
        self,
        esm2_embedder,
        standard_sequences: List[str],
    ):
        results = _run_masking_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=standard_sequences[:2],
        )
        total = len([r for r in results if r["masking_ratio"] > 0])
        violations = sum(
            1 for r in results if r["masking_ratio"] > 0 and not r["monotonic"]
        )
        ratio = violations / total if total > 0 else 0

        print(f"\n[ESM2 Monotonicity] violations = {violations}/{total} ({ratio:.1%})")
        # More lenient for a real model: allow up to 30 % violations
        assert ratio <= 0.30, (
            f"Monotonicity violated too often: {violations}/{total} ({ratio:.1%})"
        )
