# MR: Progressive X-masking — replace residues with 'X' and measure embedding divergence.

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from tests.property.oracles.embedding_metrics import (
    compute_all_metrics,
    format_metrics_table,
    write_metrics_csv,
)


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

SIGNIFICANCE_THRESHOLD = 0.1

SEED = 42


def _mask_sequence_cumulative(
    sequence: str,
    ratio: float,
    seed: int,
    prev_positions: Set[int],
) -> Tuple[str, Set[int]]:
    if ratio <= 0.0:
        return sequence, set()
    if ratio >= 1.0:
        return "X" * len(sequence), set(range(len(sequence)))

    rng = random.Random(seed)
    all_positions = set(range(len(sequence)))
    target_n_masked = max(1, int(len(sequence) * ratio))
    n_new_to_mask = target_n_masked - len(prev_positions)

    if n_new_to_mask > 0:
        available = list(all_positions - prev_positions)
        rng.shuffle(available)
        new_positions = set(available[:min(n_new_to_mask, len(available))])
        positions = prev_positions | new_positions
    else:
        positions = prev_positions

    seq_list = list(sequence)
    for pos in positions:
        seq_list[pos] = "X"
    return "".join(seq_list), positions


def _run_masking_experiment(
    embedder,
    embedder_label: str,
    sequences: List[str],
    masking_ratios: List[float] = MASKING_RATIOS,
    significance_threshold: float = SIGNIFICANCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for seq_idx, seq in enumerate(sequences):
        original_emb = embedder.embed_pooled(seq)
        prev_cosine = 0.0
        prev_positions: Set[int] = set()

        for ratio in masking_ratios:
            masked_seq, prev_positions = _mask_sequence_cumulative(
                seq, ratio, seed=SEED + seq_idx, prev_positions=prev_positions
            )
            masked_emb = embedder.embed_pooled(masked_seq)

            metrics = compute_all_metrics(original_emb, masked_emb)

            monotonic = metrics["cosine_distance"] >= prev_cosine - 1e-9

            results.append(
                {
                    "embedder": embedder_label,
                    "test_type": "x_masking",
                    "parameter": f"seq{seq_idx}_mask{int(ratio * 100)}%",
                    "masking_ratio": ratio,
                    "cosine_distance": metrics["cosine_distance"],
                    "l2_distance": metrics["l2_distance"],
                    "threshold": significance_threshold,
                    "significant": metrics["cosine_distance"] > significance_threshold,
                    "monotonic": monotonic,
                    "passed": True,
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
    seq_results = [r for r in results if r["parameter"].startswith(f"seq{seq_idx}_")]
    for r in sorted(seq_results, key=lambda x: x["masking_ratio"]):
        if r["significant"]:
            return r["masking_ratio"]
    return None


def _summarise_experiment(results: List[Dict[str, Any]], embedder_label: str) -> str:
    lines = [
        f"\n{'=' * 80}",
        f"  Progressive X-Masking Summary — {embedder_label}",
        f"{'=' * 80}",
    ]

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


class TestProgressiveXMaskingESM2:

    def test_masking_divergence_profile(
        self,
        esm2_embedder,
        standard_sequences: List[str],
        reports_dir,
    ):
        results = _run_masking_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=standard_sequences[:2],
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
        assert ratio <= 0.30, (
            f"Monotonicity violated too often: {violations}/{total} ({ratio:.1%})"
        )
