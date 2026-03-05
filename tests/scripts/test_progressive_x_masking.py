# MR: Progressive X-masking — replace residues with 'X' and measure embedding divergence.

import csv
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tests.property.oracles.embedding_metrics import (
    compute_all_metrics,
)

MASKING_RATIOS = [
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.0,
]

SIGNIFICANCE_THRESHOLD = 0.1

BASE_SEED = 42
N_RUNS = 30  # Number of repetitions

def _mask_sequence_progressive(
    sequence: str,
    ratio: float,
    seed: int,
    prev_positions: Set[int],
) -> Tuple[str, Set[int]]:
    # Progressive masking: each level contains all positions from lower levels.
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
        new_positions = set(available[: min(n_new_to_mask, len(available))])
        positions = prev_positions | new_positions
    else:
        positions = prev_positions

    seq_list = list(sequence)
    for pos in positions:
        seq_list[pos] = "X"
    return "".join(seq_list), positions

def _mask_sequence_random(
    sequence: str,
    ratio: float,
    seed: int,
) -> str:
    # Random masking: each level is completely independent.
    if ratio <= 0.0:
        return sequence
    if ratio >= 1.0:
        return "X" * len(sequence)

    rng = random.Random(seed)
    n_to_mask = max(1, int(len(sequence) * ratio))
    positions = rng.sample(range(len(sequence)), n_to_mask)

    seq_list = list(sequence)
    for pos in positions:
        seq_list[pos] = "X"
    return "".join(seq_list)

def _run_progressive_masking_experiment(
    embedder,
    embedder_label: str,
    sequences: List[str],
    masking_ratios: List[float] = MASKING_RATIOS,
    significance_threshold: float = SIGNIFICANCE_THRESHOLD,
    n_runs: int = N_RUNS,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for seq_idx, seq in enumerate(sequences):
        original_emb = embedder.embed_pooled(seq)

        for ratio in masking_ratios:
            cosine_distances = []
            l2_distances = []

            for run_idx in range(n_runs):
                # For progressive masking, we need to build up from 0%
                seed = BASE_SEED + seq_idx * 1000 + run_idx
                prev_positions: Set[int] = set()

                # Build up progressively through all ratios up to current
                for r in masking_ratios:
                    if r > ratio:
                        break
                    masked_seq, prev_positions = _mask_sequence_progressive(
                        seq, r, seed=seed, prev_positions=prev_positions
                    )

                masked_emb = embedder.embed_pooled(masked_seq)
                metrics = compute_all_metrics(original_emb, masked_emb)
                cosine_distances.append(metrics["cosine_distance"])
                l2_distances.append(metrics["l2_distance"])

            mean_cosine = np.mean(cosine_distances)
            std_cosine = np.std(cosine_distances, ddof=1) if n_runs > 1 else 0.0
            mean_l2 = np.mean(l2_distances)
            std_l2 = np.std(l2_distances, ddof=1) if n_runs > 1 else 0.0

            results.append(
                {
                    "embedder": embedder_label,
                    "test_type": "x_masking_progressive",
                    "parameter": f"seq{seq_idx}_mask{int(ratio * 100)}%",
                    "masking_ratio": ratio,
                    "cosine_distance": mean_cosine,
                    "cosine_std": std_cosine,
                    "l2_distance": mean_l2,
                    "l2_std": std_l2,
                    "threshold": significance_threshold,
                    "significant": mean_cosine > significance_threshold,
                    "n_runs": n_runs,
                    "passed": "N/A",
                    "sequence_length": len(seq),
                }
            )

    # Check monotonicity
    for seq_idx in range(len(sequences)):
        seq_results = [
            r for r in results if r["parameter"].startswith(f"seq{seq_idx}_")
        ]
        seq_results_sorted = sorted(seq_results, key=lambda x: x["masking_ratio"])
        prev_cosine = 0.0
        for r in seq_results_sorted:
            r["monotonic"] = r["cosine_distance"] >= prev_cosine - 1e-9
            prev_cosine = r["cosine_distance"]

    return results

def _run_random_masking_experiment(
    embedder,
    embedder_label: str,
    sequences: List[str],
    masking_ratios: List[float] = MASKING_RATIOS,
    significance_threshold: float = SIGNIFICANCE_THRESHOLD,
    n_runs: int = N_RUNS,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for seq_idx, seq in enumerate(sequences):
        original_emb = embedder.embed_pooled(seq)

        for ratio in masking_ratios:
            cosine_distances = []
            l2_distances = []

            for run_idx in range(n_runs):
                seed = BASE_SEED + seq_idx * 1000 + int(ratio * 100) * 10 + run_idx
                masked_seq = _mask_sequence_random(seq, ratio, seed=seed)
                masked_emb = embedder.embed_pooled(masked_seq)

                metrics = compute_all_metrics(original_emb, masked_emb)
                cosine_distances.append(metrics["cosine_distance"])
                l2_distances.append(metrics["l2_distance"])

            mean_cosine = np.mean(cosine_distances)
            std_cosine = np.std(cosine_distances, ddof=1) if n_runs > 1 else 0.0
            mean_l2 = np.mean(l2_distances)
            std_l2 = np.std(l2_distances, ddof=1) if n_runs > 1 else 0.0

            results.append(
                {
                    "embedder": embedder_label,
                    "test_type": "x_masking_random",
                    "parameter": f"seq{seq_idx}_mask{int(ratio * 100)}%",
                    "masking_ratio": ratio,
                    "cosine_distance": mean_cosine,
                    "cosine_std": std_cosine,
                    "l2_distance": mean_l2,
                    "l2_std": std_l2,
                    "threshold": significance_threshold,
                    "significant": mean_cosine > significance_threshold,
                    "n_runs": n_runs,
                    "passed": "N/A",
                    "sequence_length": len(seq),
                }
            )

    # Check monotonicity (expected to have more violations for random)
    for seq_idx in range(len(sequences)):
        seq_results = [
            r for r in results if r["parameter"].startswith(f"seq{seq_idx}_")
        ]
        seq_results_sorted = sorted(seq_results, key=lambda x: x["masking_ratio"])
        prev_cosine = 0.0
        for r in seq_results_sorted:
            r["monotonic"] = r["cosine_distance"] >= prev_cosine - 1e-9
            prev_cosine = r["cosine_distance"]

    return results

def _find_critical_ratio(
    results: List[Dict[str, Any]], seq_idx: int
) -> Optional[float]:
    seq_results = [r for r in results if r["parameter"].startswith(f"seq{seq_idx}_")]
    for r in sorted(seq_results, key=lambda x: x["masking_ratio"]):
        if r["significant"]:
            return r["masking_ratio"]
    return None

def _format_masking_table(
    results: List[Dict[str, Any]], title: Optional[str] = None
) -> str:
    if not results:
        return "No results to display."

    headers = [
        "Seq",
        "Mask%",
        "Cosine (mean±std)",
        "L2 (mean±std)",
        "Significant",
        "Monotonic",
    ]
    col_widths = [4, 6, 20, 20, 11, 9]

    lines = []
    if title:
        lines.append(f"\n{'=' * 80}")
        lines.append(f"  {title}")
        lines.append(f"{'=' * 80}")

    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_row)
    lines.append("-" * len(header_row))

    # Sort by seq index then masking ratio
    sorted_results = sorted(
        results,
        key=lambda r: (
            int(r["parameter"].split("_")[0].replace("seq", "")),
            r["masking_ratio"],
        ),
    )

    for row in sorted_results:
        seq_idx = row["parameter"].split("_")[0].replace("seq", "")
        mask_pct = f"{int(row['masking_ratio'] * 100)}%"
        cosine_str = f"{row['cosine_distance']:.4f}±{row.get('cosine_std', 0):.4f}"
        l2_str = f"{row['l2_distance']:.4f}±{row.get('l2_std', 0):.4f}"
        sig = "✓" if row.get("significant", False) else ""
        mono = "✓" if row.get("monotonic", True) else "✗"

        values = [seq_idx, mask_pct, cosine_str, l2_str, sig, mono]
        data_row = " | ".join(v.ljust(w) for v, w in zip(values, col_widths))
        lines.append(data_row)

    lines.append("")
    return "\n".join(lines)

def _summarise_experiment(
    results: List[Dict[str, Any]],
    embedder_label: str,
    masking_type: str = "Progressive",
) -> str:
    lines = [
        f"\n{'=' * 80}",
        f"  {masking_type} X-Masking Summary — {embedder_label}",
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

# Test sequences of varying lengths for length-dependent analysis
TEST_SEQUENCES_BY_LENGTH = {
    "short_15": "MKTAYIAKQRQISFV",  # 15 aa
    "medium_76": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",  # Ubiquitin 76 aa
    "long_400": "MKTAYIAK" * 50,  # 400 aa
    "very_long_1000": "ACDEFGHIKLMNPQRSTVWY" * 50,  # 1000 aa
}

class TestProgressiveXMaskingESM2:
    def test_masking_divergence_profile(
        self,
        esm2_embedder,
        standard_sequences: List[str],
        reports_dir,
    ):
        # Use sequences of different lengths for comprehensive analysis
        test_sequences = list(TEST_SEQUENCES_BY_LENGTH.values())
        seq_labels = list(TEST_SEQUENCES_BY_LENGTH.keys())

        # Run progressive masking (each level contains previous as subset)
        progressive_results = _run_progressive_masking_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=test_sequences,
            significance_threshold=SIGNIFICANCE_THRESHOLD,
        )

        # Add sequence length labels to results
        for r in progressive_results:
            seq_idx = int(r["parameter"].split("_")[0].replace("seq", ""))
            r["seq_label"] = seq_labels[seq_idx]

        table = _format_masking_table(
            progressive_results,
            title=f"Progressive X-Masking — ESM2-T6-8M (n={N_RUNS} runs)",
        )
        print(table)
        print(_summarise_experiment(progressive_results, "esm2_t6_8m", "Progressive"))
        _write_masking_csv(
            progressive_results, reports_dir / "x_masking_progressive_esm2.csv"
        )

        # Run random masking (each level is independent)
        random_results = _run_random_masking_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=test_sequences,
            significance_threshold=SIGNIFICANCE_THRESHOLD,
        )

        # Add sequence length labels to results
        for r in random_results:
            seq_idx = int(r["parameter"].split("_")[0].replace("seq", ""))
            r["seq_label"] = seq_labels[seq_idx]

        table = _format_masking_table(
            random_results, title=f"Random X-Masking — ESM2-T6-8M (n={N_RUNS} runs)"
        )
        print(table)
        print(_summarise_experiment(random_results, "esm2_t6_8m", "Random"))
        _write_masking_csv(random_results, reports_dir / "x_masking_random_esm2.csv")

    def test_monotonicity_mostly_holds(
        self,
        esm2_embedder,
        standard_sequences: List[str],
    ):
        test_sequences = list(TEST_SEQUENCES_BY_LENGTH.values())

        # Progressive masking should have better monotonicity
        progressive_results = _run_progressive_masking_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=test_sequences,
        )
        total = len([r for r in progressive_results if r["masking_ratio"] > 0])
        violations = sum(
            1
            for r in progressive_results
            if r["masking_ratio"] > 0 and not r["monotonic"]
        )
        ratio = violations / total if total > 0 else 0

        print(
            f"\n[ESM2 Progressive Monotonicity] violations = {violations}/{total} ({ratio:.1%})"
        )

        # Random masking may have more violations (positions change each level)
        random_results = _run_random_masking_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=test_sequences,
        )
        total_rand = len([r for r in random_results if r["masking_ratio"] > 0])
        violations_rand = sum(
            1 for r in random_results if r["masking_ratio"] > 0 and not r["monotonic"]
        )
        ratio_rand = violations_rand / total_rand if total_rand > 0 else 0

        print(
            f"[ESM2 Random Monotonicity] violations = {violations_rand}/{total_rand} ({ratio_rand:.1%})"
        )
        # Random masking allowed more violations since positions are independent
        assert ratio_rand <= 0.40, (
            f"Random monotonicity violated too often: {violations_rand}/{total_rand} ({ratio_rand:.1%})"
        )

# ---------------------------------------------------------------------------
# Extended CSV writer that preserves all result columns
# ---------------------------------------------------------------------------

_MASKING_CSV_FIELDS = [
    "embedder",
    "test_type",
    "parameter",
    "masking_ratio",
    "cosine_distance",
    "cosine_std",
    "l2_distance",
    "l2_std",
    "threshold",
    "significant",
    "n_runs",
    "passed",
    "sequence_length",
]

def _write_masking_csv(results: List[Dict[str, Any]], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MASKING_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(results, key=lambda r: (r.get("parameter", ""), r.get("masking_ratio", 0))):
            writer.writerow({k: row.get(k, "") for k in _MASKING_CSV_FIELDS})

# ---------------------------------------------------------------------------
# UniRef50-scale experiment (250 seqs/bin, 7 bins)
# ---------------------------------------------------------------------------

N_RUNS_UNIREF = 5  # Fewer repetitions; power comes from many sequences

class TestXMaskingUniRef50:

    def test_uniref50_masking(self, esm2_embedder, reports_dir):
        try:
            from tests.scripts.fetch_uniref50_sequences import load_uniref50_sequences
        except ImportError:
            import pytest
            pytest.skip("UniRef50 FASTA not available – run fetch_uniref50_sequences.py first")

        try:
            bin_seqs = load_uniref50_sequences()
        except FileNotFoundError:
            import pytest
            pytest.skip("UniRef50 FASTA not found – run fetch_uniref50_sequences.py first")

        all_progressive = []
        all_random = []

        for bin_target in sorted(bin_seqs.keys()):
            sequences = bin_seqs[bin_target]
            if not sequences:
                continue

            print(f"\n[UniRef50] bin={bin_target}, n_seqs={len(sequences)}")

            prog = _run_progressive_masking_experiment(
                embedder=esm2_embedder,
                embedder_label="esm2_t6_8m",
                sequences=sequences,
                n_runs=N_RUNS_UNIREF,
            )
            rand = _run_random_masking_experiment(
                embedder=esm2_embedder,
                embedder_label="esm2_t6_8m",
                sequences=sequences,
                n_runs=N_RUNS_UNIREF,
            )

            # Tag results with bin info
            for r in prog + rand:
                r["bin"] = bin_target

            all_progressive.extend(prog)
            all_random.extend(rand)

        _write_masking_csv(all_progressive, reports_dir / "x_masking_progressive_uniref50.csv")
        _write_masking_csv(all_random, reports_dir / "x_masking_random_uniref50.csv")

        print(f"\n[UniRef50] Progressive results: {len(all_progressive)} rows")
        print(f"[UniRef50] Random results:      {len(all_random)} rows")

# ---------------------------------------------------------------------------
# One-hot encoding baseline
# ---------------------------------------------------------------------------

class TestXMaskingOneHot:

    def test_one_hot_masking(self, one_hot_embedder, reports_dir):
        test_sequences = list(TEST_SEQUENCES_BY_LENGTH.values())
        seq_labels = list(TEST_SEQUENCES_BY_LENGTH.keys())

        progressive_results = _run_progressive_masking_experiment(
            embedder=one_hot_embedder,
            embedder_label="one_hot_encoding",
            sequences=test_sequences,
        )
        for r in progressive_results:
            seq_idx = int(r["parameter"].split("_")[0].replace("seq", ""))
            r["seq_label"] = seq_labels[seq_idx]

        table = _format_masking_table(
            progressive_results,
            title=f"Progressive X-Masking — One-Hot Encoding (n={N_RUNS} runs)",
        )
        print(table)
        _write_masking_csv(progressive_results, reports_dir / "x_masking_progressive_one_hot.csv")

        random_results = _run_random_masking_experiment(
            embedder=one_hot_embedder,
            embedder_label="one_hot_encoding",
            sequences=test_sequences,
        )
        for r in random_results:
            seq_idx = int(r["parameter"].split("_")[0].replace("seq", ""))
            r["seq_label"] = seq_labels[seq_idx]

        table = _format_masking_table(
            random_results,
            title=f"Random X-Masking — One-Hot Encoding (n={N_RUNS} runs)",
        )
        print(table)
        _write_masking_csv(random_results, reports_dir / "x_masking_random_one_hot.csv")
