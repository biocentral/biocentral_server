# Amino acid mutation sensitivity: replace residues with each of the 19 other
# standard amino acids (instead of 'X') and measure embedding divergence.
#
# This experiment reuses the same masking-ratio framework as X-masking, but the
# replacement character is a real amino acid.  This lets us study how sensitive
# the model is to specific substitutions, and how that compares to X-masking.

import csv
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from tests.property.oracles.embedding_metrics import compute_all_metrics
from tests.scripts.conftest import AMINO_ACIDS

MASKING_RATIOS = [
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0,
]

BASE_SEED = 42
N_RUNS = 5  # Per-sequence repetitions (power from many seqs × many AAs)

# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def _mutate_sequence_progressive(
    sequence: str,
    ratio: float,
    replacement_aa: str,
    seed: int,
    prev_positions: Set[int],
) -> Tuple[str, Set[int]]:
    if ratio <= 0.0:
        return sequence, set()
    if ratio >= 1.0:
        return replacement_aa * len(sequence), set(range(len(sequence)))

    rng = random.Random(seed)
    target_n = max(1, int(len(sequence) * ratio))
    n_new = target_n - len(prev_positions)

    if n_new > 0:
        available = list(set(range(len(sequence))) - prev_positions)
        rng.shuffle(available)
        new_pos = set(available[: min(n_new, len(available))])
        positions = prev_positions | new_pos
    else:
        positions = prev_positions

    seq_list = list(sequence)
    for pos in positions:
        seq_list[pos] = replacement_aa
    return "".join(seq_list), positions

# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def _run_mutation_experiment(
    embedder,
    embedder_label: str,
    sequences: List[str],
    replacement_aas: List[str],
    masking_ratios: List[float] = MASKING_RATIOS,
    n_runs: int = N_RUNS,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for seq_idx, seq in enumerate(sequences):
        original_emb = embedder.embed_pooled(seq)

        for aa in replacement_aas:
            for ratio in masking_ratios:
                cosine_dists = []
                l2_dists = []

                for run_idx in range(n_runs):
                    seed = BASE_SEED + seq_idx * 10000 + ord(aa) * 100 + run_idx
                    prev_positions: Set[int] = set()

                    # Build up progressively
                    for r in masking_ratios:
                        if r > ratio:
                            break
                        mutated, prev_positions = _mutate_sequence_progressive(
                            seq, r, aa, seed=seed, prev_positions=prev_positions,
                        )

                    mutated_emb = embedder.embed_pooled(mutated)
                    metrics = compute_all_metrics(original_emb, mutated_emb)
                    cosine_dists.append(metrics["cosine_distance"])
                    l2_dists.append(metrics["l2_distance"])

                results.append({
                    "embedder": embedder_label,
                    "test_type": "aa_mutation",
                    "parameter": f"seq{seq_idx}_aa{aa}_mut{int(ratio * 100)}%",
                    "seq_idx": seq_idx,
                    "replacement_aa": aa,
                    "masking_ratio": ratio,
                    "cosine_distance": float(np.mean(cosine_dists)),
                    "cosine_std": float(np.std(cosine_dists, ddof=1)) if n_runs > 1 else 0.0,
                    "l2_distance": float(np.mean(l2_dists)),
                    "l2_std": float(np.std(l2_dists, ddof=1)) if n_runs > 1 else 0.0,
                    "n_runs": n_runs,
                    "sequence_length": len(seq),
                })

    return results

# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_MUTATION_CSV_FIELDS = [
    "embedder", "test_type", "parameter", "seq_idx", "replacement_aa",
    "masking_ratio", "cosine_distance", "cosine_std", "l2_distance", "l2_std",
    "n_runs", "sequence_length", "bin",
]

def _write_mutation_csv(results: List[Dict[str, Any]], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MUTATION_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(results, key=lambda r: (
            r.get("replacement_aa", ""), r.get("masking_ratio", 0), r.get("seq_idx", 0),
        )):
            writer.writerow({k: row.get(k, "") for k in _MUTATION_CSV_FIELDS})

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_mutation_summary(results: List[Dict[str, Any]], title: str = "") -> str:
    lines = [f"\n{'=' * 80}", f"  {title}", f"{'=' * 80}"]

    header = f"{'AA':>3} | {'10% cos':>10} | {'50% cos':>10} | {'100% cos':>10} | {'10% L2':>10} | {'50% L2':>10} | {'100% L2':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    aas = sorted({r["replacement_aa"] for r in results})
    for aa in aas:
        vals = {}
        for ratio in [0.10, 0.50, 1.0]:
            subset = [r for r in results if r["replacement_aa"] == aa and abs(r["masking_ratio"] - ratio) < 0.01]
            if subset:
                vals[f"cos_{ratio}"] = np.mean([r["cosine_distance"] for r in subset])
                vals[f"l2_{ratio}"] = np.mean([r["l2_distance"] for r in subset])
            else:
                vals[f"cos_{ratio}"] = float("nan")
                vals[f"l2_{ratio}"] = float("nan")

        lines.append(
            f"  {aa} | {vals['cos_0.1']:10.4f} | {vals['cos_0.5']:10.4f} | {vals['cos_1.0']:10.4f} "
            f"| {vals['l2_0.1']:10.4f} | {vals['l2_0.5']:10.4f} | {vals['l2_1.0']:10.4f}"
        )

    lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

# Small set for the basic experiment (same 4 sequences as original masking test)
_BASIC_TEST_SEQUENCES = {
    "short_15": "MKTAYIAKQRQISFV",
    "medium_76": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "long_400": "MKTAYIAK" * 50,
    "very_long_1000": "ACDEFGHIKLMNPQRSTVWY" * 50,
}

class TestAAMutationSensitivity:

    def test_all_aa_mutations(self, esm2_embedder, reports_dir):
        sequences = list(_BASIC_TEST_SEQUENCES.values())

        results = _run_mutation_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=sequences,
            replacement_aas=AMINO_ACIDS,
        )

        summary = _format_mutation_summary(
            results, title="AA Mutation Sensitivity — ESM2-T6-8M (basic sequences)"
        )
        print(summary)

        _write_mutation_csv(results, reports_dir / "aa_mutation_sensitivity_esm2.csv")
        print(f"Wrote {len(results)} rows to aa_mutation_sensitivity_esm2.csv")

class TestAAMutationUniRef50:

    def test_uniref50_mutation(self, esm2_embedder, reports_dir):
        try:
            from tests.scripts.fetch_uniref50_sequences import load_uniref50_sequences
        except ImportError:
            import pytest
            pytest.skip("UniRef50 FASTA not available")

        try:
            bin_seqs = load_uniref50_sequences()
        except FileNotFoundError:
            import pytest
            pytest.skip("UniRef50 FASTA not found – run fetch_uniref50_sequences.py first")

        all_results: List[Dict[str, Any]] = []

        for bin_target in sorted(bin_seqs.keys()):
            sequences = bin_seqs[bin_target]
            if not sequences:
                continue

            print(f"\n[AA Mutation UniRef50] bin={bin_target}, n_seqs={len(sequences)}")

            results = _run_mutation_experiment(
                embedder=esm2_embedder,
                embedder_label="esm2_t6_8m",
                sequences=sequences,
                replacement_aas=AMINO_ACIDS,
                n_runs=3,  # Reduced; many seqs × 20 AAs gives plenty of power
            )

            for r in results:
                r["bin"] = bin_target

            all_results.extend(results)

        _write_mutation_csv(all_results, reports_dir / "aa_mutation_sensitivity_uniref50.csv")

        summary = _format_mutation_summary(
            all_results, title="AA Mutation Sensitivity — ESM2-T6-8M (UniRef50)"
        )
        print(summary)
        print(f"\n[AA Mutation UniRef50] Total rows: {len(all_results)}")
