# Random sequence baseline: generate random AA sequences and run the same masking experiments.

import random
from typing import Any, Dict, List

from tests.scripts.test_progressive_x_masking import (
    _run_progressive_masking_experiment,
    _run_random_masking_experiment,
    _write_masking_csv,
)
from tests.scripts.conftest import AMINO_ACIDS

# Match the length bins used in the original experiment
RANDOM_SEQ_LENGTHS = [50, 100, 200, 400, 600, 800, 1000]
N_RANDOM_SEQS_PER_LENGTH = 250
RANDOM_SEED = 12345

def _generate_random_sequences(
    lengths: List[int],
    n_per_length: int,
    seed: int = RANDOM_SEED,
) -> Dict[int, List[str]]:
    rng = random.Random(seed)
    result: Dict[int, List[str]] = {}
    for length in lengths:
        seqs = []
        for _ in range(n_per_length):
            seq = "".join(rng.choices(AMINO_ACIDS, k=length))
            seqs.append(seq)
        result[length] = seqs
    return result

class TestXMaskingRandomSequences:

    def test_random_seq_masking(self, esm2_embedder, reports_dir):
        random_seqs = _generate_random_sequences(
            lengths=RANDOM_SEQ_LENGTHS,
            n_per_length=N_RANDOM_SEQS_PER_LENGTH,
        )

        all_progressive: List[Dict[str, Any]] = []
        all_random: List[Dict[str, Any]] = []

        for seq_len in sorted(random_seqs.keys()):
            sequences = random_seqs[seq_len]
            print(f"\n[Random Seqs] length={seq_len}, n_seqs={len(sequences)}")

            prog = _run_progressive_masking_experiment(
                embedder=esm2_embedder,
                embedder_label="esm2_t6_8m",
                sequences=sequences,
                n_runs=5,  # Fewer runs; power from many sequences
            )
            rand = _run_random_masking_experiment(
                embedder=esm2_embedder,
                embedder_label="esm2_t6_8m",
                sequences=sequences,
                n_runs=5,
            )

            for r in prog + rand:
                r["bin"] = seq_len

            all_progressive.extend(prog)
            all_random.extend(rand)

        _write_masking_csv(
            all_progressive, reports_dir / "x_masking_progressive_random_seqs.csv"
        )
        _write_masking_csv(
            all_random, reports_dir / "x_masking_random_random_seqs.csv"
        )

        print(f"\n[Random Seqs] Progressive results: {len(all_progressive)} rows")
        print(f"[Random Seqs] Random results:      {len(all_random)} rows")
