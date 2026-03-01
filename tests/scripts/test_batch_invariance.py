# Batch invariance: embedding A+B together must yield the same per-sequence embedding as embedding them individually.

import random
import hashlib
from typing import Any, Dict, List

from tests.property.oracles.embedding_metrics import (
    compute_all_metrics,
    format_metrics_table,
    write_metrics_csv,
)


BATCH_SIZES = [2, 5, 10, 20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_batch(
    target: str,
    fillers: List[str],
    batch_size: int,
    seed: int,
) -> List[str]:
    # Create a batch of batch_size sequences with target at a random position.
    if batch_size <= 1:
        return [target]

    rng = random.Random(seed)
    n_fillers = batch_size - 1
    selected = [fillers[i % len(fillers)] for i in range(n_fillers)]
    pos = rng.randint(0, len(selected))
    selected.insert(pos, target)
    return selected


def _run_batch_invariance(
    embedder,
    embedder_label: str,
    target_sequences: List[str],
    filler_sequences: List[str],
    batch_sizes: List[int] = BATCH_SIZES,
    tolerance: float = 1e-6,
) -> List[Dict[str, Any]]:
    results = []

    for seq_idx, target in enumerate(target_sequences):
        # Reference: embed alone
        ref_emb = embedder.embed_pooled(target)

        for bs in batch_sizes:
            seed = int.from_bytes(
                hashlib.sha256(f"{embedder_label}:{seq_idx}:{bs}".encode()).digest()[
                    :4
                ],
                "big",
            )
            batch = _build_batch(target, filler_sequences, bs, seed)
            target_pos = batch.index(target)

            batch_embs = embedder.embed_batch(batch, pooled=True)
            batched_emb = batch_embs[target_pos]

            metrics = compute_all_metrics(ref_emb, batched_emb)

            results.append(
                {
                    "embedder": embedder_label,
                    "test_type": "batch_invariance",
                    "parameter": f"seq{seq_idx}_batch{bs}",
                    "cosine_distance": metrics["cosine_distance"],
                    "l2_distance": metrics["l2_distance"],
                    "kl_divergence": metrics["kl_divergence"],
                    "threshold": tolerance,
                    "passed": metrics["cosine_distance"] <= tolerance,
                    "sequence_length": len(target),
                    "batch_size": bs,
                    "target_position": target_pos,
                }
            )

    return results


# ---------------------------------------------------------------------------
# ESM2 tests
# ---------------------------------------------------------------------------


class TestBatchInvarianceESM2:
    # Real ESM2 model: padding-induced differences should be negligible (cosine distance ≤ 0.01).

    TOLERANCE = 0.01

    def test_batch_invariance_pooled(
        self,
        esm2_embedder,
        standard_sequences: List[str],
        filler_sequences: List[str],
        reports_dir,
    ):
        results = _run_batch_invariance(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            target_sequences=standard_sequences[:2],  # fewer seqs to keep runtime low
            filler_sequences=filler_sequences,
            tolerance=self.TOLERANCE,
        )

        table = format_metrics_table(results, title="Batch Invariance — ESM2-T6-8M")
        print(table)
        write_metrics_csv(results, reports_dir / "batch_invariance_esm2.csv")

        for r in results:
            assert r["passed"], (
                f"Batch invariance FAILED: {r['parameter']} — "
                f"cosine_dist={r['cosine_distance']:.8f} > {self.TOLERANCE}"
            )
