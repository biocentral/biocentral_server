# Idempotency invariant: same sequence → same embedding for the same model, across repeated calls.

import pytest
from typing import List

from tests.fixtures.fixed_embedder import FixedEmbedder
from tests.property.oracles.embedding_metrics import (
    compute_all_metrics,
    format_metrics_table,
    write_metrics_csv,
)


N_REPEATS = 5  # number of repeated embed calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_idempotency_experiment(
    embedder,
    embedder_label: str,
    sequences: List[str],
    n_repeats: int = N_REPEATS,
    pooled: bool = True,
):
    # Embed each sequence n_repeats times and return per-pair metrics.
    results = []

    for seq_idx, seq in enumerate(sequences):
        embeddings = []
        for _ in range(n_repeats):
            if pooled:
                emb = embedder.embed_pooled(seq)
            else:
                emb = embedder.embed(seq)
            embeddings.append(emb)

        # Compare every subsequent embedding to the first one (reference)
        reference = embeddings[0]
        for repeat_idx in range(1, n_repeats):
            metrics = compute_all_metrics(reference, embeddings[repeat_idx])
            results.append({
                "embedder": embedder_label,
                "test_type": "idempotency",
                "parameter": f"seq{seq_idx}_repeat{repeat_idx}",
                "cosine_distance": metrics["cosine_distance"],
                "l2_distance": metrics["l2_distance"],
                "kl_divergence": metrics["kl_divergence"],
                "threshold": 1e-6,
                "passed": metrics["cosine_distance"] <= 1e-6,
                "sequence_length": len(seq),
            })

    return results


# ---------------------------------------------------------------------------
# FixedEmbedder tests
# ---------------------------------------------------------------------------

class TestIdempotencyFixedEmbedder:
    # Idempotency must hold perfectly for the deterministic FixedEmbedder.

    def test_pooled_embedding_idempotent(
        self,
        fixed_embedder: FixedEmbedder,
        standard_sequences: List[str],
        reports_dir,
    ):
        results = _run_idempotency_experiment(
            embedder=fixed_embedder,
            embedder_label="fixed_embedder",
            sequences=standard_sequences,
            pooled=True,
        )

        table = format_metrics_table(results, title="Idempotency — FixedEmbedder (pooled)")
        print(table)
        write_metrics_csv(results, reports_dir / "idempotency_fixed_pooled.csv")

        for r in results:
            assert r["passed"], (
                f"Idempotency FAILED for {r['parameter']}: "
                f"cosine_distance={r['cosine_distance']:.10f}"
            )

    def test_per_residue_embedding_idempotent(
        self,
        fixed_embedder: FixedEmbedder,
        standard_sequences: List[str],
        reports_dir,
    ):
        results = _run_idempotency_experiment(
            embedder=fixed_embedder,
            embedder_label="fixed_embedder",
            sequences=standard_sequences,
            pooled=False,
        )

        table = format_metrics_table(results, title="Idempotency — FixedEmbedder (per-residue)")
        print(table)
        write_metrics_csv(results, reports_dir / "idempotency_fixed_per_residue.csv")

        for r in results:
            assert r["passed"], (
                f"Idempotency FAILED for {r['parameter']}: "
                f"cosine_distance={r['cosine_distance']:.10f}"
            )


# ---------------------------------------------------------------------------
# Real ESM2 tests (slow, requires model)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestIdempotencyESM2:
    # Real pLM: GPU non-determinism may introduce tiny drifts — we tolerate cosine distance ≤ 1e-5.

    TOLERANCE = 1e-5

    def test_pooled_embedding_idempotent(
        self,
        esm2_embedder,
        standard_sequences: List[str],
        reports_dir,
    ):
        results = _run_idempotency_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=standard_sequences,
            pooled=True,
        )

        # Override threshold for real model
        for r in results:
            r["threshold"] = self.TOLERANCE
            r["passed"] = r["cosine_distance"] <= self.TOLERANCE

        table = format_metrics_table(results, title="Idempotency — ESM2-T6-8M (pooled)")
        print(table)
        write_metrics_csv(results, reports_dir / "idempotency_esm2_pooled.csv")

        for r in results:
            assert r["passed"], (
                f"Idempotency FAILED for {r['parameter']}: "
                f"cosine_distance={r['cosine_distance']:.10f} > {self.TOLERANCE}"
            )
