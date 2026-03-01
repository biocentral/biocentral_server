# Idempotency invariant: same sequence -> same embedding for the same model, across repeated calls.

from typing import List

from tests.property.oracles.embedding_metrics import (
    compute_all_metrics,
    format_metrics_table,
    write_metrics_csv,
)


N_REPEATS = 5


def _run_idempotency_experiment(
    embedder,
    embedder_label: str,
    sequences: List[str],
    n_repeats: int = N_REPEATS,
    pooled: bool = True,
):
    results = []

    for seq_idx, seq in enumerate(sequences):
        embeddings = []
        for _ in range(n_repeats):
            if pooled:
                emb = embedder.embed_pooled(seq)
            else:
                emb = embedder.embed(seq)
            embeddings.append(emb)

        reference = embeddings[0]
        for repeat_idx in range(1, n_repeats):
            metrics = compute_all_metrics(reference, embeddings[repeat_idx])
            results.append(
                {
                    "embedder": embedder_label,
                    "test_type": "idempotency",
                    "parameter": f"seq{seq_idx}_repeat{repeat_idx}",
                    "cosine_distance": metrics["cosine_distance"],
                    "l2_distance": metrics["l2_distance"],
                    "threshold": 1e-6,
                    "passed": metrics["cosine_distance"] <= 1e-6,
                    "sequence_length": len(seq),
                }
            )

    return results


class TestIdempotencyESM2:
    TOLERANCE = 1e-5

    def test_pooled_embedding_idempotent(
        self,
        esm2_embedder,
        extended_sequences: List[str],
        reports_dir,
    ):
        # Use first 10 extended sequences × 4 repeats = 40 rows
        results = _run_idempotency_experiment(
            embedder=esm2_embedder,
            embedder_label="esm2_t6_8m",
            sequences=extended_sequences[:10],
            pooled=True,
        )

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
