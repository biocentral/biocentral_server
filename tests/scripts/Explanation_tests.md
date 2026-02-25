## Experiments

| # | Script | Invariant / MR | What it tests |
|---|--------|---------------|---------------|
| 1 | `test_idempotency.py` | **Idempotency** | Same sequence → same embedding (repeated calls) |
| 2 | `test_batch_invariance.py` | **Batch invariance** | Embedding A+B together == embedding A, B separately |
| 3 | `test_projection_determinism.py` | **Projection determinism** | PCA/UMAP/t-SNE on same data → same (or similar) coordinates |
| 4 | `test_progressive_x_masking.py` | **MR: X-masking** | Progressively replace residues with 'X'; measure divergence |
| 5 | `test_reversed_sequence.py` | **MR: Reversed sequence** | Reverse order; measure sensitivity to position |

CSV reports are written to `tests/reports/`.

## Experiment Details

### 1. Idempotency

calling `embed(seq)` multiple times must
return identical results (for a deterministic embedder) or results within a
very tight tolerance (for GPU models with non-deterministic kernels).

- **FixedEmbedder**: expects exact equality (cosine distance ≤ 1e-6)
- **ESM2**: tolerates cosine distance ≤ 1e-5

### 2. Batch Invariance

Embedding sequences A and B *together* in one batch must yield the same
per-sequence embedding as embedding them *individually*. Transformer attention
operates over the single sequence, not across batch members, so batch
composition must be invisible to the output.

- Tests batch sizes: 2, 5, 10, 20
- Target sequence placed at random positions within the batch

### 3. Projection Determinism

- **PCA**: deterministic → expects bit-identical coordinates across runs
- **UMAP / t-SNE**: stochastic → checks structural similarity via Procrustes distance

### 4. Progressive X-Masking (Metamorphic Relation)

Replace progressively more amino acids with `X` (the standard placeholder for
"not resolved / unknown") and measure how the embedding diverges from the
original.

**Questions answered:**
- Is divergence monotonically increasing with masking ratio?
- At what masking ratio does the embedding differ *significantly*?
- Critical ratio `r*` where cosine distance > 0.1

Masking ratios tested: 0%, 5%, 10%, 15%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%

### 5. Reversed Sequence (Metamorphic Relation)

Reverse the order and compare embeddings.

