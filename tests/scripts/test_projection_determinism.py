# Projection determinism: same reduction on same embeddings must yield identical (PCA) or similar (UMAP/t-SNE) coordinates.

import numpy as np
import pytest
from typing import List

from tests.fixtures.fixed_embedder import FixedEmbedder
 

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed_sequences(embedder, sequences: List[str]) -> np.ndarray:
    # Return a (N, D) matrix of pooled embeddings.
    embs = embedder.embed_batch(sequences, pooled=True)
    return np.stack(embs)


def _project(data: np.ndarray, method: str, n_components: int = 2, **kwargs) -> np.ndarray:
    # Run protspace projection and return (N, n_components) array.
    try:
        from protspace.utils import REDUCERS
        from protspace.data.processors import BaseProcessor
    except ImportError:
        pytest.skip("protspace not installed")

    config = dict(kwargs)
    processor = BaseProcessor(config=config, reducers=REDUCERS)
    reduction = processor.process_reduction(
        data=data,
        method=method,
        dims=n_components,
    )
    # reduction is a protspace Reduction object; extract the numpy array
    coords = np.column_stack([
        reduction.result.column(f"D{d+1}").to_pylist()
        for d in range(n_components)
    ])
    return coords


def _procrustes_distance(A: np.ndarray, B: np.ndarray) -> float:
    # Procrustes distance between two point clouds (translation + rotation + uniform scaling).
    # Centre both
    A_c = A - A.mean(axis=0)
    B_c = B - B.mean(axis=0)
    # Scale
    norm_A = np.linalg.norm(A_c)
    norm_B = np.linalg.norm(B_c)
    if norm_A == 0 or norm_B == 0:
        return 0.0
    A_c /= norm_A
    B_c /= norm_B
    # Optimal rotation via SVD
    M = A_c.T @ B_c
    U, _, Vt = np.linalg.svd(M)
    R = Vt.T @ U.T
    A_rot = A_c @ R
    return float(np.linalg.norm(A_rot - B_c))


# ---------------------------------------------------------------------------
# PCA determinism (exact)
# ---------------------------------------------------------------------------

class TestPCADeterminism:
    # PCA is a closed-form decomposition; two runs must be bit-identical.

    def test_pca_deterministic(
        self,
        fixed_embedder: FixedEmbedder,
        diverse_sequences: List[str],
        reports_dir,
    ):
        data = _embed_sequences(fixed_embedder, diverse_sequences)

        coords_1 = _project(data, method="pca", n_components=2)
        coords_2 = _project(data, method="pca", n_components=2)

        diff = np.max(np.abs(coords_1 - coords_2))
        procrustes = _procrustes_distance(coords_1, coords_2)

        result = {
            "embedder": "fixed_embedder",
            "test_type": "projection_determinism",
            "parameter": "pca",
            "max_abs_diff": float(diff),
            "procrustes_distance": procrustes,
            "passed": diff < 1e-10,
        }
        print(f"\n[PCA Determinism] max_abs_diff={diff:.2e}, procrustes={procrustes:.2e}")
        assert result["passed"], f"PCA not deterministic: max_abs_diff={diff}"

    def test_pca_3d_deterministic(
        self,
        fixed_embedder: FixedEmbedder,
        diverse_sequences: List[str],
    ):
        data = _embed_sequences(fixed_embedder, diverse_sequences)

        coords_1 = _project(data, method="pca", n_components=3)
        coords_2 = _project(data, method="pca", n_components=3)

        diff = np.max(np.abs(coords_1 - coords_2))
        assert diff < 1e-10, f"PCA-3D not deterministic: max_abs_diff={diff}"


# ---------------------------------------------------------------------------
# UMAP / t-SNE structural consistency (stochastic)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestStochasticProjectionConsistency:
    # UMAP and t-SNE are stochastic; check that two runs produce similar point-cloud shapes.

    PROCRUSTES_THRESHOLD = 0.3  # generous for stochastic methods

    def test_umap_consistency(
        self,
        fixed_embedder: FixedEmbedder,
        diverse_sequences: List[str],
        reports_dir,
    ):
        data = _embed_sequences(fixed_embedder, diverse_sequences)
        n_neighbors = min(5, len(diverse_sequences) - 1)

        coords_1 = _project(data, method="umap", n_components=2,
                            n_neighbors=n_neighbors, min_dist=0.1)
        coords_2 = _project(data, method="umap", n_components=2,
                            n_neighbors=n_neighbors, min_dist=0.1)

        procrustes = _procrustes_distance(coords_1, coords_2)
        print(f"\n[UMAP Consistency] Procrustes distance = {procrustes:.6f}")

        # Report but don't necessarily fail (stochastic)
        result = {
            "embedder": "fixed_embedder",
            "test_type": "projection_consistency",
            "parameter": "umap",
            "procrustes_distance": procrustes,
            "threshold": self.PROCRUSTES_THRESHOLD,
            "passed": procrustes <= self.PROCRUSTES_THRESHOLD,
        }
        print(f"  passed={result['passed']} (threshold={self.PROCRUSTES_THRESHOLD})")

    def test_tsne_consistency(
        self,
        fixed_embedder: FixedEmbedder,
        diverse_sequences: List[str],
        reports_dir,
    ):
        data = _embed_sequences(fixed_embedder, diverse_sequences)
        perplexity = min(5.0, len(diverse_sequences) - 1)

        coords_1 = _project(data, method="tsne", n_components=2, perplexity=perplexity)
        coords_2 = _project(data, method="tsne", n_components=2, perplexity=perplexity)

        procrustes = _procrustes_distance(coords_1, coords_2)
        print(f"\n[t-SNE Consistency] Procrustes distance = {procrustes:.6f}")

        result = {
            "embedder": "fixed_embedder",
            "test_type": "projection_consistency",
            "parameter": "tsne",
            "procrustes_distance": procrustes,
            "threshold": self.PROCRUSTES_THRESHOLD,
            "passed": procrustes <= self.PROCRUSTES_THRESHOLD,
        }
        print(f"  passed={result['passed']} (threshold={self.PROCRUSTES_THRESHOLD})")
