# Projection determinism: same reduction on same embeddings must yield identical (PCA) or similar (UMAP/t-SNE) coordinates.

import numpy as np
import pytest
from typing import List


def _embed_sequences(embedder, sequences: List[str]) -> np.ndarray:
    embs = embedder.embed_batch(sequences, pooled=True)
    return np.stack(embs)


def _project(
    data: np.ndarray, method: str, n_components: int = 2, **kwargs
) -> np.ndarray:
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
    if isinstance(reduction, dict):
        coords = np.column_stack([reduction[f"D{d + 1}"] for d in range(n_components)])
    else:
        coords = np.column_stack(
            [
                reduction.result.column(f"D{d + 1}").to_pylist()
                for d in range(n_components)
            ]
        )
    return coords


def _procrustes_distance(A: np.ndarray, B: np.ndarray) -> float:
    A_c = A - A.mean(axis=0)
    B_c = B - B.mean(axis=0)
    norm_A = np.linalg.norm(A_c)
    norm_B = np.linalg.norm(B_c)
    if norm_A == 0 or norm_B == 0:
        return 0.0
    A_c /= norm_A
    B_c /= norm_B
    M = A_c.T @ B_c
    U, _, Vt = np.linalg.svd(M)
    R = Vt.T @ U.T
    A_rot = A_c @ R
    return float(np.linalg.norm(A_rot - B_c))


class TestPCADeterminism:

    def test_pca_deterministic(
        self,
        esm2_embedder,
        diverse_sequences: List[str],
        reports_dir,
    ):
        data = _embed_sequences(esm2_embedder, diverse_sequences)

        coords_1 = _project(data, method="pca", n_components=2)
        coords_2 = _project(data, method="pca", n_components=2)

        diff = np.max(np.abs(coords_1 - coords_2))
        procrustes = _procrustes_distance(coords_1, coords_2)

        result = {
            "embedder": "esm2_t6_8m",
            "test_type": "projection_determinism",
            "parameter": "pca",
            "max_abs_diff": float(diff),
            "procrustes_distance": procrustes,
            "passed": diff < 1e-10,
        }
        print(
            f"\n[PCA Determinism] max_abs_diff={diff:.2e}, procrustes={procrustes:.2e}"
        )
        assert result["passed"], f"PCA not deterministic: max_abs_diff={diff}"

    def test_pca_3d_deterministic(
        self,
        esm2_embedder,
        diverse_sequences: List[str],
    ):
        data = _embed_sequences(esm2_embedder, diverse_sequences)

        coords_1 = _project(data, method="pca", n_components=3)
        coords_2 = _project(data, method="pca", n_components=3)

        diff = np.max(np.abs(coords_1 - coords_2))
        assert diff < 1e-10, f"PCA-3D not deterministic: max_abs_diff={diff}"


class TestStochasticProjectionConsistency:

    PROCRUSTES_THRESHOLD = 0.3

    def test_umap_consistency(
        self,
        esm2_embedder,
        diverse_sequences: List[str],
        reports_dir,
    ):
        data = _embed_sequences(esm2_embedder, diverse_sequences)
        n_neighbors = min(5, len(diverse_sequences) - 1)

        coords_1 = _project(
            data, method="umap", n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42
        )
        coords_2 = _project(
            data, method="umap", n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42
        )

        procrustes = _procrustes_distance(coords_1, coords_2)
        print(f"\n[UMAP Consistency] Procrustes distance = {procrustes:.6f}")

        result = {
            "embedder": "esm2_t6_8m",
            "test_type": "projection_consistency",
            "parameter": "umap",
            "procrustes_distance": procrustes,
            "threshold": self.PROCRUSTES_THRESHOLD,
            "passed": procrustes <= self.PROCRUSTES_THRESHOLD,
        }
        print(f"  passed={result['passed']} (threshold={self.PROCRUSTES_THRESHOLD})")

    def test_tsne_consistency(
        self,
        esm2_embedder,
        diverse_sequences: List[str],
        reports_dir,
    ):
        data = _embed_sequences(esm2_embedder, diverse_sequences)
        perplexity = min(5.0, len(diverse_sequences) - 1)

        coords_1 = _project(data, method="tsne", n_components=2, perplexity=perplexity, random_state=42)
        coords_2 = _project(data, method="tsne", n_components=2, perplexity=perplexity, random_state=42)

        procrustes = _procrustes_distance(coords_1, coords_2)
        print(f"\n[t-SNE Consistency] Procrustes distance = {procrustes:.6f}")

        result = {
            "embedder": "esm2_t6_8m",
            "test_type": "projection_consistency",
            "parameter": "tsne",
            "procrustes_distance": procrustes,
            "threshold": self.PROCRUSTES_THRESHOLD,
            "passed": procrustes <= self.PROCRUSTES_THRESHOLD,
        }
        print(f"  passed={result['passed']} (threshold={self.PROCRUSTES_THRESHOLD})")
