"""Projection oracle tests using direct protspace inference."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Protocol

import numpy as np
import pytest

from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET
from tests.fixtures.fixed_embedder import FixedEmbedder

_projection_oracle_results: List[Dict[str, Any]] = []
pytestmark = pytest.mark.property


def get_projection_oracle_results() -> List[Dict[str, Any]]:
    """Get accumulated projection oracle results."""
    return _projection_oracle_results


def add_projection_oracle_result(result: Dict[str, Any]) -> None:
    """Add a result to the projection oracle results collection."""
    if "timestamp" not in result:
        result["timestamp"] = datetime.now().isoformat()
    _projection_oracle_results.append(result)


def clear_projection_oracle_results() -> None:
    """Clear all accumulated projection oracle results."""
    _projection_oracle_results.clear()


@dataclass
class ProjectionOracleConfig:
    """Configuration for projection oracles."""

    method: str
    n_components: int = 2
    distance_correlation_threshold: float = 0.5


PROJECTION_ORACLE_CONFIGS = {
    "umap": ProjectionOracleConfig(
        method="umap",
        n_components=2,
        distance_correlation_threshold=0.3,
    ),
    "pca": ProjectionOracleConfig(
        method="pca",
        n_components=2,
        distance_correlation_threshold=0.4,
    ),
    "tsne": ProjectionOracleConfig(
        method="tsne",
        n_components=2,
        distance_correlation_threshold=0.2,
    ),
}


class ProjectorProtocol(Protocol):
    """Protocol defining the projector interface for oracle tests."""

    def project(
        self,
        embeddings: Dict[str, np.ndarray],
        method: str,
        n_components: int,
    ) -> Dict[str, np.ndarray]:
        """Project embeddings to lower dimensions."""
        ...


class ProjectionDeterminismOracle:
    """Verifies projections are deterministic across multiple runs."""

    def __init__(
        self,
        projector: ProjectorProtocol,
        config: ProjectionOracleConfig,
        num_runs: int = 3,
    ):
        self.projector = projector
        self.config = config
        self.num_runs = num_runs

    def verify(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Verify projection determinism for a set of embeddings."""
        projections_list = []
        for _ in range(self.num_runs):
            proj = self.projector.project(
                embeddings,
                self.config.method,
                self.config.n_components,
            )
            projections_list.append(proj)

        first = projections_list[0]
        all_identical = all(
            self._projections_equal(first, p) for p in projections_list[1:]
        )

        result = {
            "method": self.config.method,
            "test_type": "determinism",
            "num_sequences": len(embeddings),
            "num_runs": self.num_runs,
            "passed": all_identical,
        }
        add_projection_oracle_result(result)
        return result

    def _projections_equal(
        self,
        p1: Dict[str, np.ndarray],
        p2: Dict[str, np.ndarray],
    ) -> bool:
        """Check if two projection dicts are equal."""
        if set(p1.keys()) != set(p2.keys()):
            return False
        for key in p1:
            if not np.allclose(p1[key], p2[key], rtol=1e-5):
                return False
        return True


class DimensionalityOracle:
    """Verifies projection dimensions are correct."""

    def __init__(
        self,
        projector: ProjectorProtocol,
        config: ProjectionOracleConfig,
    ):
        self.projector = projector
        self.config = config

    def verify(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Verify projection dimensions are correct."""
        projections = self.projector.project(
            embeddings,
            self.config.method,
            self.config.n_components,
        )

        issues = []
        for seq_id, proj in projections.items():
            if proj.shape != (self.config.n_components,):
                issues.append(
                    f"{seq_id}: shape {proj.shape} != "
                    f"expected ({self.config.n_components},)"
                )

        passed = len(issues) == 0

        result = {
            "method": self.config.method,
            "test_type": "dimensionality",
            "n_components": self.config.n_components,
            "num_sequences": len(embeddings),
            "issues": issues,
            "passed": passed,
        }
        add_projection_oracle_result(result)
        return result


class ProjectionValueValidityOracle:
    """Verifies projection values are finite and within bounds."""

    def __init__(
        self,
        projector: ProjectorProtocol,
        config: ProjectionOracleConfig,
        max_abs_value: float = 1000.0,
    ):
        self.projector = projector
        self.config = config
        self.max_abs_value = max_abs_value

    def verify(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Verify projection values are valid."""
        projections = self.projector.project(
            embeddings,
            self.config.method,
            self.config.n_components,
        )

        issues = []
        for seq_id, proj in projections.items():
            if not np.isfinite(proj).all():
                issues.append(f"{seq_id}: contains NaN or Inf values")

            if np.abs(proj).max() > self.max_abs_value:
                issues.append(
                    f"{seq_id}: values exceed bounds "
                    f"(max abs: {np.abs(proj).max():.2f})"
                )

        passed = len(issues) == 0

        result = {
            "method": self.config.method,
            "test_type": "value_validity",
            "num_sequences": len(embeddings),
            "issues": issues,
            "passed": passed,
        }
        add_projection_oracle_result(result)
        return result


class DistancePreservationOracle:
    """Verifies projections approximately preserve pairwise distances."""

    def __init__(
        self,
        projector: ProjectorProtocol,
        config: ProjectionOracleConfig,
    ):
        self.projector = projector
        self.config = config

    def verify(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Verify distance preservation property."""
        if len(embeddings) < 3:
            return {
                "method": self.config.method,
                "test_type": "distance_preservation",
                "passed": True,
                "reason": "Not enough sequences for distance comparison",
            }

        projections = self.projector.project(
            embeddings,
            self.config.method,
            self.config.n_components,
        )

        seq_ids = list(embeddings.keys())
        original_dists = []
        projected_dists = []

        for i in range(len(seq_ids)):
            for j in range(i + 1, len(seq_ids)):
                id_i, id_j = seq_ids[i], seq_ids[j]

                emb_i = embeddings[id_i]
                emb_j = embeddings[id_j]
                if len(emb_i.shape) > 1:
                    emb_i = emb_i.mean(axis=0)
                if len(emb_j.shape) > 1:
                    emb_j = emb_j.mean(axis=0)
                orig_dist = np.linalg.norm(emb_i - emb_j)

                proj_dist = np.linalg.norm(
                    projections[id_i] - projections[id_j]
                )

                original_dists.append(orig_dist)
                projected_dists.append(proj_dist)

        correlation = np.corrcoef(original_dists, projected_dists)[0, 1]

        if np.isnan(correlation):
            correlation = 0.0

        passed = correlation >= self.config.distance_correlation_threshold

        result = {
            "method": self.config.method,
            "test_type": "distance_preservation",
            "correlation": float(correlation),
            "threshold": self.config.distance_correlation_threshold,
            "num_pairs": len(original_dists),
            "passed": passed,
        }
        add_projection_oracle_result(result)
        return result


class DirectProjector:
    """Projector using protspace directly without HTTP calls."""

    def __init__(self, config: ProjectionOracleConfig = None):
        self.config = config or ProjectionOracleConfig(method="pca")
        self._processor = None
        self._reducers = None

    def _ensure_initialized(self):
        if self._processor is not None:
            return

        from protspace.data.processors import BaseProcessor
        from protspace.utils import REDUCERS

        self._reducers = REDUCERS
        self._processor = BaseProcessor(config={}, reducers=REDUCERS)

    def project(
        self,
        embeddings: Dict[str, np.ndarray],
        method: str,
        n_components: int,
    ) -> Dict[str, np.ndarray]:
        """Project embeddings to lower dimensions using protspace."""
        self._ensure_initialized()

        if method.lower() not in self._reducers:
            raise ValueError(
                f"Unknown projection method: {method}. "
                f"Available: {list(self._reducers.keys())}"
            )

        seq_ids = list(embeddings.keys())
        embedding_matrix = []

        for seq_id in seq_ids:
            emb = embeddings[seq_id]
            if len(emb.shape) > 1:
                emb = emb.mean(axis=0)
            embedding_matrix.append(emb)

        embedding_matrix = np.array(embedding_matrix)

        reduction = self._processor.process_reduction(
            data=embedding_matrix,
            method=method.lower(),
            dims=n_components,
        )

        # Extract coordinates - handle both dict (new API) and Reduction object (old API)
        if isinstance(reduction, dict):
            # New protspace API returns dict with D1, D2, etc. keys
            coords = np.column_stack([
                reduction[f"D{d+1}"]
                for d in range(n_components)
            ])
        else:
            # Old API: Reduction object with .result Arrow table
            coords = np.column_stack([
                reduction.result.column(f"D{d+1}").to_pylist()
                for d in range(n_components)
            ])

        projections = {}
        for i, seq_id in enumerate(seq_ids):
            projections[seq_id] = coords[i].astype(np.float32)

        return projections


@pytest.fixture(scope="module")
def pca_config() -> ProjectionOracleConfig:
    """Oracle configuration for PCA projection."""
    return PROJECTION_ORACLE_CONFIGS["pca"]


@pytest.fixture(scope="module")
def umap_config() -> ProjectionOracleConfig:
    """Oracle configuration for UMAP projection."""
    return PROJECTION_ORACLE_CONFIGS["umap"]


@pytest.fixture(scope="module")
def tsne_config() -> ProjectionOracleConfig:
    """Oracle configuration for t-SNE projection."""
    return PROJECTION_ORACLE_CONFIGS["tsne"]


@pytest.fixture(scope="module")
def oracle_embeddings() -> Dict[str, np.ndarray]:
    """Test embeddings from canonical dataset using FixedEmbedder."""
    embedder = FixedEmbedder(model_name="esm2_t6")
    sequences = {
        "standard_001": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
        "standard_002": CANONICAL_TEST_DATASET.get_by_id("standard_002").sequence,
        "standard_003": CANONICAL_TEST_DATASET.get_by_id("standard_003").sequence,
        "real_insulin_b": CANONICAL_TEST_DATASET.get_by_id("real_insulin_b").sequence,
        "real_ubiquitin": CANONICAL_TEST_DATASET.get_by_id("real_ubiquitin").sequence,
    }
    return embedder.embed_dict(sequences, pooled=True)


@pytest.fixture(scope="module")
def diverse_test_embeddings() -> Dict[str, np.ndarray]:
    """Diverse test embeddings with varied sequence lengths."""
    embedder = FixedEmbedder(model_name="esm2_t6")
    sequences = {
        "short": CANONICAL_TEST_DATASET.get_by_id("length_short_10").sequence,
        "medium": CANONICAL_TEST_DATASET.get_by_id("length_medium_50").sequence,
        "standard": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
        "charged": CANONICAL_TEST_DATASET.get_by_id("charged_rich").sequence,
        "hydrophobic": CANONICAL_TEST_DATASET.get_by_id("hydrophobic_rich").sequence,
    }
    return embedder.embed_dict(sequences, pooled=True)


@pytest.fixture(scope="module")
def direct_pca_projector(pca_config: ProjectionOracleConfig) -> DirectProjector:
    """Create projector that runs PCA directly via protspace."""
    try:
        projector = DirectProjector(config=pca_config)
        projector._ensure_initialized()
        return projector
    except Exception as e:
        pytest.skip(f"protspace not available for PCA: {e}")


@pytest.fixture(scope="module")
def direct_umap_projector(umap_config: ProjectionOracleConfig) -> DirectProjector:
    """Create projector that runs UMAP directly via protspace."""
    try:
        projector = DirectProjector(config=umap_config)
        projector._ensure_initialized()
        return projector
    except Exception as e:
        pytest.skip(f"protspace not available for UMAP: {e}")


@pytest.fixture(scope="module")
def direct_tsne_projector(tsne_config: ProjectionOracleConfig) -> DirectProjector:
    """Create projector that runs t-SNE directly via protspace."""
    try:
        projector = DirectProjector(config=tsne_config)
        projector._ensure_initialized()
        return projector
    except Exception as e:
        pytest.skip(f"protspace not available for t-SNE: {e}")


@pytest.fixture(scope="module", params=["pca", "umap"])
def direct_projector(request):
    """Parametrized fixture for multiple projection methods."""
    method = request.param
    config = PROJECTION_ORACLE_CONFIGS[method]
    try:
        projector = DirectProjector(config=config)
        projector._ensure_initialized()
        return projector, config
    except Exception as e:
        pytest.skip(f"protspace not available for {method}: {e}")


@pytest.mark.slow
class TestProjectionDeterminism:
    """Tests for projection determinism oracle using direct protspace."""

    def test_pca_determinism(
        self,
        direct_pca_projector: DirectProjector,
        pca_config: ProjectionOracleConfig,
        oracle_embeddings: Dict[str, np.ndarray],
    ):
        """Verify PCA projections are deterministic via protspace."""
        oracle = ProjectionDeterminismOracle(
            projector=direct_pca_projector,
            config=pca_config,
        )

        result = oracle.verify(oracle_embeddings)
        assert result["passed"], (
            f"PCA determinism failed: method={result['method']}, "
            f"num_sequences={result['num_sequences']}, num_runs={result['num_runs']}"
        )

    def test_umap_determinism(
        self,
        direct_umap_projector: DirectProjector,
        umap_config: ProjectionOracleConfig,
        oracle_embeddings: Dict[str, np.ndarray],
    ):
        """Verify UMAP projections are deterministic (with fixed seed) via protspace."""
        oracle = ProjectionDeterminismOracle(
            projector=direct_umap_projector,
            config=umap_config,
        )

        result = oracle.verify(oracle_embeddings)
        assert result["passed"], (
            f"UMAP determinism failed: method={result['method']}, "
            f"num_sequences={result['num_sequences']}, num_runs={result['num_runs']}"
        )


@pytest.mark.slow
class TestDimensionality:
    """Tests for projection dimensionality oracle using direct protspace."""

    def test_pca_dimensionality_2d(
        self,
        direct_pca_projector: DirectProjector,
        pca_config: ProjectionOracleConfig,
        oracle_embeddings: Dict[str, np.ndarray],
    ):
        """Verify PCA projects to correct number of dimensions via protspace."""
        oracle = DimensionalityOracle(
            projector=direct_pca_projector,
            config=pca_config,
        )

        result = oracle.verify(oracle_embeddings)
        assert result["passed"], (
            f"Dimensionality check failed for method={result['method']} "
            f"n_components={result['n_components']}: {result['issues']}"
        )

    def test_umap_dimensionality_2d(
        self,
        direct_umap_projector: DirectProjector,
        umap_config: ProjectionOracleConfig,
        oracle_embeddings: Dict[str, np.ndarray],
    ):
        """Verify UMAP projects to correct number of dimensions via protspace."""
        oracle = DimensionalityOracle(
            projector=direct_umap_projector,
            config=umap_config,
        )

        result = oracle.verify(oracle_embeddings)
        assert result["passed"], (
            f"Dimensionality check failed for method={result['method']} "
            f"n_components={result['n_components']}: {result['issues']}"
        )


@pytest.mark.slow
class TestProjectionValueValidity:
    """Tests for projection value validity oracle using direct protspace."""

    def test_pca_values_valid(
        self,
        direct_pca_projector: DirectProjector,
        pca_config: ProjectionOracleConfig,
        oracle_embeddings: Dict[str, np.ndarray],
    ):
        """Verify PCA projection values are valid via protspace."""
        oracle = ProjectionValueValidityOracle(
            projector=direct_pca_projector,
            config=pca_config,
        )

        result = oracle.verify(oracle_embeddings)
        assert result["passed"], (
            f"Value validity failed for method={result['method']}, "
            f"num_sequences={result['num_sequences']}: {result['issues']}"
        )

    def test_umap_values_valid(
        self,
        direct_umap_projector: DirectProjector,
        umap_config: ProjectionOracleConfig,
        diverse_test_embeddings: Dict[str, np.ndarray],
    ):
        """Verify UMAP projection values are valid for diverse inputs via protspace."""
        oracle = ProjectionValueValidityOracle(
            projector=direct_umap_projector,
            config=umap_config,
        )

        result = oracle.verify(diverse_test_embeddings)
        assert result["passed"], f"Value validity failed: {result['issues']}"


@pytest.mark.slow
class TestDistancePreservation:
    """Tests for distance preservation oracle using direct protspace."""

    def test_pca_preserves_distances(
        self,
        direct_pca_projector: DirectProjector,
        pca_config: ProjectionOracleConfig,
        oracle_embeddings: Dict[str, np.ndarray],
    ):
        """Verify PCA approximately preserves pairwise distances via protspace."""
        oracle = DistancePreservationOracle(
            projector=direct_pca_projector,
            config=pca_config,
        )

        result = oracle.verify(oracle_embeddings)
        assert result["passed"], (
            f"Distance preservation failed for method={result['method']}: "
            f"correlation={result['correlation']:.6f} < threshold={result['threshold']:.6f}; "
            f"num_pairs={result['num_pairs']}"
        )

    def test_umap_preserves_local_structure(
        self,
        direct_umap_projector: DirectProjector,
        umap_config: ProjectionOracleConfig,
        oracle_embeddings: Dict[str, np.ndarray],
    ):
        """Verify UMAP preserves local structure (lower threshold than PCA) via protspace."""
        oracle = DistancePreservationOracle(
            projector=direct_umap_projector,
            config=umap_config,
        )

        result = oracle.verify(oracle_embeddings)

        print(f"UMAP distance correlation: {result.get('correlation', 'N/A')}")
        assert result["passed"], (
            f"Distance preservation failed for method={result['method']}: "
            f"correlation={result['correlation']:.6f} < threshold={result['threshold']:.6f}; "
            f"num_pairs={result['num_pairs']}"
        )


@pytest.mark.slow
class TestParametrizedProjectionOracles:
    """Parametrized oracle tests across multiple projection methods."""

    def test_projection_determinism(
        self,
        direct_projector,
        oracle_embeddings: Dict[str, np.ndarray],
    ):
        """Verify projections are deterministic for any method via protspace."""
        projector, config = direct_projector
        oracle = ProjectionDeterminismOracle(
            projector=projector,
            config=config,
        )

        result = oracle.verify(oracle_embeddings)

        assert result["passed"], (
            f"Determinism failed: method={result['method']}, "
            f"num_sequences={result['num_sequences']}"
        )

    def test_projection_dimensionality(
        self,
        direct_projector,
        oracle_embeddings: Dict[str, np.ndarray],
    ):
        """Verify correct dimensionality for any method via protspace."""
        projector, config = direct_projector
        oracle = DimensionalityOracle(
            projector=projector,
            config=config,
        )

        result = oracle.verify(oracle_embeddings)

        assert result["passed"], (
            f"Dimensionality failed: method={result['method']}, "
            f"issues={result['issues']}"
        )


@pytest.fixture(scope="module", autouse=True)
def cleanup_projection_results():
    """Clean up projection oracle results after module completes."""
    yield
    results = get_projection_oracle_results()
    if results:
        print(f"\n📊 Projection oracle tests completed: {len(results)} results")
    clear_projection_oracle_results()
