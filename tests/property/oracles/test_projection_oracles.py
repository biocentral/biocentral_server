"""
Projection Oracle Tests for Dimensionality Reduction Validity.

This module implements test oracles that verify critical projection properties:
1. Determinism: Same input always produces the same projection output.
2. Dimensionality Correctness: Output dimensions match configuration.
3. Value Validity: Projected values are finite and within reasonable bounds.
4. Distance Preservation: Relative distances are approximately preserved.

Uses the canonical test dataset for reproducible testing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pytest

from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET
from tests.fixtures.fixed_embedder import FixedEmbedder


# ============================================================================
# ORACLE RESULT STORAGE
# ============================================================================

_projection_oracle_results: List[Dict[str, Any]] = []


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


# ============================================================================
# ORACLE CONFIGURATION
# ============================================================================


@dataclass
class ProjectionOracleConfig:
    """Configuration for projection oracles."""

    method: str
    n_components: int = 2
    distance_correlation_threshold: float = 0.5  # Minimum acceptable correlation


# Pre-defined configurations for common projection methods
PROJECTION_ORACLE_CONFIGS = {
    "umap": ProjectionOracleConfig(
        method="umap",
        n_components=2,
        distance_correlation_threshold=0.3,  
    ),
    "pca": ProjectionOracleConfig(
        method="pca",
        n_components=2,
        distance_correlation_threshold=0.6, 
    ),
    "tsne": ProjectionOracleConfig(
        method="tsne",
        n_components=2,
        distance_correlation_threshold=0.2, 
    ),
}


# ============================================================================
# PROJECTOR PROTOCOL
# ============================================================================


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


# ============================================================================
# MOCK PROJECTOR FOR TESTING
# ============================================================================


class MockProjector:
    """
    Mock projector that generates deterministic projections for testing.
    
    Uses embedding hash to generate reproducible low-dimensional outputs.
    """

    def __init__(self, config: ProjectionOracleConfig):
        self.config = config

    def _get_seed(self, embeddings: Dict[str, np.ndarray]) -> int:
        """Get deterministic seed from embeddings."""
        import hashlib
        
        # Create hash from all embedding values
        combined = np.concatenate([emb.flatten() for emb in embeddings.values()])
        hash_bytes = hashlib.sha256(combined.tobytes()).digest()
        return int.from_bytes(hash_bytes[:4], "big")

    def project(
        self,
        embeddings: Dict[str, np.ndarray],
        method: str,
        n_components: int,
    ) -> Dict[str, np.ndarray]:
        """Generate deterministic mock projection."""
        seed = self._get_seed(embeddings)
        rng = np.random.default_rng(seed)

        # Create projection matrix (simulates dimensionality reduction)
        if embeddings:
            first_emb = next(iter(embeddings.values()))
            input_dim = first_emb.shape[-1] if len(first_emb.shape) > 0 else 1
        else:
            input_dim = 320  # Default ESM2 dimension

        # Generate a consistent projection matrix
        proj_matrix = rng.standard_normal((input_dim, n_components))
        proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)

        # Project each embedding
        projections = {}
        for seq_id, emb in embeddings.items():
            # Use mean pooling if per-residue
            if len(emb.shape) > 1:
                emb = emb.mean(axis=0)
            
            # Project and normalize to reasonable range
            proj = emb @ proj_matrix
            projections[seq_id] = proj.astype(np.float32)

        return projections


# ============================================================================
# PROJECTION DETERMINISM ORACLE
# ============================================================================


class ProjectionDeterminismOracle:
    """
    Oracle verifying that projections are deterministic.

    Tests that projecting the same embeddings multiple times yields
    identical results.
    """

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
        """
        Verify projection determinism for a set of embeddings.

        Args:
            embeddings: Dictionary of sequence_id -> embedding

        Returns:
            Result dictionary with pass/fail and details
        """
        projections_list = []
        for _ in range(self.num_runs):
            proj = self.projector.project(
                embeddings,
                self.config.method,
                self.config.n_components,
            )
            projections_list.append(proj)

        # Check all projections are identical
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


# ============================================================================
# DIMENSIONALITY ORACLE
# ============================================================================


class DimensionalityOracle:
    """
    Oracle verifying that projection dimensions are correct.

    Tests that output projections have the expected number of dimensions.
    """

    def __init__(
        self,
        projector: ProjectorProtocol,
        config: ProjectionOracleConfig,
    ):
        self.projector = projector
        self.config = config

    def verify(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Verify projection dimensions are correct.

        Args:
            embeddings: Dictionary of sequence_id -> embedding

        Returns:
            Result dictionary with pass/fail and details
        """
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


# ============================================================================
# VALUE VALIDITY ORACLE
# ============================================================================


class ProjectionValueValidityOracle:
    """
    Oracle verifying that projection values are valid.

    Tests that:
    - All values are finite (no NaN or Inf)
    - Values are within reasonable bounds
    """

    def __init__(
        self,
        projector: ProjectorProtocol,
        config: ProjectionOracleConfig,
        max_abs_value: float = 1000.0,  # Reasonable upper bound
    ):
        self.projector = projector
        self.config = config
        self.max_abs_value = max_abs_value

    def verify(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Verify projection values are valid.

        Args:
            embeddings: Dictionary of sequence_id -> embedding

        Returns:
            Result dictionary with pass/fail and details
        """
        projections = self.projector.project(
            embeddings,
            self.config.method,
            self.config.n_components,
        )

        issues = []
        for seq_id, proj in projections.items():
            # Check for NaN/Inf
            if not np.isfinite(proj).all():
                issues.append(f"{seq_id}: contains NaN or Inf values")

            # Check bounds
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


# ============================================================================
# DISTANCE PRESERVATION ORACLE
# ============================================================================


class DistancePreservationOracle:
    """
    Oracle verifying that projections approximately preserve distances.

    Tests that pairwise distances in the original space correlate with
    pairwise distances in the projected space.
    """

    def __init__(
        self,
        projector: ProjectorProtocol,
        config: ProjectionOracleConfig,
    ):
        self.projector = projector
        self.config = config

    def verify(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Verify distance preservation property.

        Args:
            embeddings: Dictionary of sequence_id -> embedding

        Returns:
            Result dictionary with correlation and pass/fail
        """
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

        # Compute pairwise distances in original space
        seq_ids = list(embeddings.keys())
        original_dists = []
        projected_dists = []

        for i in range(len(seq_ids)):
            for j in range(i + 1, len(seq_ids)):
                id_i, id_j = seq_ids[i], seq_ids[j]

                # Original distance (using pooled embeddings)
                emb_i = embeddings[id_i]
                emb_j = embeddings[id_j]
                if len(emb_i.shape) > 1:
                    emb_i = emb_i.mean(axis=0)
                if len(emb_j.shape) > 1:
                    emb_j = emb_j.mean(axis=0)
                orig_dist = np.linalg.norm(emb_i - emb_j)

                # Projected distance
                proj_dist = np.linalg.norm(
                    projections[id_i] - projections[id_j]
                )

                original_dists.append(orig_dist)
                projected_dists.append(proj_dist)

        # Compute correlation
        correlation = np.corrcoef(original_dists, projected_dists)[0, 1]
        
        # Handle edge case of constant values
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


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def pca_config() -> ProjectionOracleConfig:
    """Oracle configuration for PCA projection."""
    return PROJECTION_ORACLE_CONFIGS["pca"]


@pytest.fixture(scope="module")
def umap_config() -> ProjectionOracleConfig:
    """Oracle configuration for UMAP projection."""
    return PROJECTION_ORACLE_CONFIGS["umap"]


@pytest.fixture(scope="module")
def mock_pca_projector(pca_config) -> MockProjector:
    """Mock projector for PCA."""
    return MockProjector(pca_config)


@pytest.fixture(scope="module")
def mock_umap_projector(umap_config) -> MockProjector:
    """Mock projector for UMAP."""
    return MockProjector(umap_config)


@pytest.fixture(scope="module")
def test_embeddings() -> Dict[str, np.ndarray]:
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


# ============================================================================
# TEST CLASSES
# ============================================================================


class TestProjectionDeterminism:
    """Tests for projection determinism oracle."""

    def test_pca_determinism(
        self,
        mock_pca_projector: MockProjector,
        pca_config: ProjectionOracleConfig,
        test_embeddings: Dict[str, np.ndarray],
    ):
        """Verify PCA projections are deterministic."""
        oracle = ProjectionDeterminismOracle(
            projector=mock_pca_projector,
            config=pca_config,
        )

        result = oracle.verify(test_embeddings)
        assert result["passed"], "PCA projection should be deterministic"

    def test_umap_determinism(
        self,
        mock_umap_projector: MockProjector,
        umap_config: ProjectionOracleConfig,
        test_embeddings: Dict[str, np.ndarray],
    ):
        """Verify UMAP projections are deterministic (with fixed seed)."""
        oracle = ProjectionDeterminismOracle(
            projector=mock_umap_projector,
            config=umap_config,
        )

        result = oracle.verify(test_embeddings)
        assert result["passed"], "UMAP projection should be deterministic with fixed seed"


class TestDimensionality:
    """Tests for projection dimensionality oracle."""

    def test_pca_dimensionality_2d(
        self,
        mock_pca_projector: MockProjector,
        pca_config: ProjectionOracleConfig,
        test_embeddings: Dict[str, np.ndarray],
    ):
        """Verify PCA projects to correct number of dimensions."""
        oracle = DimensionalityOracle(
            projector=mock_pca_projector,
            config=pca_config,
        )

        result = oracle.verify(test_embeddings)
        assert result["passed"], f"Dimensionality check failed: {result['issues']}"

    def test_umap_dimensionality_2d(
        self,
        mock_umap_projector: MockProjector,
        umap_config: ProjectionOracleConfig,
        test_embeddings: Dict[str, np.ndarray],
    ):
        """Verify UMAP projects to correct number of dimensions."""
        oracle = DimensionalityOracle(
            projector=mock_umap_projector,
            config=umap_config,
        )

        result = oracle.verify(test_embeddings)
        assert result["passed"], f"Dimensionality check failed: {result['issues']}"


class TestProjectionValueValidity:
    """Tests for projection value validity oracle."""

    def test_pca_values_valid(
        self,
        mock_pca_projector: MockProjector,
        pca_config: ProjectionOracleConfig,
        test_embeddings: Dict[str, np.ndarray],
    ):
        """Verify PCA projection values are valid."""
        oracle = ProjectionValueValidityOracle(
            projector=mock_pca_projector,
            config=pca_config,
        )

        result = oracle.verify(test_embeddings)
        assert result["passed"], f"Value validity failed: {result['issues']}"

    def test_umap_values_valid(
        self,
        mock_umap_projector: MockProjector,
        umap_config: ProjectionOracleConfig,
        diverse_test_embeddings: Dict[str, np.ndarray],
    ):
        """Verify UMAP projection values are valid for diverse inputs."""
        oracle = ProjectionValueValidityOracle(
            projector=mock_umap_projector,
            config=umap_config,
        )

        result = oracle.verify(diverse_test_embeddings)
        assert result["passed"], f"Value validity failed: {result['issues']}"


class TestDistancePreservation:
    """Tests for distance preservation oracle."""

    def test_pca_preserves_distances(
        self,
        mock_pca_projector: MockProjector,
        pca_config: ProjectionOracleConfig,
        test_embeddings: Dict[str, np.ndarray],
    ):
        """Verify PCA approximately preserves pairwise distances."""
        oracle = DistancePreservationOracle(
            projector=mock_pca_projector,
            config=pca_config,
        )

        result = oracle.verify(test_embeddings)
        assert result["passed"], (
            f"Distance preservation failed: correlation={result['correlation']:.3f} "
            f"< threshold={result['threshold']}"
        )

    def test_umap_preserves_local_structure(
        self,
        mock_umap_projector: MockProjector,
        umap_config: ProjectionOracleConfig,
        test_embeddings: Dict[str, np.ndarray],
    ):
        """Verify UMAP preserves local structure (lower threshold than PCA)."""
        oracle = DistancePreservationOracle(
            projector=mock_umap_projector,
            config=umap_config,
        )

        result = oracle.verify(test_embeddings)
        # UMAP may have lower correlation, that's expected
        print(f"UMAP distance correlation: {result.get('correlation', 'N/A')}")
        assert result["passed"], (
            f"Distance preservation failed: correlation={result['correlation']:.3f} "
            f"< threshold={result['threshold']}"
        )


# ============================================================================
# SESSION CLEANUP
# ============================================================================


@pytest.fixture(scope="module", autouse=True)
def cleanup_projection_results():
    """Clean up projection oracle results after module completes."""
    yield
    results = get_projection_oracle_results()
    if results:
        print(f"\n📊 Projection oracle tests completed: {len(results)} results")
    clear_projection_oracle_results()
