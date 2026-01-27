#!/usr/bin/env python3
"""
Metamorphic Relations for Embedding Service Testing.

This module implements metamorphic testing principles for validating embedding
services. Metamorphic testing addresses the oracle problem in ML systems by
defining relationships (metamorphic relations) between inputs and outputs
rather than expected outputs directly.

Implemented Relations:
    1. IdempotencyRelation: embed(seq) == embed(seq) across multiple calls
    2. BatchVarianceRelation: embed([A, B]) == [embed(A), embed(B)]
    3. ProjectionDeterminismRelation: project(embeddings) is deterministic with fixed seed
    4. ReversalRelation: embed(seq) vs embed(reverse(seq)) analysis
    5. ProgressiveMaskingRelation: embed(seq) vs embed(mask(seq, ratio)) degradation

Usage:
    from tests.scripts.metamorphic_relations import (
        IdempotencyRelation,
        BatchVarianceRelation,
        run_all_relations,
    )

    # Run individual relation
    relation = IdempotencyRelation(embedder)
    results = relation.verify(sequence)

    # Run all relations
    all_results = run_all_relations(embedder, sequences)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union
import random
import itertools

import numpy as np


# ============================================================================
# EMBEDDER PROTOCOL
# ============================================================================


class EmbedderProtocol(Protocol):
    """Protocol defining the embedder interface for metamorphic testing."""

    model_name: str

    def embed(self, sequence: str) -> np.ndarray:
        """Embed a single sequence, returning per-residue embeddings."""
        ...

    def embed_pooled(self, sequence: str) -> np.ndarray:
        """Embed a single sequence, returning pooled embedding."""
        ...

    def embed_batch(
        self, sequences: List[str], pooled: bool = False
    ) -> List[np.ndarray]:
        """Embed multiple sequences."""
        ...


# ============================================================================
# RESULT TYPES
# ============================================================================


class RelationVerdict(str, Enum):
    """Verdict for a metamorphic relation test."""
    
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass
class MetricResult:
    """Result containing distance/similarity metrics."""
    
    cosine_distance: float
    l2_distance: float
    kl_divergence: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "cosine_distance": self.cosine_distance,
            "l2_distance": self.l2_distance,
            "kl_divergence": self.kl_divergence,
        }


@dataclass
class RelationResult:
    """Result of a single metamorphic relation verification."""
    
    relation_name: str
    test_case: str
    parameter: str
    verdict: RelationVerdict
    metrics: Optional[MetricResult]
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "relation_name": self.relation_name,
            "test_case": self.test_case,
            "parameter": self.parameter,
            "verdict": self.verdict.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            **self.details,
        }
        if self.metrics:
            result.update(self.metrics.to_dict())
        return result


# ============================================================================
# METRIC COMPUTATION
# ============================================================================


def _ensure_1d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 1D by pooling if necessary."""
    if arr.ndim == 1:
        return arr
    elif arr.ndim == 2:
        return arr.mean(axis=0)
    else:
        raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def compute_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two embeddings."""
    a_flat = _ensure_1d(a)
    b_flat = _ensure_1d(b)
    
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    
    if norm_a == 0 or norm_b == 0:
        return 1.0
    
    cosine_similarity = np.dot(a_flat, b_flat) / (norm_a * norm_b)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    return float(1.0 - cosine_similarity)


def compute_l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2 (Euclidean) distance between two embeddings."""
    a_flat = _ensure_1d(a)
    b_flat = _ensure_1d(b)
    return float(np.linalg.norm(a_flat - b_flat))


def compute_kl_divergence(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute KL divergence after softmax normalization."""
    from scipy.special import softmax
    
    a_flat = _ensure_1d(a)
    b_flat = _ensure_1d(b)
    
    p = softmax(a_flat)
    q = softmax(b_flat) + epsilon
    q = q / q.sum()
    
    kl = np.sum(p * np.log(p / q))
    return float(max(0.0, kl))


def compute_all_metrics(a: np.ndarray, b: np.ndarray) -> MetricResult:
    """Compute all metrics between two embeddings."""
    return MetricResult(
        cosine_distance=compute_cosine_distance(a, b),
        l2_distance=compute_l2_distance(a, b),
        kl_divergence=compute_kl_divergence(a, b),
    )


def embeddings_are_identical(a: np.ndarray, b: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Check if two embeddings are numerically identical."""
    return np.allclose(a, b, rtol=rtol, atol=atol)


# ============================================================================
# BASE METAMORPHIC RELATION
# ============================================================================


class MetamorphicRelation(ABC):
    """
    Abstract base class for metamorphic relations.
    
    A metamorphic relation defines a relationship between inputs and their
    corresponding outputs that should always hold, regardless of the
    specific input values.
    """
    
    def __init__(
        self,
        embedder: EmbedderProtocol,
        threshold: float = 0.1,
        name: Optional[str] = None,
    ):
        """
        Initialize the metamorphic relation.
        
        Args:
            embedder: Embedder instance implementing EmbedderProtocol
            threshold: Threshold for pass/fail determination (cosine distance)
            name: Optional custom name for this relation
        """
        self.embedder = embedder
        self.threshold = threshold
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def verify(self, *args, **kwargs) -> List[RelationResult]:
        """
        Verify the metamorphic relation.
        
        Returns:
            List of RelationResult objects for each test case
        """
        pass
    
    def _create_result(
        self,
        test_case: str,
        parameter: str,
        verdict: RelationVerdict,
        metrics: Optional[MetricResult] = None,
        **details,
    ) -> RelationResult:
        """Helper to create a RelationResult."""
        return RelationResult(
            relation_name=self.name,
            test_case=test_case,
            parameter=parameter,
            verdict=verdict,
            metrics=metrics,
            threshold=self.threshold,
            details=details,
        )


# ============================================================================
# IDEMPOTENCY RELATION
# ============================================================================


class IdempotencyRelation(MetamorphicRelation):
    """
    Metamorphic relation verifying idempotency of embeddings.
    
    Property: embed(seq) == embed(seq) for any sequence seq
    
    This relation tests that:
    1. Multiple calls with the same sequence produce identical embeddings
    2. The embedder maintains no state that affects subsequent calls
    3. Results are reproducible within a session
    """
    
    def __init__(
        self,
        embedder: EmbedderProtocol,
        threshold: float = 1e-6,  # Very strict for idempotency
        num_repetitions: int = 3,
    ):
        """
        Initialize IdempotencyRelation.
        
        Args:
            embedder: Embedder to test
            threshold: Maximum allowed cosine distance (default: 1e-6)
            num_repetitions: Number of times to repeat embedding (default: 3)
        """
        super().__init__(embedder, threshold, name="Idempotency")
        self.num_repetitions = num_repetitions
    
    def verify(
        self,
        sequence: str,
        use_pooled: bool = True,
    ) -> List[RelationResult]:
        """
        Verify idempotency for a single sequence.
        
        Args:
            sequence: Sequence to test
            use_pooled: Whether to use pooled embeddings (default: True)
        
        Returns:
            List of RelationResult objects comparing consecutive calls
        """
        results = []
        embeddings = []
        
        # Generate embeddings multiple times
        embed_fn = self.embedder.embed_pooled if use_pooled else self.embedder.embed
        for i in range(self.num_repetitions):
            embedding = embed_fn(sequence)
            embeddings.append(embedding)
        
        # Compare each embedding to the first one
        reference = embeddings[0]
        for i, embedding in enumerate(embeddings[1:], start=2):
            is_identical = embeddings_are_identical(reference, embedding)
            metrics = compute_all_metrics(reference, embedding)
            
            verdict = RelationVerdict.PASSED if is_identical else RelationVerdict.FAILED
            
            result = self._create_result(
                test_case=f"seq_{len(sequence)}aa",
                parameter=f"call_1_vs_{i}",
                verdict=verdict,
                metrics=metrics,
                sequence_length=len(sequence),
                is_numerically_identical=is_identical,
                embedder_name=self.embedder.model_name,
            )
            results.append(result)
        
        return results
    
    def verify_batch(self, sequences: List[str]) -> List[RelationResult]:
        """Verify idempotency for multiple sequences."""
        all_results = []
        for seq in sequences:
            all_results.extend(self.verify(seq))
        return all_results


# ============================================================================
# BATCH VARIANCE RELATION
# ============================================================================


class BatchVarianceRelation(MetamorphicRelation):
    """
    Metamorphic relation verifying batch invariance of embeddings.
    
    Property: embed([A, B])[i] == embed([A])[0] for any batch containing A at index i
    
    This relation tests that:
    1. Embedding a sequence alone yields the same result as in a batch
    2. Batch order does not affect individual embeddings
    3. Batch size does not affect individual embeddings
    """
    
    def __init__(
        self,
        embedder: EmbedderProtocol,
        threshold: float = 0.1,
        batch_sizes: Optional[List[int]] = None,
    ):
        """
        Initialize BatchVarianceRelation.
        
        Args:
            embedder: Embedder to test
            threshold: Maximum allowed cosine distance
            batch_sizes: List of batch sizes to test (default: [2, 5, 10, 20])
        """
        super().__init__(embedder, threshold, name="BatchVariance")
        self.batch_sizes = batch_sizes or [2, 5, 10, 20]
    
    def verify(
        self,
        target_sequence: str,
        filler_sequences: List[str],
        seed: int = 42,
    ) -> List[RelationResult]:
        """
        Verify batch invariance for a target sequence.
        
        Args:
            target_sequence: The sequence to test
            filler_sequences: Additional sequences to form batches
            seed: Random seed for batch construction
        
        Returns:
            List of RelationResult objects for each batch configuration
        """
        results = []
        random.seed(seed)
        
        # Ground truth: sequence embedded alone
        single_embedding = self.embedder.embed_pooled(target_sequence)
        
        for batch_size in self.batch_sizes:
            if batch_size > len(filler_sequences) + 1:
                continue  # Skip if not enough filler sequences
            
            # Test multiple positions within the batch
            for position in ["first", "middle", "last"]:
                batch = self._create_batch(
                    target_sequence, filler_sequences, batch_size, position
                )
                target_idx = batch.index(target_sequence)
                
                # Embed the batch
                batch_embeddings = self.embedder.embed_batch(batch, pooled=True)
                batched_embedding = batch_embeddings[target_idx]
                
                # Compare
                is_identical = embeddings_are_identical(single_embedding, batched_embedding)
                metrics = compute_all_metrics(single_embedding, batched_embedding)
                
                passed = metrics.cosine_distance <= self.threshold
                verdict = RelationVerdict.PASSED if passed else RelationVerdict.FAILED
                
                result = self._create_result(
                    test_case=f"seq_{len(target_sequence)}aa",
                    parameter=f"batch_{batch_size}_pos_{position}",
                    verdict=verdict,
                    metrics=metrics,
                    batch_size=batch_size,
                    target_position=position,
                    target_index=target_idx,
                    is_numerically_identical=is_identical,
                    embedder_name=self.embedder.model_name,
                )
                results.append(result)
        
        return results
    
    def verify_order_independence(
        self,
        sequences: List[str],
        num_permutations: int = 5,
        seed: int = 42,
    ) -> List[RelationResult]:
        """
        Verify that batch order doesn't affect embeddings.
        
        Args:
            sequences: List of sequences to permute
            num_permutations: Number of random permutations to test
            seed: Random seed
        
        Returns:
            List of RelationResult objects
        """
        results = []
        random.seed(seed)
        
        if len(sequences) < 2:
            return results
        
        # Reference: embed in original order
        reference_embeddings = self.embedder.embed_batch(sequences, pooled=True)
        
        # Test permutations
        for perm_idx in range(num_permutations):
            permuted = sequences.copy()
            random.shuffle(permuted)
            
            # Create mapping from permuted back to original order
            perm_to_orig = {
                permuted.index(seq): i for i, seq in enumerate(sequences)
            }
            
            # Embed permuted batch
            permuted_embeddings = self.embedder.embed_batch(permuted, pooled=True)
            
            # Compare each sequence's embedding
            for orig_idx, seq in enumerate(sequences):
                perm_idx_in_batch = permuted.index(seq)
                
                ref_emb = reference_embeddings[orig_idx]
                perm_emb = permuted_embeddings[perm_idx_in_batch]
                
                is_identical = embeddings_are_identical(ref_emb, perm_emb)
                metrics = compute_all_metrics(ref_emb, perm_emb)
                
                passed = metrics.cosine_distance <= self.threshold
                verdict = RelationVerdict.PASSED if passed else RelationVerdict.FAILED
                
                result = self._create_result(
                    test_case=f"seq_{orig_idx}",
                    parameter=f"permutation_{perm_idx}",
                    verdict=verdict,
                    metrics=metrics,
                    is_numerically_identical=is_identical,
                    embedder_name=self.embedder.model_name,
                )
                results.append(result)
        
        return results
    
    def _create_batch(
        self,
        target: str,
        fillers: List[str],
        batch_size: int,
        target_position: str,
    ) -> List[str]:
        """Create a batch with target at specified position."""
        if batch_size == 1:
            return [target]
        
        n_fillers = batch_size - 1
        selected_fillers = [fillers[i % len(fillers)] for i in range(n_fillers)]
        
        if target_position == "first":
            return [target] + selected_fillers
        elif target_position == "last":
            return selected_fillers + [target]
        else:  # middle
            mid = len(selected_fillers) // 2
            return selected_fillers[:mid] + [target] + selected_fillers[mid:]


# ============================================================================
# PROJECTION DETERMINISM RELATION
# ============================================================================


class ProjectionDeterminismRelation(MetamorphicRelation):
    """
    Metamorphic relation verifying determinism of projection methods.
    
    Property: project(embeddings, seed=S) == project(embeddings, seed=S)
    
    This relation tests that dimensionality reduction methods (UMAP, PCA, t-SNE)
    produce identical results when using the same random seed.
    """
    
    SUPPORTED_METHODS = ["umap", "pca", "tsne"]
    
    def __init__(
        self,
        embedder: EmbedderProtocol,
        threshold: float = 1e-6,
        methods: Optional[List[str]] = None,
    ):
        """
        Initialize ProjectionDeterminismRelation.
        
        Args:
            embedder: Embedder to generate embeddings for projection
            threshold: Maximum allowed distance (default: 1e-6)
            methods: Projection methods to test (default: ["pca", "umap"])
        """
        super().__init__(embedder, threshold, name="ProjectionDeterminism")
        self.methods = methods or ["pca", "umap"]
    
    def verify(
        self,
        sequences: List[str],
        n_components: int = 2,
        seed: int = 42,
        num_repetitions: int = 3,
    ) -> List[RelationResult]:
        """
        Verify projection determinism for given sequences.
        
        Args:
            sequences: Sequences to embed and project
            n_components: Number of projection dimensions
            seed: Random seed for projection
            num_repetitions: Number of times to repeat projection
        
        Returns:
            List of RelationResult objects
        """
        results = []
        
        # Generate embeddings
        embeddings = self.embedder.embed_batch(sequences, pooled=True)
        embedding_matrix = np.array(embeddings)
        
        for method in self.methods:
            projections = []
            
            for i in range(num_repetitions):
                proj = self._project(embedding_matrix, method, n_components, seed)
                projections.append(proj)
            
            # Compare each projection to the first
            reference = projections[0]
            for i, proj in enumerate(projections[1:], start=2):
                # Use Frobenius norm for matrix comparison
                diff = np.linalg.norm(reference - proj, ord='fro')
                is_identical = diff < self.threshold
                
                verdict = RelationVerdict.PASSED if is_identical else RelationVerdict.FAILED
                
                result = self._create_result(
                    test_case=f"n_seqs_{len(sequences)}",
                    parameter=f"{method}_rep_1_vs_{i}",
                    verdict=verdict,
                    metrics=None,
                    frobenius_diff=float(diff),
                    method=method,
                    n_components=n_components,
                    seed=seed,
                    is_deterministic=is_identical,
                    embedder_name=self.embedder.model_name,
                )
                results.append(result)
        
        return results
    
    def _project(
        self,
        embeddings: np.ndarray,
        method: str,
        n_components: int,
        seed: int,
    ) -> np.ndarray:
        """Apply projection method to embeddings."""
        if method == "pca":
            from sklearn.decomposition import PCA
            projector = PCA(n_components=n_components, random_state=seed)
        elif method == "umap":
            try:
                import umap
                projector = umap.UMAP(
                    n_components=n_components,
                    random_state=seed,
                    n_neighbors=min(15, len(embeddings) - 1),
                    min_dist=0.1,
                )
            except ImportError:
                # Return zeros if UMAP not available
                return np.zeros((len(embeddings), n_components))
        elif method == "tsne":
            from sklearn.manifold import TSNE
            projector = TSNE(
                n_components=n_components,
                random_state=seed,
                perplexity=min(30, len(embeddings) - 1),
            )
        else:
            raise ValueError(f"Unknown projection method: {method}")
        
        return projector.fit_transform(embeddings)


# ============================================================================
# REVERSAL RELATION (EXPERIMENTAL)
# ============================================================================


class ReversalRelation(MetamorphicRelation):
    """
    Experimental metamorphic relation analyzing sequence reversal effects.
    
    Property: Analyze relationship between embed(seq) and embed(reverse(seq))
    
    This is an exploratory relation to understand how PLMs handle reversed
    sequences. Unlike other relations, this doesn't have a strict pass/fail
    criterion but provides insights into model behavior.
    """
    
    def __init__(
        self,
        embedder: EmbedderProtocol,
        threshold: float = 0.5,  # Relaxed threshold for exploration
    ):
        """
        Initialize ReversalRelation.
        
        Args:
            embedder: Embedder to test
            threshold: Threshold for "significant" difference
        """
        super().__init__(embedder, threshold, name="SequenceReversal")
    
    def verify(
        self,
        sequence: str,
    ) -> List[RelationResult]:
        """
        Analyze embedding relationship for a sequence and its reverse.
        
        Args:
            sequence: Sequence to test
        
        Returns:
            List containing single RelationResult with analysis
        """
        reversed_seq = sequence[::-1]
        
        # Embed both
        original_emb = self.embedder.embed_pooled(sequence)
        reversed_emb = self.embedder.embed_pooled(reversed_seq)
        
        metrics = compute_all_metrics(original_emb, reversed_emb)
        
        # Categorize the relationship
        if metrics.cosine_distance < 0.1:
            relationship = "highly_similar"
        elif metrics.cosine_distance < 0.3:
            relationship = "moderately_similar"
        elif metrics.cosine_distance < 0.5:
            relationship = "somewhat_different"
        else:
            relationship = "significantly_different"
        
        # For exploratory relations, we use INCONCLUSIVE as the default
        # since we're gathering data rather than testing a strict property
        verdict = RelationVerdict.INCONCLUSIVE
        
        result = self._create_result(
            test_case=f"seq_{len(sequence)}aa",
            parameter="forward_vs_reverse",
            verdict=verdict,
            metrics=metrics,
            sequence_length=len(sequence),
            relationship=relationship,
            embedder_name=self.embedder.model_name,
        )
        
        return [result]
    
    def verify_batch(self, sequences: List[str]) -> List[RelationResult]:
        """Analyze reversal relationship for multiple sequences."""
        all_results = []
        for seq in sequences:
            all_results.extend(self.verify(seq))
        return all_results


# ============================================================================
# PROGRESSIVE MASKING RELATION (EXPERIMENTAL)
# ============================================================================


class ProgressiveMaskingRelation(MetamorphicRelation):
    """
    Experimental metamorphic relation analyzing progressive token masking.
    
    Property: Analyze how embed(seq) changes as more tokens are replaced with 'X'
    
    This relation explores the degradation curve of embeddings as the
    sequence becomes increasingly corrupted. It helps identify:
    1. At what masking ratio embeddings diverge significantly
    2. Whether the degradation is linear or follows another pattern
    3. Model robustness to unknown/unresolved amino acids
    """
    
    MASK_TOKEN = "X"
    
    def __init__(
        self,
        embedder: EmbedderProtocol,
        threshold: float = 0.3,
        masking_ratios: Optional[List[float]] = None,
    ):
        """
        Initialize ProgressiveMaskingRelation.
        
        Args:
            embedder: Embedder to test
            threshold: Threshold for "significantly different"
            masking_ratios: Ratios to test (default: [0, 0.1, 0.2, ..., 0.9])
        """
        super().__init__(embedder, threshold, name="ProgressiveMasking")
        self.masking_ratios = masking_ratios or [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]
    
    def verify(
        self,
        sequence: str,
        seed: int = 42,
    ) -> List[RelationResult]:
        """
        Analyze embedding degradation under progressive masking.
        
        Args:
            sequence: Sequence to test
            seed: Random seed for mask position selection
        
        Returns:
            List of RelationResult objects for each masking ratio
        """
        results = []
        
        # Ground truth: unmasked sequence
        original_emb = self.embedder.embed_pooled(sequence)
        
        prev_metrics: Optional[MetricResult] = None
        
        for ratio in self.masking_ratios:
            masked_seq = self._mask_sequence(sequence, ratio, seed)
            masked_emb = self.embedder.embed_pooled(masked_seq)
            
            metrics = compute_all_metrics(original_emb, masked_emb)
            
            # Track whether this ratio crosses the significance threshold
            is_significant = metrics.cosine_distance > self.threshold
            
            # Calculate delta from previous ratio
            delta_cosine = 0.0
            if prev_metrics is not None:
                delta_cosine = metrics.cosine_distance - prev_metrics.cosine_distance
            
            verdict = RelationVerdict.INCONCLUSIVE  # Exploratory
            
            result = self._create_result(
                test_case=f"seq_{len(sequence)}aa",
                parameter=f"mask_{int(ratio * 100)}pct",
                verdict=verdict,
                metrics=metrics,
                sequence_length=len(sequence),
                masking_ratio=ratio,
                num_masked=int(len(sequence) * ratio),
                is_significantly_different=is_significant,
                delta_from_previous=delta_cosine,
                embedder_name=self.embedder.model_name,
            )
            results.append(result)
            
            prev_metrics = metrics
        
        return results
    
    def find_divergence_threshold(
        self,
        sequence: str,
        seed: int = 42,
        granularity: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Find the masking ratio at which embeddings diverge significantly.
        
        Args:
            sequence: Sequence to test
            seed: Random seed
            granularity: Step size for masking ratio search
        
        Returns:
            Dictionary with threshold information
        """
        original_emb = self.embedder.embed_pooled(sequence)
        
        ratios = np.arange(0, 1.0 + granularity, granularity)
        distances = []
        
        for ratio in ratios:
            masked_seq = self._mask_sequence(sequence, ratio, seed)
            masked_emb = self.embedder.embed_pooled(masked_seq)
            distance = compute_cosine_distance(original_emb, masked_emb)
            distances.append(distance)
        
        # Find first ratio where distance exceeds threshold
        threshold_ratio = None
        for i, (ratio, dist) in enumerate(zip(ratios, distances)):
            if dist > self.threshold:
                threshold_ratio = ratio
                break
        
        return {
            "sequence_length": len(sequence),
            "threshold_ratio": threshold_ratio,
            "ratios": ratios.tolist(),
            "distances": distances,
            "embedder_name": self.embedder.model_name,
        }
    
    def _mask_sequence(
        self,
        sequence: str,
        ratio: float,
        seed: int,
    ) -> str:
        """Mask a portion of the sequence with 'X' token."""
        if ratio <= 0:
            return sequence
        if ratio >= 1:
            return self.MASK_TOKEN * len(sequence)
        
        random.seed(seed + int(ratio * 1000))
        seq_list = list(sequence)
        n_mask = int(len(sequence) * ratio)
        
        positions = random.sample(range(len(sequence)), min(n_mask, len(sequence)))
        for pos in positions:
            seq_list[pos] = self.MASK_TOKEN
        
        return "".join(seq_list)
    
    def verify_batch(self, sequences: List[str], seed: int = 42) -> List[RelationResult]:
        """Analyze masking effects for multiple sequences."""
        all_results = []
        for seq in sequences:
            all_results.extend(self.verify(seq, seed))
        return all_results


# ============================================================================
# RELATION REGISTRY AND RUNNER
# ============================================================================


class RelationRegistry:
    """Registry for all available metamorphic relations."""
    
    RELATIONS = {
        "idempotency": IdempotencyRelation,
        "batch_variance": BatchVarianceRelation,
        "projection_determinism": ProjectionDeterminismRelation,
        "reversal": ReversalRelation,
        "progressive_masking": ProgressiveMaskingRelation,
    }
    
    @classmethod
    def get_relation(cls, name: str) -> type:
        """Get relation class by name."""
        if name not in cls.RELATIONS:
            raise ValueError(f"Unknown relation: {name}. Available: {list(cls.RELATIONS.keys())}")
        return cls.RELATIONS[name]
    
    @classmethod
    def list_relations(cls) -> List[str]:
        """List all available relation names."""
        return list(cls.RELATIONS.keys())


def run_all_relations(
    embedder: EmbedderProtocol,
    sequences: List[str],
    relations: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[RelationResult]]:
    """
    Run all (or selected) metamorphic relations on given sequences.
    
    Args:
        embedder: Embedder to test
        sequences: Sequences to use for testing
        relations: List of relation names to run (default: all)
        config: Configuration overrides for relations
    
    Returns:
        Dictionary mapping relation names to their results
    """
    config = config or {}
    relations = relations or RelationRegistry.list_relations()
    
    all_results = {}
    
    for rel_name in relations:
        rel_class = RelationRegistry.get_relation(rel_name)
        rel_config = config.get(rel_name, {})
        
        relation = rel_class(embedder, **rel_config)
        
        if rel_name in ["idempotency", "reversal"]:
            results = []
            for seq in sequences:
                results.extend(relation.verify(seq))
        elif rel_name == "batch_variance":
            results = []
            for i, seq in enumerate(sequences):
                fillers = [s for j, s in enumerate(sequences) if j != i]
                results.extend(relation.verify(seq, fillers))
        elif rel_name == "projection_determinism":
            if len(sequences) >= 5:  # Need enough sequences for projection
                results = relation.verify(sequences[:10])
            else:
                results = []
        elif rel_name == "progressive_masking":
            results = relation.verify_batch(sequences)
        else:
            results = []
        
        all_results[rel_name] = results
    
    return all_results


def summarize_results(results: Dict[str, List[RelationResult]]) -> Dict[str, Any]:
    """
    Generate summary statistics for relation results.
    
    Args:
        results: Dictionary of relation results
    
    Returns:
        Summary dictionary with pass/fail counts and rates
    """
    summary = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "inconclusive": 0,
        "pass_rate": 0.0,
        "by_relation": {},
    }
    
    for rel_name, rel_results in results.items():
        rel_summary = {
            "total": len(rel_results),
            "passed": sum(1 for r in rel_results if r.verdict == RelationVerdict.PASSED),
            "failed": sum(1 for r in rel_results if r.verdict == RelationVerdict.FAILED),
            "inconclusive": sum(1 for r in rel_results if r.verdict == RelationVerdict.INCONCLUSIVE),
        }
        rel_summary["pass_rate"] = (
            rel_summary["passed"] / rel_summary["total"] * 100
            if rel_summary["total"] > 0 else 0.0
        )
        
        summary["by_relation"][rel_name] = rel_summary
        summary["total_tests"] += rel_summary["total"]
        summary["passed"] += rel_summary["passed"]
        summary["failed"] += rel_summary["failed"]
        summary["inconclusive"] += rel_summary["inconclusive"]
    
    # Calculate overall pass rate (excluding inconclusive)
    decisive = summary["passed"] + summary["failed"]
    summary["pass_rate"] = summary["passed"] / decisive * 100 if decisive > 0 else 0.0
    
    return summary
