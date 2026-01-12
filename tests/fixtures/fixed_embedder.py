"""
Fixed Embedder for deterministic, reproducible testing.

This module provides a mock embedder that generates deterministic embeddings
based on sequence content, enabling reproducible edge case testing without
model overhead.

The embeddings are generated using a seeded random number generator based on
the SHA-256 hash of the input sequence, ensuring that:
1. Different sequences produce different embeddings
2. The same sequence always produces the same embedding
3. Embeddings have realistic statistical properties

Example:
    >>> embedder = FixedEmbedder(model_name="prot_t5")
    >>> embedding = embedder.embed("MKTAYIAK...")
    >>> assert embedding.shape == (len(sequence), 1024)

    >>> # Same sequence always gives same result
    >>> embedding2 = embedder.embed("MKTAYIAK...")
    >>> np.allclose(embedding, embedding2)
    True
"""

import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class FixedEmbedderConfig:
    """Configuration for FixedEmbedder."""

    embedding_dim: int = 1024  # Default ProtT5 dimension
    seed_base: int = 42
    noise_scale: float = 0.1
    # Supported dimensions for different "models"
    model_dimensions: Dict[str, int] = field(
        default_factory=lambda: {
            "prot_t5": 1024,
            "prot_t5_xl": 1024,
            "esm2_t33": 1280,
            "esm2_t36": 2560,
            "esm2_t6": 320,
            "esm2_t12": 480,
        }
    )

class FixedEmbedder:
    """
    Deterministic mock embedder for reproducible testing.

    Generates high-dimensional, noise-based embeddings that are:
    - Deterministic: Same sequence always produces same embedding
    - Reproducible: Results are consistent across runs and machines
    - Configurable: Supports different embedding dimensions
    - Fast: No model loading or GPU required, which is convenient for tests

    The embeddings are generated using a seeded random number generator
    based on the SHA-256 hash of the input sequence.

    Attributes:
        model_name: Name of the model being emulated
        embedding_dim: Dimension of the output embeddings
        seed_base: Base seed for reproducibility
        noise_scale: Scale factor for noise component
    """

    # Amino acid vocabulary for validation
    AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
    EXTENDED_AMINO_ACIDS = AMINO_ACIDS | set("BJOUXZ")  # Include ambiguous codes

    def __init__(
        self,
        model_name: str = "prot_t5",
        embedding_dim: Optional[int] = None,
        seed_base: int = 42,
        noise_scale: float = 0.1,
    ):
        """
        Initialize FixedEmbedder.

        Args:
            model_name: Name of the model to emulate (determines default dimension)
            embedding_dim: Override embedding dimension (if None, uses model default)
            seed_base: Base seed for reproducibility
            noise_scale: Scale factor for noise component
        """
        self.model_name = model_name
        self.config = FixedEmbedderConfig(seed_base=seed_base, noise_scale=noise_scale)

        # Determine embedding dimension
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = self.config.model_dimensions.get(
                model_name, self.config.model_dimensions["prot_t5"]
            )

        self.seed_base = seed_base
        self.noise_scale = noise_scale

        # Pre-compute amino acid base embeddings for consistency
        self._aa_embeddings = self._generate_aa_base_embeddings()

    def _sequence_to_seed(self, sequence: str) -> int:
        """
        Convert sequence to deterministic seed using SHA-256.

        Args:
            sequence: Protein sequence

        Returns:
            Integer seed derived from sequence hash
        """
        # Combine sequence with seed base for reproducibility
        hash_input = f"{self.seed_base}:{sequence}".encode("utf-8")
        hash_bytes = hashlib.sha256(hash_input).digest()
        # Use first 8 bytes as seed (enough for numpy)
        seed = int.from_bytes(hash_bytes[:8], byteorder="big") % (2**32)
        return seed

    def _generate_aa_base_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Generate base embeddings for each amino acid.

        These provide a consistent "semantic" component to embeddings,
        ensuring that positions with the same amino acid have similar
        base patterns across different sequences.

        Returns:
            Dictionary mapping amino acid to base embedding vector
        """
        aa_embeddings = {}
        all_aas = sorted(self.EXTENDED_AMINO_ACIDS | {"X", "-", "*"})  # Special chars

        for i, aa in enumerate(all_aas):
            # Deterministic seed for each amino acid
            rng = np.random.default_rng(self.seed_base + i * 1000)
            # Generate base embedding with unit norm
            base = rng.standard_normal(self.embedding_dim).astype(np.float32)
            base = base / np.linalg.norm(base)
            aa_embeddings[aa] = base

        return aa_embeddings

    def _get_aa_embedding(self, aa: str) -> np.ndarray:
        """Get base embedding for amino acid, with fallback for unknown chars."""
        if aa in self._aa_embeddings:
            return self._aa_embeddings[aa]
        # Fallback to 'X' (unknown) for any unrecognized character
        return self._aa_embeddings.get(
            "X", np.zeros(self.embedding_dim, dtype=np.float32)
        )

    def embed(self, sequence: str) -> np.ndarray:
        """
        Generate deterministic per-residue embedding for a sequence.

        Args:
            sequence: Protein sequence (string of amino acids)

        Returns:
            numpy array of shape (seq_len, embedding_dim) with dtype float32
        """
        if not sequence:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        seq_len = len(sequence)

        # Get sequence-specific seed
        seq_seed = self._sequence_to_seed(sequence)
        rng = np.random.default_rng(seq_seed)

        # Build embedding: base (AA-specific) + positional + noise
        embeddings = np.zeros((seq_len, self.embedding_dim), dtype=np.float32)

        for pos, aa in enumerate(sequence):
            # 1. Amino acid base component (semantic similarity)
            aa_base = self._get_aa_embedding(aa.upper())

            # 2. Positional encoding component (position-specific)
            pos_seed = seq_seed + pos * 100
            pos_rng = np.random.default_rng(pos_seed)
            positional = (
                pos_rng.standard_normal(self.embedding_dim).astype(np.float32) * 0.3
            )

            # 3. Context noise component (sequence-specific variation)
            noise = (
                rng.standard_normal(self.embedding_dim).astype(np.float32)
                * self.noise_scale
            )

            # Combine components
            embeddings[pos] = aa_base + positional + noise

        # Normalize to have reasonable magnitude (similar to real embeddings)
        # Real PLM embeddings typically have values in [-5, 5] range
        embeddings = embeddings / np.std(embeddings) * 1.0

        return embeddings

    def embed_pooled(self, sequence: str) -> np.ndarray:
        """
        Generate deterministic per-sequence (pooled) embedding.

        Args:
            sequence: Protein sequence

        Returns:
            numpy array of shape (embedding_dim,) with dtype float32
        """
        per_residue = self.embed(sequence)
        if len(per_residue) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return np.mean(per_residue, axis=0)

    def embed_batch(
        self,
        sequences: List[str],
        pooled: bool = False,
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of sequences.

        Args:
            sequences: List of protein sequences
            pooled: If True, return per-sequence embeddings; else per-residue

        Returns:
            List of numpy arrays (shapes depend on pooled flag)
        """
        if pooled:
            return [self.embed_pooled(seq) for seq in sequences]
        else:
            return [self.embed(seq) for seq in sequences]

    def embed_dict(
        self,
        sequences: Dict[str, str],
        pooled: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for sequences in dictionary format.

        Args:
            sequences: Dictionary mapping sequence IDs to sequences
            pooled: If True, return per-sequence embeddings

        Returns:
            Dictionary mapping sequence IDs to embeddings
        """
        result = {}
        for seq_id, seq in sequences.items():
            if pooled:
                result[seq_id] = self.embed_pooled(seq)
            else:
                result[seq_id] = self.embed(seq)
        return result

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension for this embedder."""
        return self.embedding_dim

    def __repr__(self) -> str:
        return (
            f"FixedEmbedder(model_name='{self.model_name}', "
            f"embedding_dim={self.embedding_dim}, seed_base={self.seed_base})"
        )

class FixedEmbedderRegistry:
    """
    Registry for FixedEmbedder instances with different configurations.

    Provides easy access to pre-configured embedders for testing.
    Thread-safe singleton pattern for consistent embeddings across tests.
    """

    _instances: Dict[str, FixedEmbedder] = {}

    @classmethod
    def get_embedder(
        cls,
        model_name: str = "prot_t5",
        seed_base: int = 42,
    ) -> FixedEmbedder:
        """
        Get or create a FixedEmbedder instance.

        Args:
            model_name: Model to emulate
            seed_base: Base seed for reproducibility

        Returns:
            FixedEmbedder instance
        """
        key = f"{model_name}:{seed_base}"
        if key not in cls._instances:
            cls._instances[key] = FixedEmbedder(
                model_name=model_name,
                seed_base=seed_base,
            )
        return cls._instances[key]

    @classmethod
    def clear(cls):
        """Clear the registry (useful for testing)."""
        cls._instances.clear()

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_fixed_embedder(model_name: str = "prot_t5") -> FixedEmbedder:
    """
    Get a FixedEmbedder for the specified model.

    This is the primary entry point for obtaining a FixedEmbedder instance.
    Uses the registry to ensure consistent embeddings across multiple calls.

    Args:
        model_name: Name of the model to emulate (prot_t5, esm2_t33, esm2_t36)

    Returns:
        FixedEmbedder instance configured for the specified model
    """
    return FixedEmbedderRegistry.get_embedder(model_name)

def generate_test_embeddings(
    sequences: List[str],
    model_name: str = "prot_t5",
    pooled: bool = False,
) -> List[np.ndarray]:
    """
    Convenience function to generate test embeddings.

    Args:
        sequences: List of protein sequences
        model_name: Model to emulate
        pooled: If True, return per-sequence embeddings

    Returns:
        List of embedding arrays
    """
    embedder = get_fixed_embedder(model_name)
    return embedder.embed_batch(sequences, pooled=pooled)

def generate_test_embeddings_dict(
    sequences: Dict[str, str],
    model_name: str = "prot_t5",
    pooled: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate test embeddings in dictionary format (matching model input format).

    Args:
        sequences: Dictionary mapping IDs to sequences
        model_name: Model to emulate
        pooled: If True, return per-sequence embeddings

    Returns:
        Dictionary mapping IDs to embeddings
    """
    embedder = get_fixed_embedder(model_name)
    return embedder.embed_dict(sequences, pooled=pooled)
