"""
Test fixtures and utilities for FixedEmbedder integration testing.

Provides pre-defined test sequences and expected embedding properties
for reproducible edge case testing without model overhead.

This module provides optional test utilities such as pre-defined sequences or validation helpers. 
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from tests.fixtures.fixed_embedder import (
    FixedEmbedder,
    get_fixed_embedder,
)

# ============================================================================
# TEST SEQUENCES - Matching existing test patterns
# ============================================================================

# Standard test sequences (same as used in triton tests)
SINGLE_SEQUENCE = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGN"
]

FIVE_SEQUENCES = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGN",
    "MKKLVLSLSLVLAFSSATAAFAAIPQNIRAQYPAVVKEQRQVVRSQNGDLADNIKKISDNLKAKIYAMHYVDVFYNKSLEKIMKDIQVTNATKTVYISINDLKRRMGGWKYPNMQVLLGRKGKKGKKAKRQ",
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "MASMTGGQQMGRGSGMMGMGGMQGGFMGQMMGGGGFMGGMMMGGFMGGGMMGFMGGMMMGGMMGFMGGGMRP",
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
]

@dataclass
class TestSequenceSet:
    """Collection of test sequences with metadata."""

    name: str
    sequences: List[str]
    description: str
    expected_properties: Dict = field(default_factory=dict)

# Edge case sequences for boundary testing
EDGE_CASE_SEQUENCES = TestSequenceSet(
    name="edge_cases",
    sequences=[
        "M",  # Single residue
        "MK",  # Two residues
        "A" * 10,  # short
        "A" * 100,  # long
        "ACDEFGHIKLMNPQRSTVWY",  # All 20 standard amino acids
        "X" * 5,  # Unknown
        "MXKXAX",  # Mixed unknown
        "MKTAYIAK" * 50,  # Long repetitive sequence (400 residues)
    ],
    description="Edge case sequences for boundary testing",
    expected_properties={
        "min_length": 1,
        "max_length": 400,
        "contains_unknown": True,
    },
)

# ============================================================================
# EXPECTED EMBEDDING PROPERTIES
# ============================================================================

def get_expected_embedding_properties(model_name: str) -> Dict:
    """Get expected properties for embeddings from a model."""
    properties = {
        "prot_t5": {
            "dimension": 1024,
            "dtype": np.float32,
            "value_range": (-10.0, 10.0),  # Approximate range
        },
        "esm2_t33": {
            "dimension": 1280,
            "dtype": np.float32,
            "value_range": (-10.0, 10.0),
        },
        "esm2_t36": {
            "dimension": 2560,
            "dtype": np.float32,
            "value_range": (-10.0, 10.0),
        },
    }
    return properties.get(model_name, properties["prot_t5"])

# ============================================================================
# EMBEDDING GENERATION UTILITIES
# ============================================================================

def generate_fixed_embeddings_for_sequences(
    sequences: List[str],
    model_name: str = "prot_t5",
    pooled: bool = False,
) -> List[np.ndarray]:
    """
    Generate fixed embeddings for a list of sequences. Duplicate of generate_test_embeddings from fixed_embedder.

    Args:
        sequences: List of protein sequences
        model_name: Model to emulate
        pooled: Whether to return per-sequence embeddings

    Returns:
        List of embedding arrays
    """
    embedder = get_fixed_embedder(model_name)
    return embedder.embed_batch(sequences, pooled=pooled)

def generate_fixed_embeddings_dict(
    sequences: Dict[str, str],
    model_name: str = "prot_t5",
    pooled: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate fixed embeddings in dictionary format (matching model input format).

    Args:
        sequences: Dictionary mapping IDs to sequences
        model_name: Model to emulate
        pooled: Whether to return per-sequence embeddings

    Returns:
        Dictionary mapping IDs to embeddings
    """
    embedder = get_fixed_embedder(model_name)
    return embedder.embed_dict(sequences, pooled=pooled)

def convert_sequences_to_test_format(
    sequences: List[str],
    model_name: str = "prot_t5",
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """
    Convert sequences to the format expected by prediction models.

    This is the primary utility for setting up test data that mimics
    the format used in production (sequences dict + embeddings dict).

    Args:
        sequences: List of protein sequences
        model_name: Model to emulate for embeddings

    Returns:
        Tuple of (sequences_dict, embeddings_dict) where:
        - sequences_dict maps seq_id -> sequence string
        - embeddings_dict maps seq_id -> per-residue embedding array
    """
    sequences_dict = {f"seq{i}": seq for i, seq in enumerate(sequences)}
    embeddings_dict = generate_fixed_embeddings_dict(sequences_dict, model_name)
    return sequences_dict, embeddings_dict

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_embedding_shape(
    embedding: np.ndarray,
    expected_length: int,
    model_name: str = "prot_t5",
    pooled: bool = False,
) -> bool:
    """
    Validate that an embedding has the expected shape.

    Args:
        embedding: The embedding array to validate
        expected_length: Expected sequence length
        model_name: Model name (determines embedding dimension)
        pooled: Whether this is a pooled embedding

    Returns:
        True if shape is valid
    """
    props = get_expected_embedding_properties(model_name)
    expected_dim = props["dimension"]

    if pooled:
        expected_shape = (expected_dim,)
    else:
        expected_shape = (expected_length, expected_dim)

    return embedding.shape == expected_shape

def validate_embedding_properties(
    embedding: np.ndarray,
    model_name: str = "prot_t5",
) -> Dict[str, bool]:
    """
    Validate various properties of an embedding.

    Args:
        embedding: The embedding array to validate
        model_name: Model name for expected properties

    Returns:
        Dictionary of validation results
    """
    props = get_expected_embedding_properties(model_name)

    return {
        "correct_dtype": embedding.dtype == props["dtype"],
        "no_nan": not np.any(np.isnan(embedding)),
        "no_inf": not np.any(np.isinf(embedding)),
        "not_all_zeros": not np.allclose(embedding, 0),
        "within_range": np.all(embedding >= props["value_range"][0])
        and np.all(embedding <= props["value_range"][1]),
    }

def assert_embedding_valid(
    embedding: np.ndarray,
    sequence_length: int,
    model_name: str = "prot_t5",
    pooled: bool = False,
) -> None:
    """
    Assert that an embedding is valid, raising AssertionError if not.

    Args:
        embedding: The embedding array to validate
        sequence_length: Expected sequence length
        model_name: Model name for validation
        pooled: Whether this is a pooled embedding

    Raises:
        AssertionError: If embedding is invalid
    """
    # Check shape
    assert validate_embedding_shape(
        embedding, sequence_length, model_name, pooled
    ), f"Invalid shape: {embedding.shape}"

    # Check properties
    props = validate_embedding_properties(embedding, model_name)
    for prop_name, is_valid in props.items():
        assert is_valid, f"Embedding failed {prop_name} check"
