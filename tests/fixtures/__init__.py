"""Test fixtures package for biocentral_server tests."""

from .fixed_embedder import (
    FixedEmbedder,
    FixedEmbedderRegistry,
    FixedEmbedderConfig,
    get_fixed_embedder,
    generate_test_embeddings,
    generate_test_embeddings_dict,
)

from .fixed_embeddings import (
    # Test sequences
    SINGLE_SEQUENCE,
    FIVE_SEQUENCES,
    EDGE_CASE_SEQUENCES,
    MEMBRANE_SEQUENCES,
    LONG_SEQUENCES,
    TestSequenceSet,
    # Validation utilities
    validate_embedding_shape,
    validate_embedding_properties,
    assert_embedding_valid,
    get_expected_embedding_properties,
)

__all__ = [
    # FixedEmbedder
    "FixedEmbedder",
    "FixedEmbedderRegistry",
    "FixedEmbedderConfig",
    "get_fixed_embedder",
    "generate_test_embeddings",
    "generate_test_embeddings_dict",
    # Test sequences
    "SINGLE_SEQUENCE",
    "FIVE_SEQUENCES",
    "EDGE_CASE_SEQUENCES",
    "MEMBRANE_SEQUENCES",
    "LONG_SEQUENCES",
    "TestSequenceSet",
    # Validation utilities
    "validate_embedding_shape",
    "validate_embedding_properties",
    "assert_embedding_valid",
    "get_expected_embedding_properties",
]
