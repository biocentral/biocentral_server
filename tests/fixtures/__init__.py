"""Test fixtures package for biocentral_server tests."""

from .fixed_embedder import (
    FixedEmbedder,
    FixedEmbedderRegistry,
    FixedEmbedderConfig,
    get_fixed_embedder,
    generate_test_embeddings,
    generate_test_embeddings_dict,
    convert_sequences_to_test_format,
    # Validation utilities
    validate_embedding_shape,
    validate_embedding_properties,
    assert_embedding_valid,
    get_expected_embedding_properties,
)

from .test_dataset import (
    # Dataset and sequence classes
    TestSequence,
    TestDataset,
    CANONICAL_TEST_DATASET,
    # Convenience functions
    get_test_sequences,
    get_test_sequences_dict,
    get_test_embeddings,
    get_dataset_statistics,
)

__all__ = [
    # FixedEmbedder
    "FixedEmbedder",
    "FixedEmbedderRegistry",
    "FixedEmbedderConfig",
    "get_fixed_embedder",
    "generate_test_embeddings",
    "generate_test_embeddings_dict",
    "convert_sequences_to_test_format",
    # Validation utilities
    "validate_embedding_shape",
    "validate_embedding_properties",
    "assert_embedding_valid",
    "get_expected_embedding_properties",
    # Canonical test dataset
    "TestSequence",
    "TestDataset",
    "CANONICAL_TEST_DATASET",
    "get_test_sequences",
    "get_test_sequences_dict",
    "get_test_embeddings",
    "get_dataset_statistics",
]
