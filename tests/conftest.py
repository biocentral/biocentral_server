"""
Shared pytest fixtures for biocentral_server tests.

This file provides reusable fixtures for testing, including:
- FixedEmbedder instances for deterministic testing
- Pre-defined test sequences
- Pre-computed embeddings in various formats
- CLI options for embedder backend selection
"""

import os
import pytest
import numpy as np
from typing import Dict, List, Tuple

from tests.fixtures.fixed_embedder import (
    FixedEmbedder,
    FixedEmbedderRegistry,
    get_fixed_embedder,
)


# ============================================================================
# PYTEST HOOKS FOR CLI OPTIONS
# ============================================================================

def pytest_addoption(parser):
    """Add custom CLI options for test configuration."""
    parser.addoption(
        "--use-real-embedder",
        action="store_true",
        default=False,
        help="Use real ESM-2 8M embedder instead of FixedEmbedder for integration tests",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "real_embedder: mark test to only run with real embedder backend"
    )
    config.addinivalue_line(
        "markers", "fixed_embedder: mark test to only run with fixed embedder backend"
    )
    config.addinivalue_line(
        "markers", "modifies_registry: mark test that modifies FixedEmbedderRegistry"
    )


# ============================================================================
# FIXED EMBEDDER FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def fixed_embedder_prot_t5() -> FixedEmbedder:
    """Session-scoped FixedEmbedder for ProtT5 (1024-dim)."""
    return FixedEmbedder(model_name="prot_t5", seed_base=42)


@pytest.fixture(scope="session")
def fixed_embedder_esm2_t6() -> FixedEmbedder:
    """Session-scoped FixedEmbedder for ESM2-T6/8M (320-dim)."""
    return FixedEmbedder(model_name="esm2_t6", seed_base=42)


@pytest.fixture(scope="session")
def fixed_embedder_esm2_t33() -> FixedEmbedder:
    """Session-scoped FixedEmbedder for ESM2-T33 (1280-dim)."""
    return FixedEmbedder(model_name="esm2_t33", seed_base=42)


@pytest.fixture(scope="session")
def fixed_embedder_esm2_t36() -> FixedEmbedder:
    """Session-scoped FixedEmbedder for ESM2-T36 (2560-dim)."""
    return FixedEmbedder(model_name="esm2_t36", seed_base=42)


@pytest.fixture
def fixed_embedder_factory():
    """Factory fixture for creating FixedEmbedders with custom config."""

    def _factory(model_name: str = "prot_t5", seed_base: int = 42, **kwargs):
        return FixedEmbedder(model_name=model_name, seed_base=seed_base, **kwargs)

    return _factory

# ============================================================================
# TEST SEQUENCES
# ============================================================================

@pytest.fixture(scope="session")
def single_sequence() -> List[str]:
    """Single test sequence (same as used in triton tests)."""
    return [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGN"
    ]

@pytest.fixture(scope="session")
def five_sequences() -> List[str]:
    """Five test sequences of varying lengths (same as triton tests)."""
    return [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGN",
        "MKKLVLSLSLVLAFSSATAAFAAIPQNIRAQYPAVVKEQRQVVRSQNGDLADNIKKISDNLKAKIYAMHYVDVFYNKSLEKIMKDIQVTNATKTVYISINDLKRRMGGWKYPNMQVLLGRKGKKGKKAKRQ",
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        "MASMTGGQQMGRGSGMMGMGGMQGGFMGQMMGGGGFMGGMMMGGFMGGGMMGFMGGMMMGGMMGFMGGGMRP",
        "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
    ]

@pytest.fixture(scope="session")
def edge_case_sequences() -> List[str]:
    """Edge case sequences for boundary testing."""
    return [
        "M",  # Single residue
        "MK",  # Two residues
        "A" * 10,  # short
        "A" * 100,  # long
        "ACDEFGHIKLMNPQRSTVWY",  # All 20 standard amino acids
        "X" * 5,  # Unknown residues
        "MXKXAX",  # Mixed unknown
        "MKTAYIAK" * 50,  # Long repetitive (400 residues)
    ]

# ============================================================================
# PRE-COMPUTED EMBEDDINGS FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def prot_t5_embeddings_single(
    fixed_embedder_prot_t5: FixedEmbedder,
    single_sequence: List[str],
) -> List[np.ndarray]:
    """Pre-computed ProtT5 embeddings for single sequence."""
    return fixed_embedder_prot_t5.embed_batch(single_sequence, pooled=False)

@pytest.fixture(scope="session")
def prot_t5_embeddings_batch(
    fixed_embedder_prot_t5: FixedEmbedder,
    five_sequences: List[str],
) -> List[np.ndarray]:
    """Pre-computed ProtT5 embeddings for 5 sequences."""
    return fixed_embedder_prot_t5.embed_batch(five_sequences, pooled=False)

@pytest.fixture(scope="session")
def esm2_t33_embeddings_single(
    fixed_embedder_esm2_t33: FixedEmbedder,
    single_sequence: List[str],
) -> List[np.ndarray]:
    """Pre-computed ESM2-T33 embeddings for single sequence."""
    return fixed_embedder_esm2_t33.embed_batch(single_sequence, pooled=False)

@pytest.fixture(scope="session")
def esm2_t33_embeddings_batch(
    fixed_embedder_esm2_t33: FixedEmbedder,
    five_sequences: List[str],
) -> List[np.ndarray]:
    """Pre-computed ESM2-T33 embeddings for 5 sequences."""
    return fixed_embedder_esm2_t33.embed_batch(five_sequences, pooled=False)

@pytest.fixture(scope="session")
def esm2_t36_embeddings_single(
    fixed_embedder_esm2_t36: FixedEmbedder,
    single_sequence: List[str],
) -> List[np.ndarray]:
    """Pre-computed ESM2-T36 embeddings for single sequence."""
    return fixed_embedder_esm2_t36.embed_batch(single_sequence, pooled=False)

@pytest.fixture(scope="session")
def esm2_t36_embeddings_batch(
    fixed_embedder_esm2_t36: FixedEmbedder,
    five_sequences: List[str],
) -> List[np.ndarray]:
    """Pre-computed ESM2-T36 embeddings for 5 sequences."""
    return fixed_embedder_esm2_t36.embed_batch(five_sequences, pooled=False)

# ============================================================================
# DICT FORMAT FIXTURES (for model input)
# ============================================================================

def _sequences_to_dict(sequences: List[str]) -> Dict[str, str]:
    """Convert list of sequences to dict format."""
    return {f"seq{i}": seq for i, seq in enumerate(sequences)}

def _embeddings_to_dict(embeddings: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert list of embeddings to dict format."""
    return {f"seq{i}": emb for i, emb in enumerate(embeddings)}

def _convert_to_model_input(
    sequences: List[str],
    embeddings: List[np.ndarray],
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """Convert lists to dict format expected by prediction models."""
    return _sequences_to_dict(sequences), _embeddings_to_dict(embeddings)

@pytest.fixture(scope="session")
def prot_t5_model_input_single(
    single_sequence: List[str],
    prot_t5_embeddings_single: List[np.ndarray],
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """ProtT5 embeddings in model input format (single sequence)."""
    return _convert_to_model_input(single_sequence, prot_t5_embeddings_single)

@pytest.fixture(scope="session")
def prot_t5_model_input_batch(
    five_sequences: List[str],
    prot_t5_embeddings_batch: List[np.ndarray],
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """ProtT5 embeddings in model input format (5 sequences)."""
    return _convert_to_model_input(five_sequences, prot_t5_embeddings_batch)

@pytest.fixture(scope="session")
def esm2_t33_model_input_single(
    single_sequence: List[str],
    esm2_t33_embeddings_single: List[np.ndarray],
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """ESM2-T33 embeddings in model input format (single sequence)."""
    return _convert_to_model_input(single_sequence, esm2_t33_embeddings_single)

@pytest.fixture(scope="session")
def esm2_t33_model_input_batch(
    five_sequences: List[str],
    esm2_t33_embeddings_batch: List[np.ndarray],
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """ESM2-T33 embeddings in model input format (5 sequences)."""
    return _convert_to_model_input(five_sequences, esm2_t33_embeddings_batch)

@pytest.fixture(scope="session")
def esm2_t36_model_input_single(
    single_sequence: List[str],
    esm2_t36_embeddings_single: List[np.ndarray],
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """ESM2-T36 embeddings in model input format (single sequence)."""
    return _convert_to_model_input(single_sequence, esm2_t36_embeddings_single)

@pytest.fixture(scope="session")
def esm2_t36_model_input_batch(
    five_sequences: List[str],
    esm2_t36_embeddings_batch: List[np.ndarray],
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """ESM2-T36 embeddings in model input format (5 sequences)."""
    return _convert_to_model_input(five_sequences, esm2_t36_embeddings_batch)

# ============================================================================
# CLEANUP
# ============================================================================

@pytest.fixture(autouse=True, scope="function")
def reset_registry_for_isolated_tests(request):
    """
    Reset registry after tests that modify it.

    Only applies to tests marked with @pytest.mark.modifies_registry
    """
    yield
    if request.node.get_closest_marker("modifies_registry"):
        FixedEmbedderRegistry.clear()
