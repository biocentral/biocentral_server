"""
Fixtures for FixedEmbedder unit tests.

These tests verify the FixedEmbedder's own behavior with arbitrary sequences,
so we disable strict_dataset validation.
"""

import pytest

from tests.fixtures.fixed_embedder import FixedEmbedder


@pytest.fixture(autouse=True)
def disable_strict_dataset(monkeypatch):
    """
    Disable strict_dataset validation for FixedEmbedder unit tests.
    
    FixedEmbedder unit tests need to test with arbitrary sequences to verify
    edge cases, determinism, etc. The strict_dataset check is for production
    code that should only use canonical test sequences.
    """
    original_init = FixedEmbedder.__init__
    
    def patched_init(self, *args, strict_dataset=False, **kwargs):
        # Force strict_dataset=False for unit tests
        return original_init(self, *args, strict_dataset=False, **kwargs)
    
    monkeypatch.setattr(FixedEmbedder, "__init__", patched_init)
