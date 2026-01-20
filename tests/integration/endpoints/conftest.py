"""
Shared fixtures for endpoint integration tests.

This module provides two embedding approaches for testing:
1. FixedEmbedder: Deterministic mock that generates high-dimensional noise-based
   embeddings. Fast, no GPU required, suitable for CI.
2. RealEmbedder: Uses ESM-2 8M for actual inference. Requires model download,
   slower but tests real embedding pipeline.

Usage:
    pytest tests/integration/endpoints/           # Uses FixedEmbedder (default)
    pytest tests/integration/endpoints/ --use-real-embedder  # Uses ESM-2 8M
    USE_REAL_EMBEDDER=1 pytest tests/integration/endpoints/  # Via env var
"""

import os
import pytest
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Protocol, Union
from dataclasses import dataclass
from unittest.mock import MagicMock, AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.fixtures.fixed_embedder import FixedEmbedder, FixedEmbedderRegistry


# =============================================================================
# EMBEDDER BACKEND ABSTRACTION
# =============================================================================

class EmbedderBackend(str, Enum):
    """Available embedder backends for integration tests."""
    FIXED = "fixed"  # FixedEmbedder - deterministic mock
    REAL = "real"    # Real ESM-2 8M model


class EmbedderProtocol(Protocol):
    """Protocol defining the embedder interface for tests."""
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        ...
    
    def embed(self, sequence: str) -> np.ndarray:
        """Embed a single sequence, returning per-residue embeddings."""
        ...
    
    def embed_pooled(self, sequence: str) -> np.ndarray:
        """Embed a single sequence, returning pooled embedding."""
        ...
    
    def embed_batch(self, sequences: List[str], pooled: bool = False) -> List[np.ndarray]:
        """Embed a batch of sequences."""
        ...
    
    def embed_dict(self, sequences: Dict[str, str], pooled: bool = False) -> Dict[str, np.ndarray]:
        """Embed sequences from a dictionary."""
        ...


@dataclass
class RealEmbedder:
    """
    Wrapper around biotrainer's ESM-2 8M embedder for real inference.
    
    Uses facebook/esm2_t6_8M_UR50D - the smallest ESM-2 model (8M parameters,
    320-dimensional embeddings) for fast CI testing with real inference.
    """
    
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    embedding_dim: int = 320
    _embedding_service: Optional[object] = None
    
    def __post_init__(self):
        """Initialize the real embedding service."""
        try:
            from biotrainer.embedders import get_embedding_service
            self._embedding_service = get_embedding_service(
                embedder_name=self.model_name,
                use_half_precision=False,
                custom_tokenizer_config=None,
                device="cpu",
            )
        except ImportError as e:
            raise ImportError(
                f"Failed to import biotrainer for real embedder: {e}. "
                "Make sure biotrainer is installed."
            )
    
    def embed(self, sequence: str) -> np.ndarray:
        """Embed a single sequence, returning per-residue embeddings."""
        embeddings = list(self._embedding_service.generate_embeddings(
            input_data=[sequence], reduce=False
        ))
        if embeddings:
            _, emb = embeddings[0]
            return emb
        return np.zeros((len(sequence), self.embedding_dim), dtype=np.float32)
    
    def embed_pooled(self, sequence: str) -> np.ndarray:
        """Embed a single sequence, returning pooled embedding."""
        embeddings = list(self._embedding_service.generate_embeddings(
            input_data=[sequence], reduce=True
        ))
        if embeddings:
            _, emb = embeddings[0]
            return emb
        return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def embed_batch(self, sequences: List[str], pooled: bool = False) -> List[np.ndarray]:
        """Embed a batch of sequences."""
        embeddings = list(self._embedding_service.generate_embeddings(
            input_data=sequences, reduce=pooled
        ))
        return [emb for _, emb in embeddings]
    
    def embed_dict(self, sequences: Dict[str, str], pooled: bool = False) -> Dict[str, np.ndarray]:
        """Embed sequences from a dictionary."""
        seq_list = list(sequences.values())
        seq_ids = list(sequences.keys())
        embeddings = self.embed_batch(seq_list, pooled=pooled)
        return dict(zip(seq_ids, embeddings))
    
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def embedder_backend(request) -> EmbedderBackend:
    """
    Determine which embedder backend to use.
    
    Priority:
    1. --use-real-embedder CLI flag
    2. USE_REAL_EMBEDDER environment variable
    3. Default to FIXED
    """
    use_real = (
        request.config.getoption("--use-real-embedder", default=False) or
        os.environ.get("USE_REAL_EMBEDDER", "").lower() in ("1", "true", "yes")
    )
    return EmbedderBackend.REAL if use_real else EmbedderBackend.FIXED


@pytest.fixture(scope="session")
def embedder(embedder_backend: EmbedderBackend) -> Union[FixedEmbedder, RealEmbedder]:
    """
    Get the appropriate embedder based on backend selection.
    
    Returns:
        FixedEmbedder or RealEmbedder instance configured for ESM-2 8M dimensions
    """
    if embedder_backend == EmbedderBackend.REAL:
        return RealEmbedder()
    else:
        # Use esm2_t6 config in FixedEmbedder to match ESM-2 8M dimensions (320)
        return FixedEmbedder(model_name="esm2_t6", seed_base=42, strict_dataset=False)


@pytest.fixture(scope="session")
def embedder_name(embedder_backend: EmbedderBackend) -> str:
    """Get the embedder name string for API requests."""
    if embedder_backend == EmbedderBackend.REAL:
        return "facebook/esm2_t6_8M_UR50D"
    return "esm2_t6"


@pytest.fixture(scope="session")
def embedding_dim(embedder_backend: EmbedderBackend) -> int:
    """Get the embedding dimension for current backend."""
    return 320  # Both ESM-2 8M and esm2_t6 FixedEmbedder use 320 dimensions


# =============================================================================
# TEST SEQUENCES
# =============================================================================

@pytest.fixture(scope="session")
def test_sequences() -> Dict[str, str]:
    """Standard test sequences for integration tests."""
    return {
        "protein_1": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGN",
        "protein_2": "MKKLVLSLSLVLAFSSATAAFAAIPQNIRAQYPAVVKEQRQVVRSQNGDLADNIKKISDNLKAKIYAMHYVDVFYNKSLEKIMKDIQVTNATKTVYISINDLKRRMGGWKYPNMQVLLGRKGKKGKKAKRQ",
        "protein_3": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
    }


@pytest.fixture(scope="session")
def single_test_sequence() -> Dict[str, str]:
    """Single sequence for quick tests."""
    return {
        "test_seq": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGN"
    }


@pytest.fixture(scope="session")
def short_test_sequences() -> Dict[str, str]:
    """Short sequences for faster test execution."""
    return {
        "short_1": "MVLSPADKTNVKAAWGKVGAHAGE",
        "short_2": "MGHFTEEDKATITSLWGKVNVE",
    }


# =============================================================================
# APPLICATION AND CLIENT FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def integration_app():
    """
    Create a FastAPI application with all endpoint routers for integration testing.
    
    This mimics the production app setup with all relevant routers included.
    """
    from biocentral_server.embeddings import embeddings_router, projection_router
    from biocentral_server.custom_models import custom_models_router
    from biocentral_server.predict import predict_router
    
    app = FastAPI(title="Biocentral Test Server")
    app.include_router(embeddings_router)
    app.include_router(projection_router)
    app.include_router(custom_models_router)
    app.include_router(predict_router)
    
    return app


@pytest.fixture(scope="module")
def integration_client(integration_app) -> TestClient:
    """Create a TestClient for the integration app."""
    return TestClient(integration_app)


# =============================================================================
# MOCK HELPERS
# =============================================================================

@pytest.fixture
def mock_rate_limiter():
    """Disable rate limiting for tests."""
    async def no_rate_limit():
        pass
    return no_rate_limit


@pytest.fixture
def mock_user_manager():
    """Mock UserManager for tests."""
    manager = MagicMock()
    manager.get_user_id_from_request = AsyncMock(return_value="test-integration-user")
    return manager


@pytest.fixture
def mock_task_manager_with_sync_execution():
    """
    Mock TaskManager that executes tasks synchronously for testing.
    
    This allows integration tests to verify task execution without
    async polling complications.
    """
    class SyncTaskManager:
        def __init__(self):
            self._tasks = {}
            self._task_counter = 0
        
        def add_task(self, task, user_id=None, task_id=None):
            if task_id is None:
                self._task_counter += 1
                task_id = f"sync-task-{self._task_counter}"
            
            # Execute task immediately (synchronously)
            try:
                result = task.run()  # Assuming tasks have a run() method
                self._tasks[task_id] = {"status": "completed", "result": result}
            except Exception as e:
                self._tasks[task_id] = {"status": "failed", "error": str(e)}
            
            return task_id
        
        def get_unique_task_id(self, task) -> str:
            self._task_counter += 1
            return f"sync-task-{self._task_counter}"
        
        def get_task_status(self, task_id: str) -> Dict:
            return self._tasks.get(task_id, {"status": "not_found"})
    
    return SyncTaskManager()


@pytest.fixture
def mock_embeddings_database():
    """
    Mock EmbeddingsDatabase for integration tests.
    
    Stores embeddings in memory, allowing tests to verify the
    embed -> store -> retrieve flow.
    """
    class InMemoryEmbeddingsDatabase:
        def __init__(self):
            self._embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        
        def filter_existing_embeddings(
            self, sequences: Dict[str, str], embedder_name: str, reduced: bool
        ):
            key = f"{embedder_name}:{reduced}"
            existing = {}
            non_existing = {}
            
            stored = self._embeddings.get(key, {})
            for seq_id, seq in sequences.items():
                if seq_id in stored:
                    existing[seq_id] = seq
                else:
                    non_existing[seq_id] = seq
            
            return existing, non_existing
        
        def add_embeddings(
            self, embeddings: Dict[str, np.ndarray], embedder_name: str, reduced: bool
        ):
            key = f"{embedder_name}:{reduced}"
            if key not in self._embeddings:
                self._embeddings[key] = {}
            self._embeddings[key].update(embeddings)
        
        def get_embeddings(
            self, seq_ids: List[str], embedder_name: str, reduced: bool
        ) -> Dict[str, np.ndarray]:
            key = f"{embedder_name}:{reduced}"
            stored = self._embeddings.get(key, {})
            return {sid: stored[sid] for sid in seq_ids if sid in stored}
        
        def clear(self):
            self._embeddings.clear()
    
    return InMemoryEmbeddingsDatabase()


@pytest.fixture
def mock_file_manager(tmp_path):
    """Mock FileManager using temp directory for test isolation."""
    class TempFileManager:
        def __init__(self, base_path):
            self._base_path = base_path
            self._models_dir = base_path / "models"
            self._models_dir.mkdir(exist_ok=True)
        
        def get_biotrainer_model_path(self, model_hash: str):
            model_path = self._models_dir / model_hash
            model_path.mkdir(exist_ok=True)
            return str(model_path)
        
        def get_biotrainer_result_files(self, model_hash: str) -> Dict[str, str]:
            model_path = self._models_dir / model_hash
            if not model_path.exists():
                return {}
            
            result_files = {}
            for key, filename in [
                ("out_config", "config.yml"),
                ("logging_out", "training.log"),
                ("out_file", "model.pt"),
            ]:
                file_path = model_path / filename
                if file_path.exists():
                    result_files[key] = file_path.read_text()
            
            return result_files
    
    return TempFileManager(tmp_path)


# =============================================================================
# EMBEDDING GENERATION HELPERS
# =============================================================================

@pytest.fixture
def generate_embeddings(embedder):
    """
    Factory fixture to generate embeddings using the configured backend.
    
    Usage:
        def test_something(generate_embeddings, test_sequences):
            embeddings = generate_embeddings(test_sequences, pooled=True)
    """
    def _generate(sequences: Dict[str, str], pooled: bool = False) -> Dict[str, np.ndarray]:
        return embedder.embed_dict(sequences, pooled=pooled)
    
    return _generate


# =============================================================================
# SKIP CONDITIONS
# =============================================================================

@pytest.fixture
def skip_if_fixed_embedder(embedder_backend):
    """Skip test if using FixedEmbedder backend."""
    if embedder_backend == EmbedderBackend.FIXED:
        pytest.skip("Test requires real embedder backend")


@pytest.fixture
def skip_if_real_embedder(embedder_backend):
    """Skip test if using real embedder backend (for mock-specific tests)."""
    if embedder_backend == EmbedderBackend.REAL:
        pytest.skip("Test requires fixed embedder backend")
