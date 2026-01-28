"""Shared fixtures for endpoint integration tests."""

import os
import time
import pytest
import httpx
from typing import Any, Dict, List, Optional

from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET


def get_server_url() -> str:
    """Get the server URL from environment variable."""
    url = os.environ.get("CI_SERVER_URL")
    if not url:
        pytest.skip("CI_SERVER_URL environment variable not set. Start the server and set CI_SERVER_URL=http://localhost:9540")
    return url


@pytest.fixture(scope="session")
def server_url() -> str:
    """Get the server URL, skip tests if not available."""
    return get_server_url()


@pytest.fixture(scope="session")
def client(server_url) -> httpx.Client:
    """
    Create an httpx client for the integration test server.
    
    This client connects to the real running server instance.
    """
    http_client = httpx.Client(
        base_url=f"{server_url}/api/v1",
        timeout=httpx.Timeout(120.0, connect=10.0),
    )
    
    # Verify server is accessible (health endpoint is at root, not under /api/v1)
    try:
        response = httpx.get(f"{server_url}/health", timeout=10.0)
        if response.status_code != 200:
            pytest.fail(f"Server at {server_url} returned status {response.status_code}")
    except httpx.RequestError as e:
        pytest.fail(f"Cannot connect to server at {server_url}: {e}")
    
    yield http_client
    http_client.close()


@pytest.fixture(scope="session")
def poll_task(client):
    """
    Factory fixture to poll a task until completion.
    
    Usage:
        def test_async_operation(client, poll_task):
            response = client.post("/endpoint", json=data)
            task_id = response.json()["task_id"]
            result = poll_task(task_id)
            assert result["status"] == "finished"
    """
    def _poll(task_id: str, timeout: int = 120, poll_interval: float = 1.0) -> Dict[str, Any]:
        """
        Poll task status until completion or timeout.
        
        Args:
            task_id: The task ID to poll
            timeout: Maximum seconds to wait
            poll_interval: Seconds between polls
            
        Returns:
            The final task status response
            
        Raises:
            TimeoutError if task doesn't complete within timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            response = client.get(f"/biocentral_service/task_status/{task_id}")
            if response.status_code != 200:
                raise RuntimeError(f"Failed to get task status: {response.status_code}")
            
            status = response.json()
            task_status = status.get("status", "").lower()
            
            # Check for terminal states
            if task_status in ("finished", "completed", "done"):
                return status
            elif task_status in ("failed", "error"):
                return status
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
    
    return _poll


EMBEDDER_ESM2_T6_8M = "facebook/esm2_t6_8M_UR50D"
EMBEDDER_FIXED = "fixed"

EMBEDDER_MAP = {
    "esm2_t6_8m": EMBEDDER_ESM2_T6_8M,
    "fixed": EMBEDDER_FIXED,
}


def get_embedder_name() -> str:
    """Get embedder name from CI_EMBEDDER env var."""
    ci_embedder = os.environ.get("CI_EMBEDDER", "esm2_t6_8m").lower()
    
    if ci_embedder not in EMBEDDER_MAP:
        valid_options = ", ".join(EMBEDDER_MAP.keys())
        pytest.fail(f"Invalid CI_EMBEDDER='{ci_embedder}'. Valid options: {valid_options}")
    
    return EMBEDDER_MAP[ci_embedder]


def is_fixed_embedder() -> bool:
    """Check if we're using the FixedEmbedder mode."""
    return os.environ.get("CI_EMBEDDER", "esm2_t6_8m").lower() == "fixed"


@pytest.fixture(scope="session")
def embedder_name() -> str:
    """Get the embedder name to use for tests."""
    return get_embedder_name()


@pytest.fixture(scope="session")
def test_sequences() -> Dict[str, str]:
    """Standard test sequences from canonical dataset."""
    return {
        "protein_1": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
        "protein_2": CANONICAL_TEST_DATASET.get_by_id("standard_002").sequence,
        "protein_3": CANONICAL_TEST_DATASET.get_by_id("standard_003").sequence,
    }


@pytest.fixture(scope="session")
def single_test_sequence() -> Dict[str, str]:
    """Single sequence from canonical dataset."""
    return {
        "test_seq": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
    }


@pytest.fixture(scope="session")
def short_test_sequences() -> Dict[str, str]:
    """Short sequences from canonical dataset."""
    return {
        "short_1": CANONICAL_TEST_DATASET.get_by_id("length_short_10").sequence,
        "short_2": CANONICAL_TEST_DATASET.get_by_id("length_medium_50").sequence,
    }


@pytest.fixture(scope="session")
def minimum_length_sequences() -> Dict[str, str]:
    """Sequences at minimum length boundaries."""
    return {
        "min_1": CANONICAL_TEST_DATASET.get_by_id("length_min_1").sequence,
        "min_2": CANONICAL_TEST_DATASET.get_by_id("length_min_2").sequence,
        "short_5": CANONICAL_TEST_DATASET.get_by_id("length_short_5").sequence,
    }


@pytest.fixture(scope="session")
def long_sequences() -> Dict[str, str]:
    """Long sequences for performance and boundary testing."""
    return {
        "long_200": CANONICAL_TEST_DATASET.get_by_id("length_long_200").sequence,
        "very_long_400": CANONICAL_TEST_DATASET.get_by_id("length_very_long_400").sequence,
    }


@pytest.fixture(scope="session")
def unknown_token_sequences() -> Dict[str, str]:
    """Sequences containing unknown (X) residues."""
    return {
        "unknown_single": CANONICAL_TEST_DATASET.get_by_id("unknown_single").sequence,
        "unknown_multiple": CANONICAL_TEST_DATASET.get_by_id("unknown_multiple").sequence,
        "unknown_start": CANONICAL_TEST_DATASET.get_by_id("unknown_start").sequence,
        "unknown_end": CANONICAL_TEST_DATASET.get_by_id("unknown_end").sequence,
        "unknown_middle": CANONICAL_TEST_DATASET.get_by_id("unknown_middle").sequence,
        "unknown_scattered": CANONICAL_TEST_DATASET.get_by_id("unknown_scattered").sequence,
        "unknown_high_ratio": CANONICAL_TEST_DATASET.get_by_id("unknown_high_ratio").sequence,
    }


@pytest.fixture(scope="session")
def ambiguous_code_sequences() -> Dict[str, str]:
    """Sequences containing ambiguous amino acid codes (B, Z, J, U, O)."""
    return {
        "ambiguous_B": CANONICAL_TEST_DATASET.get_by_id("ambiguous_B").sequence,
        "ambiguous_Z": CANONICAL_TEST_DATASET.get_by_id("ambiguous_Z").sequence,
        "ambiguous_J": CANONICAL_TEST_DATASET.get_by_id("ambiguous_J").sequence,
        "selenocysteine": CANONICAL_TEST_DATASET.get_by_id("selenocysteine").sequence,
        "pyrrolysine": CANONICAL_TEST_DATASET.get_by_id("pyrrolysine").sequence,
    }


@pytest.fixture(scope="session")
def composition_edge_sequences() -> Dict[str, str]:
    """Sequences with unusual amino acid compositions."""
    return {
        "all_standard_aa": CANONICAL_TEST_DATASET.get_by_id("all_standard_aa").sequence,
        "homopolymer_A": CANONICAL_TEST_DATASET.get_by_id("homopolymer_A").sequence,
        "homopolymer_long": CANONICAL_TEST_DATASET.get_by_id("homopolymer_long").sequence,
        "hydrophobic_rich": CANONICAL_TEST_DATASET.get_by_id("hydrophobic_rich").sequence,
        "charged_rich": CANONICAL_TEST_DATASET.get_by_id("charged_rich").sequence,
        "proline_rich": CANONICAL_TEST_DATASET.get_by_id("proline_rich").sequence,
        "cysteine_rich": CANONICAL_TEST_DATASET.get_by_id("cysteine_rich").sequence,
    }


@pytest.fixture(scope="session")
def structural_motif_sequences() -> Dict[str, str]:
    """Sequences with structural motifs."""
    return {
        "alpha_helix": CANONICAL_TEST_DATASET.get_by_id("motif_alpha_helix").sequence,
        "beta_sheet": CANONICAL_TEST_DATASET.get_by_id("motif_beta_sheet").sequence,
        "glycine_loop": CANONICAL_TEST_DATASET.get_by_id("motif_glycine_loop").sequence,
    }


@pytest.fixture(scope="session")
def real_world_sequences() -> Dict[str, str]:
    """Real-world representative protein sequences."""
    return {
        "insulin_b": CANONICAL_TEST_DATASET.get_by_id("real_insulin_b").sequence,
        "ubiquitin": CANONICAL_TEST_DATASET.get_by_id("real_ubiquitin").sequence,
        "gfp_core": CANONICAL_TEST_DATASET.get_by_id("real_gfp_core").sequence,
    }


@pytest.fixture(scope="session")
def diverse_test_sequences() -> Dict[str, str]:
    """
    Diverse collection of sequences covering multiple categories.
    
    Useful for projection tests that need more data points.
    """
    return {
        "standard_001": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
        "standard_002": CANONICAL_TEST_DATASET.get_by_id("standard_002").sequence,
        "standard_003": CANONICAL_TEST_DATASET.get_by_id("standard_003").sequence,
        "insulin_b": CANONICAL_TEST_DATASET.get_by_id("real_insulin_b").sequence,
        "ubiquitin": CANONICAL_TEST_DATASET.get_by_id("real_ubiquitin").sequence,
        "gfp_core": CANONICAL_TEST_DATASET.get_by_id("real_gfp_core").sequence,
        "all_standard_aa": CANONICAL_TEST_DATASET.get_by_id("all_standard_aa").sequence,
        "alpha_helix": CANONICAL_TEST_DATASET.get_by_id("motif_alpha_helix").sequence,
    }


CANONICAL_STANDARD_IDS = ["standard_001", "standard_002", "standard_003"]
CANONICAL_LENGTH_EDGE_IDS = [
    "length_min_1", "length_min_2", "length_short_5",
    "length_short_10", "length_medium_50", "length_long_200",
]
CANONICAL_UNKNOWN_TOKEN_IDS = [
    "unknown_single", "unknown_multiple", "unknown_start",
    "unknown_end", "unknown_middle", "unknown_scattered", "unknown_high_ratio",
]
CANONICAL_AMBIGUOUS_CODE_IDS = [
    "ambiguous_B", "ambiguous_Z", "ambiguous_J", "selenocysteine", "pyrrolysine",
]
CANONICAL_REAL_WORLD_IDS = ["real_insulin_b", "real_ubiquitin", "real_gfp_core"]

ALL_CANONICAL_IDS = (
    CANONICAL_STANDARD_IDS
    + CANONICAL_LENGTH_EDGE_IDS
    + CANONICAL_UNKNOWN_TOKEN_IDS
    + CANONICAL_AMBIGUOUS_CODE_IDS
    + CANONICAL_REAL_WORLD_IDS
)


def get_sequence_by_id(seq_id: str) -> str:
    """Helper to get a sequence by its canonical ID."""
    return CANONICAL_TEST_DATASET.get_by_id(seq_id).sequence


@pytest.fixture(scope="session")
def all_canonical_sequences() -> Dict[str, str]:
    """All sequences from canonical dataset for comprehensive testing."""
    return {seq_id: get_sequence_by_id(seq_id) for seq_id in ALL_CANONICAL_IDS}


@pytest.fixture(scope="session")
def large_batch_sequences() -> Dict[str, str]:
    """
    Large batch of sequences for performance and batch handling tests.
    
    Creates 100 sequences using canonical sequences as templates.
    """
    base_sequences = [
        CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
        CANONICAL_TEST_DATASET.get_by_id("standard_002").sequence,
        CANONICAL_TEST_DATASET.get_by_id("standard_003").sequence,
        CANONICAL_TEST_DATASET.get_by_id("real_insulin_b").sequence,
        CANONICAL_TEST_DATASET.get_by_id("real_ubiquitin").sequence,
    ]
    
    sequences = {}
    for i in range(100):
        base_seq = base_sequences[i % len(base_sequences)]
        seq_id = f"batch_seq_{i:03d}"
        sequences[seq_id] = base_seq
    
    return sequences


def validate_task_response(response_json: Dict, expected_task_id_prefix: str = None) -> str:
    """
    Validate a task creation response and return the task_id.
    
    Args:
        response_json: The JSON response from the endpoint
        expected_task_id_prefix: Optional prefix to verify in task_id
        
    Returns:
        The task_id from the response
        
    Raises:
        AssertionError if validation fails
    """
    assert "task_id" in response_json, "Response missing 'task_id' field"
    task_id = response_json["task_id"]
    assert isinstance(task_id, str), f"task_id should be string, got {type(task_id)}"
    assert len(task_id) > 0, "task_id should not be empty"
    
    if expected_task_id_prefix:
        assert task_id.startswith(expected_task_id_prefix), \
            f"task_id '{task_id}' should start with '{expected_task_id_prefix}'"
    
    return task_id


def validate_error_response(response_json: Dict, expected_status: int = None) -> Dict:
    """
    Validate an error response structure.
    
    Args:
        response_json: The JSON response from the endpoint
        expected_status: Optional expected status code in response
        
    Returns:
        The error details dict
        
    Raises:
        AssertionError if validation fails
    """
    assert "detail" in response_json, "Error response missing 'detail' field"
    detail = response_json["detail"]
    
    if isinstance(detail, list):
        for error in detail:
            assert "loc" in error, "Validation error missing 'loc'"
            assert "msg" in error, "Validation error missing 'msg'"
            assert "type" in error, "Validation error missing 'type'"
    
    return response_json


@pytest.fixture(scope="session")
def test_run_id() -> str:
    """
    Unique identifier for this test run.
    
    Useful for creating unique resources that won't conflict
    between parallel test runs.
    """
    import uuid
    return str(uuid.uuid4())[:8]
