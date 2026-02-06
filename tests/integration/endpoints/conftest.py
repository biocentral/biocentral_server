"""Shared fixtures for endpoint integration tests."""

import os
import time
import logging
import sys
import pytest
import httpx
import redis
from urllib.parse import urlparse
from typing import Any, Dict, Generator

from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET

# Suppress verbose httpx/httpcore logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_redis_port() -> int:
    """Get Redis port from environment, accounting for CI port mapping."""
    # In CI, tests run on host, Redis is exposed on REDIS_JOBS_PORT
    return int(os.environ.get("REDIS_JOBS_PORT", "6379"))


@pytest.fixture(scope="session")
def flush_redis():
    """Factory fixture to flush Redis job queue on demand."""
    def _flush():
        redis_port = get_redis_port()
        try:
            r = redis.Redis(host="localhost", port=redis_port, db=0)
            r.flushdb()
            # Give worker time to notice queue is empty
            time.sleep(0.5)
        except redis.ConnectionError:
            pass  # Redis not available, skip flush
    return _flush


def get_server_url() -> str:
    """Get the server URL from environment variable."""
    url = os.environ.get("CI_SERVER_URL")
    if not url:
        msg = (
            "CI_SERVER_URL environment variable not set. Start the server and set "
            "CI_SERVER_URL=http://localhost:9540"
        )
        # Log and print so the reason is visible in CI/local runs before skipping
        logging.warning(msg)
        print(f"SKIPPING integration tests: {msg}", file=sys.stderr)
        pytest.skip(msg)
    return url


@pytest.fixture(scope="session")
def server_url() -> str:
    """Get the server URL, skip tests if not available."""
    return get_server_url()


@pytest.fixture(scope="session")
def client(server_url) -> Generator[httpx.Client, None, None]:
    """
    Create an httpx client for the integration test server.
    
    This client connects to the real running server instance.
    """
    # Use transport with retries for connection resilience
    transport = httpx.HTTPTransport(retries=5)
    http_client = httpx.Client(
        base_url=f"{server_url}/api/v1",
        timeout=httpx.Timeout(500.0, connect=10.0),
        transport=transport,
    )
    
    # Verify server is accessible (health endpoint is at root, not under /api/v1)
    try:
        response = httpx.get(f"{server_url}/health", timeout=10.0)
        if response.status_code != 200:
            pytest.fail(f"Server at {server_url} returned status {response.status_code}")
    except httpx.RequestError as e:
        pytest.fail(f"Cannot connect to server at {server_url}: {e}")
    
    # If CI_EMBEDDER is set to 'fixed', wrap client to intercept embed endpoints
    if os.environ.get("CI_EMBEDDER", "").lower() == "fixed":
        import uuid
        from tests.fixtures.fixed_embedder import get_fixed_embedder

        original_post = http_client.post
        original_get = http_client.get

        class FakeResponse:
            def __init__(self, status_code: int, payload: Dict):
                self.status_code = status_code
                self._payload = payload

            def json(self, *args, **kwargs):
                return self._payload

        # Track fake tasks created by this test client
        _fake_tasks: Dict[str, Dict] = {}

        def _fake_post(url, *args, **kwargs):
            # Intercept embedding and projection endpoints (match path exactly)
            try:
                path = urlparse(str(url)).path
            except Exception:
                path = str(url)

            if path.startswith("/embeddings_service/embed"):
                # Prefer explicit JSON kwarg; otherwise fall back to empty dict
                data = kwargs.get("json") or {}
                # Basic validation: require embedder_name and non-empty sequence_data
                if not data.get("embedder_name"):
                    return FakeResponse(422, {"detail": "embedder_name missing"})
                if not data.get("sequence_data"):
                    return FakeResponse(422, {"detail": "sequence_data empty"})

                # Fast local embedding: compute deterministic embeddings for provided sequences
                embedder_name = data.get("embedder_name")
                seqs = data.get("sequence_data")
                pooled = bool(data.get("reduce", False))
                try:
                    fe = get_fixed_embedder(model_name=embedder_name, strict_dataset=True)
                except Exception:
                    # Fallback to default fixed embedder
                    fe = get_fixed_embedder()

                # compute embeddings (not stored anywhere) to simulate work
                if isinstance(seqs, dict):
                    fe.embed_dict(seqs, pooled=pooled)
                elif isinstance(seqs, list):
                    fe.embed_batch(seqs, pooled=pooled)

                # Return a fake task_id and register a finished DTO for it
                task_id = f"local-{uuid.uuid4().hex[:8]}"
                _fake_tasks[task_id] = {"status": "FINISHED", "result": {}}
                return FakeResponse(200, {"task_id": task_id})

            if path.startswith("/projection_service/project"):
                # Intercept projection endpoint when using fixed embedder
                # This endpoint performs dimensionality reduction (PCA, UMAP, t-SNE)
                # on protein embeddings for visualization purposes
                data = kwargs.get("json") or {}
                if not data.get("sequence_data"):
                    return FakeResponse(422, {"detail": "sequence_data empty"})

                seqs = data.get("sequence_data")
                method = data.get("method", "pca")
                n_components = data.get("config", {}).get("n_components", 2)

                # Validate method - must be a known dimensionality reduction method
                valid_methods = {"pca", "umap", "tsne", "pacmap", "trimap"}
                if method.lower() not in valid_methods:
                    return FakeResponse(400, {"detail": f"Unknown method: {method}"})

                # Return a fake task_id with projection result
                task_id = f"local-{uuid.uuid4().hex[:8]}"
                # Create fake projection result with deterministic coordinates
                # Real service returns reduced coordinates from ProtSpace
                seq_ids = list(seqs.keys()) if isinstance(seqs, dict) else [f"seq_{i}" for i in range(len(seqs))]
                projection_result = {
                    method: {
                        "identifier": seq_ids,
                        **{f"D{d+1}": [float(i * 0.1 + d * 0.01) for i, _ in enumerate(seq_ids)] 
                           for d in range(n_components)}
                    }
                }
                _fake_tasks[task_id] = {"status": "FINISHED", "projection_result": projection_result}
                return FakeResponse(200, {"task_id": task_id})

            return original_post(url, *args, **kwargs)

        def _fake_get(url, *args, **kwargs):
            # Intercept only exact task status polling paths and return finished immediately
            try:
                path = urlparse(str(url)).path
            except Exception:
                path = str(url)

            if path.startswith("/biocentral_service/task_status/local-"):
                # extract task_id from path (last path component)
                try:
                    task_id = path.rstrip("/").split("/")[-1]
                except Exception:
                    task_id = None

                if task_id and task_id in _fake_tasks:
                    dto = _fake_tasks[task_id]
                    return FakeResponse(200, {"dtos": [dto]})

            return original_get(url, *args, **kwargs)

        http_client.post = _fake_post
        http_client.get = _fake_get

    yield http_client
    http_client.close()


def _make_request_with_retry(client, method: str, url: str, max_retries: int = 5, **kwargs) -> httpx.Response:
    """Make an HTTP request with retry logic for connection errors."""
    last_error = None
    for attempt in range(max_retries):
        try:
            if method.upper() == "GET":
                return client.get(url, **kwargs)
            elif method.upper() == "POST":
                return client.post(url, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
            last_error = e
            wait_time = min(2 ** attempt, 30)  # Cap at 30 seconds
            logging.debug(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
                continue
            raise
    raise last_error


@pytest.fixture(scope="session")
def poll_task(client):
    """
    Factory fixture to poll a task until completion with robust error handling.
    """
    def _poll(
        task_id: str,
        timeout: int = 120,
        poll_interval: float = 2.0,
        max_consecutive_errors: int = 10,
    ) -> Dict[str, Any]:
        """
        Poll task status until completion or timeout.
        
        Handles transient connection errors gracefully - only fails after
        multiple consecutive errors, not on the first hiccup.
        
        Args:
            task_id: The task ID to poll
            timeout: Maximum seconds to wait for task completion
            poll_interval: Seconds between polls
            max_consecutive_errors: Number of consecutive errors before giving up
            
        Returns:
            The final task DTO with status and results
            
        Raises:
            TimeoutError: If task doesn't complete within timeout
            RuntimeError: If too many consecutive connection errors occur
        """
        start = time.time()
        consecutive_errors = 0
        last_status = "UNKNOWN"
        
        while time.time() - start < timeout:
            elapsed = int(time.time() - start)
            
            try:
                response = _make_request_with_retry(
                    client, "GET", f"/biocentral_service/task_status/{task_id}",
                    max_retries=3,
                )
                # Reset error counter on successful request
                consecutive_errors = 0
                
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
                consecutive_errors += 1
                logging.warning(
                    f"[POLL] Connection error ({consecutive_errors}/{max_consecutive_errors}): {e}"
                )
                
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"Task {task_id} polling failed: {consecutive_errors} consecutive connection errors. "
                        f"Last error: {e}. Last known status: {last_status}"
                    )
                
                # Wait longer after errors
                time.sleep(poll_interval * 2)
                continue
                
            if response.status_code != 200:
                consecutive_errors += 1
                logging.warning(
                    f"[POLL] Bad status code {response.status_code} ({consecutive_errors}/{max_consecutive_errors})"
                )
                
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"Task {task_id} polling failed: got status {response.status_code} "
                        f"{consecutive_errors} times. Last known status: {last_status}"
                    )
                
                time.sleep(poll_interval)
                continue
            
            response_json = response.json()
            dtos = response_json.get("dtos", [])
            
            if not dtos:
                time.sleep(poll_interval)
                continue
            
            # Get the latest DTO's status
            latest_dto = dtos[-1]
            task_status = latest_dto.get("status", "").upper()
            last_status = task_status
            
            # Log progress periodically
            if elapsed % 30 == 0 and elapsed > 0:
                logging.info(f"[POLL] Task {task_id}: {task_status} ({elapsed}s elapsed)")
            
            # Check for terminal states
            if task_status in ("FINISHED", "COMPLETED", "DONE"):
                logging.info(f"[POLL] Task {task_id} completed successfully in {elapsed}s")
                return latest_dto
            elif task_status in ("FAILED", "ERROR"):
                logging.warning(f"[POLL] Task {task_id} failed: {latest_dto.get('error', 'unknown')}")
                return latest_dto
            
            time.sleep(poll_interval)
        
        raise TimeoutError(
            f"Task {task_id} did not complete within {timeout}s. Last status: {last_status}"
        )
    
    return _poll


EMBEDDER_ESM2_T6_8M = "facebook/esm2_t6_8M_UR50D"
EMBEDDER_FIXED = "fixed"

EMBEDDER_MAP = {
    "esm2_t6_8m": EMBEDDER_ESM2_T6_8M,
    "fixed": EMBEDDER_FIXED,
}


def get_embedder_name() -> str:
    ci_embedder = os.environ.get("CI_EMBEDDER", "esm2_t6_8m").lower()
    
    if ci_embedder not in EMBEDDER_MAP:
        valid_options = ", ".join(EMBEDDER_MAP.keys())
        pytest.fail(f"Invalid CI_EMBEDDER='{ci_embedder}'. Valid options: {valid_options}")
    
    embedder_name = EMBEDDER_MAP[ci_embedder]
    
    #print(f"\n[CONFIG] Using embedder: {embedder_name}")
    return embedder_name


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
    }


@pytest.fixture(scope="session")
def single_test_sequence() -> Dict[str, str]:
    """Single sequence from canonical dataset."""
    return {
        "test_seq": CANONICAL_TEST_DATASET.get_by_id("standard_001").sequence,
    }



@pytest.fixture(scope="session")
def shared_embedding_sequences() -> Dict[str, str]:
    """
    Sequences shared between embedding and projection tests.
    
    These sequences are embedded with reduce=True in test_embed_flow.py
    and then reused by projection tests, ensuring cache hits.
    Must contain at least 2 sequences for PCA to work.
    """
    return {
        "short_1": CANONICAL_TEST_DATASET.get_by_id("length_short_10").sequence,
        "short_2": CANONICAL_TEST_DATASET.get_by_id("length_medium_50").sequence,
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
    }


@pytest.fixture(scope="session")
def composition_edge_sequences() -> Dict[str, str]:
    """Sequences with unusual amino acid compositions."""
    return {
        "all_standard_aa": CANONICAL_TEST_DATASET.get_by_id("all_standard_aa").sequence,
        "homopolymer_A": CANONICAL_TEST_DATASET.get_by_id("homopolymer_A").sequence, 
    }


@pytest.fixture(scope="session")
def structural_motif_sequences() -> Dict[str, str]:
    """Sequences with structural motifs."""
    return {
        "alpha_helix": CANONICAL_TEST_DATASET.get_by_id("motif_alpha_helix").sequence,
        "beta_sheet": CANONICAL_TEST_DATASET.get_by_id("motif_beta_sheet").sequence, 
    }


@pytest.fixture(scope="session")
def real_world_sequences() -> Dict[str, str]:
    """Real-world representative protein sequences."""
    return {
        "insulin_b": CANONICAL_TEST_DATASET.get_by_id("length_short_10").sequence,
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

@pytest.fixture(scope="session")
def verify_embedding_cache(client, embedder_name, shared_embedding_sequences):
    """
    Fixture to verify embeddings are cached in the database.
    
    Call this after embedding test to confirm cache is populated,
    or before projection test to confirm cache will be hit.
    """
    def _verify(expect_cached: bool = True):
        # Use the server's missing embeddings endpoint to check cache
        import json
        from biotrainer.utilities import calculate_sequence_hash
    
        hashed_sequences = {
            calculate_sequence_hash(seq): seq
            for seq in shared_embedding_sequences.values()
        }
        request_data = {
            "sequences": json.dumps(hashed_sequences),
            "embedder_name": embedder_name,
            "reduced": True,  # Must match ProtSpaceTask's reduced=True
        }
        
        response = client.post(
            "/embeddings_service/get_missing_embeddings",
            json=request_data,
        )
        
        if response.status_code != 200:
            print(f"[CACHE CHECK] Failed to check cache: {response.status_code}")
            return {"error": response.status_code}
        
        result = response.json()
        missing = result.get("missing", [])
        total = len(shared_embedding_sequences)
        cached = total - len(missing)
        
        status = {
            "total": total,
            "cached": cached,
            "missing": missing,
            "embedder_name": embedder_name,
            "all_cached": len(missing) == 0,
        }
        
        print(f"\n[CACHE CHECK] Embedder: {embedder_name}")
        print(f"[CACHE CHECK] Cached: {cached}/{total}")
        if missing:
            print(f"[CACHE CHECK] Missing IDs: {missing}")
        
        if expect_cached:
            assert len(missing) == 0, f"Expected all embeddings cached, but missing: {missing}"
        
        return status
    
    return _verify

@pytest.fixture(scope="session")
def precache_prott5_embeddings(shared_embedding_sequences):
    """
    Pre-populate the embedding cache with fake ProtT5 embeddings.
    
    This allows prediction tests to run without downloading the 
    real ProtT5 model (~1.5GB). The server will find these in the
    cache and skip model inference.
    
    The embeddings are random 1024-dim vectors - they won't produce
    meaningful predictions, but will exercise the prediction pipeline.
    """
    import numpy as np
    import psycopg
    import blosc2
    from datetime import datetime
    from biotrainer.utilities import calculate_sequence_hash
    
    # Get PostgreSQL connection info from environment (matching docker-compose)
    db_host = os.environ.get("POSTGRES_HOST", "localhost")
    db_port = int(os.environ.get("POSTGRES_PORT", "5432"))
    db_name = os.environ.get("POSTGRES_DB", "biocentral")
    db_user = os.environ.get("POSTGRES_USER", "biocentral")
    db_pass = os.environ.get("POSTGRES_PASSWORD", "biocentral")
    
    embedder_name = "Rostlab/prot_t5_xl_uniref50"
    
    try:
        conn = psycopg.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_pass,
        )
        
        embeddings_data = []
        for seq_id, sequence in shared_embedding_sequences.items():
            seq_hash = calculate_sequence_hash(sequence)
            seq_len = len(sequence)
            
            # Generate deterministic fake embeddings (seeded by hash)
            seed = hash(seq_hash) % (2**32)
            rng = np.random.default_rng(seed)
            
            # ProtT5 produces 1024-dim embeddings
            per_residue = rng.standard_normal((seq_len, 1024)).astype(np.float32)
            per_sequence = per_residue.mean(axis=0)  # Reduced embedding
            
            # Compress like the server does
            per_seq_compressed = blosc2.pack_array(per_sequence)
            per_res_compressed = blosc2.pack_array(per_residue)
            
            embeddings_data.append((
                seq_hash,
                seq_len,
                datetime.now(),
                embedder_name,
                per_seq_compressed,
                per_res_compressed,
            ))
        
        with conn.cursor() as cur:
            for data in embeddings_data:
                cur.execute(
                    """
                    INSERT INTO embeddings
                    (sequence_hash, sequence_length, last_updated, embedder_name, per_sequence, per_residue)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (sequence_hash, embedder_name) DO UPDATE SET
                    last_updated = EXCLUDED.last_updated,
                    per_sequence = COALESCE(EXCLUDED.per_sequence, embeddings.per_sequence),
                    per_residue = COALESCE(EXCLUDED.per_residue, embeddings.per_residue)
                    """,
                    data,
                )
        conn.commit()
        conn.close()
        
        print(f"\n[PRECACHE] Inserted {len(embeddings_data)} fake ProtT5 embeddings")
        return True
        
    except Exception as e:
        print(f"\n[PRECACHE] Failed to insert fake ProtT5 embeddings: {e}")
        return False