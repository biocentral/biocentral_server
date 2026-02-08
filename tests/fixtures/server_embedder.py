"""Server-based embedder that calls the embedding endpoint via HTTP."""

import os
import time
import logging
from typing import Dict, List, Optional

import httpx
import numpy as np
import blosc2
import psycopg
from biotrainer.utilities import calculate_sequence_hash


class ServerEmbedder:
    """
    Embedder that calls the server's embedding endpoint.
    
    Implements the EmbedderProtocol interface so it can be used with
    metamorphic relations for integration testing.
    
    After submitting an embedding task and waiting for completion,
    it retrieves the embeddings directly from the PostgreSQL database.
    
    When CI_EMBEDDER=fixed, uses FixedEmbedder locally and saves to DB
    (same behavior as integration test interception in conftest.py).
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        embedder_name: str = "facebook/esm2_t6_8M_UR50D",
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ):
        """
        Initialize the server embedder.
        
        Args:
            base_url: Server base URL (default: from CI_SERVER_URL env var)
            embedder_name: Name of the embedder to use on the server
            timeout: Maximum time to wait for embedding task (seconds)
            poll_interval: Time between status polls (seconds)
        """
        self.base_url = base_url or os.environ.get("CI_SERVER_URL", "http://localhost:9540")
        self.api_url = f"{self.base_url}/api/v1"
        self.embedder_name = embedder_name
        self.model_name = embedder_name  # Alias for EmbedderProtocol compatibility
        self.timeout = timeout
        self.poll_interval = poll_interval
        
        # Check if we're in CI fixed embedder mode
        self._use_fixed_embedder = os.environ.get("CI_EMBEDDER", "").lower() == "fixed"
        self._fixed_embedder = None
        
        if self._use_fixed_embedder:
            from tests.fixtures.fixed_embedder import FixedEmbedder
            self._fixed_embedder = FixedEmbedder(model_name="esm2_t6", strict_dataset=False)
            logging.info("ServerEmbedder: Using FixedEmbedder mode (CI_EMBEDDER=fixed)")
        
        # Database connection info
        self.db_host = os.environ.get("POSTGRES_HOST", "localhost")
        self.db_port = int(os.environ.get("POSTGRES_PORT", "5432"))
        self.db_name = os.environ.get("POSTGRES_DB", "embeddings_db")
        self.db_user = os.environ.get("POSTGRES_USER", "embeddingsuser")
        self.db_pass = os.environ.get("POSTGRES_PASSWORD", "embeddingspwd")
        
        # Create HTTP client with retries
        transport = httpx.HTTPTransport(retries=3)
        self.client = httpx.Client(
            base_url=self.api_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
        )
    
    def _submit_embed_task(
        self,
        sequences: Dict[str, str],
        reduce: bool = False,
    ) -> str:
        """Submit an embedding task and return the task ID."""
        request_data = {
            "embedder_name": self.embedder_name,
            "reduce": reduce,
            "sequence_data": sequences,
            "use_half_precision": True,
        }
        
        response = self.client.post("/embeddings_service/embed", json=request_data)
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to submit embedding task: {response.status_code} - {response.text}")
        
        return response.json()["task_id"]
    
    def _poll_task(self, task_id: str) -> Dict:
        """Poll task until completion and return the result."""
        start = time.time()
        
        while time.time() - start < self.timeout:
            response = self.client.get(f"/biocentral_service/task_status/{task_id}")
            
            if response.status_code != 200:
                logging.warning(f"Poll returned {response.status_code}, retrying...")
                time.sleep(self.poll_interval)
                continue
            
            dtos = response.json().get("dtos", [])
            if not dtos:
                time.sleep(self.poll_interval)
                continue
            
            latest = dtos[-1]
            status = latest.get("status", "").upper()
            
            if status in ("FINISHED", "COMPLETED", "DONE"):
                return latest
            elif status in ("FAILED", "ERROR"):
                raise RuntimeError(f"Embedding task failed: {latest.get('error', 'unknown')}")
            
            time.sleep(self.poll_interval)
        
        raise TimeoutError(f"Task {task_id} did not complete within {self.timeout}s")
    
    def _get_embedding_from_db(
        self,
        sequence: str,
        reduce: bool = False,
    ) -> Optional[np.ndarray]:
        """Retrieve a single embedding from the PostgreSQL database."""
        seq_hash = calculate_sequence_hash(sequence)
        
        with psycopg.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_pass,
        ) as conn:
            with conn.cursor() as cur:
                column = "per_sequence" if reduce else "per_residue"
                cur.execute(
                    f"""
                    SELECT {column} FROM embeddings
                    WHERE sequence_hash = %s AND embedder_name = %s
                    """,
                    (seq_hash, self.embedder_name),
                )
                row = cur.fetchone()
                
                if row and row[0]:
                    return blosc2.unpack_array(row[0])
        
        return None
    
    def _compute_and_save_fixed_embedding(
        self,
        sequence: str,
        reduce: bool = False,
    ) -> np.ndarray:
        """
        Compute embedding using FixedEmbedder and save to database.
        
        Used when CI_EMBEDDER=fixed to bypass the server's embedding endpoint.
        """
        from datetime import datetime
        
        # Compute embedding with FixedEmbedder
        if reduce:
            embedding = self._fixed_embedder.embed_pooled(sequence)
        else:
            embedding = self._fixed_embedder.embed(sequence)
        
        # Save to database
        seq_hash = calculate_sequence_hash(sequence)
        seq_len = len(sequence)
        
        if reduce:
            per_seq_compressed = blosc2.pack_array(embedding)
            per_res_compressed = None
        else:
            per_seq_compressed = None
            per_res_compressed = blosc2.pack_array(embedding)
        
        with psycopg.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_pass,
        ) as conn:
            with conn.cursor() as cur:
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
                    (seq_hash, seq_len, datetime.now(), self.embedder_name,
                     per_seq_compressed, per_res_compressed),
                )
            conn.commit()
        
        return embedding

    def embed(self, sequence: str) -> np.ndarray:
        """
        Embed a single sequence, returning per-residue embeddings.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Per-residue embeddings as numpy array of shape (seq_len, embedding_dim)
        """
        # Use FixedEmbedder in CI fixed mode
        if self._use_fixed_embedder:
            return self._compute_and_save_fixed_embedding(sequence, reduce=False)
        
        sequences = {"seq_0": sequence}
        
        # Submit embedding task
        task_id = self._submit_embed_task(sequences, reduce=False)
        
        # Wait for completion
        self._poll_task(task_id)
        
        # Retrieve from database
        embedding = self._get_embedding_from_db(sequence, reduce=False)
        
        if embedding is None:
            raise RuntimeError("Embedding not found in database after task completion")
        
        return embedding
    
    def embed_pooled(self, sequence: str) -> np.ndarray:
        """
        Embed a single sequence, returning pooled (reduced) embedding.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Pooled embedding as numpy array of shape (embedding_dim,)
        """
        # Use FixedEmbedder in CI fixed mode
        if self._use_fixed_embedder:
            return self._compute_and_save_fixed_embedding(sequence, reduce=True)
        
        sequences = {"seq_0": sequence}
        
        # Submit embedding task with reduce=True
        task_id = self._submit_embed_task(sequences, reduce=True)
        
        # Wait for completion
        self._poll_task(task_id)
        
        # Retrieve from database
        embedding = self._get_embedding_from_db(sequence, reduce=True)
        
        if embedding is None:
            raise RuntimeError("Embedding not found in database after task completion")
        
        return embedding
    
    def embed_batch(
        self,
        sequences: List[str],
        pooled: bool = False,
    ) -> List[np.ndarray]:
        """
        Embed multiple sequences.
        
        Args:
            sequences: List of protein sequence strings
            pooled: Whether to return pooled embeddings
            
        Returns:
            List of numpy arrays with embeddings for each sequence
        """
        # Use FixedEmbedder in CI fixed mode
        if self._use_fixed_embedder:
            return [
                self._compute_and_save_fixed_embedding(seq, reduce=pooled)
                for seq in sequences
            ]
        
        seq_dict = {f"seq_{i}": seq for i, seq in enumerate(sequences)}
        
        # Submit embedding task
        task_id = self._submit_embed_task(seq_dict, reduce=pooled)
        
        # Wait for completion
        self._poll_task(task_id)
        
        # Retrieve all from database
        embeddings = []
        for seq in sequences:
            emb = self._get_embedding_from_db(seq, reduce=pooled)
            embeddings.append(emb if emb is not None else np.array([]))
        
        return embeddings
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
