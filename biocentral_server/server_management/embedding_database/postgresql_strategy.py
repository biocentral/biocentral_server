import torch
import blosc2
import logging
import psycopg
import numpy as np

from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Tuple, List, Any

from .database_strategy import DatabaseStrategy


logger = logging.getLogger(__name__)


class PostgreSQLStrategy(DatabaseStrategy):
    def __init__(self):
        self.db_config = None

    def init_app(self, app):
        self.db_config = app.config.get('POSTGRESQL_CONFIG')
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS embeddings (
                        sequence_hash TEXT,
                        sequence_length INTEGER,
                        last_updated TIMESTAMP,
                        embedder_name TEXT,
                        per_sequence BYTEA,
                        per_residue BYTEA,
                        PRIMARY KEY (sequence_hash, sequence_length, embedder_name)
                    )
                ''')
                cur.execute('''
                    CREATE INDEX IF NOT EXISTS idx_sequence_hash_length_embedder_name 
                    ON embeddings(sequence_hash, sequence_length, embedder_name)
                ''')
            conn.commit()

    @contextmanager
    def _get_connection(self):
        conn = psycopg.connect(**self.db_config)
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def compress_embedding(embedding):
        if embedding is None:
            return None
        if torch.is_tensor(embedding):
            embedding = embedding.cpu().numpy()
        elif not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        return blosc2.pack_array(embedding)

    @staticmethod
    def _decompress_embedding(compressed):
        if not compressed:
            return None
        numpy_array = blosc2.unpack_array(compressed)
        return torch.from_numpy(numpy_array)

    def save_embeddings(self, embeddings_data: List[Tuple]):
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany('''
                        INSERT INTO embeddings 
                        (sequence_hash, sequence_length, last_updated, embedder_name, per_sequence, per_residue)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (sequence_hash, sequence_length, embedder_name) DO UPDATE SET
                        last_updated = EXCLUDED.last_updated,
                        per_sequence = CASE
                            WHEN EXCLUDED.per_sequence IS NOT NULL THEN EXCLUDED.per_sequence
                            ELSE embeddings.per_sequence
                        END,
                        per_residue = CASE
                            WHEN EXCLUDED.per_residue IS NOT NULL THEN EXCLUDED.per_residue
                            ELSE embeddings.per_residue
                        END
                    ''', embeddings_data)
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False

    def get_embeddings(self, sequences: Dict[str, str], embedder_name: str) -> Dict[str, Dict[str, Any]]:
        try:
            # Create a mapping from hash to a list of seq_ids
            hash_to_seq_ids = defaultdict(list)
            for seq_id, seq in sequences.items():
                hash_key = self.generate_sequence_hash(seq)
                hash_to_seq_ids[hash_key].append(seq_id)

            hash_keys = list(hash_to_seq_ids.keys())

            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    placeholders = ','.join(['%s'] * len(hash_keys))
                    # TODO Add sequence length to query
                    cur.execute(f'''
                        SELECT sequence_hash, per_sequence, per_residue
                        FROM embeddings 
                        WHERE sequence_hash IN ({placeholders}) AND embedder_name = %s
                    ''', hash_keys + [embedder_name])

                    db_results = cur.fetchall()

            results = {}
            for result in db_results:
                hash_key = result[0]
                embedding_data = {
                    "id": hash_key,
                    "per_sequence": self._decompress_embedding(result[1]),
                    "per_residue": self._decompress_embedding(result[2])
                }
                # Assign the same embedding data to all sequence IDs with this hash
                for seq_id in hash_to_seq_ids[hash_key]:
                    results[seq_id] = embedding_data

            return results
        except Exception as e:
            logger.error(f"Error retrieving embeddings: {e}")
            return {}

    def filter_existing_embeddings(self, sequences: Dict[str, str],
                                   embedder_name: str,
                                   reduced: bool) -> Tuple[Dict[str, str], Dict[str, str]]:
        hash_keys = [self.generate_sequence_hash(seq) for seq in sequences.values()]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                placeholders = ','.join(['%s'] * len(hash_keys))
                cur.execute(f'''
                    SELECT sequence_hash
                    FROM embeddings
                    WHERE sequence_hash IN ({placeholders})
                    AND embedder_name = %s
                    AND {'per_sequence' if reduced else 'per_residue'} IS NOT NULL
                ''', hash_keys + [embedder_name])
                fetch_result = cur.fetchall()
                existing_hashes = set(row[0] for row in fetch_result)

        existing = {seq_id: seq for seq_id, seq in sequences.items()
                    if self.generate_sequence_hash(seq) in existing_hashes}
        non_existing = {seq_id: seq for seq_id, seq in sequences.items()
                        if seq_id not in existing}

        return existing, non_existing
