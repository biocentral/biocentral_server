"""
Integration tests for embeddings endpoints.

Tests the /embeddings_service/* endpoints:
- POST /embed - Submit embedding calculation task
- POST /get_missing_embeddings - Check which sequences need embedding
- POST /add_embeddings - Add pre-computed embeddings

Uses configurable embedder backend (FixedEmbedder or ESM-2 8M).
"""

import json
import base64
import io
import h5py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biocentral_server.embeddings import embeddings_router
from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET
from tests.integration.endpoints.conftest import (
    CANONICAL_LENGTH_EDGE_IDS,
    CANONICAL_UNKNOWN_TOKEN_IDS,
    CANONICAL_AMBIGUOUS_CODE_IDS,
    CANONICAL_REAL_WORLD_IDS,
    get_sequence_by_id,
)


@pytest.fixture(scope="module")
def embeddings_app():
    """Create a FastAPI app with embeddings router for testing."""
    app = FastAPI()
    app.include_router(embeddings_router)
    return app


@pytest.fixture(scope="module")
def client(embeddings_app):
    """Create test client."""
    return TestClient(embeddings_app)


class TestEmbedEndpoint:
    """
    Integration tests for POST /embeddings_service/embed.
    
    Tests the embed endpoint with both FixedEmbedder and real ESM-2 8M
    backends based on test configuration.
    """

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.embeddings_endpoint.TaskManager")
    @patch("biocentral_server.embeddings.embeddings_endpoint.UserManager")
    @patch("biocentral_server.embeddings.embeddings_endpoint.MetricsCollector")
    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_embed_request_creates_task(
        self,
        mock_rate_limiter,
        mock_metrics,
        mock_user_manager,
        mock_task_manager,
        client,
        embedder_name,
        short_test_sequences,
    ):
        """Test that embedding request creates a task and returns task ID."""
        # Setup mocks
        mock_rate_limiter.return_value = lambda: None
        mock_task_manager.return_value.add_task.return_value = "embed-task-123"
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")

        request_data = {
            "embedder_name": embedder_name,
            "reduce": False,
            "sequence_data": short_test_sequences,
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 200
        response_json = response.json()
        assert "task_id" in response_json
        assert response_json["task_id"] == "embed-task-123"
        
        # Verify task was created with correct parameters
        mock_task_manager.return_value.add_task.assert_called_once()

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.embeddings_endpoint.TaskManager")
    @patch("biocentral_server.embeddings.embeddings_endpoint.UserManager")
    @patch("biocentral_server.embeddings.embeddings_endpoint.MetricsCollector")
    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_embed_request_with_reduction(
        self,
        mock_rate_limiter,
        mock_metrics,
        mock_user_manager,
        mock_task_manager,
        client,
        embedder_name,
        single_test_sequence,
    ):
        """Test embedding request with reduce=True for per-sequence embeddings."""
        mock_rate_limiter.return_value = lambda: None
        mock_task_manager.return_value.add_task.return_value = "embed-reduced-task"
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")

        request_data = {
            "embedder_name": embedder_name,
            "reduce": True,  # Request pooled embeddings
            "sequence_data": single_test_sequence,
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 200
        assert "task_id" in response.json()

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.embeddings_endpoint.TaskManager")
    @patch("biocentral_server.embeddings.embeddings_endpoint.UserManager")
    @patch("biocentral_server.embeddings.embeddings_endpoint.MetricsCollector")
    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_embed_multiple_sequences(
        self,
        mock_rate_limiter,
        mock_metrics,
        mock_user_manager,
        mock_task_manager,
        client,
        embedder_name,
        test_sequences,
    ):
        """Test embedding request with multiple sequences."""
        mock_rate_limiter.return_value = lambda: None
        mock_task_manager.return_value.add_task.return_value = "embed-batch-task"
        mock_user_manager.get_user_id_from_request = AsyncMock(return_value="test-user")

        request_data = {
            "embedder_name": embedder_name,
            "reduce": False,
            "sequence_data": test_sequences,
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 200
        assert response.json()["task_id"] == "embed-batch-task"

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_embed_empty_sequences_rejected(
        self,
        mock_rate_limiter,
        client,
        embedder_name,
    ):
        """Test that empty sequence data is rejected."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "embedder_name": embedder_name,
            "reduce": False,
            "sequence_data": {},  # Empty - should fail validation
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 422  # Validation error

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_embed_missing_embedder_name_rejected(
        self,
        mock_rate_limiter,
        client,
        short_test_sequences,
    ):
        """Test that missing embedder name is rejected."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            # Missing embedder_name
            "reduce": False,
            "sequence_data": short_test_sequences,
            "use_half_precision": False,
        }

        response = client.post("/embeddings_service/embed", json=request_data)

        assert response.status_code == 422


class TestGetMissingEmbeddingsEndpoint:
    """
    Integration tests for POST /embeddings_service/get_missing_embeddings.
    """

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.embeddings_endpoint.EmbeddingDatabaseFactory")
    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_get_missing_embeddings_all_missing(
        self,
        mock_rate_limiter,
        mock_db_factory,
        client,
        embedder_name,
        short_test_sequences,
    ):
        """Test identifying missing embeddings when none exist."""
        mock_rate_limiter.return_value = lambda: None

        # Mock database returns all as non-existing
        mock_db = MagicMock()
        mock_db.filter_existing_embeddings.return_value = ({}, short_test_sequences)
        mock_db_factory.return_value.get_embeddings_db.return_value = mock_db

        request_data = {
            "sequences": json.dumps(short_test_sequences),
            "embedder_name": embedder_name,
            "reduced": False,
        }

        response = client.post(
            "/embeddings_service/get_missing_embeddings", json=request_data
        )

        assert response.status_code == 200
        response_json = response.json()
        assert "missing" in response_json
        assert set(response_json["missing"]) == set(short_test_sequences.keys())

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.embeddings_endpoint.EmbeddingDatabaseFactory")
    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_get_missing_embeddings_partial(
        self,
        mock_rate_limiter,
        mock_db_factory,
        client,
        embedder_name,
        short_test_sequences,
    ):
        """Test identifying missing embeddings when some exist."""
        mock_rate_limiter.return_value = lambda: None

        # Mock database returns partial results
        seq_ids = list(short_test_sequences.keys())
        existing = {seq_ids[0]: short_test_sequences[seq_ids[0]]}
        non_existing = {seq_ids[1]: short_test_sequences[seq_ids[1]]}
        
        mock_db = MagicMock()
        mock_db.filter_existing_embeddings.return_value = (existing, non_existing)
        mock_db_factory.return_value.get_embeddings_db.return_value = mock_db

        request_data = {
            "sequences": json.dumps(short_test_sequences),
            "embedder_name": embedder_name,
            "reduced": False,
        }

        response = client.post(
            "/embeddings_service/get_missing_embeddings", json=request_data
        )

        assert response.status_code == 200
        response_json = response.json()
        assert len(response_json["missing"]) == 1
        assert seq_ids[1] in response_json["missing"]

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.embeddings_endpoint.EmbeddingDatabaseFactory")
    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_get_missing_embeddings_none_missing(
        self,
        mock_rate_limiter,
        mock_db_factory,
        client,
        embedder_name,
        short_test_sequences,
    ):
        """Test when all embeddings already exist."""
        mock_rate_limiter.return_value = lambda: None

        mock_db = MagicMock()
        mock_db.filter_existing_embeddings.return_value = (short_test_sequences, {})
        mock_db_factory.return_value.get_embeddings_db.return_value = mock_db

        request_data = {
            "sequences": json.dumps(short_test_sequences),
            "embedder_name": embedder_name,
            "reduced": True,
        }

        response = client.post(
            "/embeddings_service/get_missing_embeddings", json=request_data
        )

        assert response.status_code == 200
        assert response.json()["missing"] == []


class TestAddEmbeddingsEndpoint:
    """
    Integration tests for POST /embeddings_service/add_embeddings.
    """

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.embeddings_endpoint.EmbeddingDatabaseFactory")
    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_add_embeddings_valid_h5(
        self,
        mock_rate_limiter,
        mock_db_factory,
        client,
        embedder_name,
        embedder,
        short_test_sequences,
        embedding_dim,
    ):
        """Test adding embeddings from valid H5 file."""
        mock_rate_limiter.return_value = lambda: None

        mock_db = MagicMock()
        mock_db_factory.return_value.get_embeddings_db.return_value = mock_db

        # Generate embeddings using configured backend
        embeddings = embedder.embed_dict(short_test_sequences, pooled=True)

        # Create H5 file in memory
        h5_buffer = io.BytesIO()
        with h5py.File(h5_buffer, "w") as h5f:
            for idx, (seq_id, embedding) in enumerate(embeddings.items()):
                ds = h5f.create_dataset(str(idx), data=embedding)
                ds.attrs["original_id"] = seq_id

        h5_buffer.seek(0)
        h5_bytes_b64 = base64.b64encode(h5_buffer.read()).decode("utf-8")

        request_data = {
            "sequences": json.dumps(short_test_sequences),
            "embedder_name": embedder_name,
            "reduced": True,
            "h5_bytes": h5_bytes_b64,
        }

        response = client.post("/embeddings_service/add_embeddings", json=request_data)

        assert response.status_code == 200

    @pytest.mark.integration
    @patch("biocentral_server.embeddings.embeddings_endpoint.RateLimiter")
    def test_add_embeddings_invalid_base64(
        self,
        mock_rate_limiter,
        client,
        embedder_name,
        short_test_sequences,
    ):
        """Test that invalid base64 data is rejected."""
        mock_rate_limiter.return_value = lambda: None

        request_data = {
            "sequences": json.dumps(short_test_sequences),
            "embedder_name": embedder_name,
            "reduced": True,
            "h5_bytes": "not-valid-base64!!!",
        }

        response = client.post("/embeddings_service/add_embeddings", json=request_data)

        assert response.status_code == 400


class TestEmbeddingGeneration:
    """
    Tests that verify actual embedding generation with both backends.
    
    These tests use the embedder fixture directly to verify embedding
    properties without going through the HTTP endpoints.
    """

    @pytest.mark.integration
    def test_embedder_produces_correct_dimensions(
        self,
        embedder,
        embedding_dim,
        single_test_sequence,
    ):
        """Test that embedder produces embeddings with correct dimensions."""
        seq = list(single_test_sequence.values())[0]
        
        # Per-residue embedding
        per_residue = embedder.embed(seq)
        assert per_residue.shape == (len(seq), embedding_dim)
        assert per_residue.dtype == np.float32
        
        # Pooled embedding
        pooled = embedder.embed_pooled(seq)
        assert pooled.shape == (embedding_dim,)
        assert pooled.dtype == np.float32

    @pytest.mark.integration
    def test_embedder_batch_processing(
        self,
        embedder,
        embedding_dim,
        test_sequences,
    ):
        """Test batch embedding generation."""
        seq_list = list(test_sequences.values())
        
        embeddings = embedder.embed_batch(seq_list, pooled=False)
        
        assert len(embeddings) == len(seq_list)
        for seq, emb in zip(seq_list, embeddings):
            assert emb.shape == (len(seq), embedding_dim)

    @pytest.mark.integration
    def test_embedder_dict_format(
        self,
        embedder,
        embedding_dim,
        test_sequences,
    ):
        """Test dictionary-based embedding generation."""
        embeddings = embedder.embed_dict(test_sequences, pooled=True)
        
        assert len(embeddings) == len(test_sequences)
        assert set(embeddings.keys()) == set(test_sequences.keys())
        
        for seq_id, emb in embeddings.items():
            assert emb.shape == (embedding_dim,)

    @pytest.mark.integration
    @pytest.mark.fixed_embedder
    def test_fixed_embedder_determinism(
        self,
        embedder,
        single_test_sequence,
        skip_if_real_embedder,
    ):
        """Test that FixedEmbedder produces deterministic results."""
        seq = list(single_test_sequence.values())[0]
        
        emb1 = embedder.embed(seq)
        emb2 = embedder.embed(seq)
        
        # Same sequence should produce identical embeddings
        np.testing.assert_array_equal(emb1, emb2)

    @pytest.mark.integration
    def test_different_sequences_produce_different_embeddings(
        self,
        embedder,
        short_test_sequences,
    ):
        """Test that different sequences produce different embeddings."""
        seq_list = list(short_test_sequences.values())
        
        emb1 = embedder.embed_pooled(seq_list[0])
        emb2 = embedder.embed_pooled(seq_list[1])
        
        # Different sequences should produce different embeddings
        assert not np.allclose(emb1, emb2)


class TestMinimumLengthEmbeddings:
    """
    Tests for embedding generation with minimum-length sequences.
    
    These tests verify behavior at length boundary conditions using
    canonical dataset sequences.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", ["length_min_1", "length_min_2", "length_short_5"])
    def test_embed_minimum_length_sequences(
        self,
        embedder,
        embedding_dim,
        seq_id,
    ):
        """Test embedding generation for minimum-length sequences."""
        sequence = get_sequence_by_id(seq_id)
        
        # Per-residue embedding
        per_residue = embedder.embed(sequence)
        assert per_residue.shape == (len(sequence), embedding_dim)
        assert np.isfinite(per_residue).all()
        
        # Pooled embedding
        pooled = embedder.embed_pooled(sequence)
        assert pooled.shape == (embedding_dim,)
        assert np.isfinite(pooled).all()

    @pytest.mark.integration
    def test_single_residue_embedding(
        self,
        embedder,
        embedding_dim,
        minimum_length_sequences,
    ):
        """Test embedding a single residue (length=1)."""
        seq = minimum_length_sequences["min_1"]
        assert len(seq) == 1
        
        emb = embedder.embed(seq)
        assert emb.shape == (1, embedding_dim)
        
        pooled = embedder.embed_pooled(seq)
        assert pooled.shape == (embedding_dim,)


class TestLongSequenceEmbeddings:
    """
    Tests for embedding generation with long sequences.
    
    Verifies that embedder handles sequences of varying lengths properly.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", ["length_long_200", "length_very_long_400"])
    def test_embed_long_sequences(
        self,
        embedder,
        embedding_dim,
        seq_id,
    ):
        """Test embedding generation for long sequences."""
        sequence = get_sequence_by_id(seq_id)
        
        # Per-residue embedding
        per_residue = embedder.embed(sequence)
        assert per_residue.shape == (len(sequence), embedding_dim)
        assert np.isfinite(per_residue).all()
        
        # Pooled embedding
        pooled = embedder.embed_pooled(sequence)
        assert pooled.shape == (embedding_dim,)
        assert np.isfinite(pooled).all()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_embed_400_residue_sequence(
        self,
        embedder,
        embedding_dim,
        long_sequences,
    ):
        """Test embedding a 400 residue sequence."""
        seq = long_sequences["very_long_400"]
        assert len(seq) == 400
        
        emb = embedder.embed(seq)
        assert emb.shape == (400, embedding_dim)
        
        pooled = embedder.embed_pooled(seq)
        assert pooled.shape == (embedding_dim,)


class TestUnknownTokenEmbeddings:
    """
    Tests for embedding generation with unknown (X) residues.
    
    Verifies that embedder handles X tokens appropriately across
    different positions and ratios.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_UNKNOWN_TOKEN_IDS)
    def test_embed_sequences_with_unknown_tokens(
        self,
        embedder,
        embedding_dim,
        seq_id,
    ):
        """Test embedding generation for sequences containing X residues."""
        sequence = get_sequence_by_id(seq_id)
        
        # Embedder should handle X tokens without error
        per_residue = embedder.embed(sequence)
        assert per_residue.shape == (len(sequence), embedding_dim)
        assert np.isfinite(per_residue).all()
        
        pooled = embedder.embed_pooled(sequence)
        assert pooled.shape == (embedding_dim,)
        assert np.isfinite(pooled).all()

    @pytest.mark.integration
    def test_embed_high_unknown_ratio_sequence(
        self,
        embedder,
        embedding_dim,
        unknown_token_sequences,
    ):
        """Test embedding with ~70% unknown residues."""
        seq = unknown_token_sequences["unknown_high_ratio"]
        assert seq.count("X") / len(seq) > 0.6
        
        emb = embedder.embed(seq)
        assert emb.shape == (len(seq), embedding_dim)
        assert np.isfinite(emb).all()

    @pytest.mark.integration
    def test_unknown_token_positions_produce_embeddings(
        self,
        embedder,
        unknown_token_sequences,
    ):
        """Test that X tokens at different positions all get embeddings."""
        for seq_id, seq in unknown_token_sequences.items():
            emb = embedder.embed(seq)
            # Each position including X should have an embedding
            assert emb.shape[0] == len(seq)


class TestAmbiguousCodeEmbeddings:
    """
    Tests for embedding generation with ambiguous amino acid codes.
    
    Verifies handling of B (Asx), Z (Glx), J (Xle), U (Sec), O (Pyl).
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_AMBIGUOUS_CODE_IDS)
    def test_embed_sequences_with_ambiguous_codes(
        self,
        embedder,
        embedding_dim,
        seq_id,
    ):
        """Test embedding generation for sequences with ambiguous codes."""
        sequence = get_sequence_by_id(seq_id)
        
        # Embedder should handle ambiguous codes
        per_residue = embedder.embed(sequence)
        assert per_residue.shape == (len(sequence), embedding_dim)
        assert np.isfinite(per_residue).all()
        
        pooled = embedder.embed_pooled(sequence)
        assert pooled.shape == (embedding_dim,)
        assert np.isfinite(pooled).all()

    @pytest.mark.integration
    def test_selenocysteine_embedding(
        self,
        embedder,
        embedding_dim,
        ambiguous_code_sequences,
    ):
        """Test embedding sequence containing selenocysteine (U)."""
        seq = ambiguous_code_sequences["selenocysteine"]
        assert "U" in seq
        
        emb = embedder.embed(seq)
        assert emb.shape == (len(seq), embedding_dim)
        assert np.isfinite(emb).all()


class TestCompositionEdgeCases:
    """
    Tests for embedding sequences with unusual amino acid compositions.
    """

    @pytest.mark.integration
    def test_embed_all_standard_amino_acids(
        self,
        embedder,
        embedding_dim,
        composition_edge_sequences,
    ):
        """Test embedding sequence with all 20 standard amino acids."""
        seq = composition_edge_sequences["all_standard_aa"]
        assert len(set(seq)) == 20
        
        emb = embedder.embed(seq)
        assert emb.shape == (len(seq), embedding_dim)
        assert np.isfinite(emb).all()

    @pytest.mark.integration
    def test_embed_homopolymer_sequences(
        self,
        embedder,
        embedding_dim,
        composition_edge_sequences,
    ):
        """Test embedding homopolymer sequences (all same amino acid)."""
        # Short homopolymer
        short_homo = composition_edge_sequences["homopolymer_A"]
        emb_short = embedder.embed(short_homo)
        assert emb_short.shape == (len(short_homo), embedding_dim)
        
        # Long homopolymer
        long_homo = composition_edge_sequences["homopolymer_long"]
        emb_long = embedder.embed(long_homo)
        assert emb_long.shape == (len(long_homo), embedding_dim)
        
        # Verify different lengths produce different pooled embeddings
        pooled_short = embedder.embed_pooled(short_homo)
        pooled_long = embedder.embed_pooled(long_homo)
        assert not np.allclose(pooled_short, pooled_long)

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_type", [
        "hydrophobic_rich",
        "charged_rich",
        "proline_rich",
        "cysteine_rich",
    ])
    def test_embed_composition_biased_sequences(
        self,
        embedder,
        embedding_dim,
        composition_edge_sequences,
        seq_type,
    ):
        """Test embedding sequences with biased amino acid compositions."""
        seq = composition_edge_sequences[seq_type]
        
        emb = embedder.embed(seq)
        assert emb.shape == (len(seq), embedding_dim)
        assert np.isfinite(emb).all()


class TestRealWorldSequences:
    """
    Tests for embedding real-world protein sequences.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("seq_id", CANONICAL_REAL_WORLD_IDS)
    def test_embed_real_world_proteins(
        self,
        embedder,
        embedding_dim,
        seq_id,
    ):
        """Test embedding real-world protein sequences."""
        sequence = get_sequence_by_id(seq_id)
        
        per_residue = embedder.embed(sequence)
        assert per_residue.shape == (len(sequence), embedding_dim)
        assert np.isfinite(per_residue).all()
        
        pooled = embedder.embed_pooled(sequence)
        assert pooled.shape == (embedding_dim,)
        assert np.isfinite(pooled).all()

    @pytest.mark.integration
    def test_real_world_sequences_produce_unique_embeddings(
        self,
        embedder,
        real_world_sequences,
    ):
        """Test that real-world proteins produce distinguishable embeddings."""
        embeddings = embedder.embed_dict(real_world_sequences, pooled=True)
        
        emb_list = list(embeddings.values())
        # Each pair should be different
        for i in range(len(emb_list)):
            for j in range(i + 1, len(emb_list)):
                assert not np.allclose(emb_list[i], emb_list[j])


class TestBatchEmbeddingConsistency:
    """
    Tests for consistency between individual and batch embedding generation.
    """

    @pytest.mark.integration
    def test_batch_vs_individual_consistency(
        self,
        embedder,
        diverse_test_sequences,
    ):
        """Test that batch and individual embeddings are identical."""
        # Generate individually
        individual = {}
        for seq_id, seq in diverse_test_sequences.items():
            individual[seq_id] = embedder.embed_pooled(seq)
        
        # Generate as batch
        batch = embedder.embed_dict(diverse_test_sequences, pooled=True)
        
        # Should produce identical results
        for seq_id in diverse_test_sequences.keys():
            np.testing.assert_array_almost_equal(
                individual[seq_id],
                batch[seq_id],
                decimal=5,
            )

    @pytest.mark.integration
    def test_large_batch_embedding(
        self,
        embedder,
        embedding_dim,
        diverse_test_sequences,
    ):
        """Test embedding a larger batch of diverse sequences."""
        embeddings = embedder.embed_dict(diverse_test_sequences, pooled=True)
        
        assert len(embeddings) == len(diverse_test_sequences)
        for seq_id, emb in embeddings.items():
            assert emb.shape == (embedding_dim,)
            assert np.isfinite(emb).all()
