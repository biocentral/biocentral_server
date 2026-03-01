# Fixed Embedder for deterministic, reproducible testing.

import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class FixedEmbedderConfig:
    embedding_dim: int = 320
    seed_base: int = 42
    noise_scale: float = 0.1

    model_dimensions: Dict[str, int] = field(
        default_factory=lambda: {
            "prot_t5": 1024,
            "prot_t5_xl": 1024,
            "esm2_t33": 1280,
            "esm2_t36": 2560,
            "esm2_t6": 320,
            "esm2_t12": 480,
        }
    )


class FixedEmbedder:
    # The embeddings are generated using a seeded random number generator
    # based on the SHA-256 hash of the input sequence

    AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
    EXTENDED_AMINO_ACIDS = AMINO_ACIDS | set("BJOUXZ")

    def __init__(
        self,
        model_name: str = "esm2_t6",
        embedding_dim: Optional[int] = None,
        seed_base: int = 42,
        noise_scale: float = 0.1,
        strict_dataset: bool = True,
    ):
        self.model_name = model_name
        self.config = FixedEmbedderConfig(seed_base=seed_base, noise_scale=noise_scale)

        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = self.config.model_dimensions.get(
                model_name, self.config.model_dimensions["prot_t5"]
            )

        self.seed_base = seed_base
        self.noise_scale = noise_scale
        self.strict_dataset = strict_dataset

        self._allowed_sequences: Optional[set] = None
        if self.strict_dataset:
            self._allowed_sequences = self._load_canonical_sequences()

        self._aa_embeddings = self._generate_aa_base_embeddings()

    def _load_canonical_sequences(self) -> set:
        # Load allowed sequences from canonical test dataset.
        from tests.fixtures.test_dataset import CANONICAL_TEST_DATASET

        return set(CANONICAL_TEST_DATASET.get_all_sequences())

    def _validate_sequence(self, sequence: str) -> None:
        if self.strict_dataset and self._allowed_sequences is not None:
            if sequence not in self._allowed_sequences:
                raise ValueError(
                    f"Sequence not in canonical test dataset (strict_dataset=True). "
                    f"Sequence: '{sequence[:50]}{'...' if len(sequence) > 50 else ''}' "
                    f"(length={len(sequence)}). "
                    f"Use strict_dataset=False to allow arbitrary sequences, or add "
                    f"this sequence to tests/fixtures/test_dataset.py"
                )

    def _sequence_to_seed(self, sequence: str) -> int:
        # Convert sequence to deterministic seed using SHA-256.
        normalized_sequence = sequence.upper()

        hash_input = f"{self.seed_base}:{normalized_sequence}".encode("utf-8")
        hash_bytes = hashlib.sha256(hash_input).digest()

        seed = int.from_bytes(hash_bytes[:8], byteorder="big") % (2**32)
        return seed

    def _generate_aa_base_embeddings(self) -> Dict[str, np.ndarray]:
        # Generate base embeddings for each amino acid.
        aa_embeddings = {}
        all_aas = sorted(self.EXTENDED_AMINO_ACIDS | {"X", "-", "*"})

        for i, aa in enumerate(all_aas):
            rng = np.random.default_rng(self.seed_base + i * 1000)

            base = rng.standard_normal(self.embedding_dim).astype(np.float32)
            base = base / np.linalg.norm(base)
            aa_embeddings[aa] = base

        return aa_embeddings

    def _get_aa_embedding(self, aa: str) -> np.ndarray:
        # Get base embedding for amino acid, with fallback for unknown chars.
        if aa in self._aa_embeddings:
            return self._aa_embeddings[aa]

        return self._aa_embeddings.get(
            "X", np.zeros(self.embedding_dim, dtype=np.float32)
        )

    def embed(self, sequence: str) -> np.ndarray:
        # Generate deterministic per-residue embedding for a sequence.
        if not sequence:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        self._validate_sequence(sequence)

        seq_len = len(sequence)

        seq_seed = self._sequence_to_seed(sequence)
        rng = np.random.default_rng(seq_seed)

        embeddings = np.zeros((seq_len, self.embedding_dim), dtype=np.float32)

        for pos, aa in enumerate(sequence):
            aa_base = self._get_aa_embedding(aa.upper())

            pos_seed = seq_seed + pos * 100
            pos_rng = np.random.default_rng(pos_seed)
            positional = (
                pos_rng.standard_normal(self.embedding_dim).astype(np.float32) * 0.3
            )

            noise = (
                rng.standard_normal(self.embedding_dim).astype(np.float32)
                * self.noise_scale
            )

            embeddings[pos] = aa_base + positional + noise

        embeddings = embeddings / np.std(embeddings) * 1.0

        return embeddings

    def embed_pooled(self, sequence: str) -> np.ndarray:
        # Generate deterministic per-sequence (pooled) embedding.
        per_residue = self.embed(sequence)
        if len(per_residue) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return np.mean(per_residue, axis=0)

    def embed_batch(
        self,
        sequences: List[str],
        pooled: bool = False,
    ) -> List[np.ndarray]:
        # Generate embeddings for a batch of sequences.
        if pooled:
            return [self.embed_pooled(seq) for seq in sequences]
        else:
            return [self.embed(seq) for seq in sequences]

    def embed_dict(
        self,
        sequences: Dict[str, str],
        pooled: bool = False,
    ) -> Dict[str, np.ndarray]:
        # Generate embeddings for sequences in dictionary format.
        result = {}
        for seq_id, seq in sequences.items():
            if pooled:
                result[seq_id] = self.embed_pooled(seq)
            else:
                result[seq_id] = self.embed(seq)
        return result

    def get_embedding_dimension(self) -> int:
        # Return the embedding dimension for this embedder.
        return self.embedding_dim

    def __repr__(self) -> str:
        return (
            f"FixedEmbedder(model_name='{self.model_name}', "
            f"embedding_dim={self.embedding_dim}, seed_base={self.seed_base})"
        )


class FixedEmbedderRegistry:
    # Registry for FixedEmbedder instances with different configurations.

    _instances: Dict[str, FixedEmbedder] = {}

    @classmethod
    def get_embedder(
        cls,
        model_name: str = "esm2_t6",
        seed_base: int = 42,
        strict_dataset: bool = True,
    ) -> FixedEmbedder:
        # Get or create a FixedEmbedder instance.
        key = f"{model_name}:{seed_base}:{strict_dataset}"
        if key not in cls._instances:
            cls._instances[key] = FixedEmbedder(
                model_name=model_name,
                seed_base=seed_base,
                strict_dataset=strict_dataset,
            )
        return cls._instances[key]

    @classmethod
    def clear(cls):
        # Clear the registry (useful for testing).
        cls._instances.clear()


def get_fixed_embedder(
    model_name: str = "esm2_t6",
    strict_dataset: bool = True,
) -> FixedEmbedder:
    # Get a FixedEmbedder for the specified model.
    return FixedEmbedderRegistry.get_embedder(model_name, strict_dataset=strict_dataset)


def generate_test_embeddings(
    sequences: List[str],
    model_name: str = "esm2_t6",
    pooled: bool = False,
) -> List[np.ndarray]:
    # Convenience function to generate test embeddings.
    embedder = get_fixed_embedder(model_name)
    return embedder.embed_batch(sequences, pooled=pooled)


def generate_test_embeddings_dict(
    sequences: Dict[str, str],
    model_name: str = "esm2_t6",
    pooled: bool = False,
) -> Dict[str, np.ndarray]:
    # Generate test embeddings in dictionary format (matching model input format).
    embedder = get_fixed_embedder(model_name)
    return embedder.embed_dict(sequences, pooled=pooled)


def convert_sequences_to_test_format(
    sequences: List[str],
    model_name: str = "esm2_t6",
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    # Convert sequences to the format expected by prediction models.
    sequences_dict = {f"seq{i}": seq for i, seq in enumerate(sequences)}
    embeddings_dict = generate_test_embeddings_dict(sequences_dict, model_name)
    return sequences_dict, embeddings_dict


def get_expected_embedding_properties(model_name: str) -> Dict:
    # Get expected properties for embeddings from a model.
    properties = {
        "prot_t5": {
            "dimension": 1024,
            "dtype": np.float32,
            "value_range": (-10.0, 10.0),
        },
        "esm2_t33": {
            "dimension": 1280,
            "dtype": np.float32,
            "value_range": (-10.0, 10.0),
        },
        "esm2_t36": {
            "dimension": 2560,
            "dtype": np.float32,
            "value_range": (-10.0, 10.0),
        },
    }
    return properties.get(model_name, properties["prot_t5"])


def validate_embedding_shape(
    embedding: np.ndarray,
    expected_length: int,
    model_name: str = "esm2_t6",
    pooled: bool = False,
) -> bool:
    # Validate that an embedding has the expected shape.
    props = get_expected_embedding_properties(model_name)
    expected_dim = props["dimension"]

    if pooled:
        expected_shape = (expected_dim,)
    else:
        expected_shape = (expected_length, expected_dim)

    return embedding.shape == expected_shape


def validate_embedding_properties(
    embedding: np.ndarray,
    model_name: str = "esm2_t6",
) -> Dict[str, bool]:
    # Validate various properties of an embedding.
    props = get_expected_embedding_properties(model_name)

    return {
        "correct_dtype": embedding.dtype == props["dtype"],
        "no_nan": not np.any(np.isnan(embedding)),
        "no_inf": not np.any(np.isinf(embedding)),
        "not_all_zeros": not np.allclose(embedding, 0),
        "within_range": np.all(embedding >= props["value_range"][0])
        and np.all(embedding <= props["value_range"][1]),
    }


def assert_embedding_valid(
    embedding: np.ndarray,
    sequence_length: int,
    model_name: str = "esm2_t6",
    pooled: bool = False,
) -> None:
    # Assert that an embedding is valid, raising AssertionError if not.
    assert validate_embedding_shape(embedding, sequence_length, model_name, pooled), (
        f"Invalid shape: {embedding.shape}"
    )

    props = validate_embedding_properties(embedding, model_name)
    for prop_name, is_valid in props.items():
        assert is_valid, f"Embedding failed {prop_name} check"
