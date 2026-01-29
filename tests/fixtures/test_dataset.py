"""Canonical test dataset for biocentral_server testing."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from tests.fixtures.fixed_embedder import get_fixed_embedder


@dataclass
class TestSequence:
    """A single test sequence with metadata."""

    id: str
    sequence: str
    description: str


@dataclass
class TestDataset:
    """Complete test dataset with sequences and metadata."""

    name: str
    version: str
    description: str
    sequences: List[TestSequence]

    def get_by_id(self, seq_id: str) -> Optional[TestSequence]:
        """Get a sequence by its ID."""
        for seq in self.sequences:
            if seq.id == seq_id:
                return seq
        return None

    def get_all_sequences(self) -> List[str]:
        """Get all sequences as a list of strings."""
        return [s.sequence for s in self.sequences]

    def get_sequences_dict(self) -> Dict[str, str]:
        """Get all sequences as a dictionary (id -> sequence)."""
        return {s.id: s.sequence for s in self.sequences}


CANONICAL_TEST_DATASET = TestDataset(
    name="biocentral_canonical_test_set",
    version="1.0.0",
    description="Canonical test dataset for biocentral_server testing",
    sequences=[
        # Standard sequences
        TestSequence(
            id="standard_001",
            sequence="MKTAYIAKQRQISFVKSHFSRQLALPDAQFEVVHSLAKWKRQ",
            description="Medium-length standard protein (79 aa)",
        ),
        TestSequence(
            id="standard_002",
            sequence="KEQRQVVRSQNGDLADNIKKISDNLKAKIYAMHYVDVFYNKS",
            description="Signal peptide containing sequence (79 aa)",
        ),
        TestSequence(
            id="standard_003",
            sequence="MVHLTPEEKSAVTALWGKVNVDEVGGEALG",
            description="Hemoglobin-like sequence (77 aa)",
        ),
        # Length edge cases
        TestSequence(
            id="length_min_1",
            sequence="M",
            description="Minimum length: single residue (1 aa)",
        ),
        TestSequence(
            id="length_min_2",
            sequence="MK",
            description="Very short: two residues (2 aa)",
        ),
        TestSequence(
            id="length_short_5",
            sequence="MKTAY",
            description="Short sequence (5 aa)",
        ),
        TestSequence(
            id="length_short_10",
            sequence="MKTAYIAKQR",
            description="Short sequence (10 aa)",
        ),
        TestSequence(
            id="length_medium_50",
            sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLS",
            description="Medium sequence (49 aa)",
        ),
        TestSequence(
            id="length_long_200",
            sequence="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEK",
            description="Long sequence - GFP-like (211 aa)",
        ),
        TestSequence(
            id="length_very_long_400",
            sequence="MKTAYIAK" * 50,
            description="Very long repetitive sequence (400 aa)",
        ),
        # Unknown token (X) edge cases
        TestSequence(
            id="unknown_single",
            sequence="X",
            description="Single unknown residue",
        ),
        TestSequence(
            id="unknown_multiple",
            sequence="XXXXX",
            description="Multiple consecutive unknown residues (5 aa)",
        ),
        TestSequence(
            id="unknown_start",
            sequence="XMKTAYIAKQRQISFVK",
            description="Unknown at start of sequence",
        ),
        TestSequence(
            id="unknown_end",
            sequence="MKTAYIAKQRQISFVKX",
            description="Unknown at end of sequence",
        ),
        TestSequence(
            id="unknown_middle",
            sequence="MKTAYXAKQRQISFVK",
            description="Unknown in middle of sequence",
        ),
        TestSequence(
            id="unknown_scattered",
            sequence="MXKTXYIAXQRQXSFVK",
            description="Multiple scattered unknown residues",
        ),
        TestSequence(
            id="unknown_high_ratio",
            sequence="MXXXXXKXXXXXAXXXX",
            description="High ratio of unknown residues (~70%)",
        ),
        # Amino acid composition edge cases
        TestSequence(
            id="all_standard_aa",
            sequence="ACDEFGHIKLMNPQRSTVWY",
            description="All 20 standard amino acids (20 aa)",
        ),
        TestSequence(
            id="homopolymer_A",
            sequence="AAAAAAAAAA",
            description="Homopolymer: 10 Alanines",
        ),
        TestSequence(
            id="homopolymer_long",
            sequence="A" * 50,
            description="Long homopolymer: 50 Alanines",
        ),
        TestSequence(
            id="hydrophobic_rich",
            sequence="MILVFWILVFMILVFWILVFMILVFWILVFMILVFWILVF",
            description="Hydrophobic-rich sequence (40 aa)",
        ),
        TestSequence(
            id="charged_rich",
            sequence="KKRRKKRRKKRRKKRRKKRRKKRRKKRRKKERERDE",
            description="Charged residue-rich sequence (36 aa)",
        ),
        TestSequence(
            id="proline_rich",
            sequence="PPPPGPPPPGPPPPGPPPPGPPPPG",
            description="Proline-rich sequence (25 aa)",
        ),
        TestSequence(
            id="cysteine_rich",
            sequence="MCCKCCMCCKCCKCCMCCKCCKCCM",
            description="Cysteine-rich sequence (25 aa) - disulfide potential",
        ),
        # Special characters and ambiguous codes
        TestSequence(
            id="ambiguous_B",
            sequence="MKTABIAK",
            description="Contains B (Asx: Asp or Asn)",
        ),
        TestSequence(
            id="ambiguous_Z",
            sequence="MKTAZIAK",
            description="Contains Z (Glx: Glu or Gln)",
        ),
        TestSequence(
            id="ambiguous_J",
            sequence="MKTAJIAK",
            description="Contains J (Xle: Leu or Ile)",
        ),
        TestSequence(
            id="selenocysteine",
            sequence="MKTAUIAK",
            description="Contains U (Selenocysteine)",
        ),
        TestSequence(
            id="pyrrolysine",
            sequence="MKTAOIAK",
            description="Contains O (Pyrrolysine)",
        ),
        # Structural motifs
        TestSequence(
            id="motif_alpha_helix",
            sequence="AEEAAKAAEEAAKAAEEAAKAAEEAAK",
            description="Alpha helix-forming sequence (27 aa)",
        ),
        TestSequence(
            id="motif_beta_sheet",
            sequence="VTVTVTVTVTVTVTVTVTVT",
            description="Beta sheet-forming sequence (20 aa)",
        ),
        TestSequence(
            id="motif_glycine_loop",
            sequence="GGGGGGGGGGGGGGG",
            description="Glycine-rich flexible loop (15 aa)",
        ),
        # Real-world representative sequences
        TestSequence(
            id="real_insulin_b",
            sequence="FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
            description="Human Insulin B chain (30 aa)",
        ),
        TestSequence(
            id="real_ubiquitin",
            sequence="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            description="Human Ubiquitin (76 aa)",
        ),
        TestSequence(
            id="real_gfp_core",
            sequence="SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMK",
            description="GFP chromophore region (77 aa)",
        ),
    ],
)


def get_test_sequences(categories: Optional[List[str]] = None) -> List[str]:
    """
    Get test sequences, optionally filtered by category.

    Args:
        categories: Optional list of category prefixes to filter by.
                   Supported: "standard", "length", "unknown", "ambiguous",
                   "homopolymer", "motif", "real", "edge_case".
                   If None, returns all sequences.

    Returns:
        List of sequence strings
    """
    if categories is None:
        return CANONICAL_TEST_DATASET.get_all_sequences()

    # Map category names to sequence ID prefixes
    category_prefixes = {
        "standard": ["standard_"],
        "length": ["length_"],
        "unknown": ["unknown_"],
        "ambiguous": ["ambiguous_", "selenocysteine", "pyrrolysine"],
        "homopolymer": ["homopolymer_"],
        "motif": ["motif_"],
        "real": ["real_"],
        "edge_case": ["length_", "unknown_", "ambiguous_", "homopolymer_", "motif_",
                      "selenocysteine", "pyrrolysine", "hydrophobic_", "charged_",
                      "proline_", "cysteine_", "all_standard_aa"],
    }

    # Collect all prefixes for requested categories
    prefixes = []
    for cat in categories:
        if cat in category_prefixes:
            prefixes.extend(category_prefixes[cat])

    # Filter sequences by ID prefix
    result = []
    for seq in CANONICAL_TEST_DATASET.sequences:
        if any(seq.id.startswith(prefix) for prefix in prefixes):
            result.append(seq.sequence)

    return result if result else CANONICAL_TEST_DATASET.get_all_sequences()


def get_test_sequences_dict() -> Dict[str, str]:
    """
    Get test sequences as dictionary (id -> sequence).

    Returns:
        Dictionary mapping sequence ID to sequence string
    """
    return CANONICAL_TEST_DATASET.get_sequences_dict()


def get_test_embeddings(
    model_name: str = "esm2_t6",
    pooled: bool = False,
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """
    Get test sequences and their embeddings ready for model testing.

    This is the primary function for setting up integration tests.

    Args:
        model_name: Model to emulate for embeddings
        pooled: Whether to return per-sequence (pooled) embeddings

    Returns:
        Tuple of (sequences_dict, embeddings_dict)
    """
    sequences_dict = get_test_sequences_dict()
    embedder = get_fixed_embedder(model_name)
    embeddings_dict = embedder.embed_dict(sequences_dict, pooled=pooled)
    return sequences_dict, embeddings_dict


def get_dataset_statistics() -> Dict:
    """Get statistics about the canonical test dataset."""
    sequences = CANONICAL_TEST_DATASET.get_all_sequences()
    lengths = [len(s) for s in sequences]

    return {
        "total_sequences": len(sequences),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "sequences_with_unknown": sum(1 for s in sequences if "X" in s),
    }
