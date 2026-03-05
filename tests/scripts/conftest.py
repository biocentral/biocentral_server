import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List

from biotrainer.input_files import BiotrainerSequenceRecord
from tests.fixtures.test_dataset import get_test_sequences

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

@pytest.fixture(scope="session")
def reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR

@pytest.fixture(scope="session")
def esm2_embedder():
    try:
        import torch
        from biotrainer.embedders import get_embedding_service

        svc = get_embedding_service(
            embedder_name="facebook/esm2_t6_8M_UR50D",
            use_half_precision=False,
            custom_tokenizer_config=None,
            device=torch.device("cpu"),
        )
        return _ESM2Wrapper(svc)
    except Exception as exc:
        pytest.skip(f"ESM2-T6-8M unavailable: {exc}")

@pytest.fixture(scope="session")
def one_hot_embedder():
    try:
        import torch
        from biotrainer.embedders import get_embedding_service

        svc = get_embedding_service(
            embedder_name="one_hot_encoding",
            use_half_precision=False,
            custom_tokenizer_config=None,
            device=torch.device("cpu"),
        )
        return _BaselineEmbedderWrapper(svc, "one_hot_encoding")
    except Exception as exc:
        pytest.skip(f"one_hot_encoding embedder unavailable: {exc}")

class _EmbedderMixin:

    def __init__(self, service):
        self._svc = service

    def _to_records(self, sequences: List[str]) -> List[BiotrainerSequenceRecord]:
        return [
            BiotrainerSequenceRecord(seq_id=f"seq_{i}", seq=seq)
            for i, seq in enumerate(sequences)
        ]

    def embed(self, sequence: str) -> np.ndarray:
        records = self._to_records([sequence])
        results = list(self._svc.generate_embeddings(records, reduce=False))
        return np.array(results[0][1]) if results else np.array([])

    def embed_pooled(self, sequence: str) -> np.ndarray:
        records = self._to_records([sequence])
        results = list(self._svc.generate_embeddings(records, reduce=True))
        return np.array(results[0][1]) if results else np.array([])

    def embed_batch(
        self, sequences: List[str], pooled: bool = False
    ) -> List[np.ndarray]:
        records = self._to_records(sequences)
        results = list(self._svc.generate_embeddings(records, reduce=pooled))
        emb_dict = {seq_record.seq_id: np.array(emb) for seq_record, emb in results}
        return [emb_dict[f"seq_{i}"] for i in range(len(sequences))]

class _ESM2Wrapper(_EmbedderMixin):
    pass

class _BaselineEmbedderWrapper(_EmbedderMixin):

    def __init__(self, service, name: str):
        super().__init__(service)
        self.name = name

@pytest.fixture(scope="session")
def standard_sequences() -> List[str]:
    return get_test_sequences(categories=["standard"])

@pytest.fixture(scope="session")
def diverse_sequences() -> List[str]:
    seqs = get_test_sequences(categories=["standard", "real"])
    seen = set()
    unique = []
    for s in seqs:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique

@pytest.fixture(scope="session")
def filler_sequences() -> List[str]:
    return get_test_sequences(categories=["edge_case"])

# Extended sequences for reversal and batch invariance tests (30-40 rows output)
EXTENDED_TEST_SEQUENCES = [
    # Short sequences (10-20 aa)
    "MKTAYIAKQRQISFV",  # 15 aa
    "KEQRQVVRSQNGDLADNIK",  # 19 aa
    "MVHLTPEEKSAVTALWGALG",  # 20 aa
    "ACDEFGHIKLMNPQRSTVWY",  # 20 aa - all standard AA
    "FVNQHLCGSHLVEALYLVCG",  # 20 aa - insulin-like
    # Medium sequences (30-80 aa)
    "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",  # 30 aa - insulin B chain
    "MILVFWILVFMILVFWILVFMILVFWILVFMILVFWILVF",  # 40 aa - hydrophobic
    "AEEAAKAAEEAAKAAEEAAKAAEEAAKAAEEAAKAAEEAAK",  # 42 aa - alpha helix
    "KKRRKKRRKKRRKKRRKKRRKKRRKKRRKKRRKKRRKKRR",  # 40 aa - charged
    "PPPPGPPPPGPPPPGPPPPGPPPPGPPPPGPPPPGPPPPG",  # 40 aa - proline-rich
    "MCCKCCMCCKCCKCCMCCKCCKCCMCCKCCKCCMCCKCCM",  # 40 aa - cysteine-rich
    "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",  # 40 aa - glycine-rich
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",  # 76 aa - ubiquitin
    "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMK",  # 77 aa - GFP core
    # Long sequences (100-200 aa)
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFK",  # 100 aa
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",  # 146 aa - hemoglobin
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRP",  # 100 aa
    "MKKLVLSLSLVLAFSSATAAFAAIPQNIRAQYPAVVKEQRQVVRSQNGDLADNIKKISDNLKAKIYAMHYVDVFYNKSLEKIMKDIQVTNATKTVYISINDLKRRMGGWKYPNMQVLLGRKGKKGKKAKRQ",  # 130 aa
    # Very long sequences (200-400 aa)
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",  # 238 aa - GFP
    "ACDEFGHIKLMNPQRSTVWY" * 10,  # 200 aa - all AA repeated
    "MKTAYIAK" * 30,  # 240 aa - repetitive
    "AEEAAKAAEEAAK" * 18,  # 234 aa - helix repeat
    # Mixed composition sequences
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",  # 141 aa - myoglobin-like
    "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYK",  # 164 aa
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPRVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLPEPPPPKKPLDGEY",  # 393 aa - p53-like
    # Sequences with specific structural motifs
    "GVQVETISPGDGRTFPKRGQTCVVHYTGMLEDGKKFDSSRDRNKPFKFMLGKQEVIRGWEEGVAQMSVGQRAKLTISPDYAYGATGHPGIIPPHATLVFDVELLKLE",  # 106 aa - immunoglobulin fold
    "KTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGIEIKDTKEALDKIEEEQNKSKKKAQQAAADTGNSSPFPQVKNGKANMASSPQVNGKKNGQPSVFKRVKNAGPPHKPTSYKILVGENKDHIGFGKITVEAKQPENKEEAEKDKEPENAENEKKNESSKEPEDKKNESSKEPEDKKNESSKEPEDKKNESS",  # 225 aa
    "MHSSIVLATVLFVAIASASKTRELCMKSLEHAKVGTTSKYQCSYCTNSIQSLFKLANKCPEVDTKQISLRGKASCMKFYCEEPWQNMPQVKCTCVDPKRGLMHGCPENCKVVSSQINGYSCWCVLMPQGGKSCDCPDGVFQGNIYSCWCVDRDKIDVGDWILCGSCPDDDCKAQPTIKKFAGQKFKKLLTFQVCRSCLKCSPIQEAPFTAWPSVVKDLLFCDYLAYTHD",  # 230 aa
]

@pytest.fixture(scope="session")
def extended_sequences() -> List[str]:
    return EXTENDED_TEST_SEQUENCES

@pytest.fixture(scope="session")
def extended_filler_sequences() -> List[str]:
    # Mix of different lengths for realistic batching
    return [
        "MKTAY",  # 5 aa
        "ACDEFGHIKLMNPQRSTVWY",  # 20 aa
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  # 40 aa
        "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAH",  # 80 aa
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFK"
        * 2,  # 200 aa
    ]
