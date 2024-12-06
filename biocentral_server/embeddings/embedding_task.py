import threading
from typing import Dict, Any

from biotrainer.protocols import Protocol
from biotrainer.utilities import read_FASTA, get_device

from .embed import compute_embeddings_and_save_to_db

from ..server_management import TaskStatus, TaskInterface


class EmbeddingTask(TaskInterface):

    def __init__(self, embedder_name, sequence_file_path, embeddings_out_path, protocol, use_half_precision, device):
        super().__init__()
        self.embedder_name = embedder_name
        self.sequence_file_path = sequence_file_path
        self.embeddings_out_path = embeddings_out_path
        self.protocol = Protocol.from_string(protocol) if isinstance(protocol, str) else protocol
        self.use_half_precision = use_half_precision
        self.device = get_device(device)
        self._status_lock = threading.Lock()
        self._status_changed = threading.Event()

    def run(self) -> Any:
        with self._status_lock:
            self._status = TaskStatus.RUNNING
        self._status_changed.set()

        all_seq_records = read_FASTA(str(self.sequence_file_path))
        all_seqs = {seq.id: str(seq.seq) for seq in all_seq_records}

        embedding_triples = compute_embeddings_and_save_to_db(embedder_name=self.embedder_name,
                                                              all_seqs=all_seqs,
                                                              embeddings_out_path=self.embeddings_out_path,
                                                              reduce_by_protocol=self.protocol,
                                                              use_half_precision=self.use_half_precision,
                                                              device=self.device)

        rounded_embeddings = self._round_embeddings({triple.id: triple.embd for triple in embedding_triples},
                                                    reduced=self.protocol in Protocol.per_sequence_protocols())
        # Remove huggingface / prefix
        embedder_name = self.embedder_name.split("/")[-1]
        if self.use_half_precision:
            embedder_name += "-HalfPrecision"

        with self._status_lock:
            self._status = TaskStatus.FINISHED
        self._status_changed.set()
        return {"embeddings_file": {embedder_name: rounded_embeddings}}

    @staticmethod
    def _round_embeddings(embeddings: dict, reduced: bool):
        # TODO Document rounding
        if reduced:
            return {sequence_id: [round(val, 4) for val in embedding.tolist()] for sequence_id, embedding in
                    embeddings.items()}
        return {sequence_id: [[round(val, 4) for val in perResidue.tolist()] for perResidue in embedding] for
                sequence_id, embedding in
                embeddings.items()}

    def update(self) -> Dict[str, Any]:
        self._status_changed.wait(timeout=0.1)  # Wait for status change with a timeout
        with self._status_lock:
            result = {"status": self.get_task_status().name}
        self._status_changed.clear()
        return result
