from typing import Dict, Any

from biotrainer.protocols import Protocol
from biotrainer.utilities import read_FASTA, get_device

from .embed import compute_embeddings_and_save_to_db

from ..server_management import TaskStatus, MultiprocessingTask


class EmbeddingTask(MultiprocessingTask):

    def __init__(self, embedder_name, sequence_file_path, embeddings_out_path, protocol, use_half_precision, device):
        super().__init__()
        self.embedder_name = embedder_name
        self.sequence_file_path = sequence_file_path
        self.embeddings_out_path = embeddings_out_path
        self.protocol = Protocol.from_string(protocol) if isinstance(protocol, str) else protocol
        self.use_half_precision = use_half_precision
        self.device = get_device(device)

    async def _run_task(self) -> Any:
        all_seq_records = read_FASTA(str(self.sequence_file_path))
        all_seqs = {seq.id: str(seq.seq) for seq in all_seq_records}

        embedding_triples = compute_embeddings_and_save_to_db(embedder_name=self.embedder_name,
                                                              all_seqs=all_seqs,
                                                              embeddings_out_path=self.embeddings_out_path,
                                                              reduce_by_protocol=self.protocol,
                                                              use_half_precision=self.use_half_precision,
                                                              device=self.device)
        return self._round_embeddings({triple.id: triple.embd for triple in embedding_triples},
                                              reduced=self.protocol in Protocol.per_sequence_protocols())

    @staticmethod
    def _round_embeddings(embeddings: dict, reduced: bool):
        # TODO Document rounding
        if reduced:
            return {sequence_id: [round(val, 4) for val in embedding.tolist()] for sequence_id, embedding in
                    embeddings.items()}
        return {sequence_id: [[round(val, 4) for val in perResidue.tolist()] for perResidue in embedding] for
                sequence_id, embedding in
                embeddings.items()}

    def get_task_status(self) -> TaskStatus:
        if not self.process:
            return TaskStatus.RUNNING
        if self.process.is_alive():
            return TaskStatus.RUNNING
        else:
            return TaskStatus.FINISHED

    def update(self) -> Dict[str, Any]:
        result = {"status": self.get_task_status().name}
        return result
