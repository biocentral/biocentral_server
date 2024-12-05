from typing import Dict, Any
import torch.multiprocessing as torch_mp

from biotrainer.utilities import read_FASTA, get_device

from ..server_management import TaskInterface, TaskStatus, EmbeddingsDatabase
from ..embeddings import compute_embeddings_and_save_to_db


class EmbeddingTask(TaskInterface):
    process: torch_mp.Process = None

    def __init__(self, embedder_name, sequence_file_path, embeddings_out_path, protocol, use_half_precision):
        self.embedder_name = embedder_name
        self.sequence_file_path = sequence_file_path
        self.embeddings_out_path = embeddings_out_path
        self.protocol = protocol
        self.use_half_precision = use_half_precision

    async def start(self):
        all_seq_records = read_FASTA(str(self.sequence_file_path))
        all_seqs = {seq.id: str(seq.seq) for seq in all_seq_records}
        device = get_device()

        def _embed_function():
            compute_embeddings_and_save_to_db(embedder_name=self.embedder_name,
                                              all_seqs=all_seqs,
                                              embeddings_out_path=self.embeddings_out_path,
                                              reduce_by_protocol=self.protocol,
                                              use_half_precision=self.use_half_precision,
                                              device=device)

        self.process = torch_mp.Process(target=_embed_function, args=(), )
        self.process.start()

    def get_task_status(self) -> TaskStatus:
        # TODO Adapt to pipeline
        if not self.process:
            return TaskStatus.RUNNING
        if self.process.is_alive():
            return TaskStatus.RUNNING
        else:
            return TaskStatus.FINISHED

    def update(self) -> Dict[str, Any]:
        result = {"status": self.get_task_status().name}
        return result
