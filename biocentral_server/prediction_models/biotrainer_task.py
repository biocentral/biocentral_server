import yaml
import os.path
import tempfile

from pathlib import Path
from typing import Dict, Any

from biotrainer.utilities.cli import headless_main

from ..embeddings import EmbeddingTask
from ..server_management import MultiprocessingTask, EmbeddingsDatabase


class BiotrainerTask(MultiprocessingTask):

    def __init__(self, config_path: Path, config_dict: dict, database_instance: EmbeddingsDatabase, log_path: Path):
        super().__init__()
        self.config_path = config_path
        self.config_dict = config_dict
        self.database_instance = database_instance
        self.log_path = log_path
        self._last_read_position_in_log_file = 0

    async def _run_task(self):
        await self._pre_embed_with_db()
        headless_main(config_file_path=str(self.config_path))
        self._result = 1  # TODO Result

    async def _pre_embed_with_db(self):
        sequence_file_path = self.config_dict['sequence_file']

        embedder_name = self.config_dict['embedder_name']
        protocol = self.config_dict['protocol']
        device = self.config_dict.get('device', None)
        output_path = self.config_path.parent / "embeddings.h5"
        with tempfile.TemporaryDirectory() as temp_embeddings_dir:
            temp_embeddings_path = Path(temp_embeddings_dir)
            embedding_task = EmbeddingTask(embedder_name=embedder_name,
                                           sequence_file_path=sequence_file_path,
                                           embeddings_out_path=temp_embeddings_path,
                                           protocol=protocol,
                                           use_half_precision=False,
                                           device=device)
            embedding_triples = await self.run_subtask(embedding_task)

            # TODO [Optimization] Try to avoid double reading and saving of embedding files
            EmbeddingsDatabase.export_embeddings_to_hdf5(triples=embedding_triples, output_path=output_path)
        self.config_dict.pop("embedder_name")
        self.config_dict["embeddings_file"] = str(output_path)

        # TODO Enable biotrainer to accept a dict
        config_file_yaml = yaml.dump(self.config_dict)
        with open(self.config_path, "w") as config_file:
            config_file.write(config_file_yaml)

    def update(self) -> Dict[str, Any]:
        result = {"log_file": "", "status": self.get_task_status().name}
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as log_file:
                log_file.seek(self._last_read_position_in_log_file)
                new_content = log_file.read()
                self._last_read_position_in_log_file = log_file.tell()
                result["log_file"] = new_content
        return result
