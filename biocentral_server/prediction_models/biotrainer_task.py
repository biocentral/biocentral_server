import io
import time
import yaml
import os.path
import tempfile
import threading

from pathlib import Path
from typing import Dict, Any, Callable
from contextlib import redirect_stdout

from biotrainer.utilities.cli import headless_main

from ..embeddings import EmbeddingTask
from ..server_management import TaskInterface, EmbeddingsDatabase, TaskDTO


class BiotrainerTask(TaskInterface):

    def __init__(self, config_path: Path, config_dict: dict, database_instance: EmbeddingsDatabase, log_path: Path):
        super().__init__()
        self.config_path = config_path
        self.config_dict = config_dict
        self.database_instance = database_instance
        self.log_path = log_path
        self.output_buffer = io.StringIO()
        self._stop_reading = False

    def run_task(self, update_dto_callback: Callable) -> Any:
        self._pre_embed_with_db()
        # Start a thread to read the output
        read_thread = threading.Thread(target=self._read_output, args=(update_dto_callback,))
        read_thread.start()

        try:
            with redirect_stdout(self.output_buffer):
                result = headless_main(config_file_path=str(self.config_path))
        finally:
            self._stop_reading = True
            read_thread.join()

        return result

    def _read_output(self, update_dto_callback: Callable):
        while not self._stop_reading:
            output = self.output_buffer.getvalue()
            if output:
                update_dto_callback(TaskDTO.running().update({"log_file": output}))
                self.output_buffer.truncate(0)
                self.output_buffer.seek(0)
            time.sleep(2)  # Adjust this value to control how often to check for new output

    def _pre_embed_with_db(self):
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
                                           device=device,
                                           embeddings_database=self.database_instance)
            embedding_dto = self.run_subtask(embedding_task)

            embedding_triples = embedding_dto.update  # TODO
            # TODO [Optimization] Try to avoid double reading and saving of embedding files
            EmbeddingsDatabase.export_embeddings_to_hdf5(triples=embedding_triples, output_path=output_path)
        self.config_dict.pop("embedder_name")
        self.config_dict["embeddings_file"] = str(output_path)

        # TODO Enable biotrainer to accept a dict
        config_file_yaml = yaml.dump(self.config_dict)
        with open(self.config_path, "w") as config_file:
            config_file.write(config_file_yaml)
