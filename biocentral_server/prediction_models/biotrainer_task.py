import os
import time
import torch.multiprocessing as mp

from pathlib import Path
from typing import Callable, Optional
from biotrainer.protocols import Protocol
from biotrainer.utilities.cli import headless_main

from ..embeddings import ExportEmbeddingsTask
from ..server_management import TaskInterface, TaskDTO, FileContextManager, EmbeddingDatabaseFactory, EmbeddingsDatabase


class BiotrainerTask(TaskInterface):

    def __init__(self, model_path: Path, config_dict: dict):
        super().__init__()
        self.model_path = model_path
        self.config_dict = config_dict
        self.stop_reading = False
        self._last_read_position_in_log_file = 0
        self.biotrainer_process = None

    def run_task(self, update_dto_callback: Callable) -> TaskDTO:
        server_sequence_file_path = self.config_dict["sequence_file"]

        file_context_manager = FileContextManager()
        with file_context_manager.storage_write(self.model_path) as biotrainer_out_path:
            # Set output dirs to temp dir
            self.config_dict["output_dir"] = str(biotrainer_out_path)
            log_path = biotrainer_out_path / "logger_out.log"

            # Save files temporarily on the file system
            for key, value in self.config_dict.items():
                if "_file" in key and key != 'ignore_file_inconsistencies':
                    temp_path = biotrainer_out_path / (key + ".fasta")
                    file_context_manager.save_file_temporarily(temp_path, value)
                    self.config_dict[key] = str(temp_path)

            # Save embeddings to temp dir
            embedder_name = self.config_dict["embedder_name"]
            if embedder_name != "one_hot_encoding":  # Encode OHE during biotrainer process
                h5_path = self._pre_embed_with_db(server_sequence_file_path=server_sequence_file_path,
                                                  embeddings_out_path=biotrainer_out_path)
                self.config_dict.pop("embedder_name")
                self.config_dict["embeddings_file"] = str(biotrainer_out_path / h5_path.name)

            # Run biotrainer
            self.biotrainer_process = mp.Process(target=headless_main, args=(self.config_dict,), )
            self.biotrainer_process.start()

            # Read logs until process is finished
            self._read_logs(update_dto_callback=update_dto_callback, log_path=log_path)

        return TaskDTO.finished(result={})

    def _read_logs(self, update_dto_callback: Callable, log_path: Path):
        while self.biotrainer_process.is_alive():
            new_log_content = self._read_log_file(log_path)
            if new_log_content != "":
                update_dto_callback(TaskDTO.running().add_update({"log_file": new_log_content}))
            time.sleep(1)
        # After stop reading we read the last output from the log file
        final_log_content = self._read_log_file(log_path)
        update_dto_callback(TaskDTO.running().add_update({"log_file": final_log_content}))

    def _read_log_file(self, log_path: Path):
        if os.path.exists(log_path):
            with open(log_path, "r") as log_file:
                log_file.seek(self._last_read_position_in_log_file)
                new_log_content = log_file.read()
                self._last_read_position_in_log_file = log_file.tell()
                return new_log_content
        return ""

    def _pre_embed_with_db(self, server_sequence_file_path: str, embeddings_out_path: Path) -> Path:
        embedder_name = self.config_dict['embedder_name']
        protocol = Protocol.from_string(self.config_dict['protocol'])
        reduced = protocol in Protocol.using_per_sequence_embeddings()
        device = self.config_dict.get('device', None)

        export_embedding_task = ExportEmbeddingsTask(embedder_name=embedder_name,
                                              sequence_input=Path(server_sequence_file_path),
                                              embeddings_out_path=embeddings_out_path,
                                              reduced=reduced,
                                              use_half_precision=False,
                                              device=device)
        embedding_dto: Optional[TaskDTO] = None
        for current_dto in self.run_subtask(export_embedding_task):
            embedding_dto = current_dto

        if not embedding_dto:
            raise Exception("Could not compute embeddings!")
        h5_path = embedding_dto.update["embeddings_file"]
        return h5_path
