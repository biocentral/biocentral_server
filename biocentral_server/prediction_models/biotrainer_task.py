import io
import time
import yaml
import logging
import tempfile
import threading

from pathlib import Path
from typing import Dict, Any, Callable

from biotrainer.utilities.cli import headless_main

from ..embeddings import EmbeddingTask
from ..server_management import TaskInterface, EmbeddingsDatabase, TaskDTO


class LogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_buffer = io.StringIO()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))  # biotrainer format

    def emit(self, record):
        msg = self.format(record)
        self.log_buffer.write(msg + '\n')


class BiotrainerTask(TaskInterface):

    def __init__(self, config_path: Path, config_dict: dict, database_instance: EmbeddingsDatabase, log_path: Path):
        super().__init__()
        self.config_path = config_path
        self.config_dict = config_dict
        self.database_instance = database_instance
        self.log_path = log_path
        self.log_capture_handler = LogCaptureHandler()
        self.stop_reading = False
        self._log_output = ""

    def run_task(self, update_dto_callback: Callable) -> Any:
        # Add our custom handler to the root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_capture_handler)

        # Start a thread to read the log output
        read_thread = threading.Thread(target=self._read_logs, args=(update_dto_callback,))
        read_thread.start()

        self._pre_embed_with_db()

        try:
            result = headless_main(config_file_path=str(self.config_path))
        finally:
            self.stop_reading = True
            read_thread.join()
            # Remove our custom handler
            root_logger.removeHandler(self.log_capture_handler)

        return {"log_file": self._log_output}  #  TODO Does this ensure that the whole log file is sent?

    def _read_logs(self, update_dto_callback: Callable):
        while not self.stop_reading:
            log_output = self.log_capture_handler.log_buffer.getvalue()
            if log_output:
                self._log_output = log_output
                update_dto_callback(TaskDTO.running().add_update({"log_file": log_output}))
                self.log_capture_handler.log_buffer.truncate(0)
                self.log_capture_handler.log_buffer.seek(0)
                with open(str(self.log_path), "a") as log_file:
                    log_file.write(log_output)
            time.sleep(0.2)  # Query every two seconds

        # After stop reading we read the last output from the buffer
        log_output = self.log_capture_handler.log_buffer.getvalue()
        self._log_output = log_output
        with open(str(self.log_path), "a") as log_file:
            log_file.write(log_output)

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

            embeddings_task_result = embedding_dto.update["embeddings_file"][embedder_name]
            # TODO [Optimization] Try to avoid double reading and saving of embedding files
            EmbeddingsDatabase.export_embeddings_task_result_to_hdf5(embeddings_task_result=embeddings_task_result,
                                                                     output_path=output_path)
        self.config_dict.pop("embedder_name")
        self.config_dict["embeddings_file"] = str(output_path)

        # TODO Enable biotrainer to accept a dict
        config_file_yaml = yaml.dump(self.config_dict)
        with open(self.config_path, "w") as config_file:
            config_file.write(config_file_yaml)
