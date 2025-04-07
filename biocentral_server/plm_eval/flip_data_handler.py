import shutil
import logging
import requests

from tqdm import tqdm
from Bio import SeqIO
from pathlib import Path
from typing import Dict, Any, Callable
from biotrainer.protocols import Protocol
from autoeval.utilities.FLIP import FLIP_DATASETS
from biotrainer.utilities import read_FASTA, get_attributes_from_seqrecords

logger = logging.getLogger(__name__)


class FLIPDataHandler:
    """Handles FLIP dataset operations with clear separation between path management and data processing"""

    DOWNLOAD_FILE_NAME = "all_fastas"
    DOWNLOAD_URL = f"http://data.bioembeddings.com/public/FLIP/fasta/{DOWNLOAD_FILE_NAME}.zip"
    IGNORE_SPLITS = ["mixed_vs_human_2"]
    MIN_SEQ_SIZE = 0
    MAX_SEQ_SIZE = 2000

    @staticmethod
    def download_and_preprocess(flip_path: Path) -> None:
        """Download and preprocess FLIP data to the given path"""
        FLIPDataHandler._download_flip_data(FLIPDataHandler.DOWNLOAD_URL, flip_path)
        logger.info("FLIP data downloaded, preprocessing...")

        # Preprocess all dataset files in one go
        for dataset, dataset_info in tqdm(FLIP_DATASETS.items(), desc="Preprocessing datasets"):
            dataset_dir = flip_path / dataset
            protocol = dataset_info["protocol"]
            if isinstance(protocol, str):
                protocol = Protocol.from_string(protocol)

            # Process all splits
            for split in dataset_info["splits"]:
                if split in FLIPDataHandler.IGNORE_SPLITS:
                    continue

                try:
                    if protocol in Protocol.per_sequence_protocols():
                        FLIPDataHandler._ensure_preprocessed_file(dataset_dir, split)
                    elif protocol in Protocol.per_residue_protocols():
                        FLIPDataHandler._ensure_preprocessed_file(dataset_dir, "sequences")
                except Exception as e:
                    logger.warning(f"Error preprocessing {dataset}/{split}: {e}")

        logger.info("FLIP data preprocessing completed!")

    @staticmethod
    def get_dataset_paths(flip_path: Path, check_path_exists_function: Callable[[Path], bool]) -> Dict[str, Any]:
        """Build path dictionary for all FLIP datasets"""
        result_dict = {}

        for dataset, dataset_info in FLIP_DATASETS.items():
            result_dict[dataset] = {}
            dataset_dir = flip_path / dataset
            protocol = dataset_info["protocol"]

            if isinstance(protocol, str):
                protocol = Protocol.from_string(protocol)
            result_dict[dataset]["protocol"] = protocol

            result_dict[dataset]["splits"] = []
            for split in dataset_info["splits"]:
                if split in FLIPDataHandler.IGNORE_SPLITS:
                    continue

                split_data = {
                    "name": split,
                    "sequence_file": None,
                    "labels_file": None,
                    "mask_file": None
                }

                try:
                    if protocol in Protocol.per_sequence_protocols():
                        sequence_file = FLIPDataHandler._get_sequence_file_path(dataset_dir,
                                                                                split,
                                                                                check_path_exists_function)
                        if check_path_exists_function(dataset_dir / sequence_file):
                            split_data["sequence_file"] = str(dataset_dir / sequence_file)
                        else:
                            logger.error(f"Missing sequence file for {dataset}/{split}")
                            continue

                    elif protocol in Protocol.per_residue_protocols():
                        sequences_file = FLIPDataHandler._get_sequence_file_path(dataset_dir,
                                                                                 "sequences",
                                                                                 check_path_exists_function)

                        labels_file = Path(f"{split}.fasta")
                        mask_file = Path(f"resolved.fasta")

                        if check_path_exists_function(dataset_dir / sequences_file) and check_path_exists_function(
                                dataset_dir / labels_file):
                            split_data["sequence_file"] = str(dataset_dir / sequences_file)
                            split_data["labels_file"] = str(dataset_dir / labels_file)
                        else:
                            logger.error(f"Missing required files for {dataset}/{split}")
                            continue

                        if check_path_exists_function(dataset_dir / mask_file):
                            split_data["mask_file"] = str(dataset_dir / mask_file)

                    result_dict[dataset]["splits"].append(split_data)

                except Exception as e:
                    logger.error(f"Error processing {dataset}/{split}: {e}")

        return result_dict

    @staticmethod
    def _download_flip_data(url: str, flip_data_dir: Path) -> None:
        """Download and extract FLIP data archive"""
        zip_file = flip_data_dir.with_suffix('.zip')

        try:
            logger.info(f"Downloading FLIP data from {url}..")
            headers = {
                'Accept': 'application/zip, application/octet-stream',
                'User-Agent': 'biocentral_server/alpha'
            }
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8 KB

            with open(zip_file, "wb") as f, tqdm(
                    desc="Downloading FLIP data",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    progress_bar.update(size)

            logger.info("Unpacking FLIP data archive..")
            shutil.unpack_archive(zip_file, flip_data_dir)

            # Remove the zip file after successful extraction
            zip_file.unlink()

            logger.info("FLIP data downloaded and unpacked successfully!")

        except Exception as e:
            logger.error(f"Error during FLIP data download: {e}")
            if zip_file.exists():
                zip_file.unlink()
            raise

    @staticmethod
    def _get_sequence_file_path(dataset_dir: Path, name: str,
                                check_path_exists_function: Callable[[Path], bool]) -> Path:
        """Get the appropriate sequence file path (preprocessed if available)"""
        raw_path = Path(f"{name}.fasta")
        preprocessed_path = Path("preprocessed") / raw_path

        if check_path_exists_function(dataset_dir / preprocessed_path):
            return preprocessed_path
        return raw_path

    @staticmethod
    def _ensure_preprocessed_file(dataset_dir: Path, name: str) -> Path:
        """Ensure a preprocessed version of the file exists and return its path"""
        download_path = dataset_dir / f"{name}.fasta"
        preprocessed_path = dataset_dir / "preprocessed" / f"{name}.fasta"

        # If preprocessed file already exists, return its path
        if preprocessed_path.exists():
            return preprocessed_path

        # If raw file doesn't exist, we can't proceed
        if not download_path.exists():
            raise FileNotFoundError(f"Required file {download_path} not available for: {name}")

        # Preprocess the file
        all_seq_records = read_FASTA(str(download_path))
        all_attributes = get_attributes_from_seqrecords(all_seq_records)

        keep_seqs = [
            seq for seq in all_seq_records
            if (FLIPDataHandler.MIN_SEQ_SIZE <= len(seq.seq) <= FLIPDataHandler.MAX_SEQ_SIZE and
                all_attributes[seq.id].get("SET", "") != "nan")
        ]

        preprocessed_dir = dataset_dir / "preprocessed"
        preprocessed_dir.mkdir(exist_ok=True)

        n_written = SeqIO.write(keep_seqs, preprocessed_path, "fasta")
        assert n_written == len(keep_seqs)

        logger.info(f"Preprocessed {download_path.name}: kept {n_written}/{len(all_seq_records)} sequences")
        return preprocessed_path
