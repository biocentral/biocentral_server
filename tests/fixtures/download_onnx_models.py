import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


# Download URL for PREDICT models (same as PredictInitializer.DOWNLOAD_URLS)
PREDICT_MODEL_URLS = [
    "https://nextcloud.in.tum.de/index.php/s/kxJ64RcRi7g6p6r/download"
]

DEFAULT_OUTPUT_DIR = Path("onnx_models")


def download_and_extract(
    urls: list[str],
    output_dir: Path,
    force: bool = False,
) -> Path:
    output_dir = Path(output_dir)

    # Check if already downloaded
    if output_dir.exists() and not force:
        # Check for at least one expected model directory
        expected_dirs = [
            output_dir / "prott5secondarystructure",
            output_dir / "seth",
            output_dir / "bindembed",
            output_dir / "tmbed",
        ]
        if any(d.exists() and any(d.glob("*.onnx")) for d in expected_dirs):
            print(f"Models already exist at {output_dir}, skipping download")
            return output_dir

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_file = output_dir.with_suffix(".zip")

    headers = {
        "Accept": "application/zip, application/octet-stream",
        "User-Agent": "biocentral_server/test",
    }

    for i, url in enumerate(urls, 1):
        try:
            print(f"Downloading models from {url} (attempt {i}/{len(urls)})...")
            response = requests.get(url, headers=headers, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192  # 8 KB

            with (
                open(zip_file, "wb") as f,
                tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress_bar,
            ):
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    progress_bar.update(size)

            print("Extracting archive...")
            shutil.unpack_archive(zip_file, output_dir)

            # Remove zip file after extraction
            zip_file.unlink()

            print(f"Models downloaded and extracted to {output_dir}")
            return output_dir

        except Exception as e:
            print(f"Failed to download from {url}: {e}", file=sys.stderr)
            if zip_file.exists():
                zip_file.unlink()

            if i == len(urls):
                raise Exception(
                    "Failed to download models from all provided URLs"
                ) from e

    # Should not reach here
    raise Exception("Unexpected error in download loop")


def get_or_download_models(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    # Check environment variable first
    env_path = os.environ.get("ONNX_MODELS_PATH")
    if env_path and not force:
        path = Path(env_path)
        if path.exists():
            return path

    # Download to specified or default directory
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    return download_and_extract(PREDICT_MODEL_URLS, output_dir, force=force)


def main():
    parser = argparse.ArgumentParser(
        description="Download ONNX prediction models for testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to extract models to (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if models already exist",
    )

    args = parser.parse_args()

    try:
        models_path = download_and_extract(
            PREDICT_MODEL_URLS,
            args.output_dir,
            force=args.force,
        )
        print("\nTo use these models in tests, set:")
        print(f"  export ONNX_MODELS_PATH={models_path.absolute()}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
