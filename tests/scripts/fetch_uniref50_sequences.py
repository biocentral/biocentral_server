#!/usr/bin/env python3

import argparse
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
OUTPUT_FASTA = FIXTURES_DIR / "uniref50_sequences.fasta"

# Length bins and tolerances
LENGTH_BINS: List[Tuple[int, int, int]] = [
    # (target_length, min_length, max_length)
    (50, 45, 55),
    (100, 90, 110),
    (200, 180, 220),
    (400, 360, 440),
    (600, 540, 660),
    (800, 720, 880),
    (1000, 900, 1100),
]

DEFAULT_SEQS_PER_BIN = 250

UNIREF_SEARCH_URL = "https://rest.uniprot.org/uniref/search"

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def _fetch_sequences_for_bin(
    target_len: int,
    min_len: int,
    max_len: int,
    n_sequences: int,
    seed: int = 42,
) -> List[Tuple[str, str, int]]:
    # Query UniRef50 for clusters whose representative has the desired length
    query = f"identity:0.5 AND length:[{min_len} TO {max_len}]"

    # Request more than needed to allow filtering
    request_size = min(n_sequences * 3, 500)

    params = {
        "query": query,
        "format": "fasta",
        "size": request_size,
    }

    sequences = []

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Fetching bin={target_len} (range {min_len}-{max_len}), attempt {attempt + 1}...")
            response = requests.get(UNIREF_SEARCH_URL, params=params, timeout=120)
            response.raise_for_status()

            # Parse FASTA response
            raw_sequences = _parse_fasta(response.text)

            # Filter: only keep sequences with standard amino acids
            valid = []
            for uid, seq in raw_sequences:
                seq_upper = seq.upper()
                if all(c in AMINO_ACIDS for c in seq_upper) and min_len <= len(seq_upper) <= max_len:
                    valid.append((uid, seq_upper, len(seq_upper)))

            if not valid:
                print(f"  WARNING: No valid sequences found for bin={target_len}")
                break

            # If we have a Link header for pagination, follow it
            sequences.extend(valid)

            # Follow pagination if we need more
            while len(sequences) < n_sequences and "Link" in response.headers:
                link = response.headers["Link"]
                next_url = _extract_next_link(link)
                if not next_url:
                    break
                time.sleep(1)  # Be polite to the API
                response = requests.get(next_url, timeout=120)
                response.raise_for_status()
                raw_sequences = _parse_fasta(response.text)
                for uid, seq in raw_sequences:
                    seq_upper = seq.upper()
                    if all(c in AMINO_ACIDS for c in seq_upper) and min_len <= len(seq_upper) <= max_len:
                        sequences.append((uid, seq_upper, len(seq_upper)))

            break  # Success

        except requests.RequestException as e:
            print(f"  Request failed: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ERROR: Failed after {MAX_RETRIES} attempts for bin={target_len}")

    # Deduplicate by sequence
    seen = set()
    unique = []
    for uid, seq, length in sequences:
        if seq not in seen:
            seen.add(seq)
            unique.append((uid, seq, length))

    # Randomly sample if we have more than needed
    rng = random.Random(seed)
    if len(unique) > n_sequences:
        unique = rng.sample(unique, n_sequences)

    print(f"  bin={target_len}: collected {len(unique)} sequences")
    return unique

def _parse_fasta(fasta_text: str) -> List[Tuple[str, str]]:
    entries = []
    current_id = None
    current_seq_parts = []

    for line in fasta_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                entries.append((current_id, "".join(current_seq_parts)))
            # Extract ID from header (e.g., >UniRef50_A0A000 ...)
            current_id = line[1:].split()[0]
            current_seq_parts = []
        else:
            current_seq_parts.append(line)

    if current_id is not None:
        entries.append((current_id, "".join(current_seq_parts)))

    return entries

def _extract_next_link(link_header: str) -> Optional[str]:
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return match.group(1) if match else None

def fetch_all(n_seqs_per_bin: int = DEFAULT_SEQS_PER_BIN, force: bool = False) -> Path:
    if OUTPUT_FASTA.exists() and not force:
        # Verify the file has enough sequences
        existing = load_uniref50_sequences()
        total = sum(len(seqs) for seqs in existing.values())
        expected = n_seqs_per_bin * len(LENGTH_BINS)
        if total >= expected * 0.9:  # Allow 10% tolerance
            print(f"FASTA file already exists with {total} sequences ({OUTPUT_FASTA})")
            print("Use --force to re-download.")
            return OUTPUT_FASTA

    print(f"Fetching {n_seqs_per_bin} sequences per bin from UniRef50...")
    print(f"Length bins: {[b[0] for b in LENGTH_BINS]}")

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    all_entries = []
    for target_len, min_len, max_len in LENGTH_BINS:
        entries = _fetch_sequences_for_bin(
            target_len, min_len, max_len, n_seqs_per_bin,
            seed=42 + target_len,
        )
        all_entries.extend(
            (uid, seq, length, target_len) for uid, seq, length in entries
        )
        time.sleep(2)  # Rate limiting between bins

    # Write FASTA
    with open(OUTPUT_FASTA, "w") as f:
        for uid, seq, length, bin_target in all_entries:
            f.write(f">{uid} | bin={bin_target} | length={length}\n")
            # Write sequence in 80-char lines
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")

    total = len(all_entries)
    print(f"\nWrote {total} sequences to {OUTPUT_FASTA}")
    for target, _, _ in LENGTH_BINS:
        count = sum(1 for _, _, _, b in all_entries if b == target)
        print(f"  bin={target}: {count} sequences")

    return OUTPUT_FASTA

def load_uniref50_sequences(bin_size: Optional[int] = None) -> Dict[int, List[str]]:
    if not OUTPUT_FASTA.exists():
        raise FileNotFoundError(
            f"UniRef50 FASTA not found at {OUTPUT_FASTA}. "
            "Run `python tests/scripts/fetch_uniref50_sequences.py` first."
        )

    sequences: Dict[int, List[str]] = {}
    current_bin = None
    current_seq_parts = []

    with open(OUTPUT_FASTA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Save previous entry
                if current_bin is not None and current_seq_parts:
                    seq = "".join(current_seq_parts)
                    sequences.setdefault(current_bin, []).append(seq)

                # Parse bin from header
                bin_match = re.search(r"bin=(\d+)", line)
                current_bin = int(bin_match.group(1)) if bin_match else None
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    # Don't forget last entry
    if current_bin is not None and current_seq_parts:
        seq = "".join(current_seq_parts)
        sequences.setdefault(current_bin, []).append(seq)

    if bin_size is not None:
        return {bin_size: sequences.get(bin_size, [])}

    return sequences

def main():
    parser = argparse.ArgumentParser(description="Fetch UniRef50 sequences for experiments")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument(
        "--seqs-per-bin", type=int, default=DEFAULT_SEQS_PER_BIN,
        help=f"Number of sequences per length bin (default: {DEFAULT_SEQS_PER_BIN})",
    )
    args = parser.parse_args()
    fetch_all(n_seqs_per_bin=args.seqs_per_bin, force=args.force)

if __name__ == "__main__":
    main()
