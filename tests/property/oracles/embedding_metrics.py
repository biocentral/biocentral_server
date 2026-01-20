"""
Embedding metrics utilities for oracle-based testing.

Provides divergence metrics (cosine, L2, KL) for comparing embeddings,
along with table formatting and CSV report generation.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.special import softmax


def compute_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance between two embeddings.

    Cosine distance = 1 - cosine_similarity
    Range: [0, 2] where 0 = identical direction, 2 = opposite direction

    Args:
        a: First embedding vector (1D or 2D - will be pooled if 2D)
        b: Second embedding vector (1D or 2D - will be pooled if 2D)

    Returns:
        Cosine distance as float
    """
    a_flat = _ensure_1d(a)
    b_flat = _ensure_1d(b)

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    if norm_a == 0 or norm_b == 0:
        return 1.0  # Maximum distance for zero vectors

    cosine_similarity = np.dot(a_flat, b_flat) / (norm_a * norm_b)
    # Clamp to [-1, 1] to handle numerical errors
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    return float(1.0 - cosine_similarity)


def compute_l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute L2 (Euclidean) distance between two embeddings.

    Args:
        a: First embedding vector (1D or 2D - will be pooled if 2D)
        b: Second embedding vector (1D or 2D - will be pooled if 2D)

    Returns:
        L2 distance as float
    """
    a_flat = _ensure_1d(a)
    b_flat = _ensure_1d(b)

    return float(np.linalg.norm(a_flat - b_flat))


def compute_kl_divergence(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence between two embeddings after softmax normalization.

    KL(P || Q) = sum(P * log(P / Q))
    Uses softmax to convert embeddings to probability distributions.

    Args:
        a: First embedding vector (1D or 2D - will be pooled if 2D)
        b: Second embedding vector (1D or 2D - will be pooled if 2D)
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence as float (always >= 0)
    """
    a_flat = _ensure_1d(a)
    b_flat = _ensure_1d(b)

    # Convert to probability distributions using softmax
    p = softmax(a_flat)
    q = softmax(b_flat)

    # Add epsilon to avoid log(0)
    q = q + epsilon
    q = q / q.sum()  # Re-normalize after adding epsilon

    # Compute KL divergence
    kl = np.sum(p * np.log(p / q))
    return float(max(0.0, kl))  # Ensure non-negative due to numerical errors


def compute_all_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    Compute all divergence metrics between two embeddings.

    Args:
        a: First embedding vector
        b: Second embedding vector

    Returns:
        Dictionary with keys: cosine_distance, l2_distance, kl_divergence
    """
    return {
        "cosine_distance": compute_cosine_distance(a, b),
        "l2_distance": compute_l2_distance(a, b),
        "kl_divergence": compute_kl_divergence(a, b),
    }


def _ensure_1d(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is 1D by mean-pooling if 2D.

    Args:
        arr: Input array (1D or 2D)

    Returns:
        1D numpy array
    """
    if arr.ndim == 1:
        return arr
    elif arr.ndim == 2:
        # Mean pool over sequence dimension (axis 0)
        return arr.mean(axis=0)
    else:
        raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def format_metrics_table(
    results: List[Dict[str, Any]],
    title: Optional[str] = None,
) -> str:
    """
    Format metrics results as a readable table for console output.

    Args:
        results: List of dicts with keys: embedder, test_type, parameter,
                 cosine_distance, l2_distance, kl_divergence, threshold, passed
        title: Optional title for the table

    Returns:
        Formatted string table
    """
    if not results:
        return "No results to display."

    # Column headers and widths
    headers = ["Embedder", "Test Type", "Parameter", "Cosine", "L2", "KL", "Threshold", "Passed"]
    col_widths = [15, 20, 12, 10, 10, 10, 10, 8]

    lines = []

    # Add title if provided
    if title:
        lines.append(f"\n{'=' * 97}")
        lines.append(f"  {title}")
        lines.append(f"{'=' * 97}")

    # Header row
    header_row = " | ".join(
        h.ljust(w) for h, w in zip(headers, col_widths)
    )
    lines.append(header_row)
    lines.append("-" * len(header_row))

    # Data rows
    for row in results:
        values = [
            str(row.get("embedder", ""))[:15],
            str(row.get("test_type", ""))[:20],
            str(row.get("parameter", ""))[:12],
            f"{row.get('cosine_distance', 0):.6f}",
            f"{row.get('l2_distance', 0):.4f}",
            f"{row.get('kl_divergence', 0):.6f}",
            f"{row.get('threshold', 0):.4f}",
            "✓" if row.get("passed", False) else "✗",
        ]
        data_row = " | ".join(
            v.ljust(w) for v, w in zip(values, col_widths)
        )
        lines.append(data_row)

    lines.append("")
    return "\n".join(lines)


def write_metrics_csv(
    results: List[Dict[str, Any]],
    path: Union[str, Path],
) -> None:
    """
    Write metrics results to a CSV file, overwriting any existing content.

    CSV columns: timestamp, embedder, test_type, parameter, cosine_distance,
                 l2_distance, kl_divergence, threshold, passed

    Args:
        results: List of metric result dictionaries
        path: Path to output CSV file
    """
    path = Path(path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "embedder",
        "test_type",
        "parameter",
        "cosine_distance",
        "l2_distance",
        "kl_divergence",
        "threshold",
        "passed",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            # Ensure all required fields exist
            csv_row = {
                "timestamp": row.get("timestamp", datetime.now().isoformat()),
                "embedder": row.get("embedder", ""),
                "test_type": row.get("test_type", ""),
                "parameter": row.get("parameter", ""),
                "cosine_distance": row.get("cosine_distance", 0.0),
                "l2_distance": row.get("l2_distance", 0.0),
                "kl_divergence": row.get("kl_divergence", 0.0),
                "threshold": row.get("threshold", 0.0),
                "passed": row.get("passed", False),
            }
            writer.writerow(csv_row)


def get_default_report_path() -> Path:
    """Get the default path for oracle metrics CSV report."""
    return Path(__file__).parent.parent.parent / "reports" / "oracle_metrics.csv"
