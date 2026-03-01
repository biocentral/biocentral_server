#!/usr/bin/env python3
# Plot X-masking results: cosine similarity vs masking rate.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def plot_masking_results():
    # Load both progressive and random masking results
    prog_path = REPORTS_DIR / "x_masking_progressive_esm2.csv"
    rand_path = REPORTS_DIR / "x_masking_random_esm2.csv"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors for different sequence lengths
    colors = [
        "#2E86AB",
        "#E94F37",
        "#4CAF50",
        "#9C27B0",
    ]  # Blue, red, green, purple
    seq_lengths = {0: 15, 1: 76, 2: 400, 3: 1000}  # Map seq idx to length

    for ax, (path, title, bg_color) in zip(
        axes,
        [
            (prog_path, "Progressive X-Masking", "#2E86AB"),
            (rand_path, "Random X-Masking", "#A23B72"),
        ],
    ):
        if not path.exists():
            print(f"File not found: {path}")
            continue

        df = pd.read_csv(path)

        # Extract masking ratio from parameter column
        df["masking_ratio"] = (
            df["parameter"].str.extract(r"mask(\d+)%").astype(float) / 100
        )
        df["seq_idx"] = df["parameter"].str.extract(r"seq(\d+)").astype(int)

        # Convert cosine distance to similarity
        df["cosine_similarity"] = 1 - df["cosine_distance"]

        # Add regions annotation (background) - adjusted based on actual data
        # Low p: 0-15% (short seqs tolerate more; long seqs diverge quickly)
        # Mid p: 15-70% (steady decline)
        # High p: 70-100% (reversal/collapse toward all-X embedding for long seqs)
        ax.axvspan(0, 15, alpha=0.08, color="green", label="_nolegend_")
        ax.axvspan(15, 70, alpha=0.08, color="orange", label="_nolegend_")
        ax.axvspan(70, 100, alpha=0.08, color="red", label="_nolegend_")

        # Plot each sequence
        for i, seq_idx in enumerate(sorted(df["seq_idx"].unique())):
            seq_data = df[df["seq_idx"] == seq_idx].sort_values("masking_ratio")
            seq_len = seq_lengths.get(seq_idx, "?")
            ax.plot(
                seq_data["masking_ratio"] * 100,
                seq_data["cosine_similarity"],
                marker="o",
                markersize=5,
                linewidth=2,
                label=f"len={seq_len}",
                color=colors[i % len(colors)],
                alpha=0.9,
            )

        # Add threshold line at similarity = 0.9 (distance = 0.1)
        ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(
            102, 0.9, "r*", fontsize=10, va="center", color="gray", fontweight="bold"
        )

        # Add line at similarity = 0 (orthogonal vectors)
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        # Region labels at top
        ax.text(
            7,
            0.97,
            "Low p",
            ha="center",
            fontsize=9,
            color="darkgreen",
            fontweight="bold",
        )
        ax.text(
            42,
            0.97,
            "Mid p",
            ha="center",
            fontsize=9,
            color="darkorange",
            fontweight="bold",
        )
        ax.text(
            85,
            0.97,
            "High p",
            ha="center",
            fontsize=9,
            color="darkred",
            fontweight="bold",
        )

        ax.set_xlabel("Masking Rate p (%)", fontsize=12)
        ax.set_ylabel("Cosine Similarity", fontsize=12)
        ax.set_title(
            f"{title}\n(ESM2-T6-8M, n=30 runs)", fontsize=13, fontweight="bold"
        )
        ax.set_xlim(-2, 108)
        ax.set_ylim(-0.05, 1.02)  # Allow negative similarity (distance > 1)
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="-")
        ax.set_xticks(np.arange(0, 101, 10))

    plt.tight_layout()

    # Save plot
    output_path = REPORTS_DIR / "x_masking_similarity_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Also save as PDF for thesis
    pdf_path = REPORTS_DIR / "x_masking_similarity_plot.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF saved to: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    plot_masking_results()
