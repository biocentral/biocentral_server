#!/usr/bin/env python3
# Plot X-masking results: cosine similarity AND L2 distance vs masking rate.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Colors for different sequence lengths
COLORS = [
    "#2E86AB",
    "#E94F37",
    "#4CAF50",
    "#9C27B0",
    "#FF9800",
    "#00BCD4",
    "#795548",
]

def _load_and_prepare(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"File not found: {path}")
        return None
    df = pd.read_csv(path)
    df["masking_ratio"] = (
        df["parameter"].str.extract(r"mask(\d+)%").astype(float) / 100
    )
    df["seq_idx"] = df["parameter"].str.extract(r"seq(\d+)").astype(int)
    df["cosine_similarity"] = 1 - df["cosine_distance"]
    return df

def _add_region_shading(ax):
    ax.axvspan(0, 15, alpha=0.08, color="green", label="_nolegend_")
    ax.axvspan(15, 70, alpha=0.08, color="orange", label="_nolegend_")
    ax.axvspan(70, 100, alpha=0.08, color="red", label="_nolegend_")

def _add_region_labels(ax, y_pos):
    ax.text(7, y_pos, "Low p", ha="center", fontsize=9, color="darkgreen", fontweight="bold")
    ax.text(42, y_pos, "Mid p", ha="center", fontsize=9, color="darkorange", fontweight="bold")
    ax.text(85, y_pos, "High p", ha="center", fontsize=9, color="darkred", fontweight="bold")

def _infer_seq_lengths(df: pd.DataFrame) -> dict:
    if "sequence_length" in df.columns:
        mapping = {}
        for idx in sorted(df["seq_idx"].unique()):
            lens = df.loc[df["seq_idx"] == idx, "sequence_length"].dropna().unique()
            mapping[idx] = int(lens[0]) if len(lens) > 0 else "?"
        return mapping
    # Fallback for old CSVs
    return {0: 15, 1: 76, 2: 400, 3: 1000}

def plot_masking_results():
    prog_path = REPORTS_DIR / "x_masking_progressive_esm2.csv"
    rand_path = REPORTS_DIR / "x_masking_random_esm2.csv"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (path, title) in zip(
        axes,
        [
            (prog_path, "Progressive X-Masking"),
            (rand_path, "Random X-Masking"),
        ],
    ):
        df = _load_and_prepare(path)
        if df is None:
            continue

        seq_lengths = _infer_seq_lengths(df)
        _add_region_shading(ax)

        for i, seq_idx in enumerate(sorted(df["seq_idx"].unique())):
            seq_data = df[df["seq_idx"] == seq_idx].sort_values("masking_ratio")
            seq_len = seq_lengths.get(seq_idx, "?")
            ax.plot(
                seq_data["masking_ratio"] * 100,
                seq_data["cosine_similarity"],
                marker="o", markersize=5, linewidth=2,
                label=f"len={seq_len}",
                color=COLORS[i % len(COLORS)], alpha=0.9,
            )

        ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(102, 0.9, "r*", fontsize=10, va="center", color="gray", fontweight="bold")
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        _add_region_labels(ax, 0.97)

        ax.set_xlabel("Masking Rate p (%)", fontsize=12)
        ax.set_ylabel("Cosine Similarity", fontsize=12)
        ax.set_title(f"{title}\n(ESM2-T6-8M, n=30 runs)", fontsize=13, fontweight="bold")
        ax.set_xlim(-2, 108)
        ax.set_ylim(-0.05, 1.02)
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="-")
        ax.set_xticks(np.arange(0, 101, 10))

    plt.tight_layout()
    _save_plot("x_masking_similarity_plot")
    plt.close()

def plot_masking_results_with_l2():
    prog_path = REPORTS_DIR / "x_masking_progressive_esm2.csv"
    rand_path = REPORTS_DIR / "x_masking_random_esm2.csv"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    datasets = [
        (prog_path, "Progressive X-Masking"),
        (rand_path, "Random X-Masking"),
    ]

    for col, (path, masking_label) in enumerate(datasets):
        df = _load_and_prepare(path)
        if df is None:
            continue

        seq_lengths = _infer_seq_lengths(df)

        # --- Top row: cosine similarity ---
        ax_cos = axes[0, col]
        _add_region_shading(ax_cos)

        for i, seq_idx in enumerate(sorted(df["seq_idx"].unique())):
            seq_data = df[df["seq_idx"] == seq_idx].sort_values("masking_ratio")
            seq_len = seq_lengths.get(seq_idx, "?")
            ax_cos.plot(
                seq_data["masking_ratio"] * 100,
                seq_data["cosine_similarity"],
                marker="o", markersize=4, linewidth=2,
                label=f"len={seq_len}",
                color=COLORS[i % len(COLORS)], alpha=0.9,
            )

        ax_cos.axhline(y=0.9, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax_cos.text(102, 0.9, "r*", fontsize=10, va="center", color="gray", fontweight="bold")
        ax_cos.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        _add_region_labels(ax_cos, 0.97)

        ax_cos.set_xlabel("Masking Rate p (%)", fontsize=11)
        ax_cos.set_ylabel("Cosine Similarity", fontsize=11)
        ax_cos.set_title(f"{masking_label} — Cosine Similarity\n(ESM2-T6-8M)", fontsize=12, fontweight="bold")
        ax_cos.set_xlim(-2, 108)
        ax_cos.set_ylim(-0.05, 1.02)
        ax_cos.legend(loc="lower left", fontsize=9)
        ax_cos.grid(True, alpha=0.3, linestyle="-")
        ax_cos.set_xticks(np.arange(0, 101, 10))

        # --- Bottom row: L2 distance ---
        ax_l2 = axes[1, col]
        _add_region_shading(ax_l2)

        for i, seq_idx in enumerate(sorted(df["seq_idx"].unique())):
            seq_data = df[df["seq_idx"] == seq_idx].sort_values("masking_ratio")
            seq_len = seq_lengths.get(seq_idx, "?")
            ax_l2.plot(
                seq_data["masking_ratio"] * 100,
                seq_data["l2_distance"],
                marker="s", markersize=4, linewidth=2,
                label=f"len={seq_len}",
                color=COLORS[i % len(COLORS)], alpha=0.9,
            )

        _add_region_labels(ax_l2, ax_l2.get_ylim()[1] * 0.97 if ax_l2.get_ylim()[1] > 0 else 0.97)

        ax_l2.set_xlabel("Masking Rate p (%)", fontsize=11)
        ax_l2.set_ylabel("L2 (Euclidean) Distance", fontsize=11)
        ax_l2.set_title(f"{masking_label} — L2 Distance\n(ESM2-T6-8M)", fontsize=12, fontweight="bold")
        ax_l2.set_xlim(-2, 108)
        ax_l2.legend(loc="upper left", fontsize=9)
        ax_l2.grid(True, alpha=0.3, linestyle="-")
        ax_l2.set_xticks(np.arange(0, 101, 10))

    plt.tight_layout()
    _save_plot("x_masking_cosine_and_l2_plot")
    plt.close()

def plot_baseline_comparison():
    embedder_configs = [
        ("esm2", "ESM2-T6-8M", "-"),
        ("one_hot", "One-Hot Encoding", "--"),
        ("random_seqs", "Random Sequences", ":"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for masking_type, col in [("progressive", 0), ("random", 1)]:
        ax_cos = axes[0, col]
        ax_l2 = axes[1, col]

        for embedder_key, embedder_label, linestyle in embedder_configs:
            path = REPORTS_DIR / f"x_masking_{masking_type}_{embedder_key}.csv"
            df = _load_and_prepare(path)
            if df is None:
                continue

            # Aggregate across all sequences: mean ± std per masking ratio
            agg = df.groupby("masking_ratio").agg(
                cosine_mean=("cosine_similarity", "mean"),
                cosine_std=("cosine_similarity", "std"),
                l2_mean=("l2_distance", "mean"),
                l2_std=("l2_distance", "std"),
            ).reset_index()

            x = agg["masking_ratio"] * 100

            # Cosine similarity
            ax_cos.plot(x, agg["cosine_mean"], linewidth=2, linestyle=linestyle, label=embedder_label, alpha=0.9)
            ax_cos.fill_between(x, agg["cosine_mean"] - agg["cosine_std"],
                                agg["cosine_mean"] + agg["cosine_std"], alpha=0.15)

            # L2 distance
            ax_l2.plot(x, agg["l2_mean"], linewidth=2, linestyle=linestyle, label=embedder_label, alpha=0.9)
            ax_l2.fill_between(x, agg["l2_mean"] - agg["l2_std"],
                               agg["l2_mean"] + agg["l2_std"], alpha=0.15)

        masking_title = masking_type.capitalize()

        ax_cos.set_xlabel("Masking Rate p (%)", fontsize=11)
        ax_cos.set_ylabel("Cosine Similarity", fontsize=11)
        ax_cos.set_title(f"{masking_title} Masking — Cosine Similarity", fontsize=12, fontweight="bold")
        ax_cos.set_xlim(-2, 108)
        ax_cos.legend(fontsize=9)
        ax_cos.grid(True, alpha=0.3)
        ax_cos.set_xticks(np.arange(0, 101, 10))

        ax_l2.set_xlabel("Masking Rate p (%)", fontsize=11)
        ax_l2.set_ylabel("L2 (Euclidean) Distance", fontsize=11)
        ax_l2.set_title(f"{masking_title} Masking — L2 Distance", fontsize=12, fontweight="bold")
        ax_l2.set_xlim(-2, 108)
        ax_l2.legend(fontsize=9)
        ax_l2.grid(True, alpha=0.3)
        ax_l2.set_xticks(np.arange(0, 101, 10))

    plt.tight_layout()
    _save_plot("x_masking_baseline_comparison")
    plt.close()

def _save_plot(name: str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = REPORTS_DIR / f"{name}.png"
    pdf_path = REPORTS_DIR / f"{name}.pdf"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Plot saved to: {png_path}")
    print(f"PDF saved to:  {pdf_path}")

if __name__ == "__main__":
    plot_masking_results()
    plot_masking_results_with_l2()
    plot_baseline_comparison()
