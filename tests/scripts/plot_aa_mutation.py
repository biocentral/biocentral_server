#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Amino acids ordered by physicochemical property groups for nicer heatmaps
AA_ORDER = list("GAVLIMFWPSTCYNQHKRDE")
# Hydrophobic: G A V L I M F W
# Special:     P
# Polar:       S T C Y N Q
# Charged:     H K R D E

def _load_mutation_data(filename: str = "aa_mutation_sensitivity_esm2.csv") -> pd.DataFrame | None:
    path = REPORTS_DIR / filename
    if not path.exists():
        print(f"File not found: {path}")
        return None
    return pd.read_csv(path)

def plot_heatmap():
    df = _load_mutation_data()
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, metric, label in [
        (axes[0], "cosine_distance", "Cosine Distance"),
        (axes[1], "l2_distance", "L2 Distance"),
    ]:
        # Pivot: rows = replacement_aa, cols = masking_ratio
        pivot = df.pivot_table(
            values=metric,
            index="replacement_aa",
            columns="masking_ratio",
            aggfunc="mean",
        )

        # Reorder rows
        ordered_aas = [aa for aa in AA_ORDER if aa in pivot.index]
        pivot = pivot.reindex(ordered_aas)

        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        plt.colorbar(im, ax=ax, label=label)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{int(c * 100)}%" for c in pivot.columns], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10, fontfamily="monospace")

        ax.set_xlabel("Mutation Rate", fontsize=11)
        ax.set_ylabel("Replacement Amino Acid", fontsize=11)
        ax.set_title(f"Mutation Sensitivity — {label}", fontsize=12, fontweight="bold")

    plt.suptitle("AA Mutation Sensitivity — ESM2-T6-8M", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_plot("aa_mutation_heatmap")
    plt.close()

def plot_line_overlay():
    df_mut = _load_mutation_data()
    if df_mut is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel in [
        (axes[0], "cosine_distance", "Cosine Distance"),
        (axes[1], "l2_distance", "L2 Distance"),
    ]:
        cmap = matplotlib.colormaps.get_cmap("tab20")

        for i, aa in enumerate(AA_ORDER):
            aa_data = df_mut[df_mut["replacement_aa"] == aa]
            if aa_data.empty:
                continue
            agg = aa_data.groupby("masking_ratio")[metric].mean().reset_index()
            ax.plot(
                agg["masking_ratio"] * 100, agg[metric],
                linewidth=1.2, alpha=0.7, color=cmap(i / 20),
                label=aa,
            )

        # X-masking reference (if available)
        x_path = REPORTS_DIR / "x_masking_progressive_esm2.csv"
        if x_path.exists():
            df_x = pd.read_csv(x_path)
            df_x["masking_ratio_parsed"] = (
                df_x["parameter"].str.extract(r"mask(\d+)%").astype(float) / 100
            )
            agg_x = df_x.groupby("masking_ratio_parsed")[metric].mean().reset_index()
            ax.plot(
                agg_x["masking_ratio_parsed"] * 100, agg_x[metric],
                linewidth=3, color="black", linestyle="--", label="X-masking",
                alpha=0.9,
            )

        ax.set_xlabel("Mutation Rate (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"AA Mutation Curves — {ylabel}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(np.arange(0, 101, 10))

    plt.tight_layout()
    _save_plot("aa_mutation_line_overlay")
    plt.close()

def plot_bar_ranking():
    df = _load_mutation_data()
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, label in [
        (axes[0], "cosine_distance", "Cosine Distance"),
        (axes[1], "l2_distance", "L2 Distance"),
    ]:
        # Filter to ~10% mutation rate
        subset = df[df["masking_ratio"].between(0.09, 0.11)]
        if subset.empty:
            subset = df[df["masking_ratio"].between(0.04, 0.16)]

        ranking = (
            subset.groupby("replacement_aa")[metric]
            .mean()
            .sort_values(ascending=False)
        )

        colors = []
        for aa in ranking.index:
            if aa in "GAVLIMFW":
                colors.append("#E94F37")  # Hydrophobic
            elif aa in "STCYNQ":
                colors.append("#4CAF50")  # Polar
            elif aa in "HKRDE":
                colors.append("#2E86AB")  # Charged
            else:
                colors.append("#9C27B0")  # Special (P)

        ax.bar(range(len(ranking)), ranking.values, color=colors, alpha=0.85)
        ax.set_xticks(range(len(ranking)))
        ax.set_xticklabels(ranking.index, fontsize=11, fontfamily="monospace")
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"Embedding Change at ~10% Mutation — {label}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Legend for AA types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#E94F37", label="Hydrophobic (GAVLIMFW)"),
            Patch(facecolor="#4CAF50", label="Polar (STCYNQ)"),
            Patch(facecolor="#2E86AB", label="Charged (HKRDE)"),
            Patch(facecolor="#9C27B0", label="Special (P)"),
        ]
        ax.legend(handles=legend_elements, fontsize=8)

    plt.tight_layout()
    _save_plot("aa_mutation_bar_ranking")
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
    plot_heatmap()
    plot_line_overlay()
    plot_bar_ranking()
