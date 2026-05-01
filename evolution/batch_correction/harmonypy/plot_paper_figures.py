#!/usr/bin/env python
"""
Generate publication-quality PDF figures comparing evolved Harmony vs original on TMA data.

Uses results from results_tma/ which is based on the official harmonypy implementation.
"""

import os
# Enable CPU fallback for MPS (Apple Silicon) compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import importlib.util
import sys
from pathlib import Path
from umap import UMAP
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# Setup for publication-quality figures
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,  # TrueType fonts for editability
    'ps.fonttype': 42,
})

# Setup paths
example_dir = Path(__file__).parent
data_dir = example_dir.parent / "data"


def load_module(name: str, path: Path):
    """Load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_tma_data(split: str = "test"):
    """Load TMA data with real cell type labels.

    Args:
        split: Which split to load ("train", "val", or "test").
               Default is "test" for held-out evaluation.
    """
    df = pd.read_csv(data_dir / f"tma_8000_{split}.csv")
    X = df.iloc[:, :30].values  # PC1-PC30
    batch_labels = df["donor"].values
    cell_types = df["celltype"].values
    return X, batch_labels, cell_types


def compute_batch_mixing_score(X: np.ndarray, batch_labels: np.ndarray, k: int = 50) -> float:
    """Compute batch mixing score using k-nearest neighbors."""
    n_cells = X.shape[0]
    unique_batches = np.unique(batch_labels)
    expected_props = np.array([np.sum(batch_labels == b) / n_cells for b in unique_batches])

    nn = NearestNeighbors(n_neighbors=min(k + 1, n_cells), algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    mixing_scores = []
    for i in range(n_cells):
        neighbor_batches = batch_labels[indices[i, 1:]]
        observed_props = np.array([np.sum(neighbor_batches == b) / k for b in unique_batches])
        score = 1 - np.sqrt(np.mean((observed_props - expected_props) ** 2))
        mixing_scores.append(max(0, score))

    return np.mean(mixing_scores)


def compute_bio_conservation_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute biological structure conservation using silhouette score."""
    try:
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            return (silhouette + 1) / 2
        return 0.5
    except Exception:
        return 0.5


def generate_paper_figures():
    """Generate publication-quality PDF figures."""
    print("Loading TMA TEST data (held-out evaluation)...")
    X, batch_labels, cell_types = load_tma_data("test")
    print(f"  Data shape: {X.shape}")

    # Create metadata DataFrame for official harmonypy API
    meta_data = pd.DataFrame({'batch': batch_labels, 'celltype': cell_types})

    # Load harmony implementations
    print("\nLoading Harmony implementations...")
    harmony_original = load_module("harmony_original", example_dir / "harmony.py")
    harmony_evolved = load_module("harmony_evolved", example_dir / "results_tma" / "harmony_optimized.py")

    # Run algorithms (official harmonypy API)
    print("\nRunning original Harmony...")
    start_time = time.time()
    hm_original = harmony_original.run_harmony(X, meta_data, vars_use='batch', nclust=50, max_iter_harmony=10, random_state=42, verbose=False)
    time_original = time.time() - start_time
    X_corrected_original = hm_original.Z_corr
    print(f"  Time: {time_original:.2f}s")

    print("\nRunning Harmony (Evolved)...")
    start_time = time.time()
    hm_evolved = harmony_evolved.run_harmony(X, meta_data, vars_use='batch', nclust=50, max_iter_harmony=10, random_state=42, verbose=False)
    time_evolved = time.time() - start_time
    X_corrected_evolved = hm_evolved.Z_corr
    print(f"  Time: {time_evolved:.2f}s")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = {
        "Original Data": {
            "mixing": compute_batch_mixing_score(X, batch_labels),
            "bio": compute_bio_conservation_score(X, cell_types),
            "time": 0,
        },
        "Harmony": {
            "mixing": compute_batch_mixing_score(X_corrected_original, batch_labels),
            "bio": compute_bio_conservation_score(X_corrected_original, cell_types),
            "time": time_original,
        },
        "Harmony (Evolved)": {
            "mixing": compute_batch_mixing_score(X_corrected_evolved, batch_labels),
            "bio": compute_bio_conservation_score(X_corrected_evolved, cell_types),
            "time": time_evolved,
        },
    }

    for name, m in metrics.items():
        print(f"  {name}: mixing={m['mixing']:.4f}, bio={m['bio']:.4f}")

    # Compute UMAP
    print("\nComputing UMAP embeddings...")
    umap = UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    umap_original = umap.fit_transform(X)
    umap_harmony = umap.fit_transform(X_corrected_original)
    umap_evolved = umap.fit_transform(X_corrected_evolved)

    output_dir = example_dir / "results_tma" / "paper_figures"
    output_dir.mkdir(exist_ok=True)

    # Define colors
    batch_colors = {'10x': '#E64B35', 'SS2': '#4DBBD5'}  # Nature-style colors

    # Cell type colors - use a professional colormap
    unique_celltypes = np.unique(cell_types)
    n_types = len(unique_celltypes)
    cmap = plt.cm.get_cmap('tab20', n_types)
    celltype_colors = {ct: mpl.colors.rgb2hex(cmap(i)) for i, ct in enumerate(unique_celltypes)}

    # =========================================================================
    # Figure 1: UMAP Comparison (2x3 layout)
    # =========================================================================
    print("\nGenerating Figure 1: UMAP comparison...")
    fig1, axes = plt.subplots(2, 3, figsize=(7.5, 5))  # Single column: 3.5", double column: 7.5"

    datasets = [
        (umap_original, "Uncorrected", metrics["Original Data"]),
        (umap_harmony, "Harmony", metrics["Harmony"]),
        (umap_evolved, "Harmony (Evolved)", metrics["Harmony (Evolved)"]),
    ]

    # Row 1: Color by batch
    for idx, (emb, title, m) in enumerate(datasets):
        ax = axes[0, idx]
        for batch in ['10x', 'SS2']:
            mask = batch_labels == batch
            ax.scatter(emb[mask, 0], emb[mask, 1], c=batch_colors[batch],
                      label=batch, s=2, alpha=0.5, rasterized=True)
        ax.set_title(f"{title}", fontweight='bold')
        ax.set_xlabel("UMAP1")
        if idx == 0:
            ax.set_ylabel("UMAP2")
        ax.text(0.02, 0.98, f"Mixing: {m['mixing']:.3f}", transform=ax.transAxes,
               va='top', ha='left', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])

    # Add legend to first row
    axes[0, 2].legend(title="Platform", loc='upper right', markerscale=3, framealpha=0.9)

    # Row 2: Color by cell type
    for idx, (emb, title, m) in enumerate(datasets):
        ax = axes[1, idx]
        for ct in unique_celltypes:
            mask = cell_types == ct
            ax.scatter(emb[mask, 0], emb[mask, 1], c=celltype_colors[ct],
                      label=ct, s=2, alpha=0.5, rasterized=True)
        ax.set_xlabel("UMAP1")
        if idx == 0:
            ax.set_ylabel("UMAP2")
        ax.text(0.02, 0.98, f"Bio: {m['bio']:.3f}", transform=ax.transAxes,
               va='top', ha='left', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])

    # Add cell type legend
    handles, labels = axes[1, 2].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, title="Cell Type", loc='upper right',
                      markerscale=3, framealpha=0.9, fontsize=6, ncol=2)

    # Add row labels
    fig1.text(0.01, 0.75, 'Batch', va='center', ha='left', rotation=90, fontsize=10, fontweight='bold')
    fig1.text(0.01, 0.28, 'Cell Type', va='center', ha='left', rotation=90, fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0.02, 0, 1, 1])
    fig1.savefig(output_dir / "figure1_umap_comparison.pdf", format='pdf', bbox_inches='tight')
    fig1.savefig(output_dir / "figure1_umap_comparison.png", format='png', bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_dir}/figure1_umap_comparison.pdf")

    # =========================================================================
    # Figure 2: Performance Bar Chart
    # =========================================================================
    print("\nGenerating Figure 2: Performance comparison...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(7.5, 2.5))

    methods = ["Uncorrected", "Harmony", "Harmony\n(Evolved)"]
    colors = ["#999999", "#3C5488", "#00A087"]  # Nature-style

    # Mixing score
    ax = axes2[0]
    mixing_vals = [metrics["Original Data"]["mixing"], metrics["Harmony"]["mixing"], metrics["Harmony (Evolved)"]["mixing"]]
    bars = ax.bar(methods, mixing_vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Batch Mixing Score")
    ax.set_ylim(0, 1)
    ax.set_title("Batch Integration", fontweight='bold')
    for bar, val in zip(bars, mixing_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
               ha='center', va='bottom', fontsize=8)

    # Bio conservation
    ax = axes2[1]
    bio_vals = [metrics["Original Data"]["bio"], metrics["Harmony"]["bio"], metrics["Harmony (Evolved)"]["bio"]]
    bars = ax.bar(methods, bio_vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Bio Conservation Score")
    ax.set_ylim(0, 1)
    ax.set_title("Biological Structure", fontweight='bold')
    for bar, val in zip(bars, bio_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
               ha='center', va='bottom', fontsize=8)

    # Execution time
    ax = axes2[2]
    time_methods = ["Harmony", "Harmony\n(Evolved)"]
    time_vals = [metrics["Harmony"]["time"], metrics["Harmony (Evolved)"]["time"]]
    bars = ax.bar(time_methods, time_vals, color=["#3C5488", "#00A087"], edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Computational Cost", fontweight='bold')
    speedup = time_original / time_evolved if time_evolved > 0 else 1.0
    for bar, val in zip(bars, time_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}s',
               ha='center', va='bottom', fontsize=8)
    ax.text(0.5, 0.9, f"{speedup:.1f}x faster", transform=ax.transAxes,
           ha='center', fontsize=9, color='#00A087', fontweight='bold')

    plt.tight_layout()
    fig2.savefig(output_dir / "figure2_performance.pdf", format='pdf', bbox_inches='tight')
    fig2.savefig(output_dir / "figure2_performance.png", format='png', bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_dir}/figure2_performance.pdf")

    # =========================================================================
    # Figure 3: Combined Summary Figure (single panel)
    # =========================================================================
    print("\nGenerating Figure 3: Summary figure...")
    fig3, ax3 = plt.subplots(figsize=(4, 3))

    # Scatter plot: Mixing vs Bio Conservation
    for name, m, color, marker in [
        ("Uncorrected", metrics["Original Data"], "#999999", "o"),
        ("Harmony", metrics["Harmony"], "#3C5488", "s"),
        ("Harmony (Evolved)", metrics["Harmony (Evolved)"], "#00A087", "^"),
    ]:
        ax3.scatter(m["mixing"], m["bio"], c=color, s=150, marker=marker,
                   label=name, edgecolors='black', linewidth=0.5, zorder=3)

    ax3.set_xlabel("Batch Mixing Score")
    ax3.set_ylabel("Bio Conservation Score")
    ax3.set_xlim(0.5, 0.85)
    ax3.set_ylim(0.6, 0.75)
    ax3.legend(loc='lower right', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Add arrow showing improvement direction
    ax3.annotate('', xy=(0.76, 0.69), xytext=(0.55, 0.66),
                arrowprops=dict(arrowstyle='->', color='#00A087', lw=2))
    ax3.text(0.65, 0.64, 'Optimization\ndirection', fontsize=8, ha='center', color='#00A087')

    plt.tight_layout()
    fig3.savefig(output_dir / "figure3_summary.pdf", format='pdf', bbox_inches='tight')
    fig3.savefig(output_dir / "figure3_summary.png", format='png', bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_dir}/figure3_summary.pdf")

    # =========================================================================
    # Figure 4: Evolution Progress
    # =========================================================================
    print("\nGenerating Figure 4: Evolution progress...")
    import json

    # Prefer fixed scores (recomputed with corrected fitness logic)
    fixed_path = example_dir / "results" / "score_history_fixed.json"
    state_path = example_dir / "results" / "evolution_state.json"

    if fixed_path.exists():
        with open(fixed_path, "r") as f:
            fixed_data = json.load(f)
        score_history = fixed_data.get("score_history", [])
        best_score_history = fixed_data.get("best_score_history", [])
        print("  Using corrected scores from score_history_fixed.json")
    elif state_path.exists():
        with open(state_path, "r") as f:
            state = json.load(f)
        score_history = state.get("score_history", [])
        best_score_history = state.get("best_score_history", [])
        print("  Warning: Using raw scores from evolution_state.json (may contain buggy values)")
    else:
        score_history = []
        best_score_history = []

    if score_history and best_score_history:
        fig4, ax4 = plt.subplots(figsize=(6, 3))

        iterations = np.arange(len(score_history))

        # Per-iteration fitness as scatter dots
        ax4.scatter(iterations, score_history, s=6, alpha=0.35,
                   color='#3C5488', zorder=2, label='Per-iteration fitness',
                   edgecolors='none', rasterized=True)

        # Running max fitness as a solid line
        ax4.plot(iterations, best_score_history, color='#E64B35',
                linewidth=2, zorder=3, label='Best fitness')

        # Mark best iteration with vertical dashed line
        best_score = max(best_score_history)
        best_iter = next(i for i, v in enumerate(best_score_history) if v == best_score)
        ax4.axvline(x=best_iter, color='#E64B35', linestyle='--', linewidth=1, alpha=0.7, zorder=4)
        ax4.text(best_iter + 3, best_score + 0.002, f'Best (#{best_iter})',
                 fontsize=8, color='#E64B35', va='bottom')

        # Mark #272 with vertical dashed line
        ax4.axvline(x=272, color='#3C5488', linestyle='--', linewidth=1, alpha=0.7, zorder=4)
        ax4.text(272 + 3, best_score + 0.002, f'#272',
                 fontsize=8, color='#3C5488', va='bottom')

        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Fitness Score")
        ax4.set_title("Evolution Progress", fontweight='bold')
        ax4.legend(loc='lower right', framealpha=0.9)
        ax4.set_xlim(0, len(score_history) - 1)

        # Truncate y-axis from initial best to max best (with padding)
        y_min = best_score_history[0]
        y_max = max(best_score_history)
        y_pad = (y_max - y_min) * 0.05
        ax4.set_ylim(y_min - y_pad, y_max + y_pad)

        plt.tight_layout()
        fig4_dir = example_dir / "results" / "paper_figures"
        fig4_dir.mkdir(parents=True, exist_ok=True)
        fig4.savefig(fig4_dir / "figure4_evolution_progress.pdf",
                    format='pdf', bbox_inches='tight')
        fig4.savefig(fig4_dir / "figure4_evolution_progress.png",
                    format='png', bbox_inches='tight', dpi=300)
        print(f"  Saved: {fig4_dir}/figure4_evolution_progress.pdf")
    else:
        print("  Skipped: no score history data found")

    print(f"\n{'='*60}")
    print(f"All figures saved to: {output_dir}")
    print("Files generated:")
    print("  - figure1_umap_comparison.pdf")
    print("  - figure2_performance.pdf")
    print("  - figure3_summary.pdf")
    print("  - figure4_evolution_progress.pdf")
    print(f"{'='*60}")

    return metrics


if __name__ == "__main__":
    generate_paper_figures()
