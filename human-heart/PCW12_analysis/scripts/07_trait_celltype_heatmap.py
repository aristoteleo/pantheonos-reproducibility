"""
Phase 7: Hierarchically clustered heatmap of disease-trait scores
across MERFISH-defined cell types.

Pipeline (matches 05b's trait scoring so the heatmap is consistent
with the per-trait 3D renders):

  1. Load imputed disease-gene AnnData (100k MERFISH cells × 221 genes).
  2. Z-score each gene across all cells.
  3. For each trait T (multi-membership via ';' in disease_genes_partition.tsv),
     score(cell) = mean of z-scored expression over genes mapped to T.
  4. Aggregate per MERFISH `celltype` -> matrix M (cell-type x trait).
  5. Transpose so rows = traits, cols = cell types -> M_T (trait x cell-type).
  6. Column-wise z-score on M_T (each cell-type normalised across traits) -> Mz_T.
  7. Hierarchically cluster both axes (correlation distance, average linkage)
     and render a sns.clustermap PNG. Also save raw + z-scored TSVs.

Outputs (all in figures/heatmaps/):
  trait_celltype_mean_raw.tsv         # mean z-score-of-gene per cell type (cell-type x trait)
  trait_celltype_mean_colZ.tsv        # transposed + column-wise z-scored matrix actually plotted
                                      # (trait x cell-type)
  trait_celltype_clustermap.png       # main figure (rows=traits, cols=cell types)
  trait_celltype_clustermap.pdf       # vector copy
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
IMP  = ROOT / "PCW12_analysis/data/adata_imputed_disease_genes.h5ad"
PART = ROOT / "PCW12_analysis/data/disease_genes_partition.tsv"
OUT  = ROOT / "PCW12_analysis/figures/heatmaps"
OUT.mkdir(parents=True, exist_ok=True)

MIN_GENES = 3   # skip traits with < this many imputed genes
CT_KEY    = "celltype"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    log(f"Loading {IMP.name}")
    adata = ad.read_h5ad(IMP)
    genes = adata.var_names.tolist()
    log(f"  {adata.shape}; {len(genes)} imputed genes; "
        f"{adata.obs[CT_KEY].nunique()} cell types")

    part = pd.read_csv(PART, sep="\t")
    part = part[part["Gene"].isin(genes)].copy()

    gene_traits = {
        r["Gene"]: set(str(r["traits"]).split(";"))
        for _, r in part.iterrows()
        if pd.notna(r["traits"]) and str(r["traits"]).strip()
    }
    all_traits = sorted({t for s in gene_traits.values() for t in s if t})
    log(f"Traits ({len(all_traits)}): {all_traits}")

    # --- Z-score each gene across cells -------------------------------------
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float32)
    Xz = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-9)

    # --- Per-cell mean z-score per trait ------------------------------------
    cell_scores = {}
    n_genes_map: dict[str, int] = {}
    for tr in all_traits:
        idx = [i for i, g in enumerate(genes) if tr in gene_traits.get(g, set())]
        n_genes_map[tr] = len(idx)
        if len(idx) < MIN_GENES:
            log(f"  skip {tr}: only {len(idx)} genes (<{MIN_GENES})")
            continue
        cell_scores[tr] = Xz[:, idx].mean(axis=1)

    traits_kept = list(cell_scores.keys())
    log(f"Kept {len(traits_kept)}/{len(all_traits)} traits with >= {MIN_GENES} genes")

    # --- Aggregate to cell-type means ---------------------------------------
    ct = adata.obs[CT_KEY].astype(str).values
    score_df = pd.DataFrame(cell_scores, index=adata.obs_names)
    score_df["_ct"] = ct
    M = score_df.groupby("_ct").mean()                 # cell-type x trait
    M.index.name = "celltype"
    log(f"Cell-type x trait matrix: {M.shape}")

    # Transpose so rows = traits, columns = cell types
    M_T = M.T
    M_T.index.name = "trait"

    # Column-wise z-score on the transposed matrix
    # (each CELL-TYPE column normalised across traits)
    Mz_T = (M_T - M_T.mean(axis=0)) / (M_T.std(axis=0) + 1e-9)

    # Save matrices (raw kept in original orientation for reference)
    M.to_csv(OUT / "trait_celltype_mean_raw.tsv",  sep="\t", float_format="%.4f")
    Mz_T.to_csv(OUT / "trait_celltype_mean_colZ.tsv", sep="\t", float_format="%.4f")
    log(f"  wrote {OUT/'trait_celltype_mean_raw.tsv'}")
    log(f"  wrote {OUT/'trait_celltype_mean_colZ.tsv'}")

    # --- Annotate trait rows with gene count --------------------------------
    row_labels = {t: f"{t.replace('_', ' ')}  (n={n_genes_map[t]})"
                  for t in Mz_T.index}
    Mz_plot = Mz_T.rename(index=row_labels)

    # --- Clustermap ---------------------------------------------------------
    sns.set_theme(style="white")
    # rows = traits (~10), cols = cell types (~34): wide-and-short aspect.
    fig_w = max(12, 0.50 * Mz_plot.shape[1] + 4)
    fig_h = max(6,  0.55 * Mz_plot.shape[0] + 3)

    g = sns.clustermap(
        Mz_plot,
        cmap="RdBu_r",
        center=0,
        vmin=-2.5, vmax=2.5,
        method="average",
        metric="correlation",
        figsize=(fig_w, fig_h),
        linewidths=0.0,
        cbar_kws={"label": "Mean trait score (per-cell-type z-score)"},
        dendrogram_ratio=(0.18, 0.10),
        cbar_pos=(1.02, 0.35, 0.015, 0.30),
        xticklabels=True,
        yticklabels=True,
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    plt.setp(g.ax_heatmap.get_xticklabels(),
             rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=10)
    g.fig.suptitle(
        "Disease-trait expression score across MERFISH cell types\n"
        "(per-cell mean z-score of trait genes, then mean per cell type, "
        "then column-wise z-score)",
        fontsize=13, y=1.02,
    )

    png = OUT / "trait_celltype_clustermap.png"
    pdf = OUT / "trait_celltype_clustermap.pdf"
    g.fig.savefig(png, dpi=180, bbox_inches="tight")
    g.fig.savefig(pdf,             bbox_inches="tight")
    plt.close(g.fig)
    log(f"  wrote {png}")
    log(f"  wrote {pdf}")
    log("Phase 7 complete.")


if __name__ == "__main__":
    main()
