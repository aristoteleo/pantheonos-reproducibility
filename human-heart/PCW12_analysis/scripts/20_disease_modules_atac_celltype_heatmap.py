"""
Phase 20: Module-level disease x cell-type heatmap (chromatin accessibility).

Aggregates the curated 6-group disease panel into per-cell *module activity
scores* (mean log-norm ATAC signal across module member genes), averages
per cell type, then clusters.  This is the module-level companion to
Phase-19's gene-level heatmap.

Why row-wise z-score for the main figure:
  Only 6 modules -> column-wise z over 6 values is statistically thin.
  Row-wise z (per module, across cell types) directly answers the more
  natural question for a module x celltype plot:  *which cell types
  have the highest accessibility for each disease module?*
  We also save the column-wise version for comparison.

Outputs (figures/heatmaps/):
  disease_modules_atac_celltype_mean_raw.tsv     # 6 x 34, log-norm mean
  disease_modules_atac_celltype_mean_rowZ.tsv    # 6 x 34, row z-scored (plotted)
  disease_modules_atac_celltype_mean_colZ.tsv    # 6 x 34, column z-scored
  disease_modules_atac_celltype_clustermap.png
  disease_modules_atac_celltype_clustermap.pdf
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
ATAC = ROOT / "PCW12_analysis/data/adata_imputed_atac.h5ad"
OUT  = ROOT / "PCW12_analysis/figures/heatmaps"
OUT.mkdir(parents=True, exist_ok=True)

CT_KEY = "celltype"

# Same 6-group curated panel as Phase 19 (RNA Phase 8).
GENE_GROUPS: dict[str, list[str]] = {
    "Cardiac TF (CHD)": [
        "GATA4", "GATA6", "NKX2-5", "TBX5", "TBX20", "TBX1",
        "HAND1", "HAND2", "MEIS2", "PITX2",
    ],
    "Sarcomere / CM structural": [
        "MYH6", "MYH7", "MYBPC3", "TNNT2", "TNNI3", "TNNC1",
        "ACTC1", "ACTN2", "TPM1", "TTN", "MYL2", "MYL3",
        "DES", "VCL", "PLN",
    ],
    "Conduction / channel":      ["SCN5A", "GJA1"],
    "Notch / valve":             ["NOTCH1", "JAG1", "DLL4"],
    "Aortic / smooth muscle":    ["MYH11", "ACTA2", "FBN1", "ELN"],
    "Chromatin (PCGC)":          ["CHD7", "KMT2D", "CHD4"],
}

GROUP_COLORS = {
    "Cardiac TF (CHD)":          "#E64B35",
    "Sarcomere / CM structural": "#4DBBD5",
    "Conduction / channel":      "#00A087",
    "Notch / valve":             "#F39B7F",
    "Aortic / smooth muscle":    "#8491B4",
    "Chromatin (PCGC)":          "#B09C85",
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _to_dense(x) -> np.ndarray:
    return (x.toarray() if hasattr(x, "toarray") else np.asarray(x)).astype(np.float32)


def main() -> None:
    log(f"Loading {ATAC.name}")
    a = ad.read_h5ad(ATAC)
    log(f"  atac: {a.shape}")

    var_set = set(a.var_names)

    # Resolve modules: drop genes absent from the imputed ATAC adata.
    resolved: dict[str, list[str]] = {}
    for grp, glist in GENE_GROUPS.items():
        kept = [g for g in glist if g in var_set]
        missing = [g for g in glist if g not in var_set]
        if missing:
            log(f"  {grp}: missing {missing}")
        resolved[grp] = kept
        log(f"  {grp}: {len(kept)}/{len(glist)} genes")

    cell_ct = a.obs[CT_KEY].astype(str).values

    # Compute per-cell module activity = mean of log-norm signal across
    # module member genes.  Then mean per cell type -> module x celltype.
    module_means: dict[str, pd.Series] = {}
    for grp, kept in resolved.items():
        if not kept:
            continue
        Xg = _to_dense(a[:, kept].X)               # cells x n_genes
        per_cell = Xg.mean(axis=1)                  # cells
        s = pd.Series(per_cell).groupby(cell_ct).mean()
        module_means[grp] = s

    M = pd.DataFrame(module_means).T                # rows=modules, cols=celltypes
    # Order rows by curated group order
    M = M.loc[[g for g in GENE_GROUPS if g in M.index]]
    M.index.name = "disease_module"
    M.columns.name = "celltype"
    log(f"Module x cell-type mean: {M.shape}")

    # Z-score variants
    Mz_row = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1) + 1e-9, axis=0)
    Mz_col = M.sub(M.mean(axis=0), axis=1).div(M.std(axis=0) + 1e-9, axis=1)

    M.to_csv(OUT / "disease_modules_atac_celltype_mean_raw.tsv",
             sep="\t", float_format="%.4f")
    Mz_row.to_csv(OUT / "disease_modules_atac_celltype_mean_rowZ.tsv",
                  sep="\t", float_format="%.4f")
    Mz_col.to_csv(OUT / "disease_modules_atac_celltype_mean_colZ.tsv",
                  sep="\t", float_format="%.4f")
    log("  wrote raw / rowZ / colZ TSVs")

    # Row colors for module identity
    row_group = pd.Series({m: GROUP_COLORS[m] for m in M.index}, name="Module")

    sns.set_theme(style="white")
    fig_w = max(13, 0.45 * Mz_row.shape[1] + 5)
    fig_h = max(5,  0.55 * Mz_row.shape[0] + 3)

    g = sns.clustermap(
        Mz_row,
        cmap="RdBu_r",
        center=0,
        vmin=-2.0, vmax=2.0,
        method="average",
        metric="correlation",
        figsize=(fig_w, fig_h),
        linewidths=0.0,
        row_cluster=True,
        col_cluster=True,
        row_colors=row_group.to_frame(),
        cbar_kws={"label": "Module mean ATAC (per-module z-score across cell types)"},
        dendrogram_ratio=(0.18, 0.20),
        colors_ratio=(0.025, 0.04),
        cbar_pos=(1.02, 0.32, 0.018, 0.30),
        xticklabels=True,
        yticklabels=True,
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    plt.setp(g.ax_heatmap.get_xticklabels(),
             rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=11)
    g.fig.suptitle(
        "Heart-disease modules — chromatin accessibility across "
        "fetal-heart cell types (PCW12)\n"
        "MOSCOT-imputed snATAC, per-cell module mean -> per cell-type mean, "
        "row-wise z-score",
        fontsize=13, y=1.04,
    )

    handles = [mpatches.Patch(color=c, label=k) for k, c in GROUP_COLORS.items()
               if k in M.index]
    g.fig.legend(handles=handles, title="Disease module",
                 loc="upper left", bbox_to_anchor=(1.02, 0.25),
                 frameon=False, fontsize=9, title_fontsize=10)

    png = OUT / "disease_modules_atac_celltype_clustermap.png"
    pdf = OUT / "disease_modules_atac_celltype_clustermap.pdf"
    g.fig.savefig(png, dpi=180, bbox_inches="tight")
    g.fig.savefig(pdf,             bbox_inches="tight")
    plt.close(g.fig)
    log(f"  wrote {png}")
    log(f"  wrote {pdf}")
    log("Phase 20 complete.")


if __name__ == "__main__":
    main()
