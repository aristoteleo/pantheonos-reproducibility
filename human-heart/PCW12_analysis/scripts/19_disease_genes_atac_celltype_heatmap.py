"""
Phase 19: Heatmap of representative heart-disease genes' chromatin
accessibility (snATAC, MOSCOT-imputed onto the PCW12 MERFISH cells)
across fetal heart cell types.

This is the ATAC analogue of `08_disease_genes_celltype_heatmap.py`.

Differences from the RNA version:
  - All values are MOSCOT-imputed snATAC gene-aggregated signal
    (no "direct" measurements -> no Source annotation column).
  - **Column-wise z-score** (per cell type, across genes) — as requested
    by the user.  This puts cell types on equal footing for comparing
    *which* disease-gene loci are most accessible *within* each cell
    type (raw chromatin levels vary substantially across genes due to
    locus length and baseline openness, so column-wise normalisation
    is the more informative view for ATAC).

Pipeline:
  1. Load `adata_imputed_atac.h5ad` (100 000 MERFISH cells x 234 genes,
     log-norm gene-aggregated CHD enhancer signal).
  2. Restrict to the same curated 6-group disease panel used for RNA.
  3. Mean signal per `celltype` -> matrix M (gene x celltype).
  4. **Column z-score** across genes -> highlights gene-level structure
     within each cell type.
  5. Hierarchically cluster both axes; annotate each gene with its
     functional disease group.

Outputs (figures/heatmaps/):
  disease_genes_atac_celltype_mean_raw.tsv   # gene x celltype, log-norm mean
  disease_genes_atac_celltype_mean_colZ.tsv  # gene x celltype, column z-scored
  disease_genes_atac_celltype_clustermap.png
  disease_genes_atac_celltype_clustermap.pdf
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

# Same curated functional groups as the RNA heatmap (Phase-1 HIGHLIGHTS).
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
    "Conduction / channel": ["SCN5A", "GJA1"],
    "Notch / valve": ["NOTCH1", "JAG1", "DLL4"],
    "Aortic / smooth muscle": ["MYH11", "ACTA2", "FBN1", "ELN"],
    "Chromatin (PCGC)": ["CHD7", "KMT2D", "CHD4"],
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
    panel: list[str] = []
    gene_to_group: dict[str, str] = {}
    for grp, glist in GENE_GROUPS.items():
        for g in glist:
            if g not in var_set:
                log(f"  WARN: {g} absent from imputed ATAC adata, skipping")
                continue
            panel.append(g)
            gene_to_group[g] = grp
    log(f"Panel: {len(panel)} genes across {len(GENE_GROUPS)} groups")

    # Pull expression matrix for the panel (cells x genes)
    X = _to_dense(a[:, panel].X)
    df_cells = pd.DataFrame(X, columns=panel)
    df_cells["_ct"] = a.obs[CT_KEY].astype(str).values
    M = df_cells.groupby("_ct").mean().T              # rows=genes, cols=celltypes
    M.index.name = "gene"
    M.columns.name = "celltype"
    log(f"Gene x cell-type mean: {M.shape}")

    # Column-wise z-score (per cell type, across genes).  Highlights which
    # disease-gene loci are most accessible *within* each cell type.
    Mz = (M.sub(M.mean(axis=0), axis=1)
            .div(M.std(axis=0) + 1e-9, axis=1))

    M.to_csv(OUT / "disease_genes_atac_celltype_mean_raw.tsv",
             sep="\t", float_format="%.4f")
    Mz.to_csv(OUT / "disease_genes_atac_celltype_mean_colZ.tsv",
              sep="\t", float_format="%.4f")
    log(f"  wrote {OUT/'disease_genes_atac_celltype_mean_raw.tsv'}")
    log(f"  wrote {OUT/'disease_genes_atac_celltype_mean_colZ.tsv'}")

    # Row colors: gene group only (no Source split for ATAC)
    row_group = pd.Series(
        {g: GROUP_COLORS[gene_to_group[g]] for g in M.index}, name="Group"
    )

    sns.set_theme(style="white")
    fig_w = max(12, 0.45 * Mz.shape[1] + 5)
    fig_h = max(9,  0.32 * Mz.shape[0] + 3)

    g = sns.clustermap(
        Mz,
        cmap="RdBu_r",
        center=0,
        vmin=-2.5, vmax=2.5,
        method="average",
        metric="correlation",
        figsize=(fig_w, fig_h),
        linewidths=0.0,
        row_colors=row_group.to_frame(),
        cbar_kws={"label": "Mean ATAC signal (per cell-type z-score across genes)"},
        dendrogram_ratio=(0.10, 0.12),
        colors_ratio=(0.018, 0.018),
        cbar_pos=(1.02, 0.35, 0.015, 0.30),
        xticklabels=True,
        yticklabels=True,
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    plt.setp(g.ax_heatmap.get_xticklabels(),
             rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=10, style="italic")
    g.fig.suptitle(
        "Representative heart-disease genes — chromatin accessibility "
        "across fetal-heart cell types (PCW12)\n"
        "MOSCOT-imputed snATAC, mean per cell type, column-wise z-score",
        fontsize=13, y=1.01,
    )

    group_handles = [mpatches.Patch(color=c, label=k) for k, c in GROUP_COLORS.items()]
    g.fig.legend(handles=group_handles, title="Disease group",
                 loc="upper left", bbox_to_anchor=(1.02, 0.30),
                 frameon=False, fontsize=9, title_fontsize=10)

    png = OUT / "disease_genes_atac_celltype_clustermap.png"
    pdf = OUT / "disease_genes_atac_celltype_clustermap.pdf"
    g.fig.savefig(png, dpi=180, bbox_inches="tight")
    g.fig.savefig(pdf,             bbox_inches="tight")
    plt.close(g.fig)
    log(f"  wrote {png}")
    log(f"  wrote {pdf}")
    log("Phase 19 complete.")


if __name__ == "__main__":
    main()
