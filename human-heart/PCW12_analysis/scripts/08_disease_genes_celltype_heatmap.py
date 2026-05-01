"""
Phase 8: Heatmap of representative heart-disease genes across fetal heart
cell types (PCW12 MERFISH, with MOSCOT-imputed values for genes off-panel).

Pipeline:
  1. Load the unified imputed AnnData (100k MERFISH cells x 221 genes).
     - For genes already on the MERFISH panel, expression is the direct
       measurement (log1p-normalised).
     - For off-panel disease genes, expression is the MOSCOT-imputed value.
  2. Restrict to a curated panel of ~37 representative cardiac-disease
     genes (Phase 1 highlight set).
  3. Compute mean expression per `celltype` -> matrix M (gene x celltype).
  4. Row-wise z-score across cell types -> highlights cell-type specificity.
  5. Hierarchically cluster both axes; annotate each gene with its disease
     group (Cardiac TF / Sarcomere / Conduction / Notch-Valve / Aortic-SMC
     / Chromatin) and its data source (direct vs imputed).

Orientation (per user feedback):
  x-axis = cell types, y-axis = genes.

Outputs (figures/heatmaps/):
  disease_genes_celltype_mean_raw.tsv    # gene x celltype, log-normalised mean
  disease_genes_celltype_mean_rowZ.tsv   # gene x celltype, row z-scored matrix actually plotted
  disease_genes_celltype_clustermap.png
  disease_genes_celltype_clustermap.pdf
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
IMP  = ROOT / "PCW12_analysis/data/adata_imputed_disease_genes.h5ad"
SP   = ROOT / "data/full_heart_final_aug2025_update_downsampled_100k.h5ad"
HI   = ROOT / "PCW12_analysis/data/highlight_genes.tsv"
OUT  = ROOT / "PCW12_analysis/figures/heatmaps"
OUT.mkdir(parents=True, exist_ok=True)

CT_KEY = "celltype"

# Curated functional groups (matches Phase-1 HIGHLIGHTS organisation)
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

# Distinct, color-blind-tolerant palette for the 6 groups
GROUP_COLORS = {
    "Cardiac TF (CHD)":          "#E64B35",  # red
    "Sarcomere / CM structural": "#4DBBD5",  # teal-blue
    "Conduction / channel":      "#00A087",  # green
    "Notch / valve":             "#F39B7F",  # salmon
    "Aortic / smooth muscle":    "#8491B4",  # slate
    "Chromatin (PCGC)":          "#B09C85",  # taupe
}

SOURCE_COLORS = {
    "direct":  "#222222",   # MERFISH-measured
    "imputed": "#BBBBBB",   # MOSCOT-imputed
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _to_dense(x) -> np.ndarray:
    return (x.toarray() if hasattr(x, "toarray") else np.asarray(x)).astype(np.float32)


def main() -> None:
    log(f"Loading {IMP.name} (imputed off-panel disease genes)")
    imp = ad.read_h5ad(IMP)
    log(f"  imp: {imp.shape}")
    log(f"Loading {SP.name} (MERFISH on-panel measurements)")
    sp = ad.read_h5ad(SP)
    log(f"  sp:  {sp.shape}")

    # Both AnnDatas are on the same 100k cells in the same order
    # (Phase-3 imputation operated on these exact cells).
    assert imp.n_obs == sp.n_obs and (imp.obs_names == sp.obs_names).all(), \
        "Cell ordering mismatch between imputed and MERFISH adata"

    hi = pd.read_csv(HI, sep="\t")
    src_map = dict(zip(hi["Gene"], hi["class"]))   # direct / impute

    imp_vars = set(imp.var_names)
    sp_vars  = set(sp.var_names)

    # Flatten panel preserving group order; tag each gene with its source adata
    panel: list[str] = []
    panel_source: dict[str, str] = {}            # 'direct' or 'imputed'
    gene_to_group: dict[str, str] = {}
    for grp, glist in GENE_GROUPS.items():
        for g in glist:
            if g in sp_vars:
                src = "direct"
            elif g in imp_vars:
                src = "imputed"
            else:
                log(f"  WARN: {g} absent from both adatas, skipping")
                continue
            panel.append(g)
            panel_source[g] = src
            gene_to_group[g] = grp
    n_direct  = sum(s == "direct"  for s in panel_source.values())
    n_imputed = sum(s == "imputed" for s in panel_source.values())
    log(f"Panel: {len(panel)} genes ({n_direct} direct, {n_imputed} imputed) "
        f"across {len(GENE_GROUPS)} groups")

    # Pull each gene's expression vector from its source adata (100k cells)
    direct_genes  = [g for g in panel if panel_source[g] == "direct"]
    imputed_genes = [g for g in panel if panel_source[g] == "imputed"]

    cols = {}
    if direct_genes:
        Xs = _to_dense(sp[:, direct_genes].X)
        for j, g in enumerate(direct_genes):
            cols[g] = Xs[:, j]
    if imputed_genes:
        Xi = _to_dense(imp[:, imputed_genes].X)
        for j, g in enumerate(imputed_genes):
            cols[g] = Xi[:, j]

    df_cells = pd.DataFrame({g: cols[g] for g in panel})  # preserve panel order
    df_cells["_ct"] = sp.obs[CT_KEY].astype(str).values
    M = df_cells.groupby("_ct").mean().T              # rows=genes, cols=celltypes
    M.index.name = "gene"
    M.columns.name = "celltype"
    log(f"Gene x cell-type mean: {M.shape}")

    # Row z-score (per gene, across cell types) -> shows specificity
    Mz = (M.sub(M.mean(axis=1), axis=0)
            .div(M.std(axis=1) + 1e-9, axis=0))

    # Persist matrices
    M.to_csv(OUT / "disease_genes_celltype_mean_raw.tsv",  sep="\t", float_format="%.4f")
    Mz.to_csv(OUT / "disease_genes_celltype_mean_rowZ.tsv", sep="\t", float_format="%.4f")
    log(f"  wrote {OUT/'disease_genes_celltype_mean_raw.tsv'}")
    log(f"  wrote {OUT/'disease_genes_celltype_mean_rowZ.tsv'}")

    # Row colors: gene group + data source (2 columns of annotation)
    row_group  = pd.Series({g: GROUP_COLORS[gene_to_group[g]] for g in M.index},
                           name="Group")
    row_source = pd.Series({g: SOURCE_COLORS["direct" if src_map.get(g) == "direct" else "imputed"]
                            for g in M.index}, name="Source")
    row_colors = pd.concat([row_group, row_source], axis=1)

    # Clustermap (rows=genes, cols=celltypes)
    sns.set_theme(style="white")
    fig_w = max(12, 0.45 * Mz.shape[1] + 5)        # ~34 cols
    fig_h = max(9,  0.32 * Mz.shape[0] + 3)        # ~37 rows

    g = sns.clustermap(
        Mz,
        cmap="RdBu_r",
        center=0,
        vmin=-2.5, vmax=2.5,
        method="average",
        metric="correlation",
        figsize=(fig_w, fig_h),
        linewidths=0.0,
        row_colors=row_colors,
        cbar_kws={"label": "Mean expression (per-gene z-score across cell types)"},
        dendrogram_ratio=(0.10, 0.12),
        colors_ratio=(0.012, 0.018),
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
        "Representative heart-disease genes across fetal-heart cell types (PCW12)\n"
        "MERFISH-measured + MOSCOT-imputed, mean per cell type, row-wise z-score",
        fontsize=13, y=1.01,
    )

    # Two legends (group + source) below the colorbar
    group_handles = [mpatches.Patch(color=c, label=k) for k, c in GROUP_COLORS.items()]
    src_handles = [mpatches.Patch(color=SOURCE_COLORS["direct"],  label="Direct (MERFISH)"),
                   mpatches.Patch(color=SOURCE_COLORS["imputed"], label="Imputed (MOSCOT)")]
    leg1 = g.fig.legend(handles=group_handles, title="Disease group",
                        loc="upper left", bbox_to_anchor=(1.02, 0.30),
                        frameon=False, fontsize=9, title_fontsize=10)
    g.fig.add_artist(leg1)
    g.fig.legend(handles=src_handles, title="Source",
                 loc="upper left", bbox_to_anchor=(1.02, 0.05),
                 frameon=False, fontsize=9, title_fontsize=10)

    png = OUT / "disease_genes_celltype_clustermap.png"
    pdf = OUT / "disease_genes_celltype_clustermap.pdf"
    g.fig.savefig(png, dpi=180, bbox_inches="tight")
    g.fig.savefig(pdf,             bbox_inches="tight")
    plt.close(g.fig)
    log(f"  wrote {png}")
    log(f"  wrote {pdf}")
    log("Phase 8 complete.")


if __name__ == "__main__":
    main()
