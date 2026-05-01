"""
Phase 9: Per-cell-type count of heart-disease genes whose mean expression
peaks in that cell type.

For each of the 241 heart-disease genes (20 measured directly on the
MERFISH panel + 221 MOSCOT-imputed off-panel; 6/247 are absent from both
adatas and are excluded), we compute the mean log1p expression in each of
the 34 PCW12 cell types and assign the gene to the cell type with the
highest mean. We then plot a barplot of `gene count per cell type`,
ordering cell types within their lineage and coloring bars by lineage.

Lineage groups (hand-curated from `celltype` labels):
  vCM, aCM, ncCM, Endothelial/Endocardial, Fibroblast, Mural, VIC,
  Immune, Neural.

Outputs (PCW12_analysis/figures/barplots/):
  disease_gene_peak_assignments.tsv      # gene, peak_celltype, lineage, source
  disease_gene_peak_celltype_counts.tsv  # celltype, lineage, n_genes
  disease_gene_peak_celltype_barplot.png
  disease_gene_peak_celltype_barplot.pdf
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

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
IMP  = ROOT / "PCW12_analysis/data/adata_imputed_disease_genes.h5ad"
SP   = ROOT / "data/full_heart_final_aug2025_update_downsampled_100k.h5ad"
PART = ROOT / "PCW12_analysis/data/disease_genes_partition.tsv"
OUT  = ROOT / "PCW12_analysis/figures/barplots"
OUT.mkdir(parents=True, exist_ok=True)

CT_KEY = "celltype"


# ---------------------------------------------------------------------------
# Lineage mapping for the 34 PCW12 cell types
# ---------------------------------------------------------------------------
LINEAGE_MAP: dict[str, str] = {
    # Ventricular CMs
    "vCM-IVS-His":            "vCM",
    "vCM-LV-AV":              "vCM",
    "vCM-LV-Compact I":       "vCM",
    "vCM-LV-Compact II":      "vCM",
    "vCM-LV-Hybrid":          "vCM",
    "vCM-LV-Trabecular I":    "vCM",
    "vCM-LV-Trabecular II":   "vCM",
    "vCM-LV/RV-Purkinje":     "vCM",
    "vCM-RV-AV":              "vCM",
    "vCM-RV-Compact":         "vCM",
    "vCM-RV-Hybrid":          "vCM",
    "vCM-RV-Proliferating":   "vCM",
    "vCM-RV-Trabecular":      "vCM",
    # Atrial CMs
    "aCM-LA":                 "aCM",
    "aCM-RA":                 "aCM",
    # Non-chamber CMs
    "ncCM-AVC-like":          "ncCM",
    "ncCM-IFT-like":          "ncCM",
    # Endothelial / endocardial
    "aEndocardial":           "Endothelial",
    "vEndocardial":           "Endothelial",
    "VEC":                    "Endothelial",
    "BEC":                    "Endothelial",
    "LEC":                    "Endothelial",
    # Fibroblast / epicardial-derived
    "Compact vFibro":         "Fibroblast",
    "Proliferating vFibro":   "Fibroblast",
    "Trabecular vFibro":      "Fibroblast",
    "aFibro":                 "Fibroblast",
    "adFibro":                "Fibroblast",
    "EPDC":                   "Fibroblast",
    "Epicardial":             "Fibroblast",
    # Mural
    "VSMC":                   "Mural",
    "Pericyte":               "Mural",
    # Valve interstitial
    "VIC":                    "VIC",
    # Immune
    "WBC":                    "Immune",
    # Neural
    "Neuronal":               "Neural",
}

# Ordered lineage list (left → right on the x axis)
LINEAGE_ORDER = [
    "vCM", "aCM", "ncCM",
    "Endothelial", "Fibroblast", "Mural",
    "VIC", "Immune", "Neural",
]

# Color palette — color-blind tolerant, distinct per lineage
LINEAGE_COLORS: dict[str, str] = {
    "vCM":         "#E64B35",   # red — ventricular CM
    "aCM":         "#F39B7F",   # salmon — atrial CM
    "ncCM":        "#B22222",   # firebrick — non-chamber CM
    "Endothelial": "#3C5488",   # navy
    "Fibroblast":  "#00A087",   # teal-green
    "Mural":       "#8491B4",   # slate
    "VIC":         "#7E6148",   # brown
    "Immune":      "#DC0000",   # bright red — distinct from vCM red
    "Neural":      "#B09C85",   # taupe
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _to_dense(x) -> np.ndarray:
    return (x.toarray() if hasattr(x, "toarray") else np.asarray(x)).astype(np.float32)


def main() -> None:
    log(f"Loading {IMP.name}")
    imp = ad.read_h5ad(IMP)
    log(f"  imp: {imp.shape}")
    log(f"Loading {SP.name}")
    sp = ad.read_h5ad(SP)
    log(f"  sp:  {sp.shape}")

    assert imp.n_obs == sp.n_obs and (imp.obs_names == sp.obs_names).all(), \
        "Cell ordering mismatch between imputed and MERFISH adata"

    part = pd.read_csv(PART, sep="\t")
    log(f"Disease-gene partition: {len(part)} genes")

    sp_vars  = set(sp.var_names)
    imp_vars = set(imp.var_names)

    # Resolve each disease gene to a source adata
    resolved: list[tuple[str, str]] = []           # (gene, 'direct'|'imputed')
    for g in part["Gene"]:
        if g in sp_vars:
            resolved.append((g, "direct"))
        elif g in imp_vars:
            resolved.append((g, "imputed"))
        # else: missing from both → skipped
    n_direct  = sum(s == "direct"  for _, s in resolved)
    n_imputed = sum(s == "imputed" for _, s in resolved)
    log(f"Resolved {len(resolved)} genes "
        f"({n_direct} direct, {n_imputed} imputed); "
        f"{len(part) - len(resolved)} missing from both.")
    assert len(resolved) == 241, f"Expected 241 genes, got {len(resolved)}"

    # Pull per-cell vectors
    direct_genes  = [g for g, s in resolved if s == "direct"]
    imputed_genes = [g for g, s in resolved if s == "imputed"]
    cols: dict[str, np.ndarray] = {}
    if direct_genes:
        Xs = _to_dense(sp[:, direct_genes].X)
        for j, g in enumerate(direct_genes):
            cols[g] = Xs[:, j]
    if imputed_genes:
        Xi = _to_dense(imp[:, imputed_genes].X)
        for j, g in enumerate(imputed_genes):
            cols[g] = Xi[:, j]

    panel = [g for g, _ in resolved]
    df_cells = pd.DataFrame({g: cols[g] for g in panel})
    df_cells["_ct"] = sp.obs[CT_KEY].astype(str).values
    M = df_cells.groupby("_ct").mean().T          # rows=genes, cols=celltypes
    M.index.name = "gene"
    M.columns.name = "celltype"
    log(f"Mean expression matrix: {M.shape} (genes × cell types)")

    # Per-gene peak cell type
    peak_ct = M.idxmax(axis=1)
    src_lookup = dict(resolved)
    peak_df = pd.DataFrame({
        "gene":          peak_ct.index,
        "peak_celltype": peak_ct.values,
        "lineage":       [LINEAGE_MAP[c] for c in peak_ct.values],
        "source":        [src_lookup[g] for g in peak_ct.index],
        "peak_mean":     [M.at[g, peak_ct[g]] for g in peak_ct.index],
    })
    peak_df.to_csv(OUT / "disease_gene_peak_assignments.tsv",
                   sep="\t", index=False, float_format="%.4f")
    log(f"  wrote {OUT/'disease_gene_peak_assignments.tsv'}")

    # Counts per cell type — include zeros for cell types nobody peaks at
    all_cts = list(M.columns)
    counts = peak_df["peak_celltype"].value_counts().reindex(all_cts, fill_value=0)
    counts_df = pd.DataFrame({
        "celltype": counts.index,
        "lineage":  [LINEAGE_MAP[c] for c in counts.index],
        "n_genes":  counts.values,
    })
    # Order: by lineage (LINEAGE_ORDER), then within lineage by count desc
    counts_df["_lin_rank"] = counts_df["lineage"].map(
        {l: i for i, l in enumerate(LINEAGE_ORDER)})
    counts_df = counts_df.sort_values(
        ["_lin_rank", "n_genes", "celltype"],
        ascending=[True, False, True]).drop(columns="_lin_rank")
    counts_df.to_csv(OUT / "disease_gene_peak_celltype_counts.tsv",
                     sep="\t", index=False)
    log(f"  wrote {OUT/'disease_gene_peak_celltype_counts.tsv'}")

    total = int(counts_df["n_genes"].sum())
    assert total == 241, f"Counts sum {total} != 241"
    log(f"  counts sum to {total} ✓")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    x = np.arange(len(counts_df))
    bar_colors = [LINEAGE_COLORS[l] for l in counts_df["lineage"]]
    bars = ax.bar(x, counts_df["n_genes"].values,
                  color=bar_colors, edgecolor="black", linewidth=0.4,
                  width=0.78)

    # Bar value labels (skip zeros)
    for xi, v in zip(x, counts_df["n_genes"].values):
        if v > 0:
            ax.text(xi, v + 0.4, str(int(v)),
                    ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(counts_df["celltype"].values,
                       rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=10)
    ax.set_ylabel("Number of disease genes peaking here", fontsize=11)
    ax.set_xlabel("")
    ax.set_title(
        "Cell-type assignment of heart-disease genes by peak expression\n"
        f"PCW12 fetal heart MERFISH — n = {total}/247 disease genes "
        f"({n_direct} direct + {n_imputed} imputed)",
        fontsize=12, pad=14,
    )
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7, zorder=0)
    ymax = max(counts_df["n_genes"].max() * 1.32, 6)   # headroom for top annot
    ax.set_ylim(0, ymax)

    # Lineage annotations ABOVE the bars (avoids collision with rotated x ticks)
    cur = 0
    spans: list[tuple[str, int, int]] = []
    for lin in LINEAGE_ORDER:
        n = int((counts_df["lineage"] == lin).sum())
        if n == 0:
            continue
        spans.append((lin, cur, cur + n - 1))
        cur += n
    y_band = 1.012       # axes-fraction y, just above the plot area
    band_h = 0.028
    trans = ax.get_xaxis_transform()         # x in data, y in axes-fraction
    # Colored band only — lineage names are conveyed by bar color + legend.
    # This avoids label collisions for narrow single-bar lineages
    # (VIC, Immune, Neural) and keeps the title area clean.
    for lin, s, e in spans:
        ax.add_patch(mpatches.Rectangle(
            (s - 0.5 + 0.05, y_band), (e - s + 1) - 0.10, band_h,
            transform=trans, clip_on=False,
            facecolor=LINEAGE_COLORS[lin], edgecolor="none", alpha=0.95))

    # Legend (lineage colors)
    handles = [mpatches.Patch(color=LINEAGE_COLORS[l], label=l)
               for l in LINEAGE_ORDER if (counts_df["lineage"] == l).any()]
    ax.legend(handles=handles, title="Lineage",
              loc="upper right", frameon=False,
              fontsize=9, title_fontsize=10, ncol=1)

    fig.tight_layout()
    # extra room for top lineage band + bottom rotated labels
    fig.subplots_adjust(bottom=0.22, top=0.84)

    png = OUT / "disease_gene_peak_celltype_barplot.png"
    pdf = OUT / "disease_gene_peak_celltype_barplot.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf,            bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {png}")
    log(f"  wrote {pdf}")
    log("Phase 9 complete.")


if __name__ == "__main__":
    main()
