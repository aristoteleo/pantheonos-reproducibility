"""
Phase 4: Visualize imputed disease genes on the MERFISH 3D coordinates.

Input:
  PCW12_analysis/data/adata_imputed_disease_genes.h5ad  (100k × 221, MOSCOT-imputed)

Output:
  figures/imputed/<gene>.png        for all 221 genes
  figures/imputed/<gene>.gif        for highlight subset
  figures/imputed/_mapped_celltype.png
  figures/imputed/_mapping_confidence.png
  figures/heatmaps/imputed_celltype_heatmap.png
  figures/heatmaps/imputed_disease_category_heatmap.png
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
sys.path.insert(0, str(ROOT / "PCW12_analysis/scripts"))
from _viz_utils import render_gene_3d, render_celltype_3d  # noqa: E402

IMP_PATH = ROOT / "PCW12_analysis/data/adata_imputed_disease_genes.h5ad"
PARTITION = ROOT / "PCW12_analysis/data/disease_genes_partition.tsv"
HIGHLIGHTS = ROOT / "PCW12_analysis/data/highlight_genes.tsv"
OUT_DIR = ROOT / "PCW12_analysis/figures/imputed"
OUT_HEAT_DIR = ROOT / "PCW12_analysis/figures/heatmaps"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_HEAT_DIR.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log(f"Loading imputed h5ad: {IMP_PATH}")
    adata = ad.read_h5ad(IMP_PATH)
    log(f"  shape={adata.shape}, obsm keys={list(adata.obsm.keys())}")
    log(f"  X range: {float(adata.X.min()):.2f}..{float(adata.X.max()):.2f}")

    part = pd.read_csv(PARTITION, sep="\t")
    impute_genes = adata.var_names.tolist()  # exactly the imputed set (221)

    highlight = set(pd.read_csv(HIGHLIGHTS, sep="\t")["Gene"].tolist())
    impute_highlights = [g for g in impute_genes if g in highlight]
    log(f"Impute genes: {len(impute_genes)}; highlights: {len(impute_highlights)}")

    # ---- Per-gene 3D figures (PNGs always; GIFs for highlights) ----
    fail = []
    for i, g in enumerate(impute_genes, 1):
        out_png = OUT_DIR / f"{g}.png"
        gif = OUT_DIR / f"{g}.gif" if g in highlight else None
        if i % 20 == 1 or gif is not None:
            log(f"[{i}/{len(impute_genes)}] {g}{' + gif' if gif else ''}")
        try:
            render_gene_3d(adata, g, out_png, gif_path=gif,
                           title_suffix="(imputed)")
        except Exception as e:
            log(f"   FAILED {g}: {e}")
            fail.append((g, str(e)))
    if fail:
        log(f"FAILED renders: {fail}")

    # ---- Mapped cell type & confidence overviews ----
    if "mapped_celltype" in adata.obs:
        log("Rendering mapped_celltype overview ...")
        try:
            render_celltype_3d(adata, OUT_DIR / "_mapped_celltype.png",
                               obs_key="mapped_celltype",
                               title="MOSCOT-mapped scRNA cell types (3D)")
        except Exception as e:
            log(f"  mapped_celltype overview FAILED: {e}")

    if "mapping_confidence" in adata.obs:
        log("Rendering mapping_confidence ...")
        conf = adata.obs["mapping_confidence"].astype(float).values
        # Use the gene-3D plotter with `expression=` override
        try:
            render_gene_3d(adata, gene=adata.var_names[0],
                           out_png=OUT_DIR / "_mapping_confidence.png",
                           expression=conf, cmap="viridis",
                           clim_quantile=(0.02, 0.98),
                           title_suffix="(MOSCOT confidence)")
        except Exception as e:
            log(f"  mapping_confidence FAILED: {e}")

    # ---- Heatmap: MERFISH celltype × imputed disease genes (top 60 by variance) ----
    log("Building celltype heatmap (top-60 most-varying imputed genes) ...")
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    df_full = pd.DataFrame(X, columns=impute_genes,
                           index=adata.obs["celltype"].astype(str).values)
    var_per_gene = df_full.var(axis=0).sort_values(ascending=False)
    top60 = var_per_gene.head(60).index.tolist()

    mean_df = df_full[top60].groupby(level=0).mean()
    mean_df.to_csv(OUT_HEAT_DIR / "imputed_top60_celltype_mean.tsv", sep="\t")

    z = (mean_df - mean_df.mean(axis=0)) / (mean_df.std(axis=0) + 1e-9)
    row_order = z.max(axis=1).sort_values(ascending=False).index
    col_order = z.idxmax(axis=0).map(lambda x: list(row_order).index(x)).sort_values().index
    z = z.loc[row_order, col_order]
    fig, ax = plt.subplots(figsize=(max(10, 0.30 * len(col_order)),
                                    max(7, 0.30 * len(row_order))))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=90, fontsize=8)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=9)
    ax.set_title("Imputed disease genes (top-60 by variance) — z-score per MERFISH cell type")
    fig.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(OUT_HEAT_DIR / "imputed_celltype_heatmap.png", dpi=200)
    plt.close(fig)

    # ---- Disease category heatmap ----
    log("Building disease-category × cell-type heatmap ...")
    # gene -> primary categories (a gene may belong to multiple)
    gene_cat = {}
    for _, r in part[part["Gene"].isin(impute_genes)].iterrows():
        gene_cat[r["Gene"]] = r["categories"]  # already ;-joined

    # Build gene-mean per (cell type, category): mean over genes belonging to category
    cats_all = set()
    for v in gene_cat.values():
        cats_all.update(v.split(";"))
    cats_all = sorted(cats_all)

    # Per cell type: average expression across genes in each category
    ct_idx = adata.obs["celltype"].astype(str).values
    cat_mat = np.zeros((mean_df.index.nunique() if False else len(set(ct_idx)),
                        len(cats_all)), dtype=np.float32)
    cell_types = sorted(set(ct_idx))
    full_mean = pd.DataFrame(X, columns=impute_genes, index=ct_idx) \
                  .groupby(level=0).mean()  # full set for category summarization
    for j, cat in enumerate(cats_all):
        genes_in_cat = [g for g, c in gene_cat.items() if cat in c.split(";")]
        if not genes_in_cat:
            continue
        sub = full_mean[genes_in_cat]
        cat_mat[:, j] = sub.reindex(cell_types).mean(axis=1).values
    cat_df = pd.DataFrame(cat_mat, index=cell_types, columns=cats_all)
    cat_df.to_csv(OUT_HEAT_DIR / "imputed_disease_category_mean.tsv", sep="\t")

    z2 = (cat_df - cat_df.mean(axis=0)) / (cat_df.std(axis=0) + 1e-9)
    row_order2 = z2.max(axis=1).sort_values(ascending=False).index
    z2 = z2.loc[row_order2]
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(cats_all)),
                                    max(7, 0.30 * len(row_order2))))
    im = ax.imshow(z2.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(cats_all)))
    ax.set_xticklabels([c.replace("_", " ") for c in cats_all],
                       rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(row_order2)))
    ax.set_yticklabels(row_order2, fontsize=9)
    ax.set_title("Mean imputed expression per disease category — z-score across cell types")
    fig.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(OUT_HEAT_DIR / "imputed_disease_category_heatmap.png", dpi=200)
    plt.close(fig)

    log("Phase 4 complete.")


if __name__ == "__main__":
    main()
