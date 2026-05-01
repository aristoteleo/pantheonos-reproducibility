"""
Phase 5: Aggregated 3D expression patterns per heart-disease category.

For each disease category (CHD, CardiovascularDisease, PCGC_DeNovoVariants):
- Take all imputed genes that belong to the category (multi-membership ok).
- Z-score each gene across cells, then average z-scores per cell → category score.
- Render a 3D point cloud colored by the score (PNG + rotating GIF).

Also produces a 1×N panel summary PNG.

Inputs:
  PCW12_analysis/data/adata_imputed_disease_genes.h5ad
  PCW12_analysis/data/disease_genes_partition.tsv

Outputs:
  PCW12_analysis/figures/category_aggregated/<category>.png
  PCW12_analysis/figures/category_aggregated/<category>.gif
  PCW12_analysis/figures/category_aggregated/_summary_panel.png
  PCW12_analysis/figures/category_aggregated/category_scores.tsv
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
import matplotlib.image as mpimg

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
sys.path.insert(0, str(ROOT / "PCW12_analysis/scripts"))
from _viz_utils import render_gene_3d  # noqa: E402

IMP = ROOT / "PCW12_analysis/data/adata_imputed_disease_genes.h5ad"
PART = ROOT / "PCW12_analysis/data/disease_genes_partition.tsv"
OUT = ROOT / "PCW12_analysis/figures/category_aggregated"
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log(f"Loading {IMP}")
    adata = ad.read_h5ad(IMP)
    genes = adata.var_names.tolist()
    log(f"  {adata.shape}; {len(genes)} imputed genes")

    part = pd.read_csv(PART, sep="\t")
    part = part[part["Gene"].isin(genes)].copy()

    # gene -> set of categories (multi-membership via ';')
    gene_cats = {r["Gene"]: set(str(r["categories"]).split(";"))
                 for _, r in part.iterrows()}
    all_cats = sorted({c for s in gene_cats.values() for c in s})
    log(f"Categories: {all_cats}")

    # Dense gene matrix (cells × genes), z-score each gene
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-9
    Xz = (X - mu) / sd
    log(f"  z-scored matrix shape={Xz.shape}")

    # Compute per-cell mean z-score per category
    scores = {}
    for cat in all_cats:
        idx = [i for i, g in enumerate(genes) if cat in gene_cats.get(g, set())]
        if not idx:
            log(f"  [{cat}] no genes — skipping")
            continue
        score = Xz[:, idx].mean(axis=1)
        scores[cat] = score
        log(f"  [{cat}] n_genes={len(idx)}  score range "
            f"{score.min():.2f}..{score.max():.2f}")

    # Save per-cell scores TSV (compact: one row per cell × n_cats)
    scores_df = pd.DataFrame(scores, index=adata.obs_names)
    scores_df.to_csv(OUT / "category_scores.tsv", sep="\t",
                     float_format="%.4f")
    log(f"  wrote {OUT/'category_scores.tsv'}")

    # ---- Per-category 3D render (PNG + GIF) ----
    panel_imgs = []
    for cat in all_cats:
        if cat not in scores:
            continue
        png = OUT / f"{cat}.png"
        gif = OUT / f"{cat}.gif"
        log(f"Rendering {cat} → {png.name} (+ gif)")
        n_g = sum(1 for g in genes if cat in gene_cats.get(g, set()))
        render_gene_3d(
            adata, gene=cat, out_png=png, gif_path=gif,
            expression=scores[cat], cmap="magma",
            clim_quantile=(0.02, 0.98),
            title_suffix=f"— n={n_g} genes (mean z-score)",
        )
        panel_imgs.append((cat, png, n_g))

    # ---- 1×N summary panel ----
    if panel_imgs:
        n = len(panel_imgs)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6.2),
                                 facecolor="white")
        if n == 1:
            axes = [axes]
        for ax, (cat, png, n_g) in zip(axes, panel_imgs):
            ax.imshow(mpimg.imread(png))
            ax.set_title(f"{cat}  (n={n_g} genes)", fontsize=12)
            ax.axis("off")
        fig.suptitle("Aggregated disease-gene expression by category "
                     "(mean z-score per cell)", fontsize=14)
        fig.tight_layout()
        out_panel = OUT / "_summary_panel.png"
        fig.savefig(out_panel, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  wrote {out_panel}")

    log("Phase 5 complete.")


if __name__ == "__main__":
    main()
