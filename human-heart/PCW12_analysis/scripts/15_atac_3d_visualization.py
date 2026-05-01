"""
Step 3: 3D visualization of imputed ATAC signals on the MERFISH heart.

For each ATAC-imputed gene:
  - PNG of expression in 3D (z-axis flipped: apex points down)
  - GIF (rotating) for the highlight subset
For each disease Trait in heart_disease_genes.tsv:
  - Per-gene z-score across cells, then mean over the trait's genes
  - PNG + rotating GIF
  - trait_scores.tsv (per-cell trait score)
  - _summary_panel.png with all traits side-by-side

Inputs:
  PCW12_analysis/data/adata_imputed_atac.h5ad
  PCW12_analysis/data/highlight_genes.tsv  (column: highlight)
  data/heart_disease_genes.tsv             (Gene, Trait, Category)

Outputs:
  PCW12_analysis/figures/atac_imputed/<gene>.png   (per ATAC gene)
  PCW12_analysis/figures/atac_imputed/<gene>.gif   (highlight subset)
  PCW12_analysis/figures/atac_trait_aggregated/<trait>.png
  PCW12_analysis/figures/atac_trait_aggregated/<trait>.gif
  PCW12_analysis/figures/atac_trait_aggregated/_summary_panel.png
  PCW12_analysis/figures/atac_trait_aggregated/trait_scores.tsv
  PCW12_analysis/figures/atac_trait_aggregated/trait_gene_assignments.tsv
"""
from __future__ import annotations
import os
import sys
import time
import math
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

IMP_PATH = ROOT / "PCW12_analysis/data/adata_imputed_atac.h5ad"
HIGHLIGHTS = ROOT / "PCW12_analysis/data/highlight_genes.tsv"
DISEASE_PATH = ROOT / "data/heart_disease_genes.tsv"

OUT_GENE = ROOT / "PCW12_analysis/figures/atac_imputed"
OUT_TRAIT = ROOT / "PCW12_analysis/figures/atac_trait_aggregated"
OUT_GENE.mkdir(parents=True, exist_ok=True)
OUT_TRAIT.mkdir(parents=True, exist_ok=True)

MIN_GENES_PER_TRAIT = 3


def log(m: str):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log(f"Loading imputed ATAC: {IMP_PATH}")
    adata = ad.read_h5ad(IMP_PATH)
    log(f"  shape={adata.shape}  obsm keys={list(adata.obsm.keys())}")
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float32)
    log(f"  X range: {X.min():.3f}..{X.max():.3f}; mean={X.mean():.3f}")

    genes = adata.var_names.tolist()

    # Highlight set (genes flagged in highlight_genes.tsv)
    if HIGHLIGHTS.exists():
        hi = pd.read_csv(HIGHLIGHTS, sep="\t")
        if "highlight" in hi.columns:
            hi = hi[hi["highlight"].astype(bool)]
        highlight_set = set(hi["Gene"].astype(str)) & set(genes)
    else:
        highlight_set = set()
    log(f"Highlight ATAC genes: {len(highlight_set)}")

    # ---------------------------------------------------------------
    # Per-gene 3D figures (PNG always; GIF for highlights)
    # ---------------------------------------------------------------
    log(f"Rendering per-gene 3D figures for all {len(genes)} ATAC genes ...")
    failures = []
    for i, g in enumerate(genes, 1):
        out_png = OUT_GENE / f"{g}.png"
        gif = OUT_GENE / f"{g}.gif" if g in highlight_set else None
        if i % 25 == 1 or gif is not None:
            log(f"  [{i}/{len(genes)}] {g}{' + gif' if gif else ''}")
        try:
            render_gene_3d(
                adata, g, out_png,
                gif_path=gif,
                title_suffix="(ATAC, imputed)",
            )
        except Exception as e:
            log(f"   FAILED {g}: {e}")
            failures.append((g, str(e)))
    if failures:
        log(f"FAILED renders: {len(failures)}; first few: {failures[:5]}")

    # ---------------------------------------------------------------
    # Per-trait aggregation (mean z-score)
    # ---------------------------------------------------------------
    log(f"Loading disease trait table: {DISEASE_PATH}")
    disease = pd.read_csv(DISEASE_PATH, sep="\t")
    disease = disease[disease["Gene"].isin(genes)].copy()
    log(f"  disease rows for ATAC genes: {len(disease)}; "
        f"unique traits: {disease['Trait'].nunique()}")

    # Trait -> list of ATAC genes assigned to it
    trait_genes = (
        disease.groupby("Trait")["Gene"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )
    # Save assignments table
    rows = []
    for tr, gl in trait_genes.items():
        rows.append({
            "Trait": tr,
            "n_genes": len(gl),
            "Genes": ";".join(gl),
        })
    pd.DataFrame(rows).sort_values("n_genes", ascending=False).to_csv(
        OUT_TRAIT / "trait_gene_assignments.tsv", sep="\t", index=False)

    # Z-score each gene across cells
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-9
    Xz = (X - mu) / sd
    log(f"  z-scored matrix: {Xz.shape}")

    gene2idx = {g: i for i, g in enumerate(genes)}

    scores = {}
    n_genes_map = {}
    for tr, gl in trait_genes.items():
        idx = [gene2idx[g] for g in gl if g in gene2idx]
        n_genes_map[tr] = len(idx)
        if len(idx) < MIN_GENES_PER_TRAIT:
            log(f"  [{tr}] n_genes={len(idx)} < {MIN_GENES_PER_TRAIT}; skipping")
            continue
        sc = Xz[:, idx].mean(axis=1)
        scores[tr] = sc
        log(f"  [{tr}] n_genes={len(idx)} score range "
            f"{sc.min():.2f}..{sc.max():.2f} (mean {sc.mean():.3f})")

    # Per-cell scores TSV
    sdf = pd.DataFrame(scores, index=adata.obs_names)
    sdf.to_csv(OUT_TRAIT / "trait_scores.tsv", sep="\t", float_format="%.4f")
    log(f"  wrote {OUT_TRAIT / 'trait_scores.tsv'}")

    # ---- Per-trait 3D renders (PNG + GIF) ----
    panel_imgs = []
    rendered = sorted(scores.keys(), key=lambda t: -n_genes_map[t])
    for tr in rendered:
        png = OUT_TRAIT / f"{tr}.png"
        gif = OUT_TRAIT / f"{tr}.gif"
        log(f"Rendering trait {tr} -> {png.name} (+gif)")
        try:
            render_gene_3d(
                adata, gene=tr, out_png=png, gif_path=gif,
                expression=scores[tr], cmap="magma",
                clim_quantile=(0.02, 0.98),
                title_suffix=f"- ATAC z-score, n={n_genes_map[tr]} genes",
            )
            panel_imgs.append((tr, png, n_genes_map[tr]))
        except Exception as e:
            log(f"  FAILED render {tr}: {e}")

    # ---- Summary panel ----
    if panel_imgs:
        n = len(panel_imgs)
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.2 * ncols, 5.4 * nrows),
                                 facecolor="white")
        axes = np.atleast_2d(axes).reshape(nrows, ncols)
        for k, (tr, png, n_g) in enumerate(panel_imgs):
            r, c = divmod(k, ncols)
            ax = axes[r][c]
            ax.imshow(mpimg.imread(png))
            label = tr.replace("_", " ")
            if len(label) > 38:
                label = label[:36] + "..."
            ax.set_title(f"{label}  (n={n_g})", fontsize=11)
            ax.axis("off")
        for k in range(len(panel_imgs), nrows * ncols):
            r, c = divmod(k, ncols)
            axes[r][c].axis("off")
        fig.suptitle("ATAC chromatin accessibility - per-trait mean z-score (3D)",
                     fontsize=15, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_panel = OUT_TRAIT / "_summary_panel.png"
        fig.savefig(out_panel, dpi=140, bbox_inches="tight")
        plt.close(fig)
        log(f"  wrote {out_panel}")

    log("Step 3 complete.")


if __name__ == "__main__":
    main()
