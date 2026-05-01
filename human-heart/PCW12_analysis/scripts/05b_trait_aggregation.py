"""
Phase 5b: Aggregated 3D expression patterns per heart-disease *Trait*.

For each trait in disease_genes_partition.tsv (multi-membership via ';'),
aggregate all imputed genes assigned to that trait:
- Z-score each gene across cells, then average z-scores per cell → trait score.
- Render a 3D point cloud colored by the score (PNG + rotating GIF).

Also produces a grid panel summary PNG.

Inputs:
  PCW12_analysis/data/adata_imputed_disease_genes.h5ad
  PCW12_analysis/data/disease_genes_partition.tsv  (column: traits)

Outputs:
  PCW12_analysis/figures/trait_aggregated/<trait>.png
  PCW12_analysis/figures/trait_aggregated/<trait>.gif
  PCW12_analysis/figures/trait_aggregated/_summary_panel.png
  PCW12_analysis/figures/trait_aggregated/trait_scores.tsv
"""
from __future__ import annotations
import os, sys, time, math
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
OUT = ROOT / "PCW12_analysis/figures/trait_aggregated"
OUT.mkdir(parents=True, exist_ok=True)

MIN_GENES = 3  # skip traits with fewer than this many imputed genes


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log(f"Loading {IMP}")
    adata = ad.read_h5ad(IMP)
    genes = adata.var_names.tolist()
    log(f"  {adata.shape}; {len(genes)} imputed genes")

    part = pd.read_csv(PART, sep="\t")
    part = part[part["Gene"].isin(genes)].copy()

    # gene -> set of traits
    gene_traits = {
        r["Gene"]: set(str(r["traits"]).split(";"))
        for _, r in part.iterrows()
        if pd.notna(r["traits"]) and str(r["traits"]).strip()
    }
    all_traits = sorted({t for s in gene_traits.values() for t in s if t})
    log(f"Traits ({len(all_traits)}): {all_traits}")

    # Dense gene matrix, z-score each gene
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-9
    Xz = (X - mu) / sd
    log(f"  z-scored matrix shape={Xz.shape}")

    # Per-cell mean z-score per trait
    scores = {}
    n_genes_map = {}
    for tr in all_traits:
        idx = [i for i, g in enumerate(genes) if tr in gene_traits.get(g, set())]
        n_genes_map[tr] = len(idx)
        if len(idx) < MIN_GENES:
            log(f"  [{tr}] only {len(idx)} imputed genes — skipping (<{MIN_GENES})")
            continue
        score = Xz[:, idx].mean(axis=1)
        scores[tr] = score
        log(f"  [{tr}] n_genes={len(idx)}  range "
            f"{score.min():.2f}..{score.max():.2f}")

    # Save per-cell scores TSV
    scores_df = pd.DataFrame(scores, index=adata.obs_names)
    scores_df.to_csv(OUT / "trait_scores.tsv", sep="\t", float_format="%.4f")
    log(f"  wrote {OUT/'trait_scores.tsv'}")

    # ---- Per-trait 3D render (PNG + GIF) ----
    panel_imgs = []
    # Order traits by gene count desc for nicer panel reading
    rendered_traits = sorted(scores.keys(),
                             key=lambda t: -n_genes_map[t])
    for tr in rendered_traits:
        png = OUT / f"{tr}.png"
        gif = OUT / f"{tr}.gif"
        log(f"Rendering {tr} → {png.name} (+ gif)")
        render_gene_3d(
            adata, gene=tr, out_png=png, gif_path=gif,
            expression=scores[tr], cmap="magma",
            clim_quantile=(0.02, 0.98),
            title_suffix=f"— n={n_genes_map[tr]} genes (mean z-score)",
        )
        panel_imgs.append((tr, png, n_genes_map[tr]))

    # ---- Grid summary panel ----
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
            short = tr.replace("_", " ")
            if len(short) > 38:
                short = short[:36] + "…"
            ax.set_title(f"{short}  (n={n_g})", fontsize=11)
            ax.axis("off")
        # Hide unused axes
        for k in range(len(panel_imgs), nrows * ncols):
            r, c = divmod(k, ncols)
            axes[r][c].axis("off")
        fig.suptitle("Aggregated disease-gene expression by Trait "
                     "(mean z-score per cell)", fontsize=15, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_panel = OUT / "_summary_panel.png"
        fig.savefig(out_panel, dpi=140, bbox_inches="tight")
        plt.close(fig)
        log(f"  wrote {out_panel}")

    log("Phase 5b complete.")


if __name__ == "__main__":
    main()
