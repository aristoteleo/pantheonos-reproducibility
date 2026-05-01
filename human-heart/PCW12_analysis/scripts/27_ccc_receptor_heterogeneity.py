"""27 — Subcluster CoreConductionCells on receptor expression.

Question: does CCC split into spatially-distinct subpopulations
(potentially AVN-like vs His-bundle-like) based on the receptors that
appeared significant in the CCI loop?

Approach:
  1. Subset to CCC (~292 cells from full 100k AnnData).
  2. Build a feature matrix from a curated receptor + CCS-TF panel.
  3. Standard scanpy mini-pipeline: scale → PCA → neighbors → Leiden.
  4. Visualize clusters in 3D anatomical space; produce marker heatmap.

Outputs:
  figures/cci_downstream/ccc_subclusters_umap.png
  figures/cci_downstream/ccc_subclusters_3d.png
  figures/cci_downstream/ccc_receptor_tf_heatmap.png
  results/cci_downstream/ccc_subcluster_assignments.csv
  results/cci_downstream/ccc_subcluster_markers.csv
  logs/27_ccc_receptor_heterogeneity.log
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
DATA = ROOT / "PCW12_analysis/data/adata_imputed_hvg_disease.h5ad"
FIG_DIR = ROOT / "PCW12_analysis/figures/cci_downstream"
RES_DIR = ROOT / "PCW12_analysis/results/cci_downstream"
LOG_PATH = ROOT / "PCW12_analysis/logs/27_ccc_receptor_heterogeneity.log"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ccc_het")

CT_KEY = "mapped_coarse_celltype"
COORD_KEY = "X_spateo_update"

# Feature panels
RECEPTORS = ["ERBB4", "ERBB3", "PLXNA4", "EPHA4", "FGFR2", "BMPR2", "ITGA1", "ITGB8"]
CCS_TFS = ["TBX3", "TBX5", "NKX2-5", "ISL1", "SHOX2", "HCN1", "GJA5", "GJA1", "TBX2", "TBX18"]


def main():
    t0 = time.time()
    log.info(f"Loading {DATA} …")
    a = ad.read_h5ad(DATA)
    log.info(f"  shape={a.shape}")

    is_ccc = a.obs[CT_KEY] == "CoreConductionCells"
    log.info(f"  CCC count = {int(is_ccc.sum())}")
    sub = a[is_ccc].copy()
    log.info(f"  CCC sub.shape = {sub.shape}")

    feats = [g for g in RECEPTORS + CCS_TFS if g in sub.var_names]
    miss = [g for g in RECEPTORS + CCS_TFS if g not in sub.var_names]
    if miss:
        log.warning(f"  missing genes (skipped): {miss}")
    log.info(f"  feature genes used: {feats}")

    # build feature AnnData restricted to those genes (so PCA is on them)
    feat_a = sub[:, feats].copy()
    feat_a.obs[COORD_KEY] = None  # placeholder

    # standard mini-pipeline
    sc.pp.scale(feat_a, max_value=10)
    n_comps = min(len(feats) - 1, 8)
    sc.tl.pca(feat_a, n_comps=n_comps)
    sc.pp.neighbors(feat_a, n_neighbors=15, use_rep="X_pca")
    sc.tl.umap(feat_a)
    sc.tl.leiden(feat_a, resolution=0.5, key_added="ccc_subcluster", random_state=0)

    n_clust = feat_a.obs["ccc_subcluster"].nunique()
    log.info(f"  Leiden produced {n_clust} clusters: "
             f"{feat_a.obs['ccc_subcluster'].value_counts().to_dict()}")

    # propagate cluster labels back to sub for downstream use
    sub.obs["ccc_subcluster"] = feat_a.obs["ccc_subcluster"].values
    sub.obsm["X_umap"] = feat_a.obsm["X_umap"]

    # ── UMAP figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4.5))
    palette = plt.cm.tab10.colors
    for i, c in enumerate(sorted(sub.obs["ccc_subcluster"].unique(), key=int)):
        mask = (sub.obs["ccc_subcluster"] == c).values
        ax.scatter(sub.obsm["X_umap"][mask, 0], sub.obsm["X_umap"][mask, 1],
                   s=20, c=[palette[i % 10]], label=f"c{c} (n={int(mask.sum())})", alpha=0.8)
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title(f"CCC subclusters on receptor+TF panel\nn={sub.n_obs}, k={n_clust}")
    plt.tight_layout()
    fig_path = FIG_DIR / "ccc_subclusters_umap.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved {fig_path.name}")

    # ── 3D anatomical figure ───────────────────────────────────────────
    coords = a.obsm[COORD_KEY]
    rng = np.random.default_rng(0)
    bg_mask = ~is_ccc.values
    bg_idx = np.where(bg_mask)[0]
    bg_idx = rng.choice(bg_idx, min(8000, len(bg_idx)), replace=False)
    bg = coords[bg_idx]
    ccc_coords = coords[is_ccc.values]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=0.5, c="lightgray", alpha=0.05)
    for i, c in enumerate(sorted(sub.obs["ccc_subcluster"].unique(), key=int)):
        mask = (sub.obs["ccc_subcluster"] == c).values
        ax.scatter(ccc_coords[mask, 0], ccc_coords[mask, 1], ccc_coords[mask, 2],
                   s=18, c=[palette[i % 10]], label=f"c{c} (n={int(mask.sum())})", alpha=0.85)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"CCC subclusters in 3D\n(k={n_clust})")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    fig_path = FIG_DIR / "ccc_subclusters_3d.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved {fig_path.name}")

    # ── marker heatmap: per-cluster mean expression ────────────────────
    means = pd.DataFrame(index=feats, dtype=float)
    for c in sorted(sub.obs["ccc_subcluster"].unique(), key=int):
        mask = (sub.obs["ccc_subcluster"] == c).values
        means[f"c{c}"] = np.asarray(sub[:, feats].X[mask].mean(axis=0)).ravel()
    # z-score per gene (row) for visual contrast
    means_z = means.sub(means.mean(axis=1), axis=0).div(means.std(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(0.7 * means_z.shape[1] + 2, 0.32 * len(feats) + 1.5))
    im = ax.imshow(means_z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(means_z.shape[1])); ax.set_xticklabels(means_z.columns, rotation=0)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats)
    # annotate which feats are receptors vs TFs
    for j, g in enumerate(feats):
        kind = "R" if g in RECEPTORS else "TF"
        ax.text(-0.6, j, kind, fontsize=7, ha="right", va="center", color="gray")
    plt.colorbar(im, ax=ax, label="row z-score")
    ax.set_title("CCC subcluster signatures (row z-score)")
    plt.tight_layout()
    fig_path = FIG_DIR / "ccc_receptor_tf_heatmap.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved {fig_path.name}")

    # ── markers (rank_genes_groups across full var, not just panel) ────
    sc.tl.rank_genes_groups(sub, "ccc_subcluster", method="wilcoxon",
                            n_genes=25, use_raw=False)
    rgg = sub.uns["rank_genes_groups"]
    marker_rows = []
    for c in rgg["names"].dtype.names:
        for k in range(min(25, len(rgg["names"][c]))):
            marker_rows.append({
                "cluster": c, "rank": k + 1,
                "gene": rgg["names"][c][k],
                "score": float(rgg["scores"][c][k]),
                "logfc": float(rgg["logfoldchanges"][c][k]),
                "pval_adj": float(rgg["pvals_adj"][c][k]),
            })
    pd.DataFrame(marker_rows).to_csv(RES_DIR / "ccc_subcluster_markers.csv", index=False)

    # save cluster assignments with coordinates (for downstream)
    out_assign = pd.DataFrame({
        "cell_id": sub.obs_names,
        "ccc_subcluster": sub.obs["ccc_subcluster"].values,
        "x": ccc_coords[:, 0], "y": ccc_coords[:, 1], "z": ccc_coords[:, 2],
    })
    out_assign.to_csv(RES_DIR / "ccc_subcluster_assignments.csv", index=False)
    log.info(f"saved markers + assignments to {RES_DIR}")
    log.info(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
