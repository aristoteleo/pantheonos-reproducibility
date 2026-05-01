"""28 — Niche-score → CCS-TF correlation in CoreConductionCells.

Hypothesis: if NRG3 (resp SEMA3A) ACM signaling shapes CCS identity, then
CCC cells receiving high niche input should also express more CCS TFs.

Method:
  niche_NRG3(c) = mean NRG3 across the k=10 nearest ACM cells of CCC c
  niche_SEMA3A(c) = same with SEMA3A
  Spearman-correlate each niche score with each CCS TF in the same CCC cell.
  BH-FDR over the (2 niche × n_TFs) tests.

Outputs:
  figures/cci_downstream/niche_tf_correlation.png
  figures/cci_downstream/niche_quartile_dotplot.png
  results/cci_downstream/niche_tf_corr.csv
  logs/28_niche_tf_correlation.log
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
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
DATA = ROOT / "PCW12_analysis/data/adata_imputed_hvg_disease.h5ad"
FIG_DIR = ROOT / "PCW12_analysis/figures/cci_downstream"
RES_DIR = ROOT / "PCW12_analysis/results/cci_downstream"
LOG_PATH = ROOT / "PCW12_analysis/logs/28_niche_tf_correlation.log"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("niche_tf")

CT_KEY = "mapped_coarse_celltype"
COORD_KEY = "X_spateo_update"
K_NEIGHBORS = 10
LIGANDS = ["NRG3", "SEMA3A"]
CCS_TFS = ["TBX3", "TBX5", "NKX2-5", "ISL1", "SHOX2", "HCN1", "GJA5", "GJA1", "TBX2", "TBX18"]


def expr_vec(a, gene):
    j = a.var_names.get_loc(gene)
    x = a.X[:, j]
    return np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()


def main():
    t0 = time.time()
    log.info(f"Loading {DATA} …")
    a = ad.read_h5ad(DATA)
    log.info(f"  shape={a.shape}")

    is_acm = (a.obs[CT_KEY] == "ACM").to_numpy()
    is_ccc = (a.obs[CT_KEY] == "CoreConductionCells").to_numpy()
    coords = a.obsm[COORD_KEY]
    coords_acm = coords[is_acm]
    coords_ccc = coords[is_ccc]
    log.info(f"  ACM={is_acm.sum()}, CCC={is_ccc.sum()}")

    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm="auto").fit(coords_acm)
    _, idx = nn.kneighbors(coords_ccc)  # (n_ccc, k)

    # niche scores
    niche = {}
    for lig in LIGANDS:
        if lig not in a.var_names:
            log.warning(f"  ligand {lig} missing — skipping")
            continue
        lig_acm = expr_vec(a, lig)[is_acm]
        niche[lig] = lig_acm[idx].mean(axis=1)
        log.info(f"  niche-{lig}: mean={niche[lig].mean():.3f}, "
                 f"std={niche[lig].std():.3f}, q90={np.quantile(niche[lig], 0.9):.3f}")

    # CCS TF expression in CCC
    tfs_present = [g for g in CCS_TFS if g in a.var_names]
    log.info(f"  CCS TFs available: {tfs_present}  (missing: {set(CCS_TFS)-set(tfs_present)})")
    tf_mat = np.column_stack([expr_vec(a, g)[is_ccc] for g in tfs_present])  # (n_ccc, n_tf)

    # Spearman for each (niche × TF)
    rows = []
    for lig, nv in niche.items():
        for j, g in enumerate(tfs_present):
            rho, p = spearmanr(nv, tf_mat[:, j])
            rows.append({"ligand": lig, "tf": g, "rho": float(rho), "p": float(p),
                         "n": int(len(nv))})
    df = pd.DataFrame(rows)
    df["fdr"] = multipletests(df["p"].values, method="fdr_bh")[1]
    df["sig"] = df["fdr"] < 0.05
    df = df.sort_values(["ligand", "rho"], ascending=[True, False])
    df.to_csv(RES_DIR / "niche_tf_corr.csv", index=False)
    log.info(f"\n{df.to_string(index=False)}")

    # ── correlation heatmap ──────────────────────────────────────────
    pivot = df.pivot(index="tf", columns="ligand", values="rho")
    sig_pivot = df.pivot(index="tf", columns="ligand", values="sig")
    pivot = pivot.loc[tfs_present, list(niche.keys())]
    sig_pivot = sig_pivot.loc[tfs_present, list(niche.keys())]
    fig, ax = plt.subplots(figsize=(0.9 * pivot.shape[1] + 2.5, 0.35 * len(tfs_present) + 1.5))
    vmax = max(0.2, np.nanmax(np.abs(pivot.values)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    # annotate sig
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if bool(sig_pivot.values[i, j]):
                ax.text(j, i, "*", ha="center", va="center", color="black", fontsize=12)
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title(f"Niche-score → CCS-TF correlation in CCC\nn={tf_mat.shape[0]}, k={K_NEIGHBORS}, * FDR<0.05")
    plt.tight_layout()
    fig_path = FIG_DIR / "niche_tf_correlation.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved {fig_path.name}")

    # ── quartile dotplot for the highest-correlation pair per ligand ──
    fig, axes = plt.subplots(1, len(niche), figsize=(5 * len(niche), 4), squeeze=False)
    for k, (lig, nv) in enumerate(niche.items()):
        ax = axes[0, k]
        # pick top-correlated TF (positive) for this ligand
        sub = df[df["ligand"] == lig].copy()
        sub["abs_rho"] = sub["rho"].abs()
        sub = sub.sort_values("abs_rho", ascending=False)
        if len(sub) == 0:
            continue
        top = sub.iloc[0]
        g = top["tf"]; rho = top["rho"]; fdr = top["fdr"]
        tf_vec = tf_mat[:, tfs_present.index(g)]
        q_edges = np.quantile(nv, [0, 0.25, 0.5, 0.75, 1.0])
        # tiny jitter to avoid singular bins
        q_edges = np.unique(q_edges)
        q_labels = ["Q1", "Q2", "Q3", "Q4"][: len(q_edges) - 1]
        bins = np.clip(np.digitize(nv, q_edges[1:-1]), 0, len(q_labels) - 1)
        data = [tf_vec[bins == i] for i in range(len(q_labels))]
        bp = ax.boxplot(data, labels=q_labels, showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#cce5ff")
        ax.set_xlabel(f"{lig} niche-score quartile")
        ax.set_ylabel(f"{g} expression in CCC")
        ax.set_title(f"{lig} → {g}\nρ={rho:.3f}, FDR={fdr:.2g}")
    plt.tight_layout()
    fig_path = FIG_DIR / "niche_quartile_dotplot.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved {fig_path.name}")

    log.info(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
