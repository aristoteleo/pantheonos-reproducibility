"""26 — Spatial validation of NRG3-ERBB4 and SEMA3A-PLXNA4 axes (ACM → CCC).

Hypothesis: if ACM is the paracrine source of NRG3 / SEMA3A to CCC, then
NRG3-high (resp SEMA3A-high) ACM cells should be spatially closer to
ERBB4-high (resp PLXNA4-high) CCC cells than to random CCC cells.

Test:
  For each CCC cell, find its nearest ACM cell in 3D and record the ligand
  expression of that ACM. Spearman-correlate (CCC receptor) with
  (nearest-ACM ligand). Permutation null = shuffle CCC labels among
  CCC-cell coordinates (= shuffle the ligand-from-ACM column relative to
  the receptor column). Report rho, perm-pval.

Outputs:
  figures/cci_downstream/{axis}_3d.png
  figures/cci_downstream/{axis}_distance_scatter.png
  results/cci_downstream/spatial_axis_stats.csv
  logs/26_spatial_axis_validation.log
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
DATA = ROOT / "PCW12_analysis/data/adata_imputed_hvg_disease.h5ad"
FIG_DIR = ROOT / "PCW12_analysis/figures/cci_downstream"
RES_DIR = ROOT / "PCW12_analysis/results/cci_downstream"
LOG_PATH = ROOT / "PCW12_analysis/logs/26_spatial_axis_validation.log"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("axis_val")

CT_KEY = "mapped_coarse_celltype"
COORD_KEY = "X_spateo_update"
N_PERM = 1000
RNG = np.random.default_rng(42)

AXES = [
    {"name": "NRG3_ERBB4", "ligand": "NRG3", "receptor": "ERBB4"},
    {"name": "SEMA3A_PLXNA4", "ligand": "SEMA3A", "receptor": "PLXNA4"},
]


def expr_vec(a: ad.AnnData, gene: str) -> np.ndarray:
    """Return dense 1D expression vector for `gene`."""
    j = a.var_names.get_loc(gene)
    x = a.X[:, j]
    return np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()


def run_axis(adata, ligand, receptor, axis_name):
    log.info(f"=== Axis {axis_name}: {ligand}(ACM) → {receptor}(CCC) ===")
    is_acm = (adata.obs[CT_KEY] == "ACM").to_numpy()
    is_ccc = (adata.obs[CT_KEY] == "CoreConductionCells").to_numpy()
    log.info(f"  ACM={is_acm.sum()}, CCC={is_ccc.sum()}")

    coords = adata.obsm[COORD_KEY]
    coords_acm = coords[is_acm]
    coords_ccc = coords[is_ccc]

    lig_acm = expr_vec(adata, ligand)[is_acm]
    rec_ccc = expr_vec(adata, receptor)[is_ccc]

    # nearest ACM for each CCC
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(coords_acm)
    dists, idx = nn.kneighbors(coords_ccc)
    nearest_lig = lig_acm[idx.ravel()]
    nearest_dist = dists.ravel()
    log.info(f"  nearest-ACM distance: median={np.median(nearest_dist):.2f}")

    # observed correlation
    rho, p_param = spearmanr(rec_ccc, nearest_lig)
    log.info(f"  Spearman rho={rho:.4f}, parametric p={p_param:.3e}")

    # permutation null: shuffle nearest_lig among CCC cells (breaks spatial
    # pairing while preserving marginal distributions)
    null_rhos = np.empty(N_PERM, dtype=float)
    for i in range(N_PERM):
        perm = RNG.permutation(nearest_lig)
        null_rhos[i] = spearmanr(rec_ccc, perm).statistic
    p_perm = float((np.abs(null_rhos) >= abs(rho)).mean())
    log.info(f"  permutation 2-sided p={p_perm:.4f} (n_perm={N_PERM})")

    # ── 3D figure: ACM colored by ligand, CCC colored by receptor ────
    fig = plt.figure(figsize=(12, 5.5))
    bg = coords[~(is_acm | is_ccc)]
    if len(bg) > 5000:
        bg = bg[RNG.choice(len(bg), 5000, replace=False)]

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=0.5, c="lightgray", alpha=0.05)
    sc1 = ax1.scatter(
        coords_acm[:, 0], coords_acm[:, 1], coords_acm[:, 2],
        c=lig_acm, cmap="Reds", s=4, alpha=0.6, vmin=0, vmax=np.quantile(lig_acm, 0.99),
    )
    ax1.set_title(f"ACM coloured by {ligand}\n(n={is_acm.sum()})")
    plt.colorbar(sc1, ax=ax1, shrink=0.5, label=ligand)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=0.5, c="lightgray", alpha=0.05)
    sc2 = ax2.scatter(
        coords_ccc[:, 0], coords_ccc[:, 1], coords_ccc[:, 2],
        c=rec_ccc, cmap="Blues", s=12, alpha=0.85,
        vmin=0, vmax=np.quantile(rec_ccc, 0.99) if rec_ccc.max() > 0 else 1,
    )
    ax2.set_title(f"CCC coloured by {receptor}\n(n={is_ccc.sum()})")
    plt.colorbar(sc2, ax=ax2, shrink=0.5, label=receptor)

    for a in (ax1, ax2):
        a.set_xlabel("x"); a.set_ylabel("y"); a.set_zlabel("z")
    plt.suptitle(f"{axis_name}  Spearman ρ={rho:.3f}  p_perm={p_perm:.3g}")
    plt.tight_layout()
    fig_path = FIG_DIR / f"{axis_name}_3d.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved {fig_path.name}")

    # ── scatter: nearest-ACM ligand vs CCC receptor ─────────────────
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(nearest_lig, rec_ccc, s=10, alpha=0.5, color="#444")
    ax.set_xlabel(f"{ligand} in nearest ACM cell")
    ax.set_ylabel(f"{receptor} in CCC cell")
    ax.set_title(f"{axis_name}\nρ={rho:.3f}  p_perm={p_perm:.3g}  n={len(rec_ccc)}")
    # null distribution inset
    inset = ax.inset_axes([0.65, 0.65, 0.32, 0.32])
    inset.hist(null_rhos, bins=40, color="lightgray", edgecolor="gray")
    inset.axvline(rho, color="red", lw=2)
    inset.set_xlabel("null ρ", fontsize=7)
    inset.tick_params(labelsize=6)
    plt.tight_layout()
    fig_path = FIG_DIR / f"{axis_name}_distance_scatter.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved {fig_path.name}")

    return {
        "axis": axis_name, "ligand": ligand, "receptor": receptor,
        "n_ccc": int(is_ccc.sum()), "n_acm": int(is_acm.sum()),
        "rho": float(rho), "p_param": float(p_param), "p_perm": p_perm,
        "n_perm": N_PERM,
        "median_nearest_dist": float(np.median(nearest_dist)),
    }


def main():
    t0 = time.time()
    log.info(f"Loading {DATA} …")
    adata = ad.read_h5ad(DATA)
    log.info(f"  shape={adata.shape}, coords={COORD_KEY} {adata.obsm[COORD_KEY].shape}")

    rows = [run_axis(adata, ax["ligand"], ax["receptor"], ax["name"]) for ax in AXES]
    df = pd.DataFrame(rows)
    out = RES_DIR / "spatial_axis_stats.csv"
    df.to_csv(out, index=False)
    log.info(f"\n{df.to_string(index=False)}")
    log.info(f"saved {out}")
    log.info(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
