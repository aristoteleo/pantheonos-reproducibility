"""
Spatial CCI prep (v2) — clones 15_spatial_cci_prep.py for the HVG ∪ disease
imputed AnnData. Subsamples to 20k cells (user-approved), Z-flips coords,
sets up Spateo metadata, builds neighbor graph, threshold layer, and renders
the connectivity matrix.

Inputs:
  PCW12_analysis/data/adata_imputed_hvg_disease.h5ad

Outputs:
  PCW12_analysis/data/adata_cci_ready_hvg.h5ad
  PCW12_analysis/figures/cci_hvg/celltype_connectivity.png
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import spateo as st

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
IN_H5AD = ROOT / "PCW12_analysis/data/adata_imputed_hvg_disease.h5ad"
OUT_H5AD = ROOT / "PCW12_analysis/data/adata_cci_ready_hvg.h5ad"
FIG_DIR = ROOT / "PCW12_analysis/figures/cci_hvg"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LR_DB_DIR = ROOT / "PCW12_analysis/data/spateo_lr_db"

GROUP_KEY = "mapped_coarse_celltype"
N_SUB = 20_000          # user-approved bump from 10k
SEED = 42
N_NEIGHBORS = 10
THRESHOLD = 0.5
TOP_N_CT = 8
FLIP_Z = True


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"Loading {IN_H5AD.name}")
    adata = sc.read_h5ad(IN_H5AD)
    log(f"  shape={adata.shape}, X dtype={adata.X.dtype}")
    log(f"  X range: {float(adata.X.min()):.2f}..{float(adata.X.max()):.2f}")
    if "impute_meta" in adata.uns:
        log(f"  impute_meta: {dict(adata.uns['impute_meta'])}")

    # --- LR-DB coverage check (the whole point of the rerun) ---
    try:
        import pandas as pd
        lr_db = pd.read_csv(LR_DB_DIR / "lr_db_human.csv")
        # try common column names
        lig_col = next((c for c in ["from", "ligand", "Ligand", "L"] if c in lr_db.columns), None)
        rec_col = next((c for c in ["to", "receptor", "Receptor", "R"] if c in lr_db.columns), None)
        if lig_col and rec_col:
            present = set(adata.var_names.str.upper())
            lig = set(lr_db[lig_col].astype(str).str.upper())
            rec = set(lr_db[rec_col].astype(str).str.upper())
            n_lig = len(lig & present)
            n_rec = len(rec & present)
            mask_pair_present = (
                lr_db[lig_col].astype(str).str.upper().isin(present)
                & lr_db[rec_col].astype(str).str.upper().isin(present)
            )
            n_pairs = int(mask_pair_present.sum())
            log(f"LR-DB coverage on imputed panel: {n_lig} ligands, {n_rec} receptors, "
                f"{n_pairs} testable LR pairs (vs ~11 in the prior 221-gene panel)")
    except Exception as e:
        log(f"LR-DB coverage check skipped: {e!r}")

    # --- top-N cell types ---
    vc = adata.obs[GROUP_KEY].value_counts()
    top_ct = vc.head(TOP_N_CT).index.tolist()
    log(f"Top-{TOP_N_CT} cell types: {top_ct}")
    log(f"Counts (full): {vc.head(TOP_N_CT).to_dict()}")

    # --- spatial coords (Z-flipped) ---
    coords = adata.obsm["X_spateo_update"].astype(float).copy()
    if FLIP_Z and coords.shape[1] == 3:
        coords[:, -1] = -coords[:, -1]
        log("Flipped Z axis on spatial coordinates")
    adata.obsm["spatial"] = coords

    # --- subsample ---
    if adata.n_obs > N_SUB:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(adata.n_obs, size=N_SUB, replace=False)
        idx.sort()
        adata = adata[idx].copy()
        log(f"Subsampled to {adata.n_obs} cells (seed={SEED})")
    sub_vc = adata.obs[GROUP_KEY].value_counts()
    log(f"Subsample top-8 counts: {sub_vc.head(TOP_N_CT).to_dict()}")

    # --- spateo metadata ---
    adata.uns["__type"] = "UMI"

    # --- spatial neighbor graph ---
    log(f"Building spatial neighbor graph (n_neighbors={N_NEIGHBORS}) ...")
    _, adata = st.tl.neighbors(
        adata,
        basis="spatial",
        spatial_key="spatial",
        n_neighbors=N_NEIGHBORS,
    )
    if "spatial_connectivities" in adata.obsp:
        adata.obsp["connectivities"] = adata.obsp["spatial_connectivities"].copy()
        log("Copied spatial_connectivities -> connectivities")

    # --- thresholded layer ---
    X_thresh = adata.X.copy()
    if hasattr(X_thresh, "toarray"):
        X_thresh = X_thresh.toarray()
    X_thresh = np.asarray(X_thresh)
    X_thresh[X_thresh < THRESHOLD] = 0
    adata.layers["thresh"] = X_thresh.astype(np.float32)
    nz = float((X_thresh > 0).mean())
    log(f"Thresholded layer: keep {nz*100:.1f}% of entries (threshold={THRESHOLD})")

    adata.uns["cci_top_celltypes"] = list(map(str, top_ct))

    log(f"Saving {OUT_H5AD.name}")
    adata.write_h5ad(OUT_H5AD)
    log(f"  size: {OUT_H5AD.stat().st_size/1e6:.1f} MB")

    # --- connectivity matrix figure ---
    log("Rendering cell-type spatial-connectivity matrix ...")
    try:
        st.pl.plot_connections(
            adata,
            cat_key=GROUP_KEY,
            save_show_or_return="save",
            save_kwargs={
                "path": str(FIG_DIR),
                "prefix": "celltype_connectivity",
                "ext": "png",
                "dpi": 200,
                "close": True,
                "verbose": False,
            },
        )
    except Exception as e:
        log(f"plot_connections save failed ({e!r}); falling back")
        try:
            st.pl.plot_connections(adata, cat_key=GROUP_KEY, save_show_or_return="return")
            plt.savefig(FIG_DIR / "celltype_connectivity.png", dpi=200, bbox_inches="tight")
            plt.close("all")
        except Exception as e2:
            log(f"Fallback also failed: {e2!r}")
            raise

    found = sorted(FIG_DIR.glob("celltype_connectivity*.png"))
    log(f"Connectivity figures: {[p.name for p in found]}")
    log("Done.")


if __name__ == "__main__":
    main()
