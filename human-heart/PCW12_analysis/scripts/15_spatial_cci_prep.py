"""
Spatial CCI prep — Step 1 of 3.

Subsample the MOSCOT-imputed disease-gene AnnData, install required Spateo
metadata, build the spatial neighbor graph, threshold the expression layer,
and render the cell-type spatial-connectivity matrix figure.

Inputs:
  PCW12_analysis/data/adata_imputed_disease_genes.h5ad

Outputs:
  PCW12_analysis/data/adata_cci_ready.h5ad
  PCW12_analysis/figures/cci/celltype_connectivity.png
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
IN_H5AD = ROOT / "PCW12_analysis/data/adata_imputed_disease_genes.h5ad"
OUT_H5AD = ROOT / "PCW12_analysis/data/adata_cci_ready.h5ad"
FIG_DIR = ROOT / "PCW12_analysis/figures/cci"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GROUP_KEY = "mapped_coarse_celltype"
N_SUB = 10_000
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

    # --- top-N cell types ---
    vc = adata.obs[GROUP_KEY].value_counts()
    top_ct = vc.head(TOP_N_CT).index.tolist()
    log(f"Top-{TOP_N_CT} cell types (will be analyzed): {top_ct}")
    log(f"Counts: {vc.head(TOP_N_CT).to_dict()}")

    # --- spatial coordinates from X_spateo_update (3D) ---
    coords = adata.obsm["X_spateo_update"].astype(float).copy()
    if FLIP_Z and coords.shape[1] == 3:
        coords[:, -1] = -coords[:, -1]
        log("Flipped Z axis on spatial coordinates")
    adata.obsm["spatial"] = coords
    log(f"obsm['spatial'] shape: {adata.obsm['spatial'].shape}")

    # --- subsample ---
    if adata.n_obs > N_SUB:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(adata.n_obs, size=N_SUB, replace=False)
        idx.sort()
        adata = adata[idx].copy()
        log(f"Subsampled to {adata.n_obs} cells (seed={SEED})")
    sub_vc = adata.obs[GROUP_KEY].value_counts()
    log(f"Subsample top-8 counts: {sub_vc.head(TOP_N_CT).to_dict()}")

    # --- spateo required metadata ---
    adata.uns["__type"] = "UMI"

    # --- spatial neighbor graph ---
    log(f"Building spatial neighbor graph (n_neighbors={N_NEIGHBORS}) ...")
    _, adata = st.tl.neighbors(
        adata,
        basis="spatial",
        spatial_key="spatial",
        n_neighbors=N_NEIGHBORS,
    )
    # Spateo writes 'spatial_connectivities' but downstream funcs read 'connectivities'
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

    # --- top-8 list cached for downstream scripts ---
    adata.uns["cci_top_celltypes"] = list(map(str, top_ct))

    # --- save adata ---
    log(f"Saving {OUT_H5AD.name}")
    adata.write_h5ad(OUT_H5AD)
    log(f"  size: {OUT_H5AD.stat().st_size/1e6:.1f} MB")

    # --- connectivity matrix figure ---
    log("Rendering cell-type spatial-connectivity matrix ...")
    fig_out = FIG_DIR / "celltype_connectivity.png"
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
        log(f"plot_connections save_kwargs path failed ({e!r}); fallback to return")
        try:
            res = st.pl.plot_connections(
                adata, cat_key=GROUP_KEY, save_show_or_return="return"
            )
            plt.savefig(fig_out, dpi=200, bbox_inches="tight")
            plt.close("all")
        except Exception as e2:
            log(f"plot_connections fallback also failed: {e2!r}")
            raise

    # Spateo may save with a generated suffix — find the file we created
    found = sorted(FIG_DIR.glob("celltype_connectivity*.png"))
    log(f"Connectivity figures present: {[p.name for p in found]}")
    log("Done.")


if __name__ == "__main__":
    main()
