"""
Phase 10: Section-16 valve spatial analysis.

Section 16 of the 3D MERFISH PCW12 dataset is the most valve-enriched
z-slice (18.8 % VIC/VEC/ncCM-AVC-like vs 3.3 % overall, log2 +2.52).
On this single anatomical plane we produce:

  Figure A: spatial scatter highlighting valve-associated cell types
            (VIC, VEC, ncCM-AVC-like) over a light-gray section background.
  Figure B: 2x5 grid of disease-trait expression panels (z-scored mean
            log1p over each trait's gene panel, computed from the
            MOSCOT-imputed adata). Faint outlines of valve-cell positions
            are overlaid on each panel.

Inputs:
  data/full_heart_final_aug2025_update_downsampled_100k.h5ad
    obs['celltype'], obs['section'], obsm['X_spateo_update'] (xyz, um).
  PCW12_analysis/data/adata_imputed_disease_genes.h5ad   (cells x 221 genes)
  PCW12_analysis/data/disease_genes_partition.tsv         (gene -> traits)

Outputs (PCW12_analysis/figures/spatial_section/):
  section16_valve_celltypes.png/.pdf
  section16_disease_trait_panels.png/.pdf
  section16_cells.tsv
  section_valve_enrichment.tsv
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
SP_PATH  = ROOT / "data/full_heart_final_aug2025_update_downsampled_100k.h5ad"
IMP_PATH = ROOT / "PCW12_analysis/data/adata_imputed_disease_genes.h5ad"
PART     = ROOT / "PCW12_analysis/data/disease_genes_partition.tsv"
OUT      = ROOT / "PCW12_analysis/figures/spatial_section"
OUT.mkdir(parents=True, exist_ok=True)

SECTION = 16
VALVE_CTS = ["VIC", "VEC", "ncCM-AVC-like"]
VALVE_COLORS = {
    "VIC":            "#d6336c",   # magenta-pink
    "VEC":            "#2f9e44",   # green
    "ncCM-AVC-like":  "#1971c2",   # blue
}
TRAITS = [
    "Valve_defects",
    "Atrioventricular_septal_defect",
    "Atrial_septal_defect",
    "Ventricular_septal_defect",
    "Malformation_of_the_outflow_tract",
    "HypertrophicCardiomyopathy",
    "DilatedCardiomyopathy",
    "PCGC_DeNovoVariants",
    "Single_ventricle_disease",
    "Familial_thoracic_aortic_aneurysm_and_aortic_dissection",
]
TRAIT_LABELS = {
    "Valve_defects":                                          "Valve defects",
    "Atrioventricular_septal_defect":                         "AV septal defect",
    "Atrial_septal_defect":                                   "Atrial septal defect",
    "Ventricular_septal_defect":                              "Ventricular septal defect",
    "Malformation_of_the_outflow_tract":                      "Outflow tract malf.",
    "HypertrophicCardiomyopathy":                             "Hypertrophic CM",
    "DilatedCardiomyopathy":                                  "Dilated CM",
    "PCGC_DeNovoVariants":                                    "PCGC de novo variants",
    "Single_ventricle_disease":                               "Single ventricle dz",
    "Familial_thoracic_aortic_aneurysm_and_aortic_dissection": "Thoracic aortic aneurysm",
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print(f"[load] spatial adata  : {SP_PATH.name}")
    sp_ad  = ad.read_h5ad(SP_PATH)
    print(f"        shape = {sp_ad.shape}, obs cols = {list(sp_ad.obs.columns)[:8]}...")

    print(f"[load] imputed adata  : {IMP_PATH.name}")
    imp_ad = ad.read_h5ad(IMP_PATH)
    print(f"        shape = {imp_ad.shape}")

    # Align imputed adata to spatial cell order via obs_names
    if not (sp_ad.obs_names == imp_ad.obs_names).all():
        print("[align] reindexing imputed adata to spatial obs_names ...")
        imp_ad = imp_ad[sp_ad.obs_names].copy()

    # Coords + section
    coords = sp_ad.obsm["X_spateo_update"]    # (N, 3) um
    sec    = sp_ad.obs["section"].astype(int).to_numpy()
    ct     = sp_ad.obs["celltype"].astype(str).to_numpy()

    # ------------------------------------------------------------------
    # Per-section enrichment table (for the supplementary TSV)
    # ------------------------------------------------------------------
    valve_mask_all = np.isin(ct, VALVE_CTS)
    overall_frac = valve_mask_all.mean()
    sec_unique, sec_counts = np.unique(sec, return_counts=True)
    rows = []
    for s, n in zip(sec_unique, sec_counts):
        m = sec == s
        n_valve = valve_mask_all[m].sum()
        f = n_valve / n if n else 0.0
        rows.append({
            "section":   int(s),
            "n_total":   int(n),
            "n_VIC":     int(((ct == "VIC")            & m).sum()),
            "n_VEC":     int(((ct == "VEC")            & m).sum()),
            "n_AVClike": int(((ct == "ncCM-AVC-like")  & m).sum()),
            "n_valve":   int(n_valve),
            "frac_valve": float(f),
            "log2_enrich": float(np.log2(f / overall_frac))
                           if (f > 0 and overall_frac > 0) else float("-inf"),
        })
    enr = pd.DataFrame(rows).sort_values("frac_valve", ascending=False)
    enr.to_csv(OUT / "section_valve_enrichment.tsv", sep="\t", index=False)
    print(f"[enrich] wrote {OUT/'section_valve_enrichment.tsv'}")
    print(enr.head(5).to_string(index=False))

    # ------------------------------------------------------------------
    # Subset to section 16
    # ------------------------------------------------------------------
    m16 = sec == SECTION
    n16 = int(m16.sum())
    assert n16 > 0, f"section {SECTION} not found"
    print(f"[section {SECTION}] n_cells = {n16}")
    xy   = coords[m16, :2]
    ct16 = ct[m16]

    # Sanity: report counts per valve type
    for vc in VALVE_CTS:
        print(f"  {vc}: {(ct16 == vc).sum()}")

    # ------------------------------------------------------------------
    # Trait -> gene panels
    # ------------------------------------------------------------------
    part = pd.read_csv(PART, sep="\t")
    avail = set(imp_ad.var_names)
    trait2genes: dict[str, list[str]] = {}
    for trait in TRAITS:
        # rows where the (semicolon-separated) traits column contains this trait
        mask = part["traits"].fillna("").apply(
            lambda s: trait in [t.strip() for t in s.split(";") if t.strip()]
        )
        genes = [g for g in part.loc[mask, "Gene"].tolist() if g in avail]
        trait2genes[trait] = genes
        print(f"  trait '{trait}': {len(genes)} genes available in imputed adata")

    # ------------------------------------------------------------------
    # Compute per-cell trait scores on section 16
    # ------------------------------------------------------------------
    imp16 = imp_ad[m16]
    X = imp16.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    # imputed values are already log1p-like (non-negative continuous);
    # if not, re-log here. We keep as-is to match how they were generated.
    var_idx = {g: i for i, g in enumerate(imp16.var_names)}

    trait_scores = np.zeros((n16, len(TRAITS)), dtype=np.float32)
    trait_z      = np.zeros_like(trait_scores)
    for j, trait in enumerate(TRAITS):
        idx = [var_idx[g] for g in trait2genes[trait]]
        if not idx:
            continue
        s = X[:, idx].mean(axis=1)
        trait_scores[:, j] = s
        # robust z-score: (x - median) / MAD * 1.4826 fall-back to std if MAD=0
        med = np.median(s)
        mad = np.median(np.abs(s - med))
        if mad > 0:
            trait_z[:, j] = (s - med) / (mad * 1.4826)
        else:
            sd = s.std()
            trait_z[:, j] = (s - s.mean()) / (sd if sd > 0 else 1.0)

    # ------------------------------------------------------------------
    # Save section-16 cell table
    # ------------------------------------------------------------------
    cells = pd.DataFrame({
        "cell_id":  sp_ad.obs_names[m16].to_numpy(),
        "x":        xy[:, 0],
        "y":        xy[:, 1],
        "celltype": ct16,
        "valve":    np.isin(ct16, VALVE_CTS),
    })
    for j, trait in enumerate(TRAITS):
        cells[f"score__{trait}"] = trait_scores[:, j]
        cells[f"z__{trait}"]     = trait_z[:, j]
    cells.to_csv(OUT / "section16_cells.tsv", sep="\t", index=False)
    print(f"[write] {OUT/'section16_cells.tsv'} ({len(cells)} rows)")

    # ------------------------------------------------------------------
    # Figure A: valve cell-type spatial map
    # ------------------------------------------------------------------
    print("[figA] valve cell-type map ...")
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=150)
    # Background: all cells faint gray
    other = ~np.isin(ct16, VALVE_CTS)
    ax.scatter(xy[other, 0], xy[other, 1],
               s=2.0, c="#d9d9d9", linewidths=0, alpha=0.8, rasterized=True)
    # Foreground: each valve cell type
    for vc in VALVE_CTS:
        m = ct16 == vc
        ax.scatter(xy[m, 0], xy[m, 1],
                   s=12, c=VALVE_COLORS[vc], linewidths=0,
                   alpha=0.95, label=f"{vc}  (n={int(m.sum())})",
                   rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()                # per prior feedback
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(f"PCW12 MERFISH — section {SECTION}\n"
                 f"valve-associated cell types (n={int(np.isin(ct16, VALVE_CTS).sum())} "
                 f"/ {n16}, {100*np.isin(ct16, VALVE_CTS).mean():.1f}%)")
    ax.legend(loc="best", frameon=True, framealpha=0.92, fontsize=10)
    for sp_ in ax.spines.values():
        sp_.set_color("#888")
    fig.tight_layout()
    fig.savefig(OUT / "section16_valve_celltypes.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "section16_valve_celltypes.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT/'section16_valve_celltypes.png'}")

    # ------------------------------------------------------------------
    # Figure B: 2x5 grid of trait expression panels
    # ------------------------------------------------------------------
    print("[figB] disease-trait panel grid ...")
    nrows, ncols = 2, 5
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 4.2 * nrows), dpi=150,
        sharex=True, sharey=True,
    )
    axes_flat = axes.ravel()

    # Outline coords for valve cells (for overlay on every panel)
    valve_mask16 = np.isin(ct16, VALVE_CTS)
    vx, vy = xy[valve_mask16, 0], xy[valve_mask16, 1]

    vmin, vmax = -2.0, 2.0
    cmap = "magma"

    # Order cells so high-z appear on top
    for k, trait in enumerate(TRAITS):
        ax = axes_flat[k]
        z = trait_z[:, k]
        order = np.argsort(z)            # low-to-high → high values plotted last
        sc = ax.scatter(
            xy[order, 0], xy[order, 1],
            c=z[order], cmap=cmap, vmin=vmin, vmax=vmax,
            s=3.0, linewidths=0, rasterized=True,
        )
        # Faint valve-cell outlines on top
        ax.scatter(vx, vy,
                   s=10, facecolors="none",
                   edgecolors="#a0e4ff", linewidths=0.45, alpha=0.55,
                   rasterized=True)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"{TRAIT_LABELS[trait]}\n(n={len(trait2genes[trait])} genes)",
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_color("#bbb")

    # Hide unused axes (none for 10/10) — but keep robust
    for k in range(len(TRAITS), nrows * ncols):
        axes_flat[k].axis("off")

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.64])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("trait z-score (robust)", fontsize=10)

    # Single legend element for valve outlines
    leg = [Line2D([0], [0], marker="o", linestyle="",
                  markerfacecolor="none", markeredgecolor="#1f9bd6",
                  markersize=8, label="VIC / VEC / ncCM-AVC-like cell")]
    fig.legend(handles=leg, loc="lower center", ncol=1,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        f"Section {SECTION}: per-trait disease-gene expression  "
        f"(mean log1p over panel, robust-z over n={n16} cells)",
        fontsize=13, y=0.995,
    )
    fig.subplots_adjust(left=0.03, right=0.91, top=0.93, bottom=0.07,
                        wspace=0.05, hspace=0.18)
    fig.savefig(OUT / "section16_disease_trait_panels.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "section16_disease_trait_panels.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT/'section16_disease_trait_panels.png'}")

    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
