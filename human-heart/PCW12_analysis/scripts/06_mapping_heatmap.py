"""
Heatmap of source-cell-type (single-cell) -> target-cell-type (spatial) mapping.

Each spot in the spatial AnnData carries:
  - obs['celltype']                : native spatial cell type (34 fine labels) = TARGET (columns)
  - obs['mapped_celltype']         : SC fine label assigned via argmax of OT transport (99 labels)
  - obs['mapped_coarse_celltype']  : SC coarse label (14)               = SOURCE (rows, primary)
  - obs['mapped_celltype_label']   : SC broad label (11)

We build column-normalized cross-tab heatmaps:
  rows = SC type (source),  cols = spatial type (target)
  cell value = fraction of spots labeled <col> that were mapped to <row>
            (each column sums to 1.0)

Outputs in figures/heatmaps/:
  mapping_coarseSC_x_spatialFine.png / .tsv   (14 x 34, primary)
  mapping_fineSC_x_spatialFine.png  / .tsv    (99 x 34, supplementary)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
H5AD = ROOT / "PCW12_analysis/data/adata_imputed_disease_genes.h5ad"
OUT = ROOT / "PCW12_analysis/figures/heatmaps"
OUT.mkdir(parents=True, exist_ok=True)


# ----- semantic ordering for the spatial fine labels ----------------------
def order_spatial_fine(labels):
    """Group spatial cell types into a meaningful left-to-right order."""
    groups = [
        ("vCM-LV", lambda s: s.startswith("vCM-LV")),
        ("vCM-RV", lambda s: s.startswith("vCM-RV")),
        ("vCM-other", lambda s: s.startswith("vCM-") and "LV" not in s and "RV" not in s),
        ("aCM",   lambda s: s.startswith("aCM")),
        ("CM-other", lambda s: s.startswith("CM") or s.endswith("CM")),
        ("Fibro", lambda s: "Fibro" in s or s == "VIC"),
        ("Endo",  lambda s: "Endocard" in s or "EPDC" in s),
        ("EC",    lambda s: s in ("BEC", "LEC") or "Endothel" in s or s.endswith("EC")),
        ("Mural", lambda s: s in ("VSMC", "Pericyte") or "SMC" in s or "Pericyt" in s),
        ("Epi",   lambda s: "Epicard" in s),
        ("Immune",lambda s: any(t in s for t in ("Macro","Mono","TCell","BCell","NK","Mast","Immune"))),
        ("Neural",lambda s: any(t in s for t in ("Neuron","Schwann","Glia","NC"))),
        ("Conduction", lambda s: "Conduct" in s or "AVN" in s or "SAN" in s),
    ]
    seen = set()
    ordered = []
    for _, pred in groups:
        bucket = sorted([l for l in labels if l not in seen and pred(l)])
        ordered.extend(bucket)
        seen.update(bucket)
    # remainder
    ordered.extend(sorted([l for l in labels if l not in seen]))
    return ordered


def order_sc_coarse(labels):
    preferred = ["VCM", "ACM", "CoreConductionCells", "TzConductionCells",
                 "FB", "Endocardial", "Endothelial", "MuralCells", "Epicardial",
                 "MyeloidCells", "LymphoidCells", "SympatheticNeuron",
                 "SchwannCells", "NC"]
    seen = set()
    out = [l for l in preferred if l in labels and not (l in seen or seen.add(l))]
    out.extend(sorted([l for l in labels if l not in seen]))
    return out


def order_sc_fine_by_coarse(fine_labels, coarse_for_fine, coarse_order):
    """Order fine SC labels grouped by their coarse parent following coarse_order."""
    fine_to_coarse = {f: coarse_for_fine.get(f, "zzz") for f in fine_labels}
    ck = {c: i for i, c in enumerate(coarse_order)}
    return sorted(fine_labels, key=lambda f: (ck.get(fine_to_coarse[f], 999), f))


# --------------------------------------------------------------------------
def col_normalize(df: pd.DataFrame) -> pd.DataFrame:
    s = df.sum(axis=0).replace(0, np.nan)
    return df.div(s, axis=1).fillna(0.0)


def plot_heatmap(M: pd.DataFrame, out_png: Path, *, title: str, figsize, annotate=False):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    cmap = mpl.cm.get_cmap("magma_r").copy()
    im = ax.imshow(M.values, cmap=cmap, aspect="auto", vmin=0, vmax=1.0)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(M.index, fontsize=8)
    ax.set_xlabel("Target spatial cell type (MERFISH `celltype`)", fontsize=10)
    ax.set_ylabel("Source single-cell type (mapped via moscot OT)", fontsize=10)
    ax.set_title(title, fontsize=11)

    # Faint grid between cells
    ax.set_xticks(np.arange(-0.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", length=0)

    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                if v >= 0.05:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v > 0.5 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Column-normalized fraction\n(per spatial cell type)", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def main():
    print("Reading obs ...")
    a = ad.read_h5ad(H5AD, backed="r")
    obs = a.obs[["celltype", "mapped_celltype",
                 "mapped_coarse_celltype", "mapped_celltype_label"]].astype(str).copy()
    print(f"  n_spots={len(obs)}")

    sp_fine = order_spatial_fine(obs["celltype"].unique().tolist())
    sc_coarse = order_sc_coarse(obs["mapped_coarse_celltype"].unique().tolist())

    # ---- Primary: coarse SC × fine spatial ----
    print("Building coarse-SC x spatial-fine cross-tab ...")
    ct = pd.crosstab(obs["mapped_coarse_celltype"], obs["celltype"])
    ct = ct.reindex(index=sc_coarse, columns=sp_fine, fill_value=0)
    ct.to_csv(OUT / "mapping_coarseSC_x_spatialFine_counts.tsv", sep="\t")
    M_coarse = col_normalize(ct)
    M_coarse.to_csv(OUT / "mapping_coarseSC_x_spatialFine.tsv", sep="\t",
                    float_format="%.4f")
    plot_heatmap(
        M_coarse,
        OUT / "mapping_coarseSC_x_spatialFine.png",
        title="Single-cell → spatial mapping  (coarse SC types × spatial fine types)\n"
              "column-normalized: each spatial cell type sums to 1",
        figsize=(14, 5),
        annotate=True,
    )

    # ---- Supplementary: fine SC × fine spatial ----
    print("Building fine-SC x spatial-fine cross-tab ...")
    # Build mapping fine_sc -> coarse_sc (most-common parent) for ordering
    f2c = (obs.groupby("mapped_celltype")["mapped_coarse_celltype"]
              .agg(lambda s: s.value_counts().index[0]).to_dict())
    sc_fine_all = obs["mapped_celltype"].unique().tolist()
    sc_fine = order_sc_fine_by_coarse(sc_fine_all, f2c, sc_coarse)

    ct2 = pd.crosstab(obs["mapped_celltype"], obs["celltype"])
    ct2 = ct2.reindex(index=sc_fine, columns=sp_fine, fill_value=0)
    ct2.to_csv(OUT / "mapping_fineSC_x_spatialFine_counts.tsv", sep="\t")
    M_fine = col_normalize(ct2)
    M_fine.to_csv(OUT / "mapping_fineSC_x_spatialFine.tsv", sep="\t",
                  float_format="%.4f")
    plot_heatmap(
        M_fine,
        OUT / "mapping_fineSC_x_spatialFine.png",
        title="Single-cell → spatial mapping  (fine SC types × spatial fine types)\n"
              "column-normalized: each spatial cell type sums to 1",
        figsize=(14, 18),
        annotate=False,
    )

    # ---- Diagnostic: top-1 SC per spatial type ----
    top = pd.DataFrame({
        "spatial_celltype": M_coarse.columns,
        "n_spots": ct.sum(axis=0).values,
        "top_SC_coarse": M_coarse.idxmax(axis=0).values,
        "top_frac": M_coarse.max(axis=0).values,
    })
    top.to_csv(OUT / "mapping_top_SC_per_spatial.tsv", sep="\t",
               index=False, float_format="%.3f")
    print("\nTop SC coarse per spatial type:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
