"""
Compare 3D imputed ATAC vs RNA disease gene signals.

Outputs (PCW12_analysis/figures/atac_vs_rna/):
  per_gene/<GENE>.png        3-panel triptych: ATAC | RNA | (ATAC_z - RNA_z)
  per_trait/<TRAIT>.png      same triptych at trait level (mean z-score)
  celltype_dotplot.png       2-panel dot plot (ATAC | RNA), x=gene, y=cell type
  gene_correlation.png       scatter (mean signal vs Pearson r), with hist
  gene_correlation.tsv       per-gene Pearson r, mean ATAC, mean RNA
  trait_correlation.tsv      per-trait Pearson r, mean ATAC, mean RNA
  _summary.md                short text summary of findings

Inputs:
  data/adata_imputed_atac.h5ad         (100 000 x 234, log-norm)
  data/adata_imputed_disease_genes.h5ad (100 000 x 221, log-norm)
  data/heart_disease_genes.tsv
  PCW12_analysis/data/highlight_genes.tsv (optional)
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
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
import pyvista as pv

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
sys.path.insert(0, str(ROOT / "PCW12_analysis/scripts"))
from _viz_utils import _coords_3d, DEFAULT_ELEV, DEFAULT_AZIM  # noqa: E402

ATAC_PATH = ROOT / "PCW12_analysis/data/adata_imputed_atac.h5ad"
RNA_PATH = ROOT / "PCW12_analysis/data/adata_imputed_disease_genes.h5ad"
DISEASE = ROOT / "data/heart_disease_genes.tsv"
HIGHLIGHTS = ROOT / "PCW12_analysis/data/highlight_genes.tsv"

OUT = ROOT / "PCW12_analysis/figures/atac_vs_rna"
OUT_GENE = OUT / "per_gene"
OUT_TRAIT = OUT / "per_trait"
for d in (OUT_GENE, OUT_TRAIT):
    d.mkdir(parents=True, exist_ok=True)

MIN_TRAIT_GENES = 3
HIGHLIGHT_DEFAULT = [
    "MYH7", "MYH6", "ACTC1", "ACTN2", "TNNT2", "MYL2",
    "NKX2-5", "TBX5", "GATA4", "MEF2C", "HAND1", "PITX2",
]


def log(m: str):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# -----------------------------------------------------------------------
# 3D rendering helpers (off-screen to numpy, no PNG written)
# -----------------------------------------------------------------------
def render_to_image(coords, expr, *, cmap="magma", clim=None,
                    title="", point_size=1.5, opacity=0.18,
                    window=(700, 700), text_color="#FFFFFF"):
    """Render a 3D scatter and return RGB ndarray."""
    if clim is None:
        pos = expr[expr > np.quantile(expr, 0.50)]
        if pos.size:
            lo = float(np.quantile(expr, 0.02))
            hi = float(np.quantile(expr, 0.98))
        else:
            lo, hi = float(expr.min()), float(expr.max())
        if hi <= lo:
            hi = lo + 1e-6
        clim = (lo, hi)
    cloud = pv.PolyData(coords)
    cloud["v"] = np.asarray(expr).ravel()
    p = pv.Plotter(off_screen=True, window_size=list(window))
    p.set_background("black")
    p.add_points(cloud, scalars="v", cmap=cmap, clim=clim,
                 point_size=point_size, opacity=opacity,
                 render_points_as_spheres=False,
                 scalar_bar_args={
                     "title": "", "color": text_color,
                     "n_colors": 20, "n_labels": 4,
                     "label_font_size": 10, "title_font_size": 12})
    if title:
        p.add_text(title, font_size=12, color=text_color, position="upper_left")
    p.camera_position = "iso"
    p.camera.Elevation(DEFAULT_ELEV)
    p.camera.Azimuth(DEFAULT_AZIM)
    img = p.screenshot(return_img=True)
    p.close()
    return img


def triptych(coords, atac_v, rna_v, out_png: Path, label: str):
    """Render ATAC | RNA | diff (ATAC_z - RNA_z) into a single PNG."""
    # robust normalisation for the diff panel: z-score each then subtract
    def _z(a):
        a = np.asarray(a, dtype=float)
        return (a - a.mean()) / (a.std() + 1e-9)
    diff = _z(atac_v) - _z(rna_v)

    img_atac = render_to_image(coords, atac_v, cmap="magma",
                               title=f"{label}  ATAC")
    img_rna = render_to_image(coords, rna_v, cmap="magma",
                              title=f"{label}  RNA")
    # diverging
    dlim = float(np.quantile(np.abs(diff), 0.98))
    img_diff = render_to_image(coords, diff, cmap="coolwarm",
                               clim=(-dlim, dlim),
                               title=f"{label}  ATAC_z - RNA_z")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="white")
    for ax, im, sub in zip(axes, [img_atac, img_rna, img_diff],
                           ["ATAC (imputed)", "RNA (imputed)",
                            "ATAC_z minus RNA_z"]):
        ax.imshow(im)
        ax.set_title(sub, fontsize=12)
        ax.axis("off")
    fig.suptitle(label, fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Cell-type dot plot
# -----------------------------------------------------------------------
def celltype_dotplot(atac_X, rna_X, gene_names, celltype, genes_subset,
                     out_png: Path):
    """Two-panel dot plot for ATAC and RNA on the same gene/celltype grid."""
    ct = pd.Series(celltype, name="ct")
    ct_order = sorted(ct.dropna().unique().tolist())
    g_idx = [i for i, g in enumerate(gene_names) if g in genes_subset]
    g_names = [gene_names[i] for i in g_idx]
    # order gene cols alphabetically
    sort_perm = np.argsort(g_names)
    g_idx = [g_idx[i] for i in sort_perm]
    g_names = [g_names[i] for i in sort_perm]

    def _dot_matrix(X):
        # mean expression and fraction expressing per (celltype, gene)
        mean_m = np.zeros((len(ct_order), len(g_idx)), dtype=float)
        frac_m = np.zeros_like(mean_m)
        for i, c in enumerate(ct_order):
            mask = (ct == c).to_numpy()
            sub = X[mask][:, g_idx]
            mean_m[i] = sub.mean(axis=0)
            frac_m[i] = (sub > 0).mean(axis=0)
        return mean_m, frac_m

    a_mean, a_frac = _dot_matrix(atac_X)
    r_mean, r_frac = _dot_matrix(rna_X)

    # scale mean per modality to [0, 1] per column (gene) for visual comparability
    def _col_norm(M):
        M = M.copy()
        cmax = M.max(axis=0, keepdims=True)
        cmax[cmax <= 0] = 1.0
        return M / cmax

    a_mean_n = _col_norm(a_mean)
    r_mean_n = _col_norm(r_mean)

    fig = plt.figure(figsize=(max(10, 0.45 * len(g_names) * 2 + 4),
                              0.32 * len(ct_order) + 3.5),
                     facecolor="white")
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1], sharey=ax_a)
    cax = fig.add_subplot(gs[0, 2])

    def _plot(ax, mean_n, frac, title):
        x, y = np.meshgrid(np.arange(len(g_names)), np.arange(len(ct_order)))
        sizes = (frac * 200).clip(0, 200)
        sc = ax.scatter(x.ravel(), y.ravel(),
                        s=sizes.ravel(),
                        c=mean_n.ravel(),
                        cmap="magma", vmin=0, vmax=1,
                        edgecolors="none")
        ax.set_xticks(np.arange(len(g_names)))
        ax.set_xticklabels(g_names, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(ct_order)))
        ax.set_yticklabels(ct_order, fontsize=8)
        ax.set_title(title, fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, axis="both", color="#dddddd", lw=0.4)
        ax.set_axisbelow(True)
        return sc

    sc = _plot(ax_a, a_mean_n, a_frac, "ATAC (imputed)")
    _plot(ax_r, r_mean_n, r_frac, "RNA (imputed)")
    ax_r.tick_params(labelleft=False)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("mean (col-normalised)")

    # legend for sizes
    handles = [plt.scatter([], [], s=f * 200, c="grey",
                           label=f"{int(f*100)}%", edgecolors="none")
               for f in (0.1, 0.25, 0.5, 0.75, 1.0)]
    ax_a.legend(handles=handles, title="frac > 0",
                loc="upper left", bbox_to_anchor=(0, -0.18),
                ncol=5, fontsize=8, frameon=False)

    fig.suptitle("Cell-type signal: ATAC vs RNA (imputed)",
                 fontsize=14, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    log("Loading imputed ATAC + RNA")
    A = ad.read_h5ad(ATAC_PATH)
    R = ad.read_h5ad(RNA_PATH)
    assert (A.obs_names == R.obs_names).all(), "obs ordering must match"
    shared = sorted(set(A.var_names) & set(R.var_names))
    log(f"Shared genes: {len(shared)}")

    # Align both to shared gene order
    A = A[:, shared].copy()
    R = R[:, shared].copy()

    aX = A.X.toarray() if hasattr(A.X, "toarray") else np.asarray(A.X)
    rX = R.X.toarray() if hasattr(R.X, "toarray") else np.asarray(R.X)
    aX = aX.astype(np.float32)
    rX = rX.astype(np.float32)
    log(f"  ATAC range {aX.min():.2f}..{aX.max():.2f} mean {aX.mean():.3f}")
    log(f"  RNA  range {rX.min():.2f}..{rX.max():.2f} mean {rX.mean():.3f}")

    # Cell type label - use coarse for the dot plot (14 vs 99 fine types)
    if "mapped_coarse_celltype" in R.obs.columns:
        ct = R.obs["mapped_coarse_celltype"].astype(str).values
        ct_src = "RNA.mapped_coarse_celltype"
    elif "mapped_celltype" in R.obs.columns:
        ct = R.obs["mapped_celltype"].astype(str).values
        ct_src = "RNA.mapped_celltype"
    else:
        ct = R.obs["celltype"].astype(str).values
        ct_src = "RNA.celltype"
    log(f"Cell-type labels from {ct_src}: {len(set(ct))} unique")

    coords = _coords_3d(A, "X_spateo_update")  # apex flipped down

    # ---- Highlight genes for triptychs and dot plot ----
    if HIGHLIGHTS.exists():
        try:
            hi = pd.read_csv(HIGHLIGHTS, sep="\t")
            if "highlight" in hi.columns:
                hi = hi[hi["highlight"].astype(bool)]
            highlight = sorted(set(hi["Gene"].astype(str)) & set(shared))
        except Exception:
            highlight = []
    else:
        highlight = []
    if not highlight:
        highlight = [g for g in HIGHLIGHT_DEFAULT if g in shared]
    log(f"Highlight genes: {len(highlight)} -> {highlight}")

    # ----------------------------------------------------
    # Gene-level Pearson correlation (across cells)
    # ----------------------------------------------------
    log("Computing per-gene Pearson r(ATAC, RNA) across 100k cells")
    rs = np.zeros(len(shared), dtype=float)
    for j in range(len(shared)):
        a = aX[:, j]; r = rX[:, j]
        if a.std() < 1e-9 or r.std() < 1e-9:
            rs[j] = np.nan
        else:
            rs[j] = pearsonr(a, r)[0]
    gcorr = pd.DataFrame({
        "gene": shared,
        "pearson_r": rs,
        "atac_mean": aX.mean(axis=0),
        "atac_frac_pos": (aX > 0).mean(axis=0),
        "rna_mean": rX.mean(axis=0),
        "rna_frac_pos": (rX > 0).mean(axis=0),
    }).sort_values("pearson_r", ascending=False)
    gcorr.to_csv(OUT / "gene_correlation.tsv", sep="\t", index=False,
                 float_format="%.4f")
    log(f"  median r = {gcorr['pearson_r'].median():.3f}; "
        f"top: {gcorr.head(3)[['gene','pearson_r']].values.tolist()}")

    # Correlation scatter / hist
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4),
                           gridspec_kw={"width_ratios": [1.2, 1]},
                           facecolor="white")
    ax[0].scatter(gcorr["atac_mean"], gcorr["pearson_r"],
                  s=12, alpha=0.7, color="#3a7bd5")
    for _, row in gcorr.head(8).iterrows():
        ax[0].annotate(row["gene"], (row["atac_mean"], row["pearson_r"]),
                       fontsize=7, alpha=0.85)
    for _, row in gcorr.tail(8).iterrows():
        ax[0].annotate(row["gene"], (row["atac_mean"], row["pearson_r"]),
                       fontsize=7, alpha=0.85, color="#a33")
    ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].set_xlabel("ATAC mean signal")
    ax[0].set_ylabel("Pearson r (ATAC vs RNA, across cells)")
    ax[0].set_title("Per-gene spatial concordance")
    ax[1].hist(gcorr["pearson_r"].dropna(), bins=40,
               color="#3a7bd5", alpha=0.85)
    ax[1].axvline(gcorr["pearson_r"].median(), color="k", ls="--",
                  label=f"median={gcorr['pearson_r'].median():.2f}")
    ax[1].set_xlabel("Pearson r")
    ax[1].set_ylabel("# genes")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "gene_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------
    # Per-gene 3D triptychs (highlights only)
    # ----------------------------------------------------
    log(f"Rendering {len(highlight)} per-gene triptychs ...")
    for g in highlight:
        j = shared.index(g)
        triptych(coords, aX[:, j], rX[:, j],
                 OUT_GENE / f"{g}.png", label=g)
        log(f"  {g}")

    # ----------------------------------------------------
    # Trait aggregation (mean z-score)
    # ----------------------------------------------------
    log("Loading disease trait table")
    disease = pd.read_csv(DISEASE, sep="\t")
    disease = disease[disease["Gene"].isin(shared)]
    log(f"  trait rows for shared genes: {len(disease)}; "
        f"unique traits: {disease['Trait'].nunique()}")

    # z-score each gene
    az = (aX - aX.mean(0)) / (aX.std(0) + 1e-9)
    rz = (rX - rX.mean(0)) / (rX.std(0) + 1e-9)
    g2i = {g: i for i, g in enumerate(shared)}

    trait_rows = []
    panel_data = []  # for summary panel
    for tr, sub in disease.groupby("Trait"):
        idx = [g2i[g] for g in sub["Gene"].unique() if g in g2i]
        if len(idx) < MIN_TRAIT_GENES:
            log(f"  [{tr}] only {len(idx)} genes; skipping")
            continue
        a_score = az[:, idx].mean(axis=1)
        r_score = rz[:, idx].mean(axis=1)
        rho = pearsonr(a_score, r_score)[0]
        trait_rows.append({"Trait": tr, "n_genes": len(idx),
                           "pearson_r": rho,
                           "atac_mean_z": float(a_score.mean()),
                           "rna_mean_z": float(r_score.mean())})
        # render triptych
        out_png = OUT_TRAIT / f"{tr}.png"
        log(f"  [{tr}] n={len(idx)} r={rho:.3f} -> {out_png.name}")
        triptych(coords, a_score, r_score, out_png,
                 label=f"{tr.replace('_',' ')}  (n={len(idx)})")
        panel_data.append((tr, out_png, len(idx), rho))

    pd.DataFrame(trait_rows).sort_values("pearson_r", ascending=False).to_csv(
        OUT / "trait_correlation.tsv", sep="\t", index=False,
        float_format="%.4f")

    # Summary panel of trait triptychs
    if panel_data:
        n = len(panel_data)
        fig, axes = plt.subplots(n, 1, figsize=(15, 5.0 * n),
                                 facecolor="white")
        if n == 1:
            axes = [axes]
        for ax, (tr, png, n_g, rho) in zip(axes, panel_data):
            ax.imshow(mpimg.imread(png))
            ax.set_title(f"{tr}  (n={n_g}, r={rho:.2f})", fontsize=11)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / "_trait_summary.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    # ----------------------------------------------------
    # Cell-type dot plot
    # ----------------------------------------------------
    log("Rendering cell-type dot plot")
    celltype_dotplot(aX, rX, shared, ct,
                     genes_subset=set(highlight),
                     out_png=OUT / "celltype_dotplot.png")

    # ----------------------------------------------------
    # Quick text summary
    # ----------------------------------------------------
    with open(OUT / "_summary.md", "w") as f:
        f.write("# ATAC vs RNA comparison — summary\n\n")
        f.write(f"- shared genes: {len(shared)}\n")
        f.write(f"- highlight genes: {len(highlight)} ({', '.join(highlight)})\n")
        f.write(f"- median per-gene Pearson r (across 100k cells): "
                f"{gcorr['pearson_r'].median():.3f}\n")
        f.write(f"- mean per-gene Pearson r: "
                f"{gcorr['pearson_r'].mean():.3f}\n")
        f.write(f"- top 5 concordant genes: "
                f"{gcorr.head(5)[['gene','pearson_r']].to_dict('records')}\n")
        f.write(f"- bottom 5 (most divergent) genes: "
                f"{gcorr.tail(5)[['gene','pearson_r']].to_dict('records')}\n\n")
        f.write("## Per-trait\n\n")
        df_t = (pd.DataFrame(trait_rows)
                  .sort_values("pearson_r", ascending=False))
        # write as plain markdown table (no tabulate dep)
        cols = list(df_t.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, row in df_t.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")
    log("Done.")


if __name__ == "__main__":
    main()
