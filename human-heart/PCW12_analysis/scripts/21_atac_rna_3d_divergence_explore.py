"""
Phase 21: Exploratory analysis of differences between ATAC and RNA 3D
spatial patterns.  Companion to Phase 18 (which produced per-gene/per-trait
triptychs and a global per-gene Pearson r).

Four analyses:
  A. Spatial divergence atlas (3D)
       - Per-cell panel-mean |ATAC_z - RNA_z| -> single 3D map.
       - Per-module signed (ATAC - RNA) -> 6-panel grid.
  B. Priming-asymmetry quadrant analysis
       - Per (gene, cell): Primed = aZ>1 & rZ<0; Expressed-only = rZ>1 & aZ<0.
       - Per-gene fractions ranked; per-celltype heatmap (panel genes).
  C. Within-cell-type Pearson r
       - For each coarse celltype with >=200 cells, vectorised per-gene r.
       - Clustermap of [genes x celltypes].
       - Within-VCM 3D triptychs for genes with largest |r_within_VCM - r_global|.
  D. Spatial spread (radius of gyration)
       - For each gene, top-10% cells by signal, compute centroid + r_g in 3D.
       - Scatter r_g(ATAC) vs r_g(RNA); diagonal = matched spread.

Outputs (PCW12_analysis/figures/atac_vs_rna_explore/):
  A_panel_divergence_3d.png
  A_module_divergence_3d_grid.png
  B_gene_quadrant_ranking.png
  B_celltype_quadrant_heatmap.png
  B_gene_quadrant_fractions.tsv
  C_within_celltype_pearson_clustermap.png
  C_within_celltype_pearson.tsv
  C_within_vcm_triptych_<GENE>.png   (4 selected)
  D_atac_vs_rna_spread.png
  D_gene_spread.tsv
  _explore_summary.md
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pyvista as pv

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
sys.path.insert(0, str(ROOT / "PCW12_analysis/scripts"))
from _viz_utils import _coords_3d, DEFAULT_ELEV, DEFAULT_AZIM  # noqa: E402

ATAC_PATH = ROOT / "PCW12_analysis/data/adata_imputed_atac.h5ad"
RNA_PATH  = ROOT / "PCW12_analysis/data/adata_imputed_disease_genes.h5ad"
OUT       = ROOT / "PCW12_analysis/figures/atac_vs_rna_explore"
OUT.mkdir(parents=True, exist_ok=True)

CT_KEY = "mapped_coarse_celltype"          # 14 RNA-derived coarse types
MIN_CT_CELLS = 200                          # for analysis C

# Same 6-module curated panel as Phase 19/20
GENE_GROUPS: dict[str, list[str]] = {
    "Cardiac TF (CHD)": [
        "GATA4", "GATA6", "NKX2-5", "TBX5", "TBX20", "TBX1",
        "HAND1", "HAND2", "MEIS2", "PITX2",
    ],
    "Sarcomere / CM structural": [
        "MYH6", "MYH7", "MYBPC3", "TNNT2", "TNNI3", "TNNC1",
        "ACTC1", "ACTN2", "TPM1", "TTN", "MYL2", "MYL3",
        "DES", "VCL", "PLN",
    ],
    "Conduction / channel":      ["SCN5A", "GJA1"],
    "Notch / valve":             ["NOTCH1", "JAG1", "DLL4"],
    "Aortic / smooth muscle":    ["MYH11", "ACTA2", "FBN1", "ELN"],
    "Chromatin (PCGC)":          ["CHD7", "KMT2D", "CHD4"],
}
GROUP_COLORS = {
    "Cardiac TF (CHD)":          "#E64B35",
    "Sarcomere / CM structural": "#4DBBD5",
    "Conduction / channel":      "#00A087",
    "Notch / valve":             "#F39B7F",
    "Aortic / smooth muscle":    "#8491B4",
    "Chromatin (PCGC)":          "#B09C85",
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _to_dense(x) -> np.ndarray:
    return (x.toarray() if hasattr(x, "toarray") else np.asarray(x)).astype(np.float32)


# ----------------------------------------------------------------------
# Vectorised Pearson r between matched columns of two equal-shape arrays.
# ----------------------------------------------------------------------
def vec_pearson_cols(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-column Pearson r between A and B (both n x k).  Returns (k,)."""
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    sa = (A * A).sum(axis=0)
    sb = (B * B).sum(axis=0)
    sab = (A * B).sum(axis=0)
    denom = np.sqrt(sa * sb) + 1e-12
    out = sab / denom
    out[(sa < 1e-12) | (sb < 1e-12)] = np.nan
    return out


# ----------------------------------------------------------------------
# 3D rendering helpers (off-screen, return ndarray; same as Phase 18).
# ----------------------------------------------------------------------
def render_to_image(coords, expr, *, cmap="magma", clim=None, title="",
                    point_size=1.5, opacity=0.18, window=(700, 700),
                    text_color="#FFFFFF"):
    if clim is None:
        lo = float(np.quantile(expr, 0.02))
        hi = float(np.quantile(expr, 0.98))
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
                 scalar_bar_args={"title": "", "color": text_color,
                                  "n_colors": 20, "n_labels": 4,
                                  "label_font_size": 10, "title_font_size": 12})
    if title:
        p.add_text(title, font_size=11, color=text_color, position="upper_left")
    p.camera_position = "iso"
    p.camera.Elevation(DEFAULT_ELEV)
    p.camera.Azimuth(DEFAULT_AZIM)
    img = p.screenshot(return_img=True)
    p.close()
    return img


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    log("Loading imputed ATAC + RNA")
    A = ad.read_h5ad(ATAC_PATH)
    R = ad.read_h5ad(RNA_PATH)
    assert (A.obs_names == R.obs_names).all(), "obs ordering must match"

    shared = sorted(set(A.var_names) & set(R.var_names))
    log(f"Shared genes: {len(shared)}")
    A = A[:, shared].copy()
    R = R[:, shared].copy()

    aX = _to_dense(A.X)
    rX = _to_dense(R.X)
    log(f"  ATAC mean={aX.mean():.3f} std={aX.std():.3f}")
    log(f"  RNA  mean={rX.mean():.3f} std={rX.std():.3f}")

    # Per-gene z-scores across cells (ddof=0 default).
    aZ = (aX - aX.mean(0)) / (aX.std(0) + 1e-9)
    rZ = (rX - rX.mean(0)) / (rX.std(0) + 1e-9)
    aZ = aZ.astype(np.float32); rZ = rZ.astype(np.float32)

    # Coarse cell type from RNA AnnData
    ct = R.obs[CT_KEY].astype(str).values
    log(f"Coarse cell types: {pd.Series(ct).value_counts().to_dict()}")

    # 3D coords (apex flipped down)
    coords = _coords_3d(A, "X_spateo_update")

    # Disease panel: drop genes absent from shared universe.
    panel: list[str] = []
    gene_to_group: dict[str, str] = {}
    for grp, glist in GENE_GROUPS.items():
        for g in glist:
            if g in shared:
                panel.append(g)
                gene_to_group[g] = grp
    panel_idx = np.array([shared.index(g) for g in panel])
    log(f"Disease panel: {len(panel)} genes")

    g2i = {g: i for i, g in enumerate(shared)}

    # ==================================================================
    # ANALYSIS A — Spatial divergence atlas (3D)
    # ==================================================================
    log("[A] Spatial divergence atlas")

    panel_div = np.abs(aZ[:, panel_idx] - rZ[:, panel_idx]).mean(axis=1)
    log(f"  per-cell panel divergence: mean={panel_div.mean():.3f} "
        f"q90={np.quantile(panel_div, 0.9):.3f} max={panel_div.max():.3f}")

    img_panel = render_to_image(
        coords, panel_div, cmap="magma",
        clim=(float(np.quantile(panel_div, 0.05)),
              float(np.quantile(panel_div, 0.98))),
        title="Panel mean |ATAC_z - RNA_z|",
        window=(900, 900),
    )
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    ax.imshow(img_panel); ax.axis("off")
    ax.set_title(f"ATAC–RNA spatial divergence (panel mean of "
                 f"|z-score difference|, {len(panel)} disease genes)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "A_panel_divergence_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {OUT/'A_panel_divergence_3d.png'}")

    # Per-module signed (ATAC - RNA), 6-panel grid
    module_imgs = {}
    for grp in GENE_GROUPS:
        idx = [g2i[g] for g in panel if gene_to_group[g] == grp]
        if not idx:
            continue
        idx = np.array(idx)
        signed = (aZ[:, idx].mean(1) - rZ[:, idx].mean(1))
        q = float(np.quantile(np.abs(signed), 0.98))
        img = render_to_image(
            coords, signed, cmap="coolwarm", clim=(-q, q),
            title=f"{grp}\n(red = ATAC > RNA, blue = RNA > ATAC)",
            window=(700, 700),
        )
        module_imgs[grp] = img

    n = len(module_imgs)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), facecolor="white")
    axes = np.atleast_2d(axes).ravel()
    for ax, (grp, im) in zip(axes, module_imgs.items()):
        ax.imshow(im); ax.axis("off")
        ax.set_title(grp, fontsize=11)
    for ax in axes[len(module_imgs):]:
        ax.axis("off")
    fig.suptitle("Per-module signed ATAC – RNA divergence in 3D "
                 "(red = primed [ATAC>RNA], blue = expressed-without-priming [RNA>ATAC])",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "A_module_divergence_3d_grid.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {OUT/'A_module_divergence_3d_grid.png'}")

    # ==================================================================
    # ANALYSIS B — Priming-asymmetry quadrants
    # ==================================================================
    log("[B] Priming-asymmetry quadrant analysis")

    primed     = (aZ > 1.0) & (rZ < 0.0)              # high ATAC, low RNA
    expr_only  = (rZ > 1.0) & (aZ < 0.0)              # high RNA, low ATAC
    primed_frac    = primed.mean(axis=0)               # per gene
    expr_only_frac = expr_only.mean(axis=0)

    Bdf = pd.DataFrame({
        "gene":             shared,
        "primed_frac":      primed_frac,
        "expr_only_frac":   expr_only_frac,
        "asymmetry":        primed_frac - expr_only_frac,
        "in_panel":         [g in panel for g in shared],
        "module":           [gene_to_group.get(g, "") for g in shared],
    })
    Bdf.to_csv(OUT / "B_gene_quadrant_fractions.tsv",
               sep="\t", index=False, float_format="%.4f")

    top_primed = Bdf.sort_values("primed_frac", ascending=False).head(15)
    top_expr   = Bdf.sort_values("expr_only_frac", ascending=False).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    for ax, df_, title, color in zip(
        axes, [top_primed, top_expr],
        ["Top primed-only (ATAC>1, RNA<0)",
         "Top expressed-without-priming (RNA>1, ATAC<0)"],
        ["#d8554f", "#3d6db6"],
    ):
        df_ = df_.iloc[::-1]                # so largest is at top
        bar_colors = [GROUP_COLORS[m] if m in GROUP_COLORS else color
                      for m in df_["module"]]
        ax.barh(df_["gene"], df_["primed_frac" if "primed" in title else "expr_only_frac"],
                color=bar_colors, edgecolor="white")
        ax.set_xlabel("fraction of 100k cells")
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [mpatches.Patch(color=c, label=k) for k, c in GROUP_COLORS.items()]
    handles.append(mpatches.Patch(color="#888", label="non-panel gene"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-gene priming asymmetry across all cells", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT / "B_gene_quadrant_ranking.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {OUT/'B_gene_quadrant_ranking.png'}")

    # Per-celltype heatmap on the disease panel (n_celltypes x n_panel_genes)
    # Two side-by-side panels: primed-frac and expr-only-frac.
    ct_series = pd.Series(ct)
    ct_keep = ct_series.value_counts()
    ct_order = ct_keep[ct_keep >= MIN_CT_CELLS].index.tolist()
    log(f"  cell types kept (>= {MIN_CT_CELLS} cells): {ct_order}")

    P = np.zeros((len(ct_order), len(panel_idx)), dtype=float)
    Eo = np.zeros_like(P)
    for i, c in enumerate(ct_order):
        mask = (ct_series == c).to_numpy()
        P[i]  = primed[mask][:, panel_idx].mean(axis=0)
        Eo[i] = expr_only[mask][:, panel_idx].mean(axis=0)

    Pdf  = pd.DataFrame(P,  index=ct_order, columns=panel)
    Eodf = pd.DataFrame(Eo, index=ct_order, columns=panel)

    # column order = group order, then alphabetical within group
    col_order = sorted(panel, key=lambda g: (list(GENE_GROUPS).index(gene_to_group[g]), g))
    Pdf = Pdf[col_order]; Eodf = Eodf[col_order]

    fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.32 * len(panel) + 4), 9),
                             facecolor="white", gridspec_kw={"hspace": 0.55})
    vmax = max(Pdf.values.max(), Eodf.values.max(), 0.05)
    for ax, M, title in zip(
        axes, [Pdf, Eodf],
        ["Primed only — fraction of cells with ATAC_z > 1 & RNA_z < 0",
         "Expressed only — fraction of cells with RNA_z > 1 & ATAC_z < 0"],
    ):
        sns.heatmap(M, ax=ax, cmap="magma", vmin=0, vmax=vmax,
                    cbar_kws={"label": "cell fraction"}, linewidths=0.0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
        # gene-level coloured tick labels
        for label in ax.get_xticklabels():
            g_ = label.get_text()
            label.set_color(GROUP_COLORS[gene_to_group[g_]])
            label.set_style("italic")
    fig.suptitle("Priming-asymmetry per cell type (panel genes)",
                 fontsize=13, y=0.99)
    fig.savefig(OUT / "B_celltype_quadrant_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {OUT/'B_celltype_quadrant_heatmap.png'}")

    # ==================================================================
    # ANALYSIS C — Within-cell-type Pearson r
    # ==================================================================
    log("[C] Within-cell-type Pearson r")

    r_global = vec_pearson_cols(aX, rX)            # length n_genes
    r_within = np.full((len(shared), len(ct_order)), np.nan, dtype=float)
    for j, c in enumerate(ct_order):
        m = (ct_series == c).to_numpy()
        r_within[:, j] = vec_pearson_cols(aX[m], rX[m])
        log(f"  {c:18s} n={m.sum():5d}  median r={np.nanmedian(r_within[:, j]):.3f}")

    Cdf = pd.DataFrame(r_within, index=shared, columns=ct_order)
    Cdf.insert(0, "r_global", r_global)
    Cdf.insert(1, "in_panel", [g in panel for g in shared])
    Cdf.insert(2, "module",   [gene_to_group.get(g, "") for g in shared])
    Cdf.to_csv(OUT / "C_within_celltype_pearson.tsv",
               sep="\t", float_format="%.4f")

    # Clustermap on the matrix [genes (drop all-NaN) x celltypes].
    M = Cdf[ct_order].copy()
    M = M.dropna(how="all").fillna(0.0)
    # Limit to genes that vary across celltypes (std > eps); keep all panel genes.
    keep_mask = (M.std(axis=1) > 0.05) | M.index.isin(panel)
    M = M.loc[keep_mask]
    log(f"  clustermap on {M.shape[0]} genes x {M.shape[1]} celltypes")

    row_colors = pd.Series(
        {g: GROUP_COLORS[gene_to_group[g]] if g in gene_to_group else "#dddddd"
         for g in M.index}, name="Module")

    g = sns.clustermap(
        M, cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
        method="average", metric="euclidean",
        figsize=(max(9, 0.55 * M.shape[1] + 5),
                 max(8, 0.025 * M.shape[0] + 4)),
        row_colors=row_colors,
        yticklabels=False,                          # too many to label
        xticklabels=True,
        cbar_kws={"label": "Pearson r (ATAC vs RNA, within cell type)"},
        dendrogram_ratio=(0.10, 0.18),
        colors_ratio=(0.018, 0.04),
        cbar_pos=(1.02, 0.35, 0.018, 0.30),
    )
    g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
    plt.setp(g.ax_heatmap.get_xticklabels(),
             rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    g.fig.suptitle(
        "Within-cell-type ATAC↔RNA Pearson r per gene\n"
        "(rows = genes; row strip = disease module; grey = non-panel gene)",
        fontsize=12, y=1.02,
    )
    legend_h = [mpatches.Patch(color=c, label=k) for k, c in GROUP_COLORS.items()]
    legend_h.append(mpatches.Patch(color="#dddddd", label="non-panel"))
    g.fig.legend(handles=legend_h, title="Module", loc="upper left",
                 bbox_to_anchor=(1.02, 0.30), frameon=False,
                 fontsize=8, title_fontsize=9)
    g.fig.savefig(OUT / "C_within_celltype_pearson_clustermap.png",
                  dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    log(f"  wrote {OUT/'C_within_celltype_pearson_clustermap.png'}")

    # Pick within-VCM divergent genes for triptychs:
    # genes that have decent global r but low (or strongly different) within-VCM r.
    if "VCM" in ct_order:
        vcm_r = Cdf["VCM"]
        delta = r_global - vcm_r            # positive = global concordant, within-VCM divergent
        # require gene to be reasonably expressed
        mean_signal = (aX.mean(0) + rX.mean(0)) * 0.5
        score = delta.where(mean_signal > 0.05, np.nan)
        cand = score.dropna().sort_values(ascending=False).head(8).index.tolist()
        # Keep top 4
        triptych_genes = cand[:4]
        log(f"  within-VCM divergent triptych genes: {triptych_genes} "
            f"(delta_r = global - VCM)")

        m_vcm = (ct_series == "VCM").to_numpy()
        coords_vcm = coords[m_vcm]
        for gname in triptych_genes:
            j = g2i[gname]
            a_v = aX[m_vcm, j]; r_v = rX[m_vcm, j]
            # Local z for diff
            az_v = (a_v - a_v.mean()) / (a_v.std() + 1e-9)
            rz_v = (r_v - r_v.mean()) / (r_v.std() + 1e-9)
            diff = az_v - rz_v
            dlim = float(np.quantile(np.abs(diff), 0.98))
            img_a = render_to_image(coords_vcm, a_v, cmap="magma",
                                    title=f"{gname}  ATAC (VCM)")
            img_r = render_to_image(coords_vcm, r_v, cmap="magma",
                                    title=f"{gname}  RNA (VCM)")
            img_d = render_to_image(coords_vcm, diff, cmap="coolwarm",
                                    clim=(-dlim, dlim),
                                    title=f"{gname}  ATAC_z - RNA_z (VCM)")
            fig, axx = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="white")
            for a_, im, sub in zip(
                axx, [img_a, img_r, img_d],
                [f"ATAC (within VCM)", f"RNA (within VCM)",
                 "ATAC_z – RNA_z (within VCM)"]):
                a_.imshow(im); a_.axis("off"); a_.set_title(sub, fontsize=11)
            fig.suptitle(
                f"{gname}  —  global r={r_global[j]:.2f}  "
                f"VCM r={vcm_r[gname]:.2f}  Δ={r_global[j] - vcm_r[gname]:+.2f}",
                fontsize=12, y=1.01,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(OUT / f"C_within_vcm_triptych_{gname}.png",
                        dpi=130, bbox_inches="tight")
            plt.close(fig)
            log(f"  wrote {OUT/f'C_within_vcm_triptych_{gname}.png'}")

    # ==================================================================
    # ANALYSIS D — Spatial spread (radius of gyration)
    # ==================================================================
    log("[D] Spatial spread (radius of gyration)")

    def radius_of_gyration(coords_sub: np.ndarray) -> tuple[np.ndarray, float]:
        c = coords_sub.mean(axis=0)
        rg = float(np.sqrt(((coords_sub - c) ** 2).sum(axis=1).mean()))
        return c, rg

    def per_modality_spread(X: np.ndarray) -> pd.DataFrame:
        rows = []
        for j, gname in enumerate(shared):
            v = X[:, j]
            thr = np.quantile(v, 0.90)
            sel = v > thr
            if sel.sum() < 50:
                rows.append({"gene": gname, "centroid_x": np.nan, "centroid_y": np.nan,
                             "centroid_z": np.nan, "rg": np.nan, "n_top": int(sel.sum())})
                continue
            c, rg = radius_of_gyration(coords[sel])
            rows.append({"gene": gname, "centroid_x": c[0], "centroid_y": c[1],
                         "centroid_z": c[2], "rg": rg, "n_top": int(sel.sum())})
        return pd.DataFrame(rows).set_index("gene")

    Sa = per_modality_spread(aX).add_prefix("atac_")
    Sr = per_modality_spread(rX).add_prefix("rna_")
    S = Sa.join(Sr)
    # centroid distance between modalities (per gene)
    dx = S["atac_centroid_x"] - S["rna_centroid_x"]
    dy = S["atac_centroid_y"] - S["rna_centroid_y"]
    dz = S["atac_centroid_z"] - S["rna_centroid_z"]
    S["centroid_shift"] = np.sqrt(dx**2 + dy**2 + dz**2)
    S["delta_rg"] = S["atac_rg"] - S["rna_rg"]
    S["in_panel"] = S.index.isin(panel)
    S["module"]   = [gene_to_group.get(g, "") for g in S.index]
    S.to_csv(OUT / "D_gene_spread.tsv", sep="\t", float_format="%.4f")

    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor="white")
    bg = S[~S["in_panel"]]
    ax.scatter(bg["rna_rg"], bg["atac_rg"], s=10, alpha=0.35,
               color="#aaaaaa", edgecolors="none", label="non-panel")
    # Label only the most off-diagonal panel genes to reduce overlap.
    deviation = (S["atac_rg"] - S["rna_rg"]).abs()
    panel_dev = deviation[S["in_panel"]].dropna()
    label_threshold = float(panel_dev.quantile(0.4)) if len(panel_dev) else 0.0
    for grp, color in GROUP_COLORS.items():
        sub = S[S["module"] == grp]
        if sub.empty:
            continue
        ax.scatter(sub["rna_rg"], sub["atac_rg"], s=70,
                   color=color, edgecolors="black", linewidth=0.7,
                   label=grp)
        for gname, row in sub.iterrows():
            dev = abs(row["atac_rg"] - row["rna_rg"])
            if dev >= label_threshold:
                ax.annotate(gname, (row["rna_rg"], row["atac_rg"]),
                            fontsize=8, alpha=0.9, xytext=(4, 4),
                            textcoords="offset points")
    lo = float(min(S["rna_rg"].min(), S["atac_rg"].min(), 1.0)) * 0.95
    hi = float(max(S["rna_rg"].max(), S["atac_rg"].max())) * 1.05
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--",
            label="ATAC = RNA")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("RNA spatial spread (radius of gyration of top-10% cells)")
    ax.set_ylabel("ATAC spatial spread (radius of gyration of top-10% cells)")
    ax.set_title("Per-gene spatial spread: ATAC vs RNA\n"
                 "Above the diagonal = ATAC pattern is broader than RNA "
                 "(consistent with chromatin priming in additional lineages)",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "D_atac_vs_rna_spread.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {OUT/'D_atac_vs_rna_spread.png'}")

    # ==================================================================
    # Summary
    # ==================================================================
    pos_above = int((S["delta_rg"] > 0).sum())
    pos_below = int((S["delta_rg"] < 0).sum())

    summary = []
    summary.append("# ATAC vs RNA 3D divergence — exploratory follow-up\n\n")
    summary.append(f"- Shared genes: **{len(shared)}**; disease panel: **{len(panel)}**.\n")
    summary.append(f"- Per-cell panel divergence |ATAC_z - RNA_z|: "
                   f"mean={panel_div.mean():.3f}, "
                   f"q90={np.quantile(panel_div, 0.9):.3f}, "
                   f"max={panel_div.max():.3f}.\n")
    summary.append("\n## A — Spatial divergence atlas\n\n")
    summary.append("- See `A_panel_divergence_3d.png` (single map) and "
                   "`A_module_divergence_3d_grid.png` (6 modules).\n")
    summary.append("\n## B — Priming asymmetry\n\n")
    summary.append("Top primed-only genes (high ATAC, low RNA):\n\n")
    for _, row in top_primed.iloc[::-1].head(8).iterrows():
        summary.append(f"  - **{row['gene']}** "
                       f"({row['module'] or 'non-panel'}) "
                       f"primed_frac={row['primed_frac']:.3f}\n")
    summary.append("\nTop expressed-without-priming genes (high RNA, low ATAC):\n\n")
    for _, row in top_expr.iloc[::-1].head(8).iterrows():
        summary.append(f"  - **{row['gene']}** "
                       f"({row['module'] or 'non-panel'}) "
                       f"expr_only_frac={row['expr_only_frac']:.3f}\n")
    summary.append("\n## C — Within-cell-type Pearson r\n\n")
    summary.append("Median within-celltype r per cell type:\n\n")
    for c in ct_order:
        med = float(np.nanmedian(Cdf[c]))
        summary.append(f"  - {c}: median r = {med:.3f} (n={(ct_series==c).sum()})\n")
    if "VCM" in ct_order:
        summary.append(f"\nWithin-VCM divergent triptychs: "
                       f"{', '.join(triptych_genes)}.\n")
    summary.append("\n## D — Spatial spread (radius of gyration)\n\n")
    summary.append(f"- Genes with ATAC spread > RNA spread: **{pos_above}** / "
                   f"genes with ATAC < RNA: **{pos_below}** "
                   f"(of {pos_above + pos_below} non-NaN).\n")
    summary.append(f"- Median Δrg (ATAC − RNA): "
                   f"**{S['delta_rg'].median():.2f}**.\n")
    summary.append(f"- Median centroid shift (ATAC vs RNA): "
                   f"**{S['centroid_shift'].median():.2f}**.\n")

    (OUT / "_explore_summary.md").write_text("".join(summary))
    log(f"  wrote {OUT/'_explore_summary.md'}")
    log("Phase 21 complete.")


if __name__ == "__main__":
    main()
