"""
Whole-heart 3D MERFISH visualizations for the data overview.

Outputs (all in PCW12_analysis/figures/overview/):
  - merfish_3d_celltype.png        : 4-panel static view (4 azimuths) coloured by celltype
  - merfish_3d_celltype.gif        : rotating GIF coloured by celltype
  - merfish_3d_chamber.png         : 4-panel static view by anatomical region (atria vs ventricles)

Coordinates: obsm['X_spateo_update'] with z-axis flipped so apex points down
(matches the convention used by all prior 3D scripts; reversal-bug-fix retained).
"""
from __future__ import annotations
import os
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import sys
import time
from pathlib import Path

import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyvista as pv
import seaborn as sns

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
sys.path.insert(0, str(ROOT / "PCW12_analysis/scripts"))
from _viz_utils import _coords_3d  # noqa: E402

SP_PATH = ROOT / "data/full_heart_final_aug2025_update_downsampled_100k.h5ad"
OUT = ROOT / "PCW12_analysis/figures/overview"
OUT.mkdir(parents=True, exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _render_panel(coords, rgb, *, point_size=2.5, opacity=0.85,
                  azimuth=-60.0, elevation=-15.0,
                  window_size=(900, 900), background="black"):
    cloud = pv.PolyData(coords)
    cloud["rgb"] = rgb
    p = pv.Plotter(off_screen=True, window_size=list(window_size))
    p.set_background(background)
    p.add_points(cloud, scalars="rgb", rgb=True,
                 render_points_as_spheres=False,
                 point_size=point_size, opacity=opacity)
    p.camera_position = "iso"
    p.camera.Elevation(elevation)
    p.camera.Azimuth(azimuth)
    img = p.screenshot(return_img=True)
    p.close()
    return img


def _render_gif(coords, rgb, out_gif, *, n_frames=36,
                point_size=2.5, opacity=0.85,
                window_size=(900, 900), background="black"):
    cloud = pv.PolyData(coords)
    cloud["rgb"] = rgb
    p = pv.Plotter(off_screen=True, window_size=list(window_size))
    p.set_background(background)
    p.add_points(cloud, scalars="rgb", rgb=True,
                 render_points_as_spheres=False,
                 point_size=point_size, opacity=opacity)
    p.camera_position = "iso"
    p.camera.Elevation(-15.0)
    p.camera.Azimuth(-60.0)
    p.open_gif(str(out_gif))
    for _ in range(n_frames):
        p.camera.Azimuth(360 / n_frames)
        p.write_frame()
    p.close()


def _make_legend_image(cats, colors, *, ncol=2, title=""):
    """Render a legend-only matplotlib figure and return RGB array."""
    fig = plt.figure(figsize=(4, max(2.0, 0.18 * len(cats) / ncol + 0.6)),
                     facecolor="white")
    ax = fig.add_subplot(111)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=c, markeredgecolor="none",
                          markersize=8, label=str(n))
               for n, c in zip(cats, colors)]
    ax.legend(handles=handles, loc="center", ncol=ncol, frameon=False,
              fontsize=8, title=title, title_fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _composite_panels(panels, labels, legend_img, out_png, suptitle):
    n = len(panels)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="white",
                             gridspec_kw={"width_ratios": [1, 1, 0.85]})
    grid_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for k, (img, lbl) in enumerate(zip(panels[:n], labels[:n])):
        r, c = grid_positions[k]
        axes[r][c].imshow(img)
        axes[r][c].set_title(lbl, fontsize=11)
        axes[r][c].axis("off")
    # Right column: legend spanning both rows
    gs = axes[0][2].get_gridspec()
    for ax in [axes[0][2], axes[1][2]]:
        ax.remove()
    ax_leg = fig.add_subplot(gs[:, 2])
    ax_leg.imshow(legend_img)
    ax_leg.axis("off")
    fig.suptitle(suptitle, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    log(f"Loading MERFISH: {SP_PATH}")
    adata = ad.read_h5ad(SP_PATH)
    log(f"  shape={adata.shape}; obsm keys={list(adata.obsm.keys())}")
    if "X_spateo_update" not in adata.obsm:
        raise SystemExit("X_spateo_update not in obsm")

    coords = _coords_3d(adata, "X_spateo_update")  # z-flipped (apex-down)
    log(f"  coords range x={coords[:,0].min():.0f}..{coords[:,0].max():.0f} "
        f"y={coords[:,1].min():.0f}..{coords[:,1].max():.0f} "
        f"z={coords[:,2].min():.0f}..{coords[:,2].max():.0f}")

    # --- 1) Coloured by celltype (34 classes) ---
    labels = adata.obs["celltype"].astype("category")
    cats = labels.cat.categories.tolist()
    codes = labels.cat.codes.to_numpy()
    palette = (sns.color_palette("tab20", 20)
               + sns.color_palette("tab20b", 20))[:len(cats)]
    palette = np.asarray(palette)
    rgb = (palette[codes] * 255).astype(np.uint8)

    log(f"Rendering 4-panel celltype views ({len(cats)} classes) ...")
    azs = [-60, 30, 120, 210]
    panels = []
    for az in azs:
        panels.append(_render_panel(coords, rgb, azimuth=az))
    legend_img = _make_legend_image(cats, palette, ncol=2,
                                    title="Cell type")
    _composite_panels(
        panels, [f"azimuth {a}°" for a in azs], legend_img,
        OUT / "merfish_3d_celltype.png",
        "MERFISH whole-heart 3D — coloured by cell type "
        "(z-axis: apex points down)",
    )
    log(f"  wrote {OUT / 'merfish_3d_celltype.png'}")

    log("Rendering rotating GIF ...")
    _render_gif(coords, rgb, OUT / "merfish_3d_celltype.gif", n_frames=36)
    log(f"  wrote {OUT / 'merfish_3d_celltype.gif'}")

    # --- 2) Coloured by anatomical region (atrial vs ventricular vs other) ---
    ct = labels.astype(str)
    region = np.full(len(ct), "Other / connective", dtype=object)
    region[ct.str.startswith("vCM") | (ct == "Compact vFibro")
           | (ct == "Trabecular vFibro") | (ct == "Proliferating vFibro")] = (
               "Ventricular CM / vFibro")
    region[ct.str.startswith("aCM") | (ct == "aFibro")
           | (ct == "adFibro")] = "Atrial CM / aFibro"
    region[ct == "aEndocardial"] = "Atrial endocardium"
    region[ct.isin(["BEC", "VEC", "LEC"])] = "Endothelial"
    region[ct.isin(["Pericyte", "VSMC", "VIC"])] = "Mural / valve interstitial"
    region[ct == "Epicardial"] = "Epicardial"
    region[ct == "EPDC"] = "EPDC"
    region[ct == "Neuronal"] = "Neuronal"
    region[ct == "WBC"] = "WBC"

    region_cats = ["Ventricular CM / vFibro",
                   "Atrial CM / aFibro",
                   "Atrial endocardium",
                   "Endothelial",
                   "Mural / valve interstitial",
                   "Epicardial",
                   "EPDC",
                   "Neuronal",
                   "WBC",
                   "Other / connective"]
    region_palette = np.asarray(sns.color_palette("Set2", len(region_cats)))
    region_to_idx = {r: i for i, r in enumerate(region_cats)}
    r_codes = np.array([region_to_idx[r] for r in region])
    r_rgb = (region_palette[r_codes] * 255).astype(np.uint8)

    log("Rendering 4-panel anatomical region views ...")
    panels = []
    for az in azs:
        panels.append(_render_panel(coords, r_rgb, azimuth=az,
                                    opacity=0.85, point_size=2.5))
    # Also a rotating GIF for the chamber view (more readable than 34-class)
    log("Rendering anatomical-group rotating GIF ...")
    _render_gif(coords, r_rgb, OUT / "merfish_3d_chamber.gif", n_frames=36)
    log(f"  wrote {OUT / 'merfish_3d_chamber.gif'}")
    legend_img = _make_legend_image(region_cats, region_palette, ncol=1,
                                    title="Anatomical group")
    _composite_panels(
        panels, [f"azimuth {a}°" for a in azs], legend_img,
        OUT / "merfish_3d_chamber.png",
        "MERFISH whole-heart 3D — coloured by anatomical group "
        "(z-axis: apex points down)",
    )
    log(f"  wrote {OUT / 'merfish_3d_chamber.png'}")
    log("Done.")


if __name__ == "__main__":
    main()
