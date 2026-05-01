"""Shared 3D visualization utilities for PCW12 disease gene analysis."""
from __future__ import annotations
import os
# Off-screen rendering for batch figure production
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from pathlib import Path
import numpy as np
import pyvista as pv

# Heart-anatomy-friendly camera defaults (per spatial visualization skill)
DEFAULT_ELEV = -15.0
DEFAULT_AZIM = -60.0
DEFAULT_BG = "black"
DEFAULT_TXT = "#FFFFFF"


def _coords_3d(adata, coords_key: str = "X_spateo_update", flip_z: bool = True):
    """Return 3D coordinates. By default flips Z so apex points down (anatomical)."""
    coords = np.asarray(adata.obsm[coords_key], dtype=float).copy()
    if flip_z:
        coords[:, 2] = -coords[:, 2]
    return coords


def _expression_vec(adata, gene: str):
    if gene not in adata.var_names:
        raise KeyError(gene)
    x = adata[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x).ravel()


def render_gene_3d(
    adata,
    gene: str,
    out_png: str | os.PathLike,
    *,
    coords_key: str = "X_spateo_update",
    point_size: float = 1.5,
    opacity: float = 0.18,
    cmap: str = "magma",
    clim_quantile: tuple[float, float] = (0.0, 0.99),
    background: str = DEFAULT_BG,
    text_color: str = DEFAULT_TXT,
    elevation: float = DEFAULT_ELEV,
    azimuth: float = DEFAULT_AZIM,
    window_size: tuple[int, int] = (900, 900),
    gif_path: str | os.PathLike | None = None,
    n_frames: int = 30,
    title_suffix: str = "",
    expression: np.ndarray | None = None,
):
    """Render a 3D scatter colored by expression of `gene`.

    If `gene` is not in `adata.var_names`, pass `expression` array directly.
    Saves PNG to `out_png`. If `gif_path`, also renders a rotating GIF.
    """
    coords = _coords_3d(adata, coords_key)
    if expression is None:
        expr = _expression_vec(adata, gene)
    else:
        expr = np.asarray(expression).ravel()
    assert expr.shape[0] == coords.shape[0], (expr.shape, coords.shape)

    # Robust color limits
    pos = expr[expr > 0]
    if pos.size == 0:
        clim = (0.0, 1.0)
    else:
        clim = (
            float(np.quantile(expr, clim_quantile[0])),
            float(max(np.quantile(pos, clim_quantile[1]), 1e-6)),
        )
        if clim[1] <= clim[0]:
            clim = (clim[0], clim[0] + 1e-6)

    cloud = pv.PolyData(coords)
    cloud["expression"] = expr

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background(background)
    plotter.add_points(
        cloud,
        render_points_as_spheres=False,
        point_size=point_size,
        cmap=cmap,
        scalars="expression",
        opacity=opacity,
        clim=clim,
        scalar_bar_args={
            "title": "Expression",
            "color": text_color,
            "n_colors": 20,
            "n_labels": 4,
            "label_font_size": 12,
            "title_font_size": 14,
        },
    )
    label = gene + (f"  {title_suffix}" if title_suffix else "")
    plotter.add_text(label, font_size=18, color=text_color, position="upper_left")
    plotter.camera_position = "iso"
    plotter.camera.Elevation(elevation)
    plotter.camera.Azimuth(azimuth)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_png))

    if gif_path:
        gif_path = Path(gif_path)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.open_gif(str(gif_path))
        for _ in range(n_frames):
            plotter.camera.Azimuth(360 / n_frames)
            plotter.write_frame()
        plotter.close()
    else:
        plotter.close()


def render_celltype_3d(
    adata,
    out_png: str | os.PathLike,
    *,
    obs_key: str = "celltype",
    coords_key: str = "X_spateo_update",
    point_size: float = 1.0,
    opacity: float = 0.35,
    palette: str = "tab20",
    background: str = DEFAULT_BG,
    elevation: float = DEFAULT_ELEV,
    azimuth: float = DEFAULT_AZIM,
    window_size: tuple[int, int] = (1100, 900),
    title: str | None = None,
):
    import seaborn as sns

    coords = _coords_3d(adata, coords_key)
    labels = adata.obs[obs_key].astype("category")
    cats = labels.cat.categories.tolist()
    codes = labels.cat.codes.to_numpy()
    n = len(cats)
    colors = np.array(sns.color_palette(palette, n_colors=max(3, n))[:n])
    rgb = (colors[codes] * 255).astype(np.uint8)

    cloud = pv.PolyData(coords)
    cloud["rgb"] = rgb

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background(background)
    plotter.add_points(
        cloud,
        scalars="rgb",
        rgb=True,
        render_points_as_spheres=False,
        point_size=point_size,
        opacity=opacity,
    )
    plotter.add_text(title or obs_key, font_size=18, color="#FFFFFF",
                     position="upper_right")
    legend = [(str(c), tuple(colors[i])) for i, c in enumerate(cats)]
    plotter.add_legend(legend, size=(0.22, min(0.85, 0.025 * n + 0.05)),
                       loc="upper left", bcolor=(0, 0, 0), face=None)
    plotter.camera_position = "iso"
    plotter.camera.Elevation(elevation)
    plotter.camera.Azimuth(azimuth)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_png))
    plotter.close()
