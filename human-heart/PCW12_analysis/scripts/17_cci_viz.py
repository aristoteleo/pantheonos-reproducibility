"""
Spatial CCI visualization — Step 3 of 3.

For the most interesting cell-type pairs (highest LR co-expression):
  1. 3D PyVista sender/receiver spatial highlight (Z-flipped)
  2. LR co-expression heatmap

Inputs:
  PCW12_analysis/data/adata_cci_ready.h5ad
  PCW12_analysis/results/cci/all_lr_top3_per_pair.csv

Outputs:
  PCW12_analysis/figures/cci/<sender>__<receiver>__3d.png
  PCW12_analysis/figures/cci/<sender>__<receiver>__heatmap.png
  PCW12_analysis/figures/cci/significant_pairs_overview.png
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
ADATA_PATH = ROOT / "PCW12_analysis/data/adata_cci_ready.h5ad"
SUMMARY_CSV = ROOT / "PCW12_analysis/results/cci/all_lr_top3_per_pair.csv"
PER_PAIR_DIR = ROOT / "PCW12_analysis/results/cci/per_pair"
FIG_DIR = ROOT / "PCW12_analysis/figures/cci"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GROUP_KEY = "mapped_coarse_celltype"
N_TOP_PAIRS = 5  # number of "most interesting" pairs to visualize


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_top_pairs(summary: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Pick the ordered cell-type pairs with the highest LR co-expression.
    Use the row with maximum lr_co_exp_ratio (preferring significant) per sr_pair,
    sorted by absolute count and ratio.
    """
    df = summary.copy()
    # Per ordered cell-type pair, keep the best LR row
    df = df.sort_values(["sr_pair", "lr_co_exp_ratio", "lr_co_exp_num"],
                        ascending=[True, False, False])
    best = df.groupby("sr_pair", as_index=False).head(1)
    # Prefer significant pairs, then by num and ratio
    if "is_significant" in best.columns:
        best = best.sort_values(
            ["is_significant", "lr_co_exp_num", "lr_co_exp_ratio"],
            ascending=[False, False, False],
        )
    else:
        best = best.sort_values(["lr_co_exp_num", "lr_co_exp_ratio"], ascending=False)
    # Filter to non-zero coexpression
    best = best[best["lr_co_exp_num"] > 0].head(n)
    return best.reset_index(drop=True)


def render_3d_pair(adata, sender: str, receiver: str, out_png: Path,
                   point_size_other: float = 1.0,
                   point_size_focus: float = 3.0) -> None:
    coords = np.asarray(adata.obsm["spatial"]).astype(float)
    labels = adata.obs[GROUP_KEY].astype(str).values

    plotter = pv.Plotter(off_screen=True, window_size=[1400, 1000])
    plotter.set_background("black")

    mask_other = (labels != sender) & (labels != receiver)
    if mask_other.any():
        plotter.add_points(
            pv.PolyData(coords[mask_other]),
            color="#888888", point_size=point_size_other, opacity=0.05,
            render_points_as_spheres=False,
        )

    mask_s = labels == sender
    n_s = int(mask_s.sum())
    if mask_s.any():
        plotter.add_points(
            pv.PolyData(coords[mask_s]),
            color="#FF3030", point_size=point_size_focus, opacity=0.55,
            render_points_as_spheres=False,
            label=f"{sender} (sender, n={n_s})",
        )

    mask_r = labels == receiver
    n_r = int(mask_r.sum())
    if mask_r.any():
        plotter.add_points(
            pv.PolyData(coords[mask_r]),
            color="#3399FF", point_size=point_size_focus, opacity=0.55,
            render_points_as_spheres=False,
            label=f"{receiver} (receiver, n={n_r})",
        )

    plotter.add_text(f"{sender} -> {receiver}", font_size=18,
                     color="white", position="upper_left")
    # Anchor the legend to a top-left corner of the viewport (loc="upper left"),
    # which avoids the right-edge clipping seen with the default placement.
    plotter.add_legend(
        bcolor=(0.08, 0.08, 0.08),
        face=None,
        size=(0.32, 0.10),
        loc="lower left",
    )
    plotter.camera.Elevation(-15)
    plotter.camera.Azimuth(-60)
    plotter.screenshot(str(out_png))
    plotter.close()


def render_lr_heatmap(per_pair_csv: Path, sender: str, receiver: str,
                      out_png: Path, top_k: int = 10) -> None:
    df = pd.read_csv(per_pair_csv)
    if df.empty:
        log(f"  (heatmap) per-pair CSV empty: {per_pair_csv.name}")
        return
    sort_col = "lr_co_exp_ratio" if "lr_co_exp_ratio" in df.columns else "lr_co_exp_num"
    df = df.sort_values(sort_col, ascending=False).head(top_k)

    if df["lr_co_exp_num"].max() <= 0:
        # fallback: still produce a placeholder bar plot of expression numbers
        fig, ax = plt.subplots(figsize=(6, 3.5))
        labels = [f"{a}-{b}" for a, b in zip(df["from"], df["to"])]
        ax.bar(labels, df[sort_col].values, color="#aaaaaa")
        ax.set_title(f"No spatially co-expressed LR pairs\n{sender} → {receiver}")
        ax.set_ylabel(sort_col)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    # Pivot ligand x receptor
    heat = df.pivot_table(index="from", columns="to",
                          values=sort_col, aggfunc="max").fillna(0)
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * heat.shape[1] + 3),
                                    max(3, 0.6 * heat.shape[0] + 2)))
    sns.heatmap(
        heat, cmap="winter", square=True, linewidths=0.4,
        annot=True, fmt=".3f", cbar_kws={"label": sort_col},
        ax=ax,
    )
    ax.set_xlabel(f"Receptor in {receiver}")
    ax.set_ylabel(f"Ligand in {sender}")
    ax.set_title(f"{sender} → {receiver} — top-{len(df)} LR co-expression")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def overview_significant(summary: pd.DataFrame, out_png: Path) -> None:
    """Bar plot of all significant LR pairs across cell-type pairs."""
    sig = summary[summary.get("is_significant", False) == True].copy() \
        if "is_significant" in summary.columns else pd.DataFrame()
    if sig.empty:
        log("No significant pairs to plot in overview")
        return
    sig["pair"] = sig["sender"] + " → " + sig["receiver"] + "\n" + sig["from"] + "-" + sig["to"]
    sig = sig.sort_values("lr_co_exp_ratio", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(sig) + 2))
    bars = ax.barh(sig["pair"], sig["lr_co_exp_ratio"], color="#3399FF")
    for b, n in zip(bars, sig["lr_co_exp_num"]):
        ax.text(b.get_width() * 1.02, b.get_y() + b.get_height() / 2,
                f"n={int(n)}", va="center", fontsize=9)
    ax.set_xlabel("LR co-expression ratio")
    ax.set_title("Significant spatial LR pairs (5/56 ordered pairs)")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    log(f"Loading {ADATA_PATH.name}")
    adata = sc.read_h5ad(ADATA_PATH)
    log(f"  shape={adata.shape}")

    summary = pd.read_csv(SUMMARY_CSV)
    top = select_top_pairs(summary, n=N_TOP_PAIRS)
    log(f"Selected {len(top)} most interesting pairs:")
    print(top[["sender", "receiver", "from", "to",
               "lr_co_exp_num", "lr_co_exp_ratio", "is_significant"]].to_string(index=False))

    # Overview
    overview_significant(summary, FIG_DIR / "significant_pairs_overview.png")
    log("Wrote significant_pairs_overview.png")

    # Per-pair viz
    for _, row in top.iterrows():
        s = str(row["sender"])
        r = str(row["receiver"])
        tag = f"{s}__{r}"
        log(f"Rendering {tag} ...")
        try:
            render_3d_pair(adata, s, r, FIG_DIR / f"{tag}__3d.png")
        except Exception as e:
            log(f"  3D failed: {e!r}")
        try:
            per_pair_csv = PER_PAIR_DIR / f"{tag}.csv"
            if per_pair_csv.exists():
                render_lr_heatmap(per_pair_csv, s, r, FIG_DIR / f"{tag}__heatmap.png")
            else:
                log(f"  per-pair CSV missing: {per_pair_csv}")
        except Exception as e:
            log(f"  heatmap failed: {e!r}")

    log("Done.")
    log("Outputs:")
    for p in sorted(FIG_DIR.glob("*.png")):
        log(f"  {p.name}  ({p.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
