"""
Visualize disease genes that are already in the MERFISH 238-gene panel.

For each direct gene:
  - Render a static 3D PNG (all 20 direct genes)
  - Render a rotating GIF if also in highlight set
Also produces a celltype overview figure and a celltype × direct-gene heatmap.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

# Off-screen rendering before pyvista import
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
sys.path.insert(0, str(ROOT / "PCW12_analysis/scripts"))
from _viz_utils import render_gene_3d, render_celltype_3d  # noqa: E402

SP_PATH = ROOT / "data/full_heart_final_aug2025_update_downsampled_100k.h5ad"
PARTITION = ROOT / "PCW12_analysis/data/disease_genes_partition.tsv"
HIGHLIGHTS = ROOT / "PCW12_analysis/data/highlight_genes.tsv"
OUT_FIG_DIR = ROOT / "PCW12_analysis/figures/direct"
OUT_HEAT_DIR = ROOT / "PCW12_analysis/figures/heatmaps"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_HEAT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading MERFISH data ...")
t0 = time.time()
adata = ad.read_h5ad(SP_PATH)
print(f"  loaded {adata.shape} in {time.time()-t0:.1f}s")
print(f"  layers: {list(adata.layers.keys())}")
print(f"  X range: {float(adata.X.min()):.2f} .. {float(adata.X.max()):.2f}")

partition = pd.read_csv(PARTITION, sep="\t")
direct = partition[partition["class"] == "direct"]["Gene"].tolist()
print(f"Direct disease genes: {len(direct)} -> {direct}")

highlight_set = set(pd.read_csv(HIGHLIGHTS, sep="\t")["Gene"].tolist())
direct_highlights = [g for g in direct if g in highlight_set]
print(f"Direct genes in highlight set (will get GIFs): {direct_highlights}")

# -------- 3D PNGs (and GIFs for highlights) --------
fail = []
for i, g in enumerate(direct, 1):
    out_png = OUT_FIG_DIR / f"{g}.png"
    gif = OUT_FIG_DIR / f"{g}.gif" if g in highlight_set else None
    print(f"[{i}/{len(direct)}] {g} -> {out_png.name}{' + gif' if gif else ''}")
    try:
        render_gene_3d(adata, g, out_png, gif_path=gif,
                       title_suffix="(direct, MERFISH)")
    except Exception as e:
        print(f"   FAILED {g}: {e}")
        fail.append((g, str(e)))

if fail:
    print(f"FAILED renders: {fail}")

# -------- Cell-type overview (one figure, reused often) --------
ct_out = OUT_FIG_DIR / "_celltype_overview.png"
print(f"Rendering cell-type overview -> {ct_out.name}")
try:
    render_celltype_3d(adata, ct_out, obs_key="celltype",
                       title="MERFISH cell types (3D)")
except Exception as e:
    print(f"   celltype overview FAILED: {e}")

# -------- Heatmap: celltype x direct gene (mean log1p expression) --------
print("Building celltype × direct-gene mean expression heatmap ...")
sub = adata[:, direct].copy()
X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
df = pd.DataFrame(X, columns=direct, index=adata.obs["celltype"].values)
mean_df = df.groupby(level=0).mean()  # celltype rows, gene cols
mean_df.to_csv(OUT_HEAT_DIR / "direct_celltype_mean.tsv", sep="\t")

# z-score per gene (column) for visual contrast
z = (mean_df - mean_df.mean(axis=0)) / (mean_df.std(axis=0) + 1e-9)

# Order rows by hierarchical-ish convenience: sort by max gene
row_order = z.max(axis=1).sort_values(ascending=False).index
col_order = z.idxmax(axis=0).map(lambda x: list(row_order).index(x)).sort_values().index
z = z.loc[row_order, col_order]

fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(col_order)),
                                max(6, 0.32 * len(row_order))))
im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
ax.set_xticks(range(len(col_order)))
ax.set_xticklabels(col_order, rotation=90, fontsize=9)
ax.set_yticks(range(len(row_order)))
ax.set_yticklabels(row_order, fontsize=9)
ax.set_title("Direct disease genes (MERFISH) — z-scored mean per cell type")
fig.colorbar(im, ax=ax, label="z-score")
fig.tight_layout()
fig.savefig(OUT_HEAT_DIR / "direct_celltype_heatmap.png", dpi=200)
plt.close(fig)

print("Done — direct visualization phase complete.")
