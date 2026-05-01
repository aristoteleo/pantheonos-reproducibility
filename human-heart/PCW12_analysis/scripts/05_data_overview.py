"""Generate dataset overview / QC figures for PCW12 analysis.

Outputs to PCW12_analysis/figures/overview/:
- scrna_qc.png
- scrna_celltype.png
- scrna_donor_pcw.png
- merfish_qc.png
- merfish_celltype.png
- merfish_section.png
- merfish_spatial.png
- crossmod_overlap.png
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Aesthetic defaults
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
OUT = ROOT / "PCW12_analysis" / "figures" / "overview"
OUT.mkdir(parents=True, exist_ok=True)

SCRNA = ROOT / "data" / "all_healthy_RoundedPCW11-13.h5ad"
MERFISH = ROOT / "data" / "full_heart_final_aug2025_update_downsampled_100k.h5ad"


def violin_panel(ax, data_dict, title, ylabel, log=False):
    """Compact violin/box hybrid for QC."""
    labels = list(data_dict.keys())
    values = [np.asarray(data_dict[k]) for k in labels]
    parts = ax.violinplot(values, showmeans=False, showmedians=True, widths=0.85)
    for pc, c in zip(parts["bodies"], plt.cm.tab10(np.linspace(0, 1, max(len(labels), 10)))):
        pc.set_facecolor(c)
        pc.set_alpha(0.65)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.6)
    for k in ("cmedians", "cmaxes", "cmins", "cbars"):
        if k in parts:
            parts[k].set_color("black")
            parts[k].set_linewidth(0.8)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")


# --- scRNA-seq ---
print("Loading scRNA-seq...", flush=True)
ascrna = ad.read_h5ad(SCRNA)
print(f"  {ascrna.shape}", flush=True)

# QC metrics already present: nFeature_RNA, nCount_RNA, ratio.mt
obs = ascrna.obs.copy()
obs["pct_mt"] = obs["ratio.mt"] * 100.0

# 1) scRNA QC violins by donor
print("Plotting scrna_qc.png...", flush=True)
donors = sorted(obs["source"].unique().tolist())
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, col, title, ylabel, log in [
    (axes[0], "nFeature_RNA", "Genes detected per cell", "n_genes", True),
    (axes[1], "nCount_RNA", "UMI counts per cell", "n_counts", True),
    (axes[2], "pct_mt", "Mitochondrial fraction", "% mito", False),
]:
    data = {d: obs.loc[obs["source"] == d, col].values for d in donors}
    violin_panel(ax, data, title, ylabel, log=log)
fig.suptitle(f"scRNA-seq QC by donor  (n = {len(obs):,} cells, {len(donors)} donors)",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "scrna_qc.png")
plt.close(fig)

# 2) scRNA celltype composition (broad celltype_label)
print("Plotting scrna_celltype.png...", flush=True)
ct = obs["celltype_label"].value_counts().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.tab20(np.linspace(0, 1, len(ct)))
ax.barh(ct.index, ct.values, color=colors, edgecolor="black", linewidth=0.5)
for i, v in enumerate(ct.values):
    ax.text(v + max(ct.values) * 0.01, i, f"{v:,}", va="center", fontsize=8)
ax.set_xlabel("Number of cells")
ax.set_title(f"scRNA-seq cell type composition  ({len(ct)} broad classes)")
ax.set_xlim(0, ct.values.max() * 1.12)
fig.savefig(OUT / "scrna_celltype.png")
plt.close(fig)

# 3) scRNA donor x PCW + Sex + Phase
print("Plotting scrna_donor_pcw.png...", flush=True)
fig = plt.figure(figsize=(15, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1, 1], wspace=0.35)

# 3a stacked bar: donor x PCW
ax1 = fig.add_subplot(gs[0, 0])
pcw_levels = sorted(obs["Rounded.PCW"].unique())
pivot = obs.groupby(["source", "Rounded.PCW"]).size().unstack(fill_value=0)[pcw_levels]
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
pcw_colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(pcw_levels)))
bottom = np.zeros(len(pivot))
for col, c in zip(pivot.columns, pcw_colors):
    ax1.bar(pivot.index, pivot[col], bottom=bottom, label=f"PCW {col}", color=c,
            edgecolor="black", linewidth=0.4)
    bottom += pivot[col].values
ax1.set_ylabel("Number of cells")
ax1.set_title("Cells per donor × developmental stage (PCW)")
ax1.legend(title="Stage", frameon=False, fontsize=8)
ax1.tick_params(axis="x", rotation=30)
for lbl in ax1.get_xticklabels():
    lbl.set_ha("right")

# 3b sex pie
ax2 = fig.add_subplot(gs[0, 1])
sex_counts = obs["Sex"].value_counts()
ax2.pie(sex_counts.values, labels=[f"{s}\n{n:,}" for s, n in sex_counts.items()],
        colors=["#f4a8a8", "#a8c8f4"], autopct="%1.0f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2))
ax2.set_title("Sex distribution")

# 3c Phase
ax3 = fig.add_subplot(gs[0, 2])
phase_counts = obs["Phase"].value_counts().reindex(["G1", "S", "G2M"])
ax3.bar(phase_counts.index, phase_counts.values,
        color=["#67a9cf", "#d6604d", "#f4a582"], edgecolor="black", linewidth=0.5)
for i, (p, n) in enumerate(phase_counts.items()):
    ax3.text(i, n + max(phase_counts.values) * 0.01, f"{n:,}",
             ha="center", va="bottom", fontsize=9)
ax3.set_ylabel("Number of cells")
ax3.set_title("Cell-cycle phase")
ax3.set_ylim(0, phase_counts.max() * 1.12)

fig.suptitle("scRNA-seq sample composition", fontsize=12, fontweight="bold", y=1.02)
fig.savefig(OUT / "scrna_donor_pcw.png")
plt.close(fig)

# Free scRNA memory
del ascrna
import gc; gc.collect()

# --- MERFISH ---
print("\nLoading MERFISH...", flush=True)
am = ad.read_h5ad(MERFISH)
print(f"  {am.shape}", flush=True)

mobs = am.obs.copy()

# 4) MERFISH QC violins by batch
print("Plotting merfish_qc.png...", flush=True)
batches = sorted(mobs["batch"].unique().tolist())
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, col, title, ylabel, log in [
    (axes[0], "n_genes_by_counts", "Genes detected per cell", "n_genes", False),
    (axes[1], "total_counts", "Total transcripts per cell", "total counts", True),
    (axes[2], "volume", "Cell volume", "volume (a.u.)", True),
]:
    data = {b: mobs.loc[mobs["batch"] == b, col].values for b in batches}
    violin_panel(ax, data, title, ylabel, log=log)
fig.suptitle(f"MERFISH QC by batch  (n = {len(mobs):,} cells, "
             f"{len(batches)} batches, 238 genes)",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "merfish_qc.png")
plt.close(fig)

# 5) MERFISH celltype composition
print("Plotting merfish_celltype.png...", flush=True)
mct = mobs["celltype"].value_counts().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 9))
colors = plt.cm.tab20b(np.linspace(0, 1, len(mct)))
ax.barh(mct.index, mct.values, color=colors, edgecolor="black", linewidth=0.4)
for i, v in enumerate(mct.values):
    ax.text(v + max(mct.values) * 0.01, i, f"{v:,}", va="center", fontsize=7)
ax.set_xlabel("Number of cells")
ax.set_title(f"MERFISH cell type composition  ({len(mct)} fine classes)")
ax.set_xlim(0, mct.values.max() * 1.12)
fig.savefig(OUT / "merfish_celltype.png")
plt.close(fig)

# 6) MERFISH section composition (sample-level: section x batch)
print("Plotting merfish_section.png...", flush=True)
sec_counts = mobs.groupby(["section", "batch"]).size().unstack(fill_value=0)
sec_counts = sec_counts.reindex(sorted(sec_counts.index))
fig, ax = plt.subplots(figsize=(14, 4.5))
bottom = np.zeros(len(sec_counts))
batch_colors = {"ATR": "#1b9e77", "VEN": "#d95f02"}
for b in sec_counts.columns:
    ax.bar(sec_counts.index.astype(str), sec_counts[b], bottom=bottom,
           label=b, color=batch_colors.get(b, "gray"),
           edgecolor="black", linewidth=0.3)
    bottom += sec_counts[b].values
ax.set_xlabel("Section ID")
ax.set_ylabel("Number of cells")
ax.set_title(f"MERFISH cells per section  ({len(sec_counts)} sections, "
             f"{mobs['fov'].nunique():,} FOVs)")
ax.legend(title="Batch", frameon=False)
ax.tick_params(axis="x", labelsize=7, rotation=0)
fig.savefig(OUT / "merfish_section.png")
plt.close(fig)

# 7) MERFISH spatial 2D scatter colored by celltype
print("Plotting merfish_spatial.png...", flush=True)
xy = am.obsm["X_spatial"]
ct_arr = mobs["celltype"].values
unique_ct = mobs["celltype"].cat.categories.tolist() \
    if hasattr(mobs["celltype"], "cat") else sorted(set(ct_arr))
n = len(unique_ct)
cmap = plt.cm.tab20(np.linspace(0, 1, 20))
cmap2 = plt.cm.tab20b(np.linspace(0, 1, 20))
palette = np.vstack([cmap, cmap2])[:n]
ct_to_color = {c: palette[i] for i, c in enumerate(unique_ct)}
colors_arr = np.array([ct_to_color[c] for c in ct_arr])

# Subsample for plot speed/clarity
rng = np.random.default_rng(0)
sub = rng.choice(len(xy), size=min(60000, len(xy)), replace=False)
xy_s = xy[sub]
colors_s = colors_arr[sub]

fig, ax = plt.subplots(figsize=(11, 9))
ax.scatter(xy_s[:, 0], xy_s[:, 1], s=1.0, c=colors_s,
           alpha=0.55, linewidths=0)
ax.set_aspect("equal")
ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")
ax.set_title(f"MERFISH 2D anatomical layout  (subsample: {len(sub):,} of {len(xy):,})")
# Compact legend
handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                      markerfacecolor=ct_to_color[c], markeredgecolor="none",
                      markersize=6, label=c) for c in unique_ct]
ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
          frameon=False, fontsize=7, ncol=1, handletextpad=0.3)
fig.savefig(OUT / "merfish_spatial.png")
plt.close(fig)

# 8) Cross-modality cell type overlap (broad-class comparison)
# Map MERFISH 34-class fine -> broad class similar to scRNA's 11 classes
print("Plotting crossmod_overlap.png...", flush=True)

# Manual mapping from MERFISH celltype -> broad class
merfish_to_broad = {
    # Ventricular CMs
    "vCM-IVS-His": "VCM", "vCM-LV-AV": "VCM", "vCM-LV-Compact I": "VCM",
    "vCM-LV-Compact II": "VCM", "vCM-LV-Hybrid": "VCM",
    "vCM-LV-Trabecular I": "VCM", "vCM-LV-Trabecular II": "VCM",
    "vCM-LV/RV-Purkinje": "ConductionCells", "vCM-RV-AV": "VCM",
    "vCM-RV-Compact": "VCM", "vCM-RV-Hybrid": "VCM",
    "vCM-RV-Proliferating": "VCM", "vCM-RV-Trabecular": "VCM",
    # Atrial CMs
    "aCM-LA": "ACM", "aCM-RA": "ACM",
    # Non-chamber CMs
    "ncCM-AVC-like": "ConductionCells", "ncCM-IFT-like": "ConductionCells",
    # Endo / Endothelial
    "vEndocardial": "Endocardial", "aEndocardial": "Endocardial",
    "BEC": "Endothelial", "LEC": "Endothelial", "VEC": "Endothelial",
    # Fibroblasts / Mural / Epicardial
    "Compact vFibro": "FB", "Trabecular vFibro": "FB",
    "Proliferating vFibro": "FB", "aFibro": "FB", "adFibro": "FB",
    "VIC": "FB",
    "VSMC": "MuralCells", "Pericyte": "MuralCells",
    "Epicardial": "Epicardial", "EPDC": "Epicardial",
    # Other
    "Neuronal": "NC",
    "WBC": "ImmuneCells",
}

mobs["broad_class"] = mobs["celltype"].map(merfish_to_broad).fillna("Other")
merfish_broad_counts = mobs["broad_class"].value_counts()

# Reload scRNA broad counts (cheap from cache file)
import scanpy as sc
sc.settings.verbosity = 0
ascrna_obs = ad.read_h5ad(SCRNA, backed="r").obs
scrna_broad_counts = ascrna_obs["celltype_label"].value_counts()

# Align
all_classes = sorted(set(scrna_broad_counts.index) | set(merfish_broad_counts.index))
scrna_vals = np.array([scrna_broad_counts.get(c, 0) for c in all_classes])
merfish_vals = np.array([merfish_broad_counts.get(c, 0) for c in all_classes])

# Convert to fractions for fair comparison
scrna_frac = scrna_vals / scrna_vals.sum() * 100
merfish_frac = merfish_vals / merfish_vals.sum() * 100

# Sort by combined abundance
order = np.argsort(-(scrna_frac + merfish_frac))
all_classes = [all_classes[i] for i in order]
scrna_frac = scrna_frac[order]
merfish_frac = merfish_frac[order]

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(all_classes))
w = 0.4
b1 = ax.bar(x - w/2, scrna_frac, w, label="scRNA-seq", color="#377eb8",
            edgecolor="black", linewidth=0.4)
b2 = ax.bar(x + w/2, merfish_frac, w, label="MERFISH", color="#e41a1c",
            edgecolor="black", linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels(all_classes, rotation=30, ha="right")
ax.set_ylabel("% of cells")
ax.set_title("Cross-modality cell-type composition  (broad classes, MERFISH mapped)")
ax.legend(frameon=False)

# Annotate "shared" classes
shared = set(scrna_broad_counts.index) & set(merfish_broad_counts.index)
for tick, cls in zip(ax.get_xticklabels(), all_classes):
    tick.set_color("black" if cls in shared else "#888888")

fig.tight_layout()
fig.savefig(OUT / "crossmod_overlap.png")
plt.close(fig)

# Summary stats JSON
summary = {
    "scrna": {
        "n_cells": int(len(obs)),
        "n_genes": 31019,
        "n_donors": int(obs["source"].nunique()),
        "n_samples": int(obs["sample_id"].nunique()),
        "pcw_distribution": obs["Rounded.PCW"].value_counts().sort_index().to_dict(),
        "sex_distribution": obs["Sex"].value_counts().to_dict(),
        "phase_distribution": obs["Phase"].value_counts().to_dict(),
        "celltype_label_classes": int(obs["celltype_label"].nunique()),
        "celltype_fine_classes": int(obs["celltype"].nunique()),
        "median_n_genes": float(np.median(obs["nFeature_RNA"])),
        "median_n_counts": float(np.median(obs["nCount_RNA"])),
        "median_pct_mt": float(np.median(obs["pct_mt"])),
    },
    "merfish": {
        "n_cells": int(len(mobs)),
        "n_genes": 238,
        "n_batches": int(mobs["batch"].nunique()),
        "n_sections": int(mobs["section"].nunique()),
        "n_fovs": int(mobs["fov"].nunique()),
        "celltype_classes": int(mobs["celltype"].nunique()),
        "median_n_genes": float(np.median(mobs["n_genes_by_counts"])),
        "median_n_counts": float(np.median(mobs["total_counts"])),
        "median_volume": float(np.median(mobs["volume"])),
    },
    "shared_broad_classes": sorted(list(shared)),
}
with open(OUT / "_summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\nDone. Saved figures to:", OUT)
print("Summary stats written to:", OUT / "_summary_stats.json")
