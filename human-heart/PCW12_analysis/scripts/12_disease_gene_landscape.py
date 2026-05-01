"""Heart-disease gene expression landscape across PCW12 cell types.

Builds 4 figures from the same imputed (cell × 221-gene) matrix used in
script 11:

A. Trait × celltype heatmap (mean per-cell trait score, row-z-scored).
B. Gene × celltype heatmap (mean log1p expression, row-z-scored,
   hierarchically clustere annotation track).
C. Per-gene Yanai's tau specificity score; histogram + top-specific +
   top-broad genes.
D. Trait-trait similarity:
   - Jaccard of gene-membership sets (panel overlap)
   - Pearson correlation of celltype-score vectors (functional similarity)
   - scatter Jaccard vs Pearson (high Pearson + low Jaccard = traits that
     hit the same celltypes via different genes).

Outputs (PCW12_analysis/figures/disease_gene_landscape/):
- trait_by_celltype_means.tsv                   extent=(-0.5, n_ct - 0.5, 0, 1),
        row z-scored matrix used in Fig A
- gene_specificity_tau.tsv            tau per gene + best celltype + traits
- trait_similarity.tsv                jaccard, pearson, n_overlap
- figA_trait_by_celltype.{png,pdf}
- figB_gene_by_celltype.{png,pdf}
- figC_gene_specificity.{png,pdf}
- figD_trait_similarity.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_rgba
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
DATA = ROOT / "PCW12_analysis" / "data"
OUT = ROOT / "PCW12_analysis" / "figures" / "disease_gene_landscape"
OUT.mkdir(parents=True, exist_ok=True)

# Lineage assignment for column ordering / coloring in Fig A.
# Anything not listed falls into "Other".
LINEAGE_ORDER = [
    "CM", "Conduction", "Valve", "Fibroblast", "Endothelial",
    "Mural", "Hematopoietic", "Neural", "Epicardial", "Other",
]
LINEAGE_COLORS = {
    "CM": "#c0392b",
    "Conduction": "#e67e22",
    "Valve": "#9b59b6",
    "Fibroblast": "#16a085",
    "Endothelial": "#3498db",
    "Mural": "#2c3e50",
    "Hematopoietic": "#f1c40f",
    "Neural": "#1abc9c",
    "Epicardial": "#7f8c8d",
    "Other": "#bdc3c7",
}


def assign_lineage(ct: str) -> str:
    s = ct.lower()
    if s in {"vic", "vec", "nccm-avc-like"}:
        return "Valve"
    if "fibro" in s or "epdc" in s:
        return "Fibroblast" if "epdc" not in s else "Epicardial"
    if "epicard" in s:
        return "Epicardial"
    if any(k in s for k in ["cm", "cardiomyo", "myocyt"]) and "nccm-avc-like" not in s:
        # conduction subtypes contain explicit terms
        if any(k in s for k in ["pacemaker", "san", "avn", "purkinje", "his", "conduction"]):
            return "Conduction"
        return "CM"
    if any(k in s for k in ["pacemaker", "san", "avn", "purkinje", "conduction"]):
        return "Conduction"
    if any(k in s for k in ["endo", "endothel", "ec ", " ec", "lec", "vec_endo"]) and "vec" != s:
        return "Endothelial"
    if any(k in s for k in ["smc", "pericyte", "mural", "mesothelial"]):
        return "Mural"
    if any(k in s for k in ["macrop", "myeloid", "lympho", "mast", "immune", "blood", "ery", "neutroph"]):
        return "Hematopoietic"
    if any(k in s for k in ["neur", "schwann", "glia", "ganglion"]):
        return "Neural"
    return "Other"


def yanai_tau(x: np.ndarray) -> float:
    """Yanai's specificity index. x is a 1D vector of (non-negative) means."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.nan
    m = x.max()
    if m <= 0:
        return np.nan
    xn = x / m
    return float((1.0 - xn).sum() / (x.size - 1))


def main() -> None:
    # ---- Load --------------------------------------------------------------
    print("[1/8] Loading imputed adata...", flush=True)
    a = ad.read_h5ad(DATA / "adata_imputed_disease_genes.h5ad")
    print(f"      shape={a.shape}", flush=True)

    X = a.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_log = np.log1p(X).astype(np.float32)
    universe = a.var_names.to_numpy()
    gene_to_idx = {g: i for i, g in enumerate(universe)}

    ct_series = a.obs["celltype"].astype(str)
    celltypes = sorted(ct_series.unique())
    n_ct = len(celltypes)
    print(f"      celltypes={n_ct}: {celltypes}", flush=True)

    # ---- (gene × celltype) mean log1p matrix -------------------------------
    print("[2/8] Computing per-celltype gene means...", flush=True)
    G = len(universe)
    gc = np.zeros((G, n_ct), dtype=np.float32)
    sizes = np.zeros(n_ct, dtype=int)
    for j, c in enumerate(celltypes):
        m = (ct_series == c).to_numpy()
        sizes[j] = int(m.sum())
        gc[:, j] = X_log[m, :].mean(axis=0)
    gc_df = pd.DataFrame(gc, index=universe, columns=celltypes)

    # ---- Trait gene sets ---------------------------------------------------
    print("[3/8] Parsing trait panels...", flush=True)
    part = pd.read_csv(DATA / "disease_genes_partition.tsv", sep="\t")
    trait_to_genes: dict[str, list[str]] = {}
    gene_to_traits: dict[str, list[str]] = {g: [] for g in universe}
    for _, row in part.iterrows():
        if pd.isna(row["traits"]):
            continue
        gene = row["Gene"]
        if gene not in gene_to_idx:
            continue
        for t in str(row["traits"]).split(";"):
            trait_to_genes.setdefault(t, []).append(gene)
            gene_to_traits[gene].append(t)
    trait_to_genes = {t: gs for t, gs in trait_to_genes.items() if len(gs) >= 3}
    traits = sorted(trait_to_genes.keys())
    print(f"      {len(traits)} traits", flush=True)

    # ---- (trait × celltype) mean trait score -------------------------------
    print("[4/8] Computing trait × celltype matrix...", flush=True)
    T = len(traits)
    tc = np.zeros((T, n_ct), dtype=np.float32)
    for i, t in enumerate(traits):
        idx = np.array([gene_to_idx[g] for g in trait_to_genes[t]])
        # mean over panel genes of (mean log1p over cells of that celltype)
        tc[i, :] = gc[idx, :].mean(axis=0)
    tc_df = pd.DataFrame(tc, index=traits, columns=celltypes)
    tc_df.to_csv(OUT / "trait_by_celltype_means.tsv", sep="\t")

    # Row z-score for visualisation (across celltypes per trait)
    tc_z = (tc - tc.mean(axis=1, keepdims=True)) / (tc.std(axis=1, keepdims=True) + 1e-9)

    # ---- Yanai tau per gene ------------------------------------------------
    print("[5/8] Computing per-gene tau...", flush=True)
    tau = np.array([yanai_tau(gc[i, :]) for i in range(G)])
    best_ct_idx = np.argmax(gc, axis=1)
    best_ct = np.array(celltypes)[best_ct_idx]
    tau_df = pd.DataFrame({
        "gene": universe,
        "tau": tau,
        "max_celltype": best_ct,
        "max_mean_log1p": gc[np.arange(G), best_ct_idx],
        "n_traits": [len(gene_to_traits[g]) for g in universe],
        "traits": [";".join(gene_to_traits[g]) for g in universe],
    }).sort_values("tau", ascending=False).reset_index(drop=True)
    tau_df.to_csv(OUT / "gene_specificity_tau.tsv", sep="\t", index=False)

    # ---- Trait similarity --------------------------------------------------
    print("[6/8] Computing trait similarity...", flush=True)
    # Jaccard
    sets = {t: set(trait_to_genes[t]) for t in traits}
    jacc = np.zeros((T, T))
    overlap = np.zeros((T, T), dtype=int)
    for i, ti in enumerate(traits):
        for j, tj in enumerate(traits):
            inter = sets[ti] & sets[tj]
            union = sets[ti] | sets[tj]
            jacc[i, j] = len(inter) / max(len(union), 1)
            overlap[i, j] = len(inter)
    # Pearson on celltype-score vectors
    pear = np.corrcoef(tc)
    # tidy table (upper triangle)
    rows = []
    for i, ti in enumerate(traits):
        for j, tj in enumerate(traits):
            if j <= i:
                continue
            rows.append(dict(
                trait_a=ti, trait_b=tj,
                jaccard=jacc[i, j], n_overlap=int(overlap[i, j]),
                pearson_celltype=pear[i, j],
            ))
    sim_df = pd.DataFrame(rows).sort_values("pearson_celltype", ascending=False)
    sim_df.to_csv(OUT / "trait_similarity.tsv", sep="\t", index=False)

    # ---- Lineage colours for celltypes -------------------------------------
    lineages = np.array([assign_lineage(c) for c in celltypes])
    # Order celltypes: by lineage, then alpha within
    lineage_rank = {l: r for r, l in enumerate(LINEAGE_ORDER)}
    ct_order = sorted(range(n_ct), key=lambda j: (lineage_rank.get(lineages[j], 99), celltypes[j]))
    ct_sorted = [celltypes[j] for j in ct_order]
    lin_sorted = lineages[ct_order]
    sizes_sorted = sizes[ct_order]

    # =======================================================================
    # FIGURE A: trait × celltype heatmap (row z-score)
    # =======================================================================
    print("[7/8] Drawing Fig A (trait × celltype)...", flush=True)
    plt.rcParams.update({"font.size": 10})
    # Order traits by hierarchical clustering on z matrix
    if T >= 2:
        link_T = hierarchy.linkage(pdist(tc_z, metric="correlation"), method="average")
        leaf_T = hierarchy.leaves_list(link_T)
    else:
        leaf_T = np.arange(T)
    tc_z_disp = tc_z[np.ix_(leaf_T, ct_order)]
    trait_disp = [traits[i].replace("_", " ") for i in leaf_T]

    fig = plt.figure(figsize=(max(11, 0.32 * n_ct + 5), 0.55 * T + 4.5))
    gs = fig.add_gridspec(2, 2,
                          width_ratios=[1, 0.022],
                          height_ratios=[0.06, 1.0],
                          hspace=0.12, wspace=0.02,
                          left=0.30, right=0.94, top=0.86, bottom=0.30)
    ax_lin = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[1, 0])
    ax_cb = fig.add_subplot(gs[1, 1])

    # Lineage strip
    lin_to_int = {l: i for i, l in enumerate(LINEAGE_ORDER)}
    lin_arr = np.array([[lin_to_int[l] for l in lin_sorted]])
    cmap_lin = ListedColormap([LINEAGE_COLORS[l] for l in LINEAGE_ORDER])
    ax_lin.imshow(lin_arr, aspect="auto", cmap=cmap_lin,
                  extent=(-0.5, n_ct - 0.5, 0, 1),
                  vmin=0, vmax=len(LINEAGE_ORDER) - 1, interpolation="nearest")
    ax_lin.set_yticks([])
    ax_lin.set_xticks([])
    ax_lin.set_xlim(-0.5, n_ct - 0.5)
    ax_lin.set_ylabel("lineage", rotation=0, labelpad=28, va="center", fontsize=9)

    # Main heatmap
    vmax = float(np.nanpercentile(np.abs(tc_z_disp), 98))
    im = ax_h.imshow(tc_z_disp, aspect="auto", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_h.set_yticks(range(T))
    ax_h.set_yticklabels(trait_disp, fontsize=9)
    ax_h.set_xticks(range(n_ct))
    ax_h.set_xticklabels([f"{c} (n={s})" for c, s in zip(ct_sorted, sizes_sorted)],
                         rotation=90, fontsize=8)
    ax_h.set_xlabel("")
    ax_h.set_xlim(-0.5, n_ct - 0.5)
    fig.colorbar(im, cax=ax_cb, label="row z-score\n(trait score across celltypes)")

    # Lineage legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=LINEAGE_COLORS[l]) for l in LINEAGE_ORDER]
    fig.legend(handles, LINEAGE_ORDER, loc="lower center", ncol=len(LINEAGE_ORDER),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.04))
    fig.suptitle("Trait × celltype: mean trait-panel expression (row z-score)",
                 fontsize=13, y=0.95)
    fig.savefig(OUT / "figA_trait_by_celltype.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "figA_trait_by_celltype.pdf", bbox_inches="tight")
    plt.close(fig)

    # =======================================================================
    # FIGURE B: gene × celltype heatmap, hierarchically clustered
    # =======================================================================
    print("[7/8] Drawing Fig B (gene × celltype)...", flush=True)
    # row-z-score for genes; drop genes with zero variance
    g_mu = gc.mean(axis=1, keepdims=True)
    g_sd = gc.std(axis=1, keepdims=True)
    keep = (g_sd[:, 0] > 1e-6)
    gc_z = np.zeros_like(gc)
    gc_z[keep, :] = (gc[keep, :] - g_mu[keep, :]) / g_sd[keep, :]
    print(f"      {int(keep.sum())}/{G} genes have non-zero variance", flush=True)

    # Cluster the genes that vary
    gc_z_keep = gc_z[keep, :]
    link_g = hierarchy.linkage(pdist(gc_z_keep, metric="correlation"), method="average")
    leaf_g = hierarchy.leaves_list(link_g)
    gene_order_idx = np.where(keep)[0][leaf_g]  # absolute gene indices in clustered order
    gc_z_disp = gc_z[gene_order_idx, :][:, ct_order]

    # Trait membership matrix (genes_kept × T) — strip on the left
    member = np.zeros((len(gene_order_idx), T), dtype=int)
    for col, t in enumerate(traits):
        gs = set(trait_to_genes[t])
        for row, gi in enumerate(gene_order_idx):
            if universe[gi] in gs:
                member[row, col] = 1

    fig = plt.figure(figsize=(max(11, 0.32 * n_ct + 5), 0.04 * len(gene_order_idx) + 5.5))
    gs = fig.add_gridspec(2, 3,
                          width_ratios=[0.18, 1.0, 0.022],
                          height_ratios=[0.025, 1.0],
                          hspace=0.018, wspace=0.012,
                          left=0.04, right=0.94, top=0.91, bottom=0.18)
    ax_lin = fig.add_subplot(gs[0, 1])
    ax_mem = fig.add_subplot(gs[1, 0])
    ax_h = fig.add_subplot(gs[1, 1], sharey=ax_mem)
    ax_cb = fig.add_subplot(gs[1, 2])

    # Lineage strip
    ax_lin.imshow(lin_arr, aspect="auto", cmap=cmap_lin,
                  extent=(-0.5, n_ct - 0.5, 0, 1),
                  vmin=0, vmax=len(LINEAGE_ORDER) - 1, interpolation="nearest")
    ax_lin.set_xticks([]); ax_lin.set_yticks([])
    ax_lin.set_xlim(-0.5, n_ct - 0.5)

    # Trait membership panel
    # Use a categorical colormap mapping trait index → color
    trait_palette = plt.get_cmap("tab10")(np.linspace(0, 1, T))
    member_rgba = np.ones((member.shape[0], member.shape[1], 4), dtype=float)
    for col in range(T):
        rgba = trait_palette[col]
        for row in range(member.shape[0]):
            if member[row, col] == 1:
                member_rgba[row, col] = rgba
    ax_mem.imshow(member_rgba, aspect="auto", interpolation="nearest")
    ax_mem.set_xticks(range(T))
    ax_mem.set_xticklabels([t.replace("_", " ") for t in traits],
                           rotation=90, fontsize=7)
    ax_mem.tick_params(axis="x", which="both", length=0)
    ax_mem.xaxis.tick_bottom()
    ax_mem.xaxis.set_label_position("bottom")
    ax_mem.set_yticks([])
    ax_mem.set_xlabel("")

    # Main heatmap
    vmax = float(np.nanpercentile(np.abs(gc_z_disp), 98))
    im = ax_h.imshow(gc_z_disp, aspect="auto", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_h.set_yticks([])
    ax_h.set_xticks(range(n_ct))
    ax_h.set_xticklabels(ct_sorted, rotation=90, fontsize=8)
    fig.colorbar(im, cax=ax_cb, label="gene z-score\nacross celltypes")

    fig.suptitle(f"Gene × celltype: {len(gene_order_idx)} disease panel genes "
                 "(hierarchically clustered)\nleft strip = trait membership · top strip = lineage colour",
                 fontsize=12, y=0.97)
    fig.savefig(OUT / "figB_gene_by_celltype.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "figB_gene_by_celltype.pdf", bbox_inches="tight")
    plt.close(fig)

    # =======================================================================
    # FIGURE C: gene specificity (tau)
    # =======================================================================
    print("[7/8] Drawing Fig C (tau specificity)...", flush=True)
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0],
                          hspace=1.15, wspace=0.45,
                          left=0.07, right=0.98, top=0.93, bottom=0.08)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[0, 1])
    ax_top = fig.add_subplot(gs[1, 0])
    ax_bot = fig.add_subplot(gs[1, 1])

    valid = tau_df.dropna(subset=["tau"])
    ax_hist.hist(valid["tau"], bins=40, color="#34495e", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax_hist.axvline(0.5, color="grey", ls="--", lw=1)
    ax_hist.set_xlabel("Yanai's tau (0 = ubiquitous, 1 = single-celltype specific)")
    ax_hist.set_ylabel("# disease genes")
    ax_hist.set_title(f"Specificity distribution ({len(valid)} genes with non-zero variance)",
                      fontsize=10)
    med = float(valid["tau"].median())
    ax_hist.text(0.98, 0.95, f"median tau = {med:.2f}",
                 transform=ax_hist.transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))

    # Tau by trait (boxplot — does the trait panel have intrinsically broad / specific genes?)
    box_data = []
    box_labels = []
    for t in traits:
        gs2 = trait_to_genes[t]
        vals = valid.loc[valid["gene"].isin(gs2), "tau"].values
        if len(vals) > 0:
            box_data.append(vals)
            box_labels.append(f"{t.replace('_', ' ')}\n(k={len(vals)})")
    bp = ax_box.boxplot(box_data, labels=box_labels, vert=True, patch_artist=True,
                        showfliers=False, widths=0.6,
                        medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], plt.get_cmap("tab10")(np.linspace(0, 1, len(bp["boxes"])))):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax_box.set_xticklabels(box_labels, rotation=40, ha="right", fontsize=7)
    ax_box.set_ylabel("tau")
    ax_box.set_title("Tau distribution per trait panel", fontsize=10)
    ax_box.axhline(med, color="grey", ls="--", lw=0.7)

    # Top-20 most specific
    top = tau_df.head(20)
    bars = ax_top.barh(range(len(top))[::-1], top["tau"], color="#c0392b", alpha=0.85,
                       edgecolor="black", linewidth=0.4)
    ax_top.set_yticks(range(len(top))[::-1])
    ax_top.set_yticklabels([f"{g}  →  {c}" for g, c in zip(top["gene"], top["max_celltype"])],
                           fontsize=8)
    ax_top.set_xlabel("tau")
    ax_top.set_xlim(0, 1)
    ax_top.set_title("Top-20 most celltype-specific disease genes", fontsize=10)

    # Bottom-20 broadest (with positive variance)
    bot = valid.tail(20).iloc[::-1]
    ax_bot.barh(range(len(bot))[::-1], bot["tau"], color="#16a085", alpha=0.85,
                edgecolor="black", linewidth=0.4)
    ax_bot.set_yticks(range(len(bot))[::-1])
    ax_bot.set_yticklabels([f"{g}  ({c})" for g, c in zip(bot["gene"], bot["max_celltype"])],
                           fontsize=8)
    ax_bot.set_xlabel("tau")
    ax_bot.set_xlim(0, 1)
    ax_bot.set_title("Bottom-20 broadest disease genes", fontsize=10)

    fig.suptitle("Per-gene celltype specificity (Yanai's tau)", fontsize=12, y=0.985)
    fig.savefig(OUT / "figC_gene_specificity.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "figC_gene_specificity.pdf", bbox_inches="tight")
    plt.close(fig)

    # =======================================================================
    # FIGURE D: trait similarity
    # =======================================================================
    print("[7/8] Drawing Fig D (trait similarity)...", flush=True)
    # Reorder both matrices by hierarchical clustering on (1 - pearson)
    dist = 1 - pear
    np.fill_diagonal(dist, 0)
    link_d = hierarchy.linkage(squareform(dist, checks=False), method="average")
    leaf_d = hierarchy.leaves_list(link_d)
    pear_o = pear[np.ix_(leaf_d, leaf_d)]
    jacc_o = jacc[np.ix_(leaf_d, leaf_d)]
    labels_o = [traits[i].replace("_", " ") for i in leaf_d]

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1],
                          left=0.05, right=0.98, top=0.86, bottom=0.34, wspace=0.95)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Pearson heatmap
    im1 = ax1.imshow(pear_o, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax1.set_xticks(range(T)); ax1.set_yticks(range(T))
    ax1.set_xticklabels(labels_o, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(labels_o, fontsize=8)
    ax1.set_title("Trait–trait Pearson on celltype-score profile", fontsize=10)
    fig.colorbar(im1, ax=ax1, shrink=0.7, pad=0.04, label="pearson r")

    # Jaccard heatmap (same ordering for visual comparison)
    im2 = ax2.imshow(jacc_o, cmap="Greens", vmin=0, vmax=max(0.05, jacc_o[~np.eye(T, dtype=bool)].max()),
                     interpolation="nearest")
    ax2.set_xticks(range(T)); ax2.set_yticks(range(T))
    ax2.set_xticklabels(labels_o, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(labels_o, fontsize=8)
    ax2.set_title("Trait–trait Jaccard on gene panels", fontsize=10)
    fig.colorbar(im2, ax=ax2, shrink=0.7, pad=0.04, label="jaccard")

    # Scatter: jaccard vs pearson (off-diagonal pairs)
    iu = np.triu_indices(T, k=1)
    j_vec = jacc[iu]
    p_vec = pear[iu]
    short = {
        "Atrial_septal_defect": "ASD",
        "Atrioventricular_septal_defect": "AVSD",
        "DilatedCardiomyopathy": "DCM",
        "Familial_thoracic_aortic_aneurysm_and_aortic_dissection": "TAA",
        "HypertrophicCardiomyopathy": "HCM",
        "Malformation_of_the_outflow_tract": "OFT",
        "PCGC_DeNovoVariants": "PCGC",
        "Single_ventricle_disease": "SingleVentricle",
        "Valve_defects": "Valve",
        "Ventricular_septal_defect": "VSD",
    }
    ax3.scatter(j_vec, p_vec, s=60, c="#2c3e50", alpha=0.75, edgecolors="black", linewidths=0.4)
    # Highlight: high pearson with low panel overlap = functional convergence via different genes
    interesting = (p_vec > 0.7) & (j_vec < 0.04)
    if interesting.any():
        ax3.scatter(j_vec[interesting], p_vec[interesting], s=110, facecolor="none",
                    edgecolors="#c0392b", linewidths=1.5,
                    label="high r, low jaccard\n(convergence via different genes)")
        # Spread annotations vertically to avoid stacking. Sort by descending p, alternate offsets.
        idxs = sorted(np.where(interesting)[0], key=lambda k: -p_vec[k])
        for rank, k_ in enumerate(idxs):
            i, j = iu[0][k_], iu[1][k_]
            # alternate dx to spread horizontally; dy stair-steps down
            dx = 14 if rank % 2 == 0 else 60
            dy = 12 - rank * 14
            ax3.annotate(f"{short.get(traits[i], traits[i][:6])} ↔ {short.get(traits[j], traits[j][:6])}",
                         (j_vec[k_], p_vec[k_]),
                         xytext=(dx, dy), textcoords="offset points", fontsize=8,
                         arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.5))
    ax3.axhline(0, color="grey", lw=0.7, ls=":")
    ax3.set_xlabel("Jaccard of gene panels")
    ax3.set_ylabel("Pearson r of celltype-score vectors")
    ax3.set_title("Panel overlap vs functional similarity", fontsize=10)
    ax3.set_xlim(left=-0.02)
    if interesting.any():
        ax3.legend(loc="lower right", fontsize=8, frameon=False)

    fig.suptitle("Trait–trait similarity: gene-set overlap vs celltype-profile correlation",
                 fontsize=13, y=0.96)
    fig.savefig(OUT / "figD_trait_similarity.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "figD_trait_similarity.pdf", bbox_inches="tight")
    plt.close(fig)

    # =======================================================================
    # Console summary
    # =======================================================================
    print("[8/8] Summary tables", flush=True)
    print("\n=== Top celltype per trait ===")
    for t in traits:
        v = tc_df.loc[t]
        ranking = v.sort_values(ascending=False)
        print(f"  {t:<48s}  top3 = {list(ranking.index[:3])}")

    print("\n=== Top-15 most-specific genes ===")
    print(tau_df.head(15)[["gene", "tau", "max_celltype", "traits"]].to_string(index=False))

    print("\n=== Top trait pairs by Pearson celltype-profile correlation ===")
    print(sim_df.head(10)[["trait_a", "trait_b", "pearson_celltype", "jaccard", "n_overlap"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
