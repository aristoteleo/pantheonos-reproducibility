"""Statistical test for valve enrichment of 10 disease-trait gene panels.

For each trait T:
1. Compute per-cell trait score = mean over g in G_T of log1p(X_imp[cell, g]).
2. Mann-Whitney U one-sided (valve > non-valve), AUC, Cliff's delta.
3. Permutation null: 2 000 random gene sets of size |G_T| from the 221-gene
   universe \ G_T; empirical p on observed Δμ.
4. BH-FDR across the 10 traits for both p-value flavours.

Secondary:
- Per-subtype (VIC, VEC, ncCM-AVC-like) vs all non-valve: MWU only.
- Robustness: valve vs fibroblast subset (Compact / Trabecular / aFibro /
  adFibro / Proliferating vFibro): MWU only.

Outputs (PCW12_analysis/figures/valve_enrichment_stats/):
- valve_enrichment_stats.tsv         primary table
- valve_enrichment_by_subtype.tsv    per-subtype MWU
- valve_vs_fibroblast.tsv            robustness comparator
- fig1_bar_auc.{png,pdf}              traits by AUC, colored by -log10 q_perm
- fig2_volcano.{png,pdf}              log2FC vs -log10 q_perm
- fig3_top_distributions.{png,pdf}    boxplots for top-3 enriched traits
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
DATA = ROOT / "PCW12_analysis" / "data"
OUT = ROOT / "PCW12_analysis" / "figures" / "valve_enrichment_stats"
OUT.mkdir(parents=True, exist_ok=True)

VALVE_TYPES = ["VIC", "VEC", "ncCM-AVC-like"]
FIBRO_TYPES = [
    "Compact vFibro",
    "Trabecular vFibro",
    "Proliferating vFibro",
    "aFibro",
    "adFibro",
    "EPDC",
]
N_PERM = 2000
SEED = 20251015


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR. Returns q-values aligned with input order."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def trait_score(X_log: np.ndarray, gene_idx: np.ndarray) -> np.ndarray:
    """Mean log1p over the chosen gene columns. X_log is already log1p."""
    if len(gene_idx) == 0:
        return np.zeros(X_log.shape[0])
    return np.asarray(X_log[:, gene_idx]).mean(axis=1)


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ---- Load data ---------------------------------------------------------
    print("[1/6] Loading imputed adata...", flush=True)
    a = ad.read_h5ad(DATA / "adata_imputed_disease_genes.h5ad")
    print(f"      shape={a.shape}  celltypes={a.obs['celltype'].nunique()}")

    # log1p once (X is already on a scaled count-like scale; consistent with
    # how spatial scoring was done)
    X = a.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_log = np.log1p(X)
    universe = a.var_names.to_numpy()
    gene_to_idx = {g: i for i, g in enumerate(universe)}

    # ---- Trait gene sets ---------------------------------------------------
    part = pd.read_csv(DATA / "disease_genes_partition.tsv", sep="\t")
    trait_to_genes: dict[str, list[str]] = {}
    for _, row in part.iterrows():
        if pd.isna(row["traits"]):
            continue
        for t in str(row["traits"]).split(";"):
            trait_to_genes.setdefault(t, []).append(row["Gene"])
    # intersect with universe
    trait_to_idx: dict[str, np.ndarray] = {}
    for t, gs in trait_to_genes.items():
        idx = np.array([gene_to_idx[g] for g in gs if g in gene_to_idx])
        if len(idx) >= 3:  # need at least 3 genes
            trait_to_idx[t] = idx
    traits = sorted(trait_to_idx.keys())
    print(f"      {len(traits)} traits with >=3 genes in universe")
    for t in traits:
        print(f"        {t}: {len(trait_to_idx[t])} genes")

    # ---- Group masks -------------------------------------------------------
    ct = a.obs["celltype"].astype(str)
    valve = ct.isin(VALVE_TYPES).to_numpy()
    fibro = ct.isin(FIBRO_TYPES).to_numpy()
    n_valve = valve.sum()
    n_nonvalve = (~valve).sum()
    print(f"      valve={n_valve}  non-valve={n_nonvalve}  fibro={fibro.sum()}")

    # ---- Primary test: valve vs non-valve ---------------------------------
    print(f"[2/6] Running MWU + {N_PERM} permutations per trait...", flush=True)
    rows = []
    for t in traits:
        idx = trait_to_idx[t]
        s = trait_score(X_log, idx)
        s_v = s[valve]
        s_n = s[~valve]
        mu_v = float(s_v.mean())
        mu_n = float(s_n.mean())
        delta = mu_v - mu_n
        # log2FC of expm1 of means (back to linear-ish space, +1 to avoid /0)
        log2fc = float(np.log2((np.expm1(mu_v) + 1e-9) / (np.expm1(mu_n) + 1e-9)))
        # MWU one-sided greater
        mwu = stats.mannwhitneyu(s_v, s_n, alternative="greater")
        auc = float(mwu.statistic / (n_valve * n_nonvalve))
        cliffs = 2 * auc - 1

        # Permutation: random gene sets of same size from universe \ G_T
        avail = np.setdiff1d(np.arange(len(universe)), idx, assume_unique=False)
        k = len(idx)
        null_delta = np.empty(N_PERM)
        for i in range(N_PERM):
            rs = rng.choice(avail, size=k, replace=False)
            ss = trait_score(X_log, rs)
            null_delta[i] = ss[valve].mean() - ss[~valve].mean()
        # one-sided p: how often does null Δμ ≥ observed?
        perm_p = (np.sum(null_delta >= delta) + 1) / (N_PERM + 1)
        null_mean = float(null_delta.mean())
        null_sd = float(null_delta.std(ddof=1) + 1e-12)
        z_perm = (delta - null_mean) / null_sd

        rows.append(
            dict(
                trait=t,
                n_genes=int(k),
                n_valve=int(n_valve),
                n_nonvalve=int(n_nonvalve),
                mean_valve=mu_v,
                mean_nonvalve=mu_n,
                delta=delta,
                log2FC=log2fc,
                AUC=auc,
                cliffs_delta=cliffs,
                MWU_U=float(mwu.statistic),
                MWU_p=float(mwu.pvalue),
                perm_p=float(perm_p),
                null_mean=null_mean,
                null_sd=null_sd,
                z_perm=float(z_perm),
                n_perm=N_PERM,
            )
        )
        print(
            f"        {t:<48s} k={k:>3d} AUC={auc:.3f} log2FC={log2fc:+.3f} "
            f"MWU_p={mwu.pvalue:.2e} perm_p={perm_p:.4f} z_perm={z_perm:+.2f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df["MWU_q"] = bh_fdr(df["MWU_p"].values)
    df["perm_q"] = bh_fdr(df["perm_p"].values)
    df["sig_perm_q05"] = df["perm_q"] < 0.05
    df = df.sort_values("AUC", ascending=False).reset_index(drop=True)
    df.to_csv(OUT / "valve_enrichment_stats.tsv", sep="\t", index=False)
    print(f"      wrote {OUT / 'valve_enrichment_stats.tsv'}")

    # ---- Per-subtype breakdown --------------------------------------------
    print("[3/6] Per-subtype MWU vs all non-valve...", flush=True)
    sub_rows = []
    for sub in VALVE_TYPES:
        m = (ct == sub).to_numpy()
        n_sub = int(m.sum())
        for t in traits:
            idx = trait_to_idx[t]
            s = trait_score(X_log, idx)
            s_sub = s[m]
            s_other = s[~valve]  # all non-valve
            if n_sub < 3:
                sub_rows.append(
                    dict(subtype=sub, trait=t, n=n_sub, AUC=np.nan, MWU_p=np.nan)
                )
                continue
            mwu = stats.mannwhitneyu(s_sub, s_other, alternative="greater")
            auc = float(mwu.statistic / (n_sub * len(s_other)))
            sub_rows.append(
                dict(
                    subtype=sub,
                    trait=t,
                    n=n_sub,
                    n_genes=int(len(idx)),
                    mean_sub=float(s_sub.mean()),
                    mean_nonvalve=float(s_other.mean()),
                    AUC=auc,
                    cliffs_delta=2 * auc - 1,
                    MWU_p=float(mwu.pvalue),
                )
            )
    sub_df = pd.DataFrame(sub_rows)
    # FDR within each subtype
    for sub in VALVE_TYPES:
        m = sub_df["subtype"] == sub
        valid = sub_df.loc[m, "MWU_p"].notna()
        q = np.full(m.sum(), np.nan)
        if valid.any():
            q[valid.values] = bh_fdr(sub_df.loc[m & sub_df["MWU_p"].notna(), "MWU_p"].values)
        sub_df.loc[m, "MWU_q"] = q
    sub_df.to_csv(OUT / "valve_enrichment_by_subtype.tsv", sep="\t", index=False)
    print(f"      wrote {OUT / 'valve_enrichment_by_subtype.tsv'}")

    # ---- Robustness: valve vs fibroblast ----------------------------------
    print("[4/6] Robustness: valve vs fibroblast...", flush=True)
    rob_rows = []
    n_f = int(fibro.sum())
    for t in traits:
        idx = trait_to_idx[t]
        s = trait_score(X_log, idx)
        s_v = s[valve]
        s_f = s[fibro]
        mwu = stats.mannwhitneyu(s_v, s_f, alternative="greater")
        auc = float(mwu.statistic / (n_valve * n_f))
        rob_rows.append(
            dict(
                trait=t,
                n_valve=int(n_valve),
                n_fibro=n_f,
                n_genes=int(len(idx)),
                mean_valve=float(s_v.mean()),
                mean_fibro=float(s_f.mean()),
                AUC=auc,
                cliffs_delta=2 * auc - 1,
                MWU_p=float(mwu.pvalue),
            )
        )
    rob_df = pd.DataFrame(rob_rows)
    rob_df["MWU_q"] = bh_fdr(rob_df["MWU_p"].values)
    rob_df = rob_df.sort_values("AUC", ascending=False).reset_index(drop=True)
    rob_df.to_csv(OUT / "valve_vs_fibroblast.tsv", sep="\t", index=False)
    print(f"      wrote {OUT / 'valve_vs_fibroblast.tsv'}")

    # ---- Figure 1: bar chart by AUC, colored by -log10(q_perm) ------------
    print("[5/6] Drawing figures...", flush=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    df_plot = df.copy()
    nlogq = -np.log10(df_plot["perm_q"].clip(lower=1e-12))
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=max(2.0, float(nlogq.max())))
    colors = cmap(norm(nlogq.values))
    # show traits on y, AUC on x; pretty labels (replace underscores)
    pretty = [t.replace("_", " ") for t in df_plot["trait"]]
    bars = ax.barh(pretty[::-1], df_plot["AUC"][::-1], color=colors[::-1], edgecolor="black", linewidth=0.4)
    ax.axvline(0.5, color="grey", lw=1, ls="--", label="AUC=0.5 (no enrichment)")
    ax.set_xlabel("AUC (P[score(valve cell) > score(non-valve cell)])")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(f"Valve enrichment for disease-trait gene panels\nN={N_PERM} permutations · BH-FDR")
    # annotate sig
    for i, (auc, qp, k) in enumerate(zip(df_plot["AUC"][::-1], df_plot["perm_q"][::-1], df_plot["n_genes"][::-1])):
        marker = "***" if qp < 0.001 else ("**" if qp < 0.01 else ("*" if qp < 0.05 else "ns"))
        # Place annotation on the far side of the bar so it's always readable
        if auc >= 0.5:
            ax.text(auc + 0.01, i, f"{marker}  (k={k})", va="center", ha="left", fontsize=8)
        else:
            ax.text(auc - 0.01, i, f"(k={k})  {marker}", va="center", ha="right", fontsize=8)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, label="-log10 q_perm")
    fig.savefig(OUT / "fig1_bar_auc.png", dpi=200)
    fig.savefig(OUT / "fig1_bar_auc.pdf")
    plt.close(fig)

    # ---- Figure 2: volcano ------------------------------------------------
    # Use z_perm on x-axis instead of log2FC (more interpretable: SDs above null)
    # because most q_perm hit the floor of 0.0005, the volcano on q is uninformative.
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sizes = 40 + 4 * df["n_genes"]
    col = np.where(df["z_perm"] >= 0, "#c0392b", "#2980b9")
    ax.scatter(df["z_perm"], df["log2FC"], s=sizes, c=col, alpha=0.85, edgecolors="black", linewidths=0.4)
    # Manual label offsets to avoid overlap. Order by z_perm.
    label_offsets = {
        "Atrial_septal_defect": (8, -10),
        "Atrioventricular_septal_defect": (-10, 12),
        "DilatedCardiomyopathy": (8, 6),
        "Familial_thoracic_aortic_aneurysm_and_aortic_dissection": (-10, 12),
        "HypertrophicCardiomyopathy": (8, 6),
        "Malformation_of_the_outflow_tract": (8, 8),
        "PCGC_DeNovoVariants": (8, -10),
        "Single_ventricle_disease": (8, -8),
        "Valve_defects": (-8, 12),
        "Ventricular_septal_defect": (8, -8),
    }
    for _, r in df.iterrows():
        dx, dy = label_offsets.get(r["trait"], (6, 6))
        ha = "left" if dx > 0 else "right"
        ax.annotate(r["trait"].replace("_", " "),
                    (r["z_perm"], r["log2FC"]),
                    fontsize=8, xytext=(dx, dy), textcoords="offset points",
                    ha=ha,
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
    # significance reference: |z| >= 1.96 ~ p<0.05 in normal approx
    ax.axvline(1.96, color="grey", ls="--", lw=1, alpha=0.6, label="z=±1.96")
    ax.axvline(-1.96, color="grey", ls="--", lw=1, alpha=0.6)
    ax.axvline(0, color="black", ls=":", lw=0.8)
    ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("permutation z-score (SDs of valve-vs-nonvalve Δμ above random-panel null)")
    ax.set_ylabel("log2 fold change (mean valve / mean non-valve)")
    ax.set_title("Trait-panel enrichment in valve cells\n(red=enriched in valve, blue=depleted; bubble size ∝ #genes)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.savefig(OUT / "fig2_volcano.png", dpi=200)
    fig.savefig(OUT / "fig2_volcano.pdf")
    plt.close(fig)

    # ---- Figure 3: trait score distributions for top-3 by AUC -------------
    top3 = df.head(3)["trait"].tolist()
    bot3 = df.tail(3)["trait"].tolist()
    show = top3 + bot3
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False, constrained_layout=True)
    for ax, t in zip(axes.flat, show):
        idx = trait_to_idx[t]
        s = trait_score(X_log, idx)
        s_v = s[valve]
        s_n = s[~valve]
        bp = ax.boxplot([s_v, s_n], labels=["valve", "non-valve"], showfliers=False, widths=0.6,
                        patch_artist=True, medianprops=dict(color="black"))
        for patch, c in zip(bp["boxes"], ["#e67e22", "#7f8c8d"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        row = df[df["trait"] == t].iloc[0]
        ax.set_title(f"{t.replace('_', ' ')}\nAUC={row['AUC']:.3f}, q_perm={row['perm_q']:.2g}",
                     fontsize=9)
        ax.set_ylabel("trait score (mean log1p)")
    fig.suptitle("Top-3 enriched and bottom-3 (depleted) traits by AUC", fontsize=11)
    fig.savefig(OUT / "fig3_top_distributions.png", dpi=200)
    fig.savefig(OUT / "fig3_top_distributions.pdf")
    plt.close(fig)

    # ---- Save markdown summary -------------------------------------------
    print("[6/6] Writing summary table...", flush=True)
    print("\n=== PRIMARY: valve vs non-valve ===")
    cols_show = ["trait", "n_genes", "AUC", "log2FC", "MWU_q", "perm_p", "perm_q", "z_perm"]
    print(df[cols_show].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== ROBUSTNESS: valve vs fibroblast ===")
    print(rob_df[["trait", "n_genes", "AUC", "MWU_q"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== PER-SUBTYPE (top trait per subtype) ===")
    for sub in VALVE_TYPES:
        s = sub_df[sub_df["subtype"] == sub].sort_values("AUC", ascending=False).head(3)
        print(f"\n{sub}:")
        print(s[["trait", "n", "AUC", "MWU_q"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
