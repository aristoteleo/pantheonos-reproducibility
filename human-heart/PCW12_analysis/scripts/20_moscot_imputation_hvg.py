"""
MOSCOT scRNA->MERFISH imputation with an EXPANDED gene set:
    impute_genes = (top-5000 HVG on SC) ∪ (heart_disease_genes.tsv)

This is the "v2" companion to 03_moscot_imputation.py. The mapping cost is
unchanged (alpha=0, tau_a=1, tau_b=0.9, 5×20k batches, cell-cycle filtered).
Only the imputed gene set is broader, so the resulting AnnData covers far more
of the Spateo human LR database (the previous 221-gene panel matched only
2 ligands + 3 receptors).

Inputs:
  data/all_healthy_RoundedPCW11-13.h5ad        # SC reference (user approved)
  data/full_heart_final_aug2025_update_downsampled_100k.h5ad
  data/heart_disease_genes.tsv                  # column: Gene

Output:
  PCW12_analysis/data/adata_imputed_hvg_disease.h5ad
"""
from __future__ import annotations
import os, sys, time, gc
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
from scipy import sparse

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
SC_PATH = ROOT / "data/all_healthy_RoundedPCW11-13.h5ad"          # user choice: PCW11-13
SP_PATH = ROOT / "data/full_heart_final_aug2025_update_downsampled_100k.h5ad"
DISEASE_TSV = ROOT / "data/heart_disease_genes.tsv"
OUT_DIR = ROOT / "PCW12_analysis/data"
LOG_DIR = ROOT / "PCW12_analysis/logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

OUT_H5AD = OUT_DIR / "adata_imputed_hvg_disease.h5ad"

BATCH_SIZE = 20_000
RANDOM_STATE = 42
N_HVG = 5000
HVG_FLAVOR = "seurat"   # user-confirmed; computed on normalize_total + log1p

# Cell-cycle genes from the skill — dropped only from the MAPPING cost,
# NOT from the imputation set (we still want to be able to impute them).
S_GENES = ["RRM2","DSCC1","PRIM1","GMNN","CCNE2","E2F8","EXO1","RAD51AP1","WDR76",
           "USP1","NASP","CASP8AP2","RAD51","MSH2","PCNA","FEN1","RRM1","CDC6",
           "CLSPN","POLA1","TYMS","SLBP","CENPU","MCM5","TIPIN","MCM4","MCM6",
           "RFC2","UNG","CHAF1B","CDC45","HELLS","MRPL36","POLR1B","BLM","CDCA7",
           "DTL","UHRF1","UBR7","MCM7","GINS2"]
G2M_GENES = ["NEK2","CDCA8","SMC4","LBR","ANP32E","HMMR","AURKB","CDC20","KIF11",
             "RANGAP1","CDK1","GTSE1","TPX2","NDC80","CKAP2","MKI67","ECT2","G2E3",
             "CENPE","NCAPD2","PIMREG","CDC25C","CENPF","TUBB4B","CENPA","BUB1",
             "PSRC1","NUF2","TOP2A","GAS2L3","NUSAP1","TACC3","CBX5","AURKA",
             "CDCA3","KIF2C","BIRC5","HMGB2","KIF20B","TTK","TMPO","UBE2C","CKS2",
             "DLGAP5","CKAP2L","ANLN","CKAP5","HJURP","CCNB2","CKS1B","CDCA2",
             "KIF23","CTCF"]
CC_GENES = {g.upper() for g in S_GENES + G2M_GENES}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    # ---------- load ----------
    log(f"Loading SC: {SC_PATH.name}")
    adata_sc = ad.read_h5ad(SC_PATH)
    log(f"  SC: {adata_sc.shape}, X dtype {adata_sc.X.dtype}")

    log(f"Loading SP: {SP_PATH.name}")
    adata_sp = ad.read_h5ad(SP_PATH)
    log(f"  SP: {adata_sp.shape}, X dtype {adata_sp.X.dtype}")

    if adata_sc.X.dtype != np.float32:
        log("Casting SC.X to float32 ...")
        adata_sc.X = adata_sc.X.astype(np.float32)

    log("Normalizing SC (normalize_total + log1p) ...")
    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.log1p(adata_sc)
    log(f"  SC X range: {float(adata_sc.X.min()):.2f}..{float(adata_sc.X.max()):.2f}")
    log(f"  SP X range: {float(adata_sp.X.min()):.2f}..{float(adata_sp.X.max()):.2f}")

    # ---------- compute HVGs on SC ----------
    log(f"Computing top-{N_HVG} HVGs on SC (flavor={HVG_FLAVOR}) ...")
    sc.pp.highly_variable_genes(adata_sc, n_top_genes=N_HVG, flavor=HVG_FLAVOR)
    hvg_set = set(adata_sc.var_names[adata_sc.var["highly_variable"]])
    log(f"  HVGs: {len(hvg_set)}")

    # ---------- disease gene list ----------
    log(f"Loading disease genes from {DISEASE_TSV.name}")
    dis_df = pd.read_csv(DISEASE_TSV, sep="\t")
    if "Gene" not in dis_df.columns:
        raise ValueError(f"Expected 'Gene' column in {DISEASE_TSV}; got {list(dis_df.columns)}")
    disease_set = set(dis_df["Gene"].astype(str).str.strip().str.upper().unique())
    disease_set = {g for g in disease_set if g and g != "NAN"}
    log(f"  Unique disease genes (uppercased): {len(disease_set)}")

    # ---------- impute set = (HVG ∪ disease) ∩ SC vars ----------
    sc_vars_upper = {g.upper(): g for g in adata_sc.var_names}
    union_upper = {g.upper() for g in hvg_set} | disease_set
    impute_genes = sorted({sc_vars_upper[u] for u in union_upper if u in sc_vars_upper})
    n_hvg_in = sum(1 for g in impute_genes if g in hvg_set)
    n_dis_in = sum(1 for g in impute_genes if g.upper() in disease_set)
    log(f"  Final impute_genes = HVG ∪ disease ∩ SC: {len(impute_genes)} "
        f"(HVG-only contribution {n_hvg_in}, disease contribution {n_dis_in})")

    # ---------- shared genes (mapping cost) ----------
    shared = adata_sc.var_names.intersection(adata_sp.var_names)
    log(f"Shared SC∩SP genes (for mapping cost): {len(shared)}")
    map_genes = [g for g in shared if g.upper() not in CC_GENES]
    log(f"  After dropping CC genes: {len(map_genes)} mapping genes")

    adata_sc_map = adata_sc[:, map_genes].copy()
    adata_sp_map = adata_sp[:, map_genes].copy()

    # spatial coordinates for MOSCOT (raw, NOT Z-flipped here — flip in CCI prep)
    if "spatial" not in adata_sp_map.obsm:
        adata_sp_map.obsm["spatial"] = np.asarray(
            adata_sp_map.obsm["X_spateo_update"], dtype=np.float32
        ).copy()
        log("  Copied obsm['X_spateo_update'] -> obsm['spatial'] for MOSCOT")

    # ---------- pre-extract impute matrix ----------
    sc_impute_X = adata_sc[:, impute_genes].X
    if sparse.issparse(sc_impute_X):
        sc_impute_X = sc_impute_X.toarray()
    sc_impute_X = np.asarray(sc_impute_X, dtype=np.float32)
    mem_mb = sc_impute_X.nbytes / 1e6
    log(f"  sc_impute_X: {sc_impute_X.shape}  ({mem_mb:.1f} MB)")

    sc_celltype = adata_sc.obs["celltype"].astype(str).values
    sc_coarse = adata_sc.obs["coarse_celltype"].astype(str).values \
        if "coarse_celltype" in adata_sc.obs else None
    sc_label = adata_sc.obs["celltype_label"].astype(str).values \
        if "celltype_label" in adata_sc.obs else None

    # ---------- batches ----------
    n_sp = adata_sp_map.n_obs
    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(n_sp)
    batch_indices = [perm[i:i + BATCH_SIZE] for i in range(0, n_sp, BATCH_SIZE)]
    log(f"SP batches: {len(batch_indices)} of size up to {BATCH_SIZE}")

    # Free SC
    del adata_sc
    gc.collect()

    imputed_full = np.zeros((n_sp, len(impute_genes)), dtype=np.float32)
    mapped_celltype = np.empty(n_sp, dtype=object)
    mapped_coarse = np.empty(n_sp, dtype=object) if sc_coarse is not None else None
    mapped_label = np.empty(n_sp, dtype=object) if sc_label is not None else None
    confidence = np.zeros(n_sp, dtype=np.float32)

    from moscot.problems.space import MappingProblem  # noqa: E402

    for bi, idx in enumerate(batch_indices):
        log(f"=== Batch {bi+1}/{len(batch_indices)}  size={len(idx)} ===")
        sp_batch = adata_sp_map[idx].copy()
        mp = MappingProblem(adata_sc_map, sp_batch)
        t0 = time.time()
        mp = mp.prepare(sc_attr=None, xy_callback="local-pca")
        log(f"  prepare done in {time.time()-t0:.1f}s")
        t0 = time.time()
        mp = mp.solve(alpha=0, tau_a=1, tau_b=0.9, device="cpu")
        log(f"  solve done in {time.time()-t0:.1f}s")

        sol = list(mp.solutions.values())[0]
        pi = np.array(sol.transport_matrix, dtype=np.float32, copy=True)
        pi *= float(pi.shape[0])
        log(f"  pi shape {pi.shape}")

        t0 = time.time()
        imputed_full[idx] = pi @ sc_impute_X
        log(f"  impute matmul {time.time()-t0:.1f}s")

        for src_labels, dst_array in (
            (sc_celltype, mapped_celltype),
            (sc_coarse, mapped_coarse),
            (sc_label, mapped_label),
        ):
            if src_labels is None or dst_array is None:
                continue
            uniq, inv = np.unique(src_labels, return_inverse=True)
            M = np.eye(len(uniq), dtype=np.float32)[inv]
            scores = pi @ M
            best = scores.argmax(axis=1)
            dst_array[idx] = uniq[best]
            if dst_array is mapped_celltype:
                row_sum = scores.sum(axis=1) + 1e-9
                conf = scores[np.arange(scores.shape[0]), best] / row_sum
                confidence[idx] = conf

        del pi, mp, sp_batch
        gc.collect()

    # ---------- assemble output AnnData ----------
    log("Building imputed AnnData ...")
    obs_out = adata_sp.obs.copy()
    obs_out["mapped_celltype"] = mapped_celltype
    if mapped_coarse is not None:
        obs_out["mapped_coarse_celltype"] = mapped_coarse
    if mapped_label is not None:
        obs_out["mapped_celltype_label"] = mapped_label
    obs_out["mapping_confidence"] = confidence

    var_out = pd.DataFrame(index=impute_genes)
    var_out["is_hvg"] = [g in hvg_set for g in impute_genes]
    var_out["is_disease"] = [g.upper() in disease_set for g in impute_genes]

    adata_imp = AnnData(
        X=sparse.csr_matrix(imputed_full),
        obs=obs_out,
        var=var_out,
        obsm={k: v for k, v in adata_sp.obsm.items()},
    )
    adata_imp.uns["impute_meta"] = {
        "n_hvg_top": N_HVG,
        "hvg_flavor": HVG_FLAVOR,
        "n_hvg_in_set": int(var_out["is_hvg"].sum()),
        "n_disease_in_set": int(var_out["is_disease"].sum()),
        "n_overlap_hvg_disease": int((var_out["is_hvg"] & var_out["is_disease"]).sum()),
        "sc_reference": SC_PATH.name,
    }
    adata_imp.write_h5ad(OUT_H5AD)
    log(f"Saved -> {OUT_H5AD}  ({adata_imp.shape}, "
        f"{OUT_H5AD.stat().st_size/1e6:.1f} MB)")

    log(f"  is_hvg={var_out['is_hvg'].sum()}, is_disease={var_out['is_disease'].sum()}, "
        f"both={(var_out['is_hvg'] & var_out['is_disease']).sum()}")
    log(f"Mapping confidence: mean={confidence.mean():.3f} median={np.median(confidence):.3f}")
    log("Mapped celltype counts (top 15):")
    log(pd.Series(mapped_celltype).value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
