"""
MOSCOT-based mapping of scRNA-seq -> MERFISH spatial, then impute disease genes.

Strategy (per `omics/spatial/single_cell_spatial_mapping.md`):
  1. Load SC (PCW 11-13) and SP (100k MERFISH).
  2. Normalize SC (normalize_total + log1p); SP is already log1p.
  3. Restrict to shared genes; drop cell-cycle genes.
  4. Split SP into 5 batches of 20k cells.
  5. For each batch:
       - MappingProblem(SC, SP_batch).prepare(...).solve(alpha=0, tau_a=1, tau_b=0.9)
       - Scale pi by pi.shape[0]
       - Impute ONLY the disease impute_genes (saves memory)
       - Map celltype labels via transport
       - Free pi
  6. Concatenate per-batch results in original SP order; save h5ad.

JAX has CPU device only (no Metal/CUDA on this machine), so MOSCOT runs on CPU.
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
SC_PATH = ROOT / "data/all_healthy_RoundedPCW11-13.h5ad"
SP_PATH = ROOT / "data/full_heart_final_aug2025_update_downsampled_100k.h5ad"
PARTITION = ROOT / "PCW12_analysis/data/disease_genes_partition.tsv"
OUT_DIR = ROOT / "PCW12_analysis/data"
LOG_DIR = ROOT / "PCW12_analysis/logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 20000
RANDOM_STATE = 42

# Cell cycle genes from skill
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
    log("Loading SC ...")
    adata_sc = ad.read_h5ad(SC_PATH)
    log(f"  SC: {adata_sc.shape}, X dtype {adata_sc.X.dtype}")

    log("Loading SP (MERFISH) ...")
    adata_sp = ad.read_h5ad(SP_PATH)
    log(f"  SP: {adata_sp.shape}, X dtype {adata_sp.X.dtype}, layers {list(adata_sp.layers.keys())}")

    # SC may have float64 sparse — cast to float32 for memory
    if adata_sc.X.dtype != np.float32:
        log("Casting SC.X to float32 ...")
        adata_sc.X = adata_sc.X.astype(np.float32)

    log("Normalizing SC (normalize_total + log1p) ...")
    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.log1p(adata_sc)
    log(f"  SC X range: {float(adata_sc.X.min()):.2f}..{float(adata_sc.X.max()):.2f}")
    log(f"  SP X range: {float(adata_sp.X.min()):.2f}..{float(adata_sp.X.max()):.2f}")

    # Shared genes
    shared = adata_sc.var_names.intersection(adata_sp.var_names)
    log(f"Shared genes: {len(shared)} / SP {adata_sp.n_vars}")

    # Drop cell-cycle genes for the mapping cost
    map_genes = [g for g in shared if g.upper() not in CC_GENES]
    log(f"After dropping cell-cycle genes: {len(map_genes)} mapping genes")

    adata_sc_map = adata_sc[:, map_genes].copy()
    adata_sp_map = adata_sp[:, map_genes].copy()

    # MOSCOT MappingProblem looks for adata.obsm['spatial'] by default.
    # MERFISH coords live in obsm['X_spateo_update'] — copy under 'spatial'.
    if "spatial" not in adata_sp_map.obsm:
        adata_sp_map.obsm["spatial"] = np.asarray(
            adata_sp_map.obsm["X_spateo_update"], dtype=np.float32
        ).copy()
        log("  Copied obsm['X_spateo_update'] -> obsm['spatial'] for MOSCOT.")

    # Disease genes to impute (only those present in SC)
    part = pd.read_csv(PARTITION, sep="\t")
    impute_genes = part[part["class"] == "impute"]["Gene"].tolist()
    impute_genes = [g for g in impute_genes if g in adata_sc.var_names]
    log(f"Disease genes to impute: {len(impute_genes)}")

    # Pre-extract full-resolution SC expression for impute genes
    sc_impute_X = adata_sc[:, impute_genes].X
    if sparse.issparse(sc_impute_X):
        sc_impute_X = sc_impute_X.toarray()
    sc_impute_X = np.asarray(sc_impute_X, dtype=np.float32)
    log(f"  sc_impute_X: {sc_impute_X.shape}")

    sc_celltype = adata_sc.obs["celltype"].astype(str).values
    sc_coarse = adata_sc.obs["coarse_celltype"].astype(str).values \
        if "coarse_celltype" in adata_sc.obs else None
    sc_label = adata_sc.obs["celltype_label"].astype(str).values \
        if "celltype_label" in adata_sc.obs else None

    # ---- Split SP into batches ----
    n_sp = adata_sp_map.n_obs
    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(n_sp)
    batch_indices = [perm[i:i + BATCH_SIZE] for i in range(0, n_sp, BATCH_SIZE)]
    log(f"SP batches: {len(batch_indices)} of size up to {BATCH_SIZE}")

    # Free SC raw data we no longer need
    del adata_sc
    gc.collect()

    # Storage for imputed expression and mapped labels in SP order
    imputed_full = np.zeros((n_sp, len(impute_genes)), dtype=np.float32)
    mapped_celltype = np.empty(n_sp, dtype=object)
    mapped_coarse = np.empty(n_sp, dtype=object) if sc_coarse is not None else None
    mapped_label = np.empty(n_sp, dtype=object) if sc_label is not None else None
    confidence = np.zeros(n_sp, dtype=np.float32)

    from moscot.problems.space import MappingProblem  # noqa: E402

    for bi, idx in enumerate(batch_indices):
        log(f"=== Batch {bi+1}/{len(batch_indices)}  size={len(idx)} ===")
        sp_batch = adata_sp_map[idx].copy()
        # Per skill: at single-cell resolution, alpha=0 (linear OT), tau_a=1, tau_b=0.9
        mp = MappingProblem(adata_sc_map, sp_batch)
        t0 = time.time()
        mp = mp.prepare(sc_attr=None, xy_callback="local-pca")
        log(f"  prepare done in {time.time()-t0:.1f}s")
        t0 = time.time()
        mp = mp.solve(alpha=0, tau_a=1, tau_b=0.9, device="cpu")
        log(f"  solve done in {time.time()-t0:.1f}s")

        # transport matrix shape (n_sp, n_sc)
        sol = list(mp.solutions.values())[0]
        # JAX-backed arrays are read-only; force a writable float32 copy.
        pi = np.array(sol.transport_matrix, dtype=np.float32, copy=True)
        pi *= float(pi.shape[0])  # required scaling per skill
        log(f"  pi shape {pi.shape}, sum/row first: {pi.sum(axis=1)[:3]}")

        # Impute disease genes
        imputed_full[idx] = pi @ sc_impute_X
        log(f"  imputed batch -> imputed_full[{idx[:3]}...] sample max {imputed_full[idx].max():.3f}")

        # Map celltype labels via transport
        for src_labels, dst_array in (
            (sc_celltype, mapped_celltype),
            (sc_coarse, mapped_coarse),
            (sc_label, mapped_label),
        ):
            if src_labels is None or dst_array is None:
                continue
            uniq, inv = np.unique(src_labels, return_inverse=True)
            M = np.eye(len(uniq), dtype=np.float32)[inv]  # (n_sc, n_lab)
            scores = pi @ M  # (batch_n, n_lab)
            best = scores.argmax(axis=1)
            dst_array[idx] = uniq[best]
            if dst_array is mapped_celltype:
                row_sum = scores.sum(axis=1) + 1e-9
                conf = scores[np.arange(scores.shape[0]), best] / row_sum
                confidence[idx] = conf

        del pi, mp, sp_batch
        gc.collect()

    log("Building imputed AnnData ...")
    obs_out = adata_sp.obs.copy()
    obs_out["mapped_celltype"] = mapped_celltype
    if mapped_coarse is not None:
        obs_out["mapped_coarse_celltype"] = mapped_coarse
    if mapped_label is not None:
        obs_out["mapped_celltype_label"] = mapped_label
    obs_out["mapping_confidence"] = confidence

    var_out = pd.DataFrame(index=impute_genes)
    var_out["disease_gene"] = True

    adata_imp = AnnData(
        X=sparse.csr_matrix(imputed_full),
        obs=obs_out,
        var=var_out,
        obsm={k: v for k, v in adata_sp.obsm.items()},  # keep coords
    )
    out_path = OUT_DIR / "adata_imputed_disease_genes.h5ad"
    adata_imp.write_h5ad(out_path)
    log(f"Saved -> {out_path}  ({adata_imp.shape})")

    # Brief summary
    log("Mapping confidence stats:")
    log(f"  mean={confidence.mean():.3f} median={np.median(confidence):.3f} "
        f"q25={np.quantile(confidence,0.25):.3f} q75={np.quantile(confidence,0.75):.3f}")
    log("Mapped celltype counts (top 15):")
    log(pd.Series(mapped_celltype).value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
