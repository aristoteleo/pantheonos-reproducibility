"""
Step 2: MOSCOT cross-modality imputation.

Build a transport matrix per SP batch using paired scRNA-seq -> MERFISH,
then reuse the same pi to project ATAC (gene-aggregated, normalized,
per-cell QC'ed) onto the spatial coordinates.

Inputs:
  PCW12_analysis/data/for_atac_mapping_rna.h5ad   - RNA raw counts
  PCW12_analysis/data/for_atac_mapping_atac.h5ad  - ATAC normalized
  data/full_heart_final_aug2025_update_downsampled_100k.h5ad  - 3D MERFISH

Output:
  PCW12_analysis/data/adata_imputed_atac.h5ad
    shape: (n_sp, n_atac_genes)
    obsm keeps X_spateo_update (3D coords)
    obs adds mapped_celltype / mapping_confidence
"""
from __future__ import annotations
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
from scipy import sparse

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
RNA_PATH = ROOT / "PCW12_analysis/data/for_atac_mapping_rna.h5ad"
ATAC_PATH = ROOT / "PCW12_analysis/data/for_atac_mapping_atac.h5ad"
SP_PATH = ROOT / "data/full_heart_final_aug2025_update_downsampled_100k.h5ad"
OUT_DIR = ROOT / "PCW12_analysis/data"
LOG_DIR = ROOT / "PCW12_analysis/logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 20000
RANDOM_STATE = 42

# Cell-cycle genes (same as 03_moscot_imputation.py)
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


def log(m: str):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    # --- Load paired SC RNA + ATAC ---
    log(f"Loading paired RNA: {RNA_PATH}")
    rna = ad.read_h5ad(RNA_PATH)
    log(f"  RNA: {rna.shape} dtype={rna.X.dtype}")
    log(f"Loading paired ATAC: {ATAC_PATH}")
    atac = ad.read_h5ad(ATAC_PATH)
    log(f"  ATAC: {atac.shape} dtype={atac.X.dtype}")
    assert (rna.obs_names == atac.obs_names).all(), \
        "RNA and ATAC must be in identical cell order"

    # --- Load MERFISH SP ---
    log(f"Loading MERFISH SP: {SP_PATH}")
    sp = ad.read_h5ad(SP_PATH)
    log(f"  SP: {sp.shape} dtype={sp.X.dtype}, obsm keys={list(sp.obsm.keys())}")

    # --- Cast & normalize RNA (raw -> log1p) ---
    if rna.X.dtype != np.float32:
        log("Casting RNA.X -> float32 ...")
        rna.X = rna.X.astype(np.float32)
    log("Normalizing RNA: normalize_total(target_sum=1e4) + log1p ...")
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    log(f"  RNA X range: {float(rna.X.min()):.2f}..{float(rna.X.max()):.2f}")
    log(f"  SP  X range: {float(sp.X.min()):.2f}..{float(sp.X.max()):.2f}")

    # --- Shared genes for OT cost (drop CC genes) ---
    shared = rna.var_names.intersection(sp.var_names)
    map_genes = [g for g in shared if g.upper() not in CC_GENES]
    log(f"Shared genes (RNA ∩ SP): {len(shared)}; "
        f"after CC-gene drop: {len(map_genes)}")

    rna_map = rna[:, map_genes].copy()
    sp_map = sp[:, map_genes].copy()
    if "spatial" not in sp_map.obsm:
        sp_map.obsm["spatial"] = np.asarray(
            sp_map.obsm["X_spateo_update"], dtype=np.float32).copy()
        log("  Copied obsm['X_spateo_update'] -> obsm['spatial'] for MOSCOT")

    # Pre-extract dense ATAC matrix (cells x genes) in paired-cell order
    atac_X = atac.X.toarray() if sparse.issparse(atac.X) else np.asarray(atac.X)
    atac_X = atac_X.astype(np.float32, copy=False)
    atac_genes = atac.var_names.tolist()
    log(f"  ATAC dense matrix: {atac_X.shape}, "
        f"max={atac_X.max():.2f} mean={atac_X.mean():.3f}")

    # Cell-type label transfer (from RNA paired cells)
    sc_celltype = rna.obs["celltype"].astype(str).values \
        if "celltype" in rna.obs else None
    sc_coarse = rna.obs["coarse_celltype"].astype(str).values \
        if "coarse_celltype" in rna.obs else None

    # Free big RNA reference (we keep rna_map for OT)
    del rna, atac
    gc.collect()

    # --- Batch loop over SP ---
    n_sp = sp_map.n_obs
    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(n_sp)
    batches = [perm[i:i + BATCH_SIZE] for i in range(0, n_sp, BATCH_SIZE)]
    log(f"SP batches: {len(batches)} of size up to {BATCH_SIZE}")

    imputed_full = np.zeros((n_sp, len(atac_genes)), dtype=np.float32)
    mapped_celltype = np.empty(n_sp, dtype=object) if sc_celltype is not None else None
    mapped_coarse = np.empty(n_sp, dtype=object) if sc_coarse is not None else None
    confidence = np.zeros(n_sp, dtype=np.float32)

    from moscot.problems.space import MappingProblem  # noqa: E402

    for bi, idx in enumerate(batches):
        log(f"=== Batch {bi+1}/{len(batches)}  size={len(idx)} ===")
        sp_batch = sp_map[idx].copy()
        mp = MappingProblem(rna_map, sp_batch)
        t0 = time.time()
        mp = mp.prepare(sc_attr=None, xy_callback="local-pca")
        log(f"  prepare done in {time.time()-t0:.1f}s")
        t0 = time.time()
        mp = mp.solve(alpha=0, tau_a=1, tau_b=0.9, device="cpu")
        log(f"  solve done in {time.time()-t0:.1f}s")

        sol = list(mp.solutions.values())[0]
        pi = np.array(sol.transport_matrix, dtype=np.float32, copy=True)
        pi *= float(pi.shape[0])  # rescale per skill
        log(f"  pi shape {pi.shape}, mean row sum: {pi.sum(axis=1).mean():.3f}")

        # Cross-modality imputation: pi (sp_batch x sc) @ atac_X (sc x gene)
        imputed_full[idx] = pi @ atac_X
        log(f"  ATAC imputed batch -> max={imputed_full[idx].max():.3f}, "
            f"mean={imputed_full[idx].mean():.3f}")

        # Cell-type label transfer
        for src_labels, dst_array in (
            (sc_celltype, mapped_celltype),
            (sc_coarse, mapped_coarse),
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

    # --- Build output AnnData (full SP rows, ATAC gene columns) ---
    log("Building output AnnData ...")
    obs_out = sp.obs.copy()
    if mapped_celltype is not None:
        obs_out["mapped_celltype"] = mapped_celltype
    if mapped_coarse is not None:
        obs_out["mapped_coarse_celltype"] = mapped_coarse
    obs_out["mapping_confidence"] = confidence

    var_out = pd.DataFrame(index=pd.Index(atac_genes, name="TargetGene"))
    var_out["modality"] = "atac"

    adata_imp = AnnData(
        X=sparse.csr_matrix(imputed_full),
        obs=obs_out,
        var=var_out,
        obsm={k: np.asarray(v) for k, v in sp.obsm.items()},
    )
    out_path = OUT_DIR / "adata_imputed_atac.h5ad"
    adata_imp.write_h5ad(out_path)
    log(f"Saved -> {out_path}  ({adata_imp.shape})")

    log("Mapping confidence stats:")
    log(f"  mean={confidence.mean():.3f} median={np.median(confidence):.3f} "
        f"q25={np.quantile(confidence,0.25):.3f} q75={np.quantile(confidence,0.75):.3f}")
    if mapped_celltype is not None:
        log("Mapped celltype counts (top 15):")
        log(pd.Series(mapped_celltype).value_counts().head(15).to_string())

    log("Step 2 complete.")


if __name__ == "__main__":
    main()
