"""
Step 1: Prepare paired scRNA / scATAC AnnDatas for MOSCOT cross-modality
imputation onto MERFISH 3D coordinates.

Inputs:
  data/chd_enhancers_counts.h5ad     - cells x enhancers (CSR, ~95k x 108k)
  data/chd_enhancer_gene_map.tsv     - enhancer index -> TargetGene (108k rows)
  data/all_healthy_RoundedPCW11-13.h5ad - paired multiome scRNA-seq (95k x 31k)

Pairing: RNA.obs_names == ATAC.obs_names (verified, full intersection 95684).

Outputs:
  PCW12_analysis/data/for_atac_mapping_atac.h5ad
  PCW12_analysis/data/for_atac_mapping_rna.h5ad
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
from scipy import sparse

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
ATAC_PATH = ROOT / "data/chd_enhancers_counts.h5ad"
MAP_PATH = ROOT / "data/chd_enhancer_gene_map.tsv"
RNA_PATH = ROOT / "data/all_healthy_RoundedPCW11-13.h5ad"
OUT_DIR = ROOT / "PCW12_analysis/data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(m: str):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    # ------------------------------------------------------------------
    # 1. Load the enhancer count matrix
    # ------------------------------------------------------------------
    log(f"Loading ATAC enhancer counts: {ATAC_PATH}")
    atac_e = ad.read_h5ad(ATAC_PATH)
    log(f"  ATAC shape (cells x enhancers): {atac_e.shape}")
    log(f"  X type: {type(atac_e.X).__name__}, dtype: {atac_e.X.dtype}")

    # ------------------------------------------------------------------
    # 2. Load enhancer -> TargetGene map and aggregate by gene
    # ------------------------------------------------------------------
    log(f"Loading enhancer->gene map: {MAP_PATH}")
    emap = pd.read_csv(MAP_PATH, sep="\t")
    log(f"  enhancer map rows: {len(emap)}; columns: {list(emap.columns)}")
    assert {"name", "TargetGene"}.issubset(emap.columns)

    # var_names of ATAC are positional indices "0".."N-1". Confirm.
    if not (atac_e.var_names.astype(str)[:5] == np.array(["0", "1", "2", "3", "4"])).all():
        log("  WARNING: ATAC var_names do not look like positional indices; "
            "using row order from the map TSV directly.")
    if len(emap) != atac_e.n_vars:
        log(f"  WARNING: map rows ({len(emap)}) != ATAC n_vars ({atac_e.n_vars}); "
            "will match by row order up to min length.")

    n_match = min(len(emap), atac_e.n_vars)
    target_gene = emap["TargetGene"].astype(str).values[:n_match]

    # Build sparse aggregator A: (n_enh, n_gene) one-hot for TargetGene
    genes, gene_codes = np.unique(target_gene, return_inverse=True)
    log(f"  unique TargetGenes: {len(genes)}")

    # Drop rows with missing/empty gene labels
    keep_mask = np.array([
        bool(g) and g.lower() not in ("nan", "none", "")
        for g in genes
    ])
    if not keep_mask.all():
        log(f"  dropping {np.sum(~keep_mask)} blank/NaN gene labels")
    valid_gene_idx = np.where(keep_mask)[0]
    valid_gene_set = set(valid_gene_idx.tolist())
    enh_mask = np.array([c in valid_gene_set for c in gene_codes])
    enh_rows = np.arange(n_match)[enh_mask]
    enh_codes = gene_codes[enh_mask]
    # Re-index gene_codes to compact range over kept genes
    new_genes = genes[valid_gene_idx]
    code_map = {old: new for new, old in enumerate(valid_gene_idx)}
    enh_new_codes = np.array([code_map[c] for c in enh_codes], dtype=np.int32)

    # Sparse aggregator: shape (n_enh, n_gene)
    A = sparse.csr_matrix(
        (np.ones(len(enh_rows), dtype=np.float32),
         (enh_rows, enh_new_codes)),
        shape=(atac_e.n_vars, len(new_genes)),
    )
    log(f"  aggregator shape: {A.shape} ({A.nnz} nz)")

    # cells x genes = ATAC.X (cells x enh) @ A (enh x gene)
    log("  multiplying ATAC.X @ A to aggregate enhancer counts by gene ...")
    if sparse.issparse(atac_e.X):
        gene_counts = atac_e.X @ A          # CSR @ CSR -> CSR
    else:
        gene_counts = sparse.csr_matrix(np.asarray(atac_e.X, dtype=np.float32) @ A)
    if not sparse.isspmatrix_csr(gene_counts):
        gene_counts = gene_counts.tocsr()
    gene_counts = gene_counts.astype(np.float32)
    log(f"  gene-aggregated matrix: {gene_counts.shape} (nnz={gene_counts.nnz})")

    var = pd.DataFrame(index=pd.Index(new_genes, name="TargetGene"))
    atac_g = AnnData(X=gene_counts, obs=atac_e.obs.copy(), var=var)
    atac_g.obs_names = atac_e.obs_names
    log(f"  AnnData (cells x genes): {atac_g.shape}")

    # Free enhancer-level AnnData
    del atac_e

    # ------------------------------------------------------------------
    # 3. Normalize: normalize_total + log1p
    # ------------------------------------------------------------------
    log("Normalizing: normalize_total(target_sum=1e4) + log1p ...")
    sc.pp.normalize_total(atac_g, target_sum=1e4)
    sc.pp.log1p(atac_g)
    log(f"  X range after norm: {float(atac_g.X.min()):.3f}..{float(atac_g.X.max()):.3f}")

    # ------------------------------------------------------------------
    # 4. QC filter: keep cells with 10 <= n_genes_by_counts <= 200
    # NOTE: n_genes_by_counts is a count metric (number of nonzero
    # genes per cell); compute on the post-aggregation matrix BEFORE
    # filtering so the threshold reflects the gene-level matrix.
    # ------------------------------------------------------------------
    sc.pp.calculate_qc_metrics(atac_g, percent_top=None, log1p=False, inplace=True)
    n_genes = atac_g.obs["n_genes_by_counts"].values
    log(f"  n_genes_by_counts: min={n_genes.min()}, max={n_genes.max()}, "
        f"median={np.median(n_genes):.0f}")
    keep_cells = (n_genes >= 10) & (n_genes <= 200)
    log(f"  cells passing QC (10<=n_genes<=200): "
        f"{keep_cells.sum()}/{len(keep_cells)} "
        f"({100*keep_cells.mean():.1f}%)")
    atac_g = atac_g[keep_cells].copy()
    log(f"  ATAC AnnData after QC: {atac_g.shape}")

    # ------------------------------------------------------------------
    # 5. Load RNA, subset to paired cells (ATAC obs_names), align order
    # ------------------------------------------------------------------
    log(f"Loading scRNA-seq: {RNA_PATH}")
    rna = ad.read_h5ad(RNA_PATH)
    log(f"  RNA shape: {rna.shape}")

    pair_idx = rna.obs_names.intersection(atac_g.obs_names)
    log(f"  paired cells (RNA.obs_names ∩ ATAC.obs_names): {len(pair_idx)}")
    if len(pair_idx) == 0:
        raise RuntimeError("No paired cells found by obs_names; "
                           "fallback by ATAC_barcode not yet implemented.")

    # Use the ATAC ordering (post-QC) as the canonical paired order.
    pair_order = [c for c in atac_g.obs_names if c in set(pair_idx)]
    rna_paired = rna[pair_order].copy()
    atac_paired = atac_g[pair_order].copy()
    assert (rna_paired.obs_names == atac_paired.obs_names).all()
    log(f"  RNA paired:  {rna_paired.shape}")
    log(f"  ATAC paired: {atac_paired.shape}")

    # ------------------------------------------------------------------
    # 6. Save outputs (RNA = raw counts, ATAC = aggregated/normalized/QCed)
    # ------------------------------------------------------------------
    out_atac = OUT_DIR / "for_atac_mapping_atac.h5ad"
    out_rna = OUT_DIR / "for_atac_mapping_rna.h5ad"
    log(f"Writing {out_atac}")
    atac_paired.write_h5ad(out_atac)
    log(f"Writing {out_rna}")
    rna_paired.write_h5ad(out_rna)

    log("Step 1 complete.")
    log(f"  ATAC out: {atac_paired.shape}  ({out_atac})")
    log(f"  RNA  out: {rna_paired.shape}  ({out_rna})")
    log(f"  ATAC X dtype/sparsity: {atac_paired.X.dtype} "
        f"nnz={atac_paired.X.nnz if sparse.issparse(atac_paired.X) else 'dense'}")
    log(f"  RNA  X dtype: {rna_paired.X.dtype}")


if __name__ == "__main__":
    main()
