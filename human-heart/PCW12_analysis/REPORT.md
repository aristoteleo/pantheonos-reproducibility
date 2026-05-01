# PCW12 Heart Disease Gene Spatial Analysis — Report

## Executive Summary

Visualized the 3D spatial expression patterns of **247 heart disease genes** in the developing human heart at PCW 12 by combining a 95k-cell scRNA-seq atlas (PCW 11–13) with a 100k-cell 3D MERFISH dataset. Of the 247 disease genes, **20** are present in the 238-gene MERFISH panel and were plotted directly; **221** are full-transcriptome only and were spatially imputed via MOSCOT optimal-transport mapping; **6** are absent from both datasets (FOXE3, FOXH1, GDF1, GET3, TAFAZZIN, ZIC3). Sanity checks confirm biologically expected localization (e.g. MYH11/ACTA2 focal in vascular smooth muscle; MYBPC3/TNNT2/TTN broad in ventricular cardiomyocytes; NOTCH1/JAG1/GJA1 in endothelial/endocardial cells).

## Inputs

| Dataset | File | Cells | Genes |
|---|---|---|---|
| scRNA-seq (PCW 11–13) | `data/all_healthy_RoundedPCW11-13.h5ad` | 95,684 | 31,019 |
| MERFISH 3D (downsampled) | `data/full_heart_final_aug2025_update_downsampled_100k.h5ad` | 100,000 | 238 |
| Disease gene list | `data/heart_disease_genes.tsv` | — | 247 unique |

Categories in the disease list: `CHD` (CHD subtypes: ASD, VSD, AVSD, valve, OFT, single ventricle), `CardiovascularDisease` (DCM, HCM, aortic aneurysm/dissection), `PCGC_DeNovoVariants`.

## Method

1. **Gene partitioning** ([01_partition_genes.py](scripts/01_partition_genes.py)) → `data/disease_genes_partition.tsv`:
   - 20 *direct* (in MERFISH panel)
   - 221 *impute* (in scRNA only)
   - 6 *missing*
2. **Direct visualization** ([02_direct_visualization.py](scripts/02_direct_visualization.py)) — PyVista off-screen 3D scatter on `obsm['X_spateo_update']`, colored by log1p expression.
3. **MOSCOT mapping** ([03_moscot_imputation.py](scripts/03_moscot_imputation.py)) — `MappingProblem(SC, SP).prepare(xy_callback="local-pca").solve(alpha=0, tau_a=1, tau_b=0.9)`; 5 spatial batches of 20k cells, mean ~130 s per batch on CPU. Cell-cycle genes excluded (229 mapping genes from 237 shared). Disease genes (221) imputed via scaled transport matrix; cell-type labels transferred via argmax of `pi @ one_hot`. JAX has no Metal/CUDA on this machine, so mapping ran on CPU.
4. **Imputed visualization** ([04_imputed_visualization.py](scripts/04_imputed_visualization.py)) — same renderer, plus per-cell-type and per-disease-category mean-expression heatmaps.

## Outputs

```
PCW12_analysis/
├── REPORT.md                                    (this file)
├── scripts/                                     5 .py
├── data/
│   ├── disease_genes_partition.tsv              247 rows
│   ├── highlight_genes.tsv                      37 curated
│   └── adata_imputed_disease_genes.h5ad         100k × 221, 274 MB
├── figures/
│   ├── direct/   20 PNGs + 15 GIFs + celltype overview
│   ├── imputed/  221 PNGs + 22 GIFs + mapped_celltype + confidence
│   └── heatmaps/
│       ├── direct_celltype_heatmap.png
│       ├── imputed_celltype_heatmap.png         (top-60 by variance)
│       └── imputed_disease_category_heatmap.png
└── logs/
```

Total: **247 PNGs, 37 GIFs, 3 heatmaps, 1 imputed h5ad**.

## Key Findings

**Direct genes (in MERFISH panel)** — see [direct_celltype_heatmap.png](figures/heatmaps/direct_celltype_heatmap.png):

| Gene | Disease | Where it localizes (MERFISH) |
|---|---|---|
| MYH7, MYH6, TTN, PLN, DES | DCM/HCM | broad ventricular CMs (vCM-LV/RV-Compact, Trabecular, Hybrid) ✓ |
| MYH11 | Aortic aneurysm | tightly focal — VSMC-rich region ✓ |
| MFAP5 | Aortic aneurysm | adFibro / VSMC ✓ |
| GJA1 | VSD / single-ventricle | LEC/BEC + atrial ncCM ✓ |
| NOTCH1 | Valve / OFT | Endocardial + LEC ✓ |
| JAG1 | OFT / Alagille-like | Neuronal + adFibro ✓ |
| TBX5, NKX2-5, HAND1, HAND2 | CHD TFs | CM clusters (atrial + ventricular trabecular) ✓ |
| PITX2 | CHD (left-right) | Epicardial / aFibro ✓ |
| NR2F2 | CHD | Pericyte / EPDC ✓ |
| FOXC1 | CHD | Endocardial / VEC ✓ |

**Imputed genes** — see [imputed_celltype_heatmap.png](figures/heatmaps/imputed_celltype_heatmap.png) (top 60 by variance):

A clean cardiomyocyte block (right half: MYBPC3, TNNT2, TPM1, MYL2, MYL3, ACTC1, ACTN2, NEXN, CSRP3, CRYAB, LDB3, FHOD3, DMD, RBM20, KLHL24, …) is up in *all* vCM populations — consistent with their role in cardiomyopathy. A second block (PRKAG2, ACTA2, ELN, FBN1, CORIN, NF1) is up in VSMC / aortic-vascular cells — consistent with thoracic-aortic-aneurysm genes.

**Disease-category × cell-type summary** — see [imputed_disease_category_heatmap.png](figures/heatmaps/imputed_disease_category_heatmap.png):

- `CardiovascularDisease` (DCM/HCM-dominant) is clearly up in vCM-LV/RV populations (Compact, Trabecular, Hybrid, RV-Compact) and down in non-CM compartments — biologically expected.
- `CHD` genes spread across vEndocardial, aEndocardial, VIC, VSMC, VEC, EPDC, adFibro — consistent with CHD affecting valve/septation/OFT lineages.
- `PCGC_DeNovoVariants` is broader (it overlaps both above sets) and is highest in VSMC, vEndocardial, Trabecular vFibro, and Neuronal compartments.

> [!NOTE]
> **MOSCOT mapping confidence** is moderate (mean 0.487, median 0.449, q25–q75: 0.351–0.594). This is typical for unbalanced OT mapping with 100k×95k pairs and a 229-gene shared cost — lower-confidence cells should be interpreted with care. The confidence map is in [_mapping_confidence.png](figures/imputed/_mapping_confidence.png).

> [!WARNING]
> **Imputed expression should not be over-interpreted at single-cell resolution.** Imputed values are weighted averages over scRNA-seq donors via the transport matrix; they recover *cell-type-level* spatial patterns well but smooth out single-cell variability.

## Verification

| Check | Result |
|---|---|
| Direct cardiomyopathy markers (MYH7/MYH6/TTN/DES) high in vCM clusters | ✓ |
| Direct smooth-muscle marker MYH11 focal in VSMC-rich region | ✓ |
| Imputed ACTA2 focal in vascular rim, depleted in central CM mass | ✓ |
| Imputed MYBPC3/TNNT2/GATA4 broadly distributed across heart | ✓ |
| Heatmap blocks for cardiomyocyte vs. vascular vs. valve genes | ✓ |
| 5/5 MOSCOT batches solved successfully (transport row sums ≈ 1) | ✓ |
| Visual verification of 9 representative figures via `observe_images` | ✓ |

## Reproduce

```bash
PY=/Users/weizexu/micromamba/envs/pantheon/bin/python
cd /Users/weizexu/Projects/pantheonos-reproducibility/human-heart
$PY PCW12_analysis/scripts/01_partition_genes.py
$PY PCW12_analysis/scripts/02_direct_visualization.py
$PY PCW12_analysis/scripts/03_moscot_imputation.py     # ~12 min CPU
$PY PCW12_analysis/scripts/04_imputed_visualization.py
```

Logs saved under `PCW12_analysis/logs/`.

## Next Steps (suggested)

1. **Try alternate gene symbols** for the 6 missing genes (e.g. TAFAZZIN→TAZ, ZIC3 likely a casing/symbol mismatch worth checking).
2. **Anatomical region annotation**: Project the 3D coords onto a coarse anatomical mask (LV/RV/atria/OFT) and re-summarize disease genes by region rather than by cell type.
3. **Trait-specific aggregation**: Replace the 3-way category collapse with finer trait labels (ASD, VSD, OFT, valve, single-ventricle, DCM, HCM, aortic) — the gene-set sizes per trait are large enough to support it.
4. **Compare to cell-type-specific differential expression** (DE in scRNA-seq) for the same genes, to flag mismatches that may be imputation artifacts.
