# Valve disease-trait enrichment — statistical report

## Question
Are valve cells (VIC + VEC + ncCM-AVC-like, n=3 278) enriched for the gene
panel of any disease trait, relative to non-valve cells?

## Method (one-line)
Per-cell trait score = mean log1p over the trait's gene panel; compared
between valve and non-valve via Mann–Whitney U (one-sided, "valve >
non-valve") with effect size AUC, against a permutation null of 2 000
size-matched random gene sets drawn from the 221-gene universe.
BH-FDR across 10 traits. **q_perm is the headline statistic.**

## Headline result

6 / 10 traits are significantly enriched in valve cells (q_perm < 0.05);
2 are strongly depleted (myocardial controls); 2 are not significant.

| Trait | k | AUC | log2FC | z_perm | q_perm | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Familial thoracic aortic aneurysm | 9 | **0.775** | +0.43 | +2.5 | 3.0e-3 | enriched |
| Valve defects | 74 | **0.757** | +0.22 | +7.6 | 1.2e-3 | enriched |
| AV septal defect | 23 | **0.726** | +0.21 | +2.5 | 7.5e-3 | enriched |
| Outflow tract malformation | 57 | **0.725** | +0.18 | +4.7 | 1.2e-3 | enriched |
| PCGC de novo variants | 61 | 0.631 | +0.09 | +4.1 | 1.2e-3 | enriched |
| Single ventricle disease | 17 | 0.625 | +0.13 | +1.7 | 4.7e-2 | borderline |
| Atrial septal defect | 88 | 0.551 | +0.03 | +6.3 | 1.2e-3 | weak (small effect, but reliable) |
| Ventricular septal defect | 34 | 0.475 | −0.02 | +1.6 | 7.1e-2 | n.s. |
| Dilated cardiomyopathy | 41 | **0.181** | −1.43 | −12.7 | 1.0 | depleted |
| Hypertrophic cardiomyopathy | 27 | **0.177** | −2.01 | −13.4 | 1.0 | depleted |

> [!NOTE]
> AUC = P(score(random valve cell) > score(random non-valve cell)).
> 0.5 = no enrichment; > 0.5 = enriched, < 0.5 = depleted.
> Sanity ✓: the two myocardial gene sets (HCM, DCM) are heavily depleted
> in valve cells exactly as expected; the test discriminates correctly.

## Robustness — valve vs fibroblast (the more honest test)

Comparing valve cells to other mesenchymal cells (Compact / Trabecular /
aFibro / adFibro / Proliferating vFibro / EPDC; n = 14 146) removes the
trivial "valve is not myocardium" effect.

| Trait | AUC vs fibro | Verdict |
|---|---:|---|
| Single ventricle disease | 0.673 | enriched in valve over fibro |
| AV septal defect | 0.668 | enriched |
| Outflow tract malformation | 0.605 | enriched |
| Atrial septal defect | 0.598 | mildly enriched |
| Valve defects | 0.598 | mildly enriched |
| Ventricular septal defect | 0.594 | mildly enriched |
| PCGC de novo variants | 0.539 | barely enriched |
| HCM | 0.494 | n.s. |
| DCM | 0.429 | depleted vs fibro |
| **Familial TAA** | **0.414** | **depleted vs fibro — NOT valve-specific** |

> [!IMPORTANT]
> The TAA panel (ACTA2, MYH11, FBN1, COL3A1, …) is a **generic
> mesenchymal/aortopathy signature shared by fibroblasts**, not unique to
> valve. The strong primary AUC was driven by the ~80% of non-valve cells
> that are myocytes. Fibroblasts actually score *higher* than valve
> cells for this panel.

## Per-subtype breakdown — which valve cells drive each signal?

Top-3 enriched traits per subtype (vs all non-valve, MWU):

| Subtype (n) | Top trait | AUC | 2nd | 3rd |
|---|---|---:|---|---|
| **VIC** (2 372) | Familial TAA | 0.826 | Valve defects (0.764) | OFT malformation (0.753) |
| **VEC** (720) | AV septal defect | 0.799 | Valve defects (0.781) | Familial TAA (0.694) |
| **ncCM-AVC-like** (186) | Single ventricle | 0.651 | VSD (0.646) | ASD (0.608) |

### Biological reading
- **VIC** (the bulk of valve mesenchyme) drives the **connective-tissue /
  ECM signal**: aortopathy genes, valve-defect genes, OFT-cushion genes
  — all converging on the EMT → cushion-mesenchyme program shared by
  semilunar valves and the proximal aorta.
- **VEC** drives the **endocardial-cushion signal**: AV septal defect
  and valve defects together — exactly the AVC-cushion program that
  gives rise to the AV valves and inferior atrial septum.
- **ncCM-AVC-like** drives the **septal-CHD signal**: it's a
  myocardial population at the AV junction, so its top hits are the
  septal/CM defects (single ventricle, VSD, ASD), not connective-tissue
  traits. This is the *only* valve-region cell type with any HCM/DCM-
  like signal.

## Top biological takeaways

1. **Valve_defects gene panel really is enriched in valve cells**
   (q_perm = 1.2e-3, AUC = 0.76). The signal survives correction
   against random gene panels and against fibroblast-only comparisons,
   so this is not an artifact of "valve cells just express more genes"
   or "valve cells aren't myocytes".
2. **AV septal defect → AV-canal valve cells**. The strongest valve-
   localised CHD trait. Driven especially by VEC (AUC=0.80), consistent
   with the well-established shared developmental origin of AV valves
   and septum primum from the AV cushions.
3. **Outflow tract malformation co-enriches with valve cells**. This
   reflects shared developmental cushion biology between OFT (semilunar
   valves) and AV cushions — both derive from EMT-derived mesenchyme.
4. **Familial TAA is mesenchymal, not valve-specific**. Useful caveat:
   when you see enrichment vs all-non-valve, always ask "is this
   valve-specific or just non-myocardial?" — only the fibroblast
   comparator answers that.
5. **HCM/DCM panels are correctly silent in valve cells**, validating
   the test pipeline.

## Caveats
- Imputed expression (`adata_imputed_disease_genes.h5ad`) inherits scVI-
  style smoothing → cell–cell correlations would inflate raw MWU
  p-values. The permutation null reuses the exact same matrix, so the
  null distribution carries the same correlation structure → q_perm is
  robust to this. Trust q_perm over q_MWU (which hits zero
  almost always).
- 221-gene universe is small and CHD-biased. The permutation null is
  drawn from this biased pool, which is the *correct* null for "given
  this CHD-relevant gene panel, is valve enriched relative to other CHD-
  relevant gene panels?" — not "vs the entire genome".
- Single sample, 2 batches — no donor-level pseudoreplication possible;
  cells treated as independent within celltype groups.

## Files
- `valve_enrichment_stats.tsv` — primary table (10 traits × 16 columns)
- `valve_enrichment_by_subtype.tsv` — VIC / VEC / ncCM-AVC-like × 10 traits
- `valve_vs_fibroblast.tsv` — robustness comparator
- `fig1_bar_auc.png/.pdf` — traits ranked by AUC
- `fig2_volcano.png/.pdf` — z_perm vs log2FC volcano
- `fig3_top_distributions.png/.pdf` — score distributions for top-3 enriched + bottom-3 depleted
- Script: `PCW12_analysis/scripts/11_valve_enrichment_stats.py`
