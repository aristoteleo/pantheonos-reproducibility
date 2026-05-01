# ATAC vs RNA 3D divergence — exploratory follow-up

- Shared genes: **215**; disease panel: **22**.
- Per-cell panel divergence |ATAC_z - RNA_z|: mean=0.710, q90=1.178, max=3.511.

## A — Spatial divergence atlas

- See `A_panel_divergence_3d.png` (single map) and `A_module_divergence_3d_grid.png` (6 modules).

## B — Priming asymmetry

Top primed-only genes (high ATAC, low RNA):

  - **HRAS** (non-panel) primed_frac=0.089
  - **RPL5** (non-panel) primed_frac=0.090
  - **CHD4** (Chromatin (PCGC)) primed_frac=0.091
  - **PIGV** (non-panel) primed_frac=0.091
  - **MAP2K2** (non-panel) primed_frac=0.093
  - **TBX1** (Cardiac TF (CHD)) primed_frac=0.094
  - **CLUH** (non-panel) primed_frac=0.094
  - **MYRF** (non-panel) primed_frac=0.095

Top expressed-without-priming genes (high RNA, low ATAC):

  - **KDM5B** (non-panel) expr_only_frac=0.078
  - **TMEM260** (non-panel) expr_only_frac=0.080
  - **NSD2** (non-panel) expr_only_frac=0.082
  - **SMAD2** (non-panel) expr_only_frac=0.082
  - **SMC1A** (non-panel) expr_only_frac=0.082
  - **INVS** (non-panel) expr_only_frac=0.086
  - **KDM6A** (non-panel) expr_only_frac=0.088
  - **PTEN** (non-panel) expr_only_frac=0.089

## C — Within-cell-type Pearson r

Median within-celltype r per cell type:

  - VCM: median r = 0.065 (n=48595)
  - FB: median r = 0.055 (n=23620)
  - ACM: median r = 0.076 (n=9955)
  - Endothelial: median r = 0.067 (n=4254)
  - Endocardial: median r = 0.049 (n=3813)
  - MuralCells: median r = 0.039 (n=3483)
  - Epicardial: median r = 0.047 (n=1448)
  - LymphoidCells: median r = 0.052 (n=1183)
  - MyeloidCells: median r = 0.067 (n=1124)
  - SympatheticNeuron: median r = 0.058 (n=1049)
  - SchwannCells: median r = 0.034 (n=755)
  - CoreConductionCells: median r = 0.043 (n=292)
  - NC: median r = 0.051 (n=224)
  - TzConductionCells: median r = 0.055 (n=205)

Within-VCM divergent triptychs: DSP, ACTC1, MYL2, GATA4.

## D — Spatial spread (radius of gyration)

- Genes with ATAC spread > RNA spread: **115** / genes with ATAC < RNA: **98** (of 213 non-NaN).
- Median Δrg (ATAC − RNA): **10.77**.
- Median centroid shift (ATAC vs RNA): **358.75**.
